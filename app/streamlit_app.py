"""
PitchSense — Streamlit dashboard (EA-FC-inspired).

Tabs
    1. Processing       — Pipeline run + ring-style progress indicator
    2. Match Centre     — Possession, timeline, team-DNA radar, territory control
    3. Pitch Analysis   — Interactive density heatmaps, formation scatter, region charts
    4. Outputs          — Generated output videos

Run:
    streamlit run app/streamlit_app.py
"""




from __future__ import annotations

import sys
import re
import json
from pathlib import Path
from collections import Counter

# Project-root import shim ----------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for _p in (_PROJECT_ROOT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st
import cv2
import numpy as np

from keypoint_pipeline import KeypointPipeline
from game_analyzer import GameAnalyzer
from frame_scrubber import (
    seek_to_frame, frame_to_rgb, regions_for_frame, find_closest_entry,
    get_video_meta,
)
from demo_loader import (
    DEMO_CACHE_DIR, DemoLoadError, list_demos, load_demo, validate_demo,
)
from config import MODEL_PATHS, OUTPUT_BASE, SUPPORTED_EXTENSIONS, TEST_DATA_DIR
import ui_theme as ui
import charts as ch

OUTPUT_VIDEOS = [
    ("final_draft.mp4",          "Final Draft", "Original + PiP top-down pitch map"),
    ("deep_analysis.mp4",        "Deep Analysis", "Segmentation overlay + ball"),
    ("full_pitch_debug_map.mp4", "Pitch Map",   "Top-down view + ball trail"),
    ("keypoint_annotations.mp4", "Keypoints",   "Skeleton over the original"),
]

SEG_CLASS_LABELS = {
    "18Yard":               "Penalty Area (18yd)",
    "18Yard Circle":        "Penalty Arc",
    "5Yard":                "Goal Area (6yd)",
    "Half Central Circle":  "Center Circle",
    "Half Field":           "Half Field",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def ensure_contrasting_bgr(t1_bgr, t2_bgr, min_dist: float = 100.0,
                           scale: float = 2.0):
    """Push two team colours apart when they are perceptually too close.

    Returns ``(t1_bgr, t2_bgr)`` (unchanged if either is None or they are
    already far enough apart). When the BGR euclidean distance is below
    ``min_dist`` the existing difference direction is amplified by
    ``scale``; if the two colours are nearly identical the second team is
    pushed to a complementary hue so the two teams stay distinguishable in
    the dashboard (hero bar, donut, radar, heatmaps, etc.).
    """
    if t1_bgr is None or t2_bgr is None:
        return t1_bgr, t2_bgr
    a = np.array(t1_bgr, dtype=float)
    b = np.array(t2_bgr, dtype=float)
    diff = b - a
    dist = float(np.linalg.norm(diff))
    if dist >= min_dist:
        return t1_bgr, t2_bgr
    if dist < 1e-3:
        # Nearly identical jerseys — rotate team1's hue by 180° and make
        # sure the result is reasonably saturated/bright.
        import colorsys
        b1, g1, r1 = t1_bgr
        h, s, v = colorsys.rgb_to_hsv(r1 / 255.0, g1 / 255.0, b1 / 255.0)
        r2, g2, b2 = colorsys.hsv_to_rgb((h + 0.5) % 1.0,
                                         max(s, 0.65), max(v, 0.55))
        return t1_bgr, (int(b2 * 255), int(g2 * 255), int(r2 * 255))
    new_b = np.clip(a + diff * scale, 0, 255)
    return t1_bgr, (int(new_b[0]), int(new_b[1]), int(new_b[2]))


def _ball_owner_by_frame(game_data) -> dict:
    """Cache and return a ``{frame_idx: owner_team}`` map built from
    ``GameAnalyzer.compute_ball_owner_per_frame``.

    Uses the SAME canonical-team + bbox-overlap + sticky carry logic as the
    Overview tab so the scrubber's per-frame "possession: Team X" line
    never disagrees with it. Recomputed only when ``game_data`` changes
    length (i.e. a new run / more frames appended).
    """
    n = len(game_data) if game_data is not None else 0
    cache = st.session_state.get("ball_owner_map")
    if cache is not None and cache.get("_n") == n:
        return cache["map"]
    owner_map: dict = {}
    if game_data:
        owners = GameAnalyzer.compute_ball_owner_per_frame(game_data)
        for entry, owner in zip(game_data, owners):
            fi = int(entry.get("frame_idx", -1))
            if fi >= 0 and owner is not None:
                owner_map[fi] = int(owner)
    st.session_state["ball_owner_map"] = {"_n": n, "map": owner_map}
    return owner_map


def _half_possession_summary(game_data: list[dict]) -> dict:
    """First/second half possession without computing hidden ball-rate KPIs."""
    if not game_data:
        return {"first": {}, "second": {}, "first_frames": 0, "second_frames": 0,
                "split_frame": 0}
    frame_indices = [int(e.get("frame_idx", i)) for i, e in enumerate(game_data)]
    if not frame_indices:
        return {"first": {}, "second": {}, "first_frames": 0, "second_frames": 0,
                "split_frame": 0}
    mid = (min(frame_indices) + max(frame_indices)) // 2
    first = [e for e, fi in zip(game_data, frame_indices) if fi <= mid]
    second = [e for e, fi in zip(game_data, frame_indices) if fi > mid]
    return {
        "first_frames": len(first),
        "second_frames": len(second),
        "split_frame": mid,
        "first": GameAnalyzer.compute_possession(first),
        "second": GameAnalyzer.compute_possession(second),
    }


# ─── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PitchSense — Match Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "analytics_data" not in st.session_state:
    st.session_state.analytics_data = None
if "game_data" not in st.session_state:
    st.session_state.game_data = None
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False
if "last_output_dir" not in st.session_state:
    st.session_state.last_output_dir = None
if "last_video_name" not in st.session_state:
    st.session_state.last_video_name = None
if "scrubber_frame_idx" not in st.session_state:
    st.session_state.scrubber_frame_idx = 1
if "scrubber_min_total" not in st.session_state:
    st.session_state.scrubber_min_total = 0
if "scrubber_fps" not in st.session_state:
    st.session_state.scrubber_fps = 30.0
if "stabilizer_report" not in st.session_state:
    st.session_state.stabilizer_report = None
if "mode" not in st.session_state:
    st.session_state.mode = "live"
if "selected_demo_id" not in st.session_state:
    st.session_state.selected_demo_id = None

# Inject theme CSS (re-injected every rerun)
st.markdown(ui.inject_css(st.session_state.theme), unsafe_allow_html=True)
# Inject Inter font
st.markdown(
    "<link rel='preconnect' href='https://fonts.googleapis.com'>"
    "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap' rel='stylesheet'>",
    unsafe_allow_html=True,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _remove_emojis(text: str) -> str:
    return re.sub(r"[^\w\s.\-]", "", text).strip()


def _render_demo_card(d, current_id: str | None) -> None:
    """Render a single demo card with summary KPIs and a Load button."""
    poss = d.meta.get("possession") or {}
    t1_pct = poss.get("team1_pct")
    t2_pct = poss.get("team2_pct")
    ball_pct = d.meta.get("ball_detection_pct")
    frames = d.meta.get("processed_frames") or d.total_frames

    badges = []
    if isinstance(t1_pct, (int, float)):
        badges.append(f"T1 {t1_pct:.0f}%")
    if isinstance(t2_pct, (int, float)):
        badges.append(f"T2 {t2_pct:.0f}%")
    if isinstance(ball_pct, (int, float)):
        badges.append(f"Ball {ball_pct:.0f}%")
    if frames:
        badges.append(f"{int(frames):,} fr")
    badge_line = " · ".join(badges)

    warnings = validate_demo(d.id)
    warn_html = ""
    if warnings:
        warn_html = (
            "<div class='ps-card__sub' style='margin-top:6px;color:#e0a23a;'>"
            "⚠ " + "; ".join(warnings[:3]) + "</div>"
        )

    is_current = (current_id == d.id)
    accent = "var(--ps-accent)" if is_current else "var(--ps-text)"

    st.markdown(
        f"""
        <div class='ps-card' style='margin-bottom:12px;border-left:3px solid {accent};'>
          <div style='font-weight:700;font-size:1rem;color:var(--ps-text);'>
            {_remove_emojis(d.title)[:80]}
          </div>
          <div class='ps-card__sub' style='margin-top:4px;'>
            {_remove_emojis(d.source_video)[:80]}
          </div>
          <div class='ps-card__sub' style='margin-top:6px;'>
            {badge_line}
          </div>
          {warn_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    btn_label = "✓ Loaded" if is_current else "Load Demo"
    btn_type = "secondary" if is_current else "primary"
    if st.button(btn_label, key=f"demo_load_{d.id}", type=btn_type,
                 use_container_width=True):
        try:
            loaded = load_demo(d.id)
        except DemoLoadError as exc:
            st.error(f"Could not load demo: {exc}")
            return
        st.session_state.game_data = loaded["game_data"]
        st.session_state.analytics_data = loaded.get("analytics_data") or []
        st.session_state.last_output_dir = loaded["last_output_dir"]
        st.session_state.last_video_name = loaded["info"].title
        st.session_state.processing_done = True
        st.session_state.selected_demo_id = d.id
        st.session_state.scrubber_fps = float(loaded.get("fps") or 30.0)
        st.session_state.scrubber_frame_idx = 1
        st.session_state.ball_owner_map = None
        st.session_state.stabilizer_report = None
        st.success(f"Loaded '{d.title}'. Open Match Centre, Pitch Analysis, etc.")
        st.rerun()


def get_video_files() -> list[Path]:
    if not TEST_DATA_DIR.exists():
        return []
    files: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(TEST_DATA_DIR.glob(f"*{ext}"))
    return sorted(files)


def get_total_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(total, 0)


def get_output_videos(output_dir: Path) -> list[tuple[Path, str, str]]:
    out: list[tuple[Path, str, str]] = []
    for filename, label, desc in OUTPUT_VIDEOS:
        p = output_dir / filename
        if p.exists():
            out.append((p, label, desc))
    return out


def get_scrubber_bounds(output_dir: Path) -> tuple[int, float, list[tuple[Path, str, int]]]:
    """Return (min_total_frames, fps, [(path, label, frames), ...]) for all
    output videos that exist. ``min_total_frames`` is the smallest frame
    count across displayed outputs, which is the safe upper bound for the
    scrubber slider.
    """
    fps = 30.0
    rows: list[tuple[Path, str, int]] = []
    for filename, label, _desc in OUTPUT_VIDEOS:
        p = output_dir / filename
        if not p.exists():
            continue
        meta = get_video_meta(str(p))
        if meta is None:
            continue
        f, total, _w, _h = meta
        if f > 0.0:
            fps = f  # they should all be the same; last one wins
        rows.append((p, label, int(total)))
    if not rows:
        return (0, fps, [])
    min_total = min(r[2] for r in rows)
    return (int(min_total), float(fps), rows)


def check_models() -> dict[str, bool]:
    return {name: Path(path).exists() for name, path in MODEL_PATHS.items()}


def _mode_badge_label(statuses: dict, mode: str) -> str:
    """Return the small status badge text under the header.

    Defined BEFORE its first use in the module so that downstream scripts
    (e.g. ``scripts/setup_demo_cache.py``) can ``from streamlit_app import
    MODEL_PATHS`` without triggering a NameError when the entire module
    is executed top-to-bottom during the import.
    """
    if mode == "demo":
        return "Demo Mode · pre-cached"
    if not all(statuses.values()):
        return "Models missing"
    return "All models loaded"


def build_seg_analytics(analytics_data: list) -> dict:
    cls_counter: Counter = Counter()
    frames_with_seg = 0
    for entry in analytics_data:
        segs = entry.get("segments", [])
        if segs:
            frames_with_seg += 1
        for seg in segs:
            cls_counter[seg.get("class_name", "unknown")] += 1
    total_det = sum(cls_counter.values())
    items = [
        {
            "key": k,
            "label": SEG_CLASS_LABELS.get(k, k),
            "count": v,
            "pct": round((v / total_det) * 100, 1) if total_det else 0.0,
        }
        for k, v in sorted(cls_counter.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return {
        "class_frequency": items,
        "frames_with_seg": frames_with_seg,
        "total_frames": len(analytics_data),
        "total_detections": total_det,
        "coverage_pct": round((frames_with_seg / max(len(analytics_data), 1)) * 100, 1),
    }


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
        <div style="display:flex;gap:10px;align-items:center;padding:6px 2px 14px;">
          <div style="width:42px;height:42px;border-radius:12px;
                      background:linear-gradient(135deg,var(--ps-accent-1),var(--ps-accent-2));
                      display:flex;align-items:center;justify-content:center;
                      font-size:22px;box-shadow:0 8px 22px -8px var(--ps-accent-2);">⚽</div>
          <div>
            <div style="font-weight:800;font-size:1.05rem;">PitchSense</div>
            <div style="font-size:0.75rem;color:var(--ps-text-dim);">Match Intelligence</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Theme switcher
    st.markdown("<div style='font-weight:600;font-size:0.82rem;margin-bottom:4px;color:var(--ps-text-dim);'>APPEARANCE</div>", unsafe_allow_html=True)
    theme_cols = st.columns(2)
    with theme_cols[0]:
        if st.button("🌙  Dark", use_container_width=True,
                     disabled=(st.session_state.theme == "dark")):
            st.session_state.theme = "dark"
            st.rerun()
    with theme_cols[1]:
        if st.button("☀️  Light", use_container_width=True,
                     disabled=(st.session_state.theme == "light")):
            st.session_state.theme = "light"
            st.rerun()

    st.markdown("---")

    # Mode switcher: Live = run the full pipeline; Demo = load a pre-cached
    # match from demo_cache/. Reset all session-state results when the
    # mode changes so the other tabs always render the right data.
    st.markdown("<div style='font-weight:600;font-size:0.82rem;margin-bottom:6px;color:var(--ps-text-dim);'>MODE</div>", unsafe_allow_html=True)
    mode_cols = st.columns(2)
    with mode_cols[0]:
        if st.button("▶  Live", use_container_width=True,
                     disabled=(st.session_state.mode == "live")):
            st.session_state.mode = "live"
            st.session_state.analytics_data = None
            st.session_state.game_data = None
            st.session_state.processing_done = False
            st.session_state.last_output_dir = None
            st.session_state.last_video_name = None
            st.session_state.selected_demo_id = None
            st.session_state.ball_owner_map = None
            st.session_state.stabilizer_report = None
            st.rerun()
    with mode_cols[1]:
        if st.button("🎬  Demo", use_container_width=True,
                     disabled=(st.session_state.mode == "demo")):
            st.session_state.mode = "demo"
            st.session_state.analytics_data = None
            st.session_state.game_data = None
            st.session_state.processing_done = False
            st.session_state.last_output_dir = None
            st.session_state.last_video_name = None
            st.session_state.selected_demo_id = None
            st.session_state.ball_owner_map = None
            st.session_state.stabilizer_report = None
            st.rerun()

    # Demo picker: only meaningful in Demo mode, but always visible so the
    # user can see which match is currently loaded.
    if st.session_state.mode == "demo":
        demos = list_demos()
        if demos:
            demo_options = {d.title: d.id for d in demos}
            current_id = st.session_state.selected_demo_id
            current_title = next(
                (t for t, i in demo_options.items() if i == current_id),
                None,
            )
            selected_title = st.selectbox(
                "Loaded Demo",
                options=list(demo_options.keys()),
                index=(list(demo_options.keys()).index(current_title)
                       if current_title in demo_options else 0),
                label_visibility="visible",
            )
            st.session_state.selected_demo_id = demo_options[selected_title]
        else:
            st.caption(f"No demos in `{DEMO_CACHE_DIR}`.")
            st.caption("Run `python scripts/setup_demo_cache.py` to populate.")

    st.markdown("---")

    # Model status
    st.markdown("<div style='font-weight:600;font-size:0.82rem;margin-bottom:6px;color:var(--ps-text-dim);'>SYSTEM STATUS</div>", unsafe_allow_html=True)
    statuses = check_models()
    if st.session_state.mode == "demo":
        # Demos don't need model weights; show them as ready so the badge
        # isn't red while the user is browsing demos.
        for k in statuses:
            statuses[k] = True
    rows_html = "".join(
        f'<div class="ps-status-row">'
        f'  <span class="ps-status-row__name">{name}</span>'
        f'  <span class="ps-status-row__state {"ok" if ok else "err"}">'
        f'{"● READY" if ok else "● MISSING"}</span>'
        f'</div>'
        for name, ok in statuses.items()
    )
    st.markdown(rows_html, unsafe_allow_html=True)

    st.markdown("---")

    # Processing options
    st.markdown("<div style='font-weight:600;font-size:0.82rem;margin-bottom:6px;color:var(--ps-text-dim);'>OPTIONS</div>", unsafe_allow_html=True)
    max_frames = st.number_input(
        "Max frames (0 = all)",
        min_value=0, max_value=100000, value=0, step=100,
        help="Limit processing for faster testing.",
    )
    enable_team_colors = st.checkbox("Team colour clustering", value=True)
    flip_projection_x = st.checkbox("Flip projection (X — long axis)", value=False,
                                    help="Mirror the long axis of the pitch if the camera is behind the opposite goal.")
    flip_projection_y = st.checkbox("Flip projection (Y — short axis)", value=False,
                                    help="Mirror the short axis if the ball/players appear on the wrong side of the top-down pitch canvas.")
    with st.expander("Advanced team calibration", expanded=False):
        team_calibration_text = st.text_area(
            "Calibration / overrides JSON",
            value="",
            height=120,
            placeholder='{"team1_bgr": [255, 0, 0], "team2_bgr": [0, 0, 255], "track_role_overrides": {"12": "gk"}}',
            help="Optional BGR seed colours and track/identity team or role overrides. Leave blank for automatic stabilization.",
        )


palette = ui.palette(st.session_state.theme)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;">
      <div>
        <h1 style="margin:0;font-size:2rem;letter-spacing:-0.02em;">PitchSense ⚽</h1>
        <p style="margin:0;color:var(--ps-text-dim);font-size:0.95rem;">
          Tactical computer-vision analytics — possession, heatmaps, formation DNA.
        </p>
      </div>
      <div class="ps-badge {'bad' if not all(statuses.values()) else ''}">
        <span class="dot"></span>
        {_mode_badge_label(statuses, st.session_state.mode)}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_processing, tab_match, tab_pitch, tab_players, tab_videos = st.tabs(
    ["Processing", "Match Centre", "Pitch Analysis", "Player Analytics", "Outputs"]
)






# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PROCESSING
# ═════════════════════════════════════════════════════════════════════════════
with tab_processing:
    if st.session_state.mode == "demo":
        # ─── Demo Mode: picker over pre-cached bundles ─────────────────────
        demos = list_demos()
        if not demos:
            st.markdown(
                f"""
                <div class='ps-card' style='text-align:center;padding:36px 24px;'>
                  <div style='font-size:3rem;margin-bottom:10px;'>🎬</div>
                  <h3 style='margin:0 0 8px;'>Demo cache is empty</h3>
                  <p style='color:var(--ps-text-dim);max-width:560px;margin:0 auto 18px;'>
                    Pre-cached output bundles live under
                    <code>{DEMO_CACHE_DIR}</code>. Populate them with:
                  </p>
                  <code style='background:var(--ps-bg-alt);padding:8px 12px;
                               border-radius:8px;display:inline-block;'>
                    python scripts/setup_demo_cache.py
                  </code>
                  <p style='color:var(--ps-text-dim);margin-top:18px;max-width:560px;margin-left:auto;margin-right:auto;'>
                    Or pass <code>--regenerate --source &lt;stem&gt;</code> to re-run the
                    pipeline on a specific video under <code>data/matches/</code>.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # No demos cached → stay in demo mode but stop so the live
            # picker (which is for the "live" branch) doesn't render.
            st.stop()
        elif not st.session_state.game_data:
            # No demo selected yet — show the picker and stop so the other
            # tabs don't render an empty-state for missing data.
            st.markdown(ui.card_open("Demo Matches",
                                     "Pick a pre-cached match — every tab renders from cached "
                                     "analytics without re-running inference."),
                        unsafe_allow_html=True)
            cols = st.columns(2)
            for idx, d in enumerate(demos):
                col = cols[idx % 2]
                with col:
                    _render_demo_card(d, current_id=st.session_state.selected_demo_id)
            st.markdown(ui.card_close(), unsafe_allow_html=True)
            st.stop()
        else:
            # A demo is already loaded — show a compact banner with a
            # "Switch demo" expander so the user can change matches without
            # losing context, then FALL THROUGH so Match Centre / Pitch
            # Analysis / Player Analytics / Outputs all render their
            # charts from the cached game_data.
            loaded_title = st.session_state.get("last_video_name") or "Demo"
            loaded_id = st.session_state.get("selected_demo_id")
            st.markdown(
                f"""
                <div class='ps-card' style='display:flex;align-items:center;
                                            justify-content:space-between;gap:12px;
                                            padding:14px 18px;'>
                  <div>
                    <div style='font-weight:700;color:var(--ps-text);'>
                      🎬 Loaded: {_remove_emojis(loaded_title)[:80]}
                    </div>
                    <div class='ps-card__sub' style='margin-top:4px;'>
                      Demo bundle · all tabs render from cached analytics.
                      Switch demo, or jump to another tab.
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Switch demo", expanded=False):
                cols = st.columns(2)
                for idx, d in enumerate(demos):
                    col = cols[idx % 2]
                    with col:
                        _render_demo_card(d, current_id=loaded_id)

    video_files = get_video_files()

    if not video_files:
        st.warning(f"No video files found in `{TEST_DATA_DIR}`. Drop a match into that folder and refresh.")
        st.stop()

    video_options = {_remove_emojis(f.stem) or f.name: str(f) for f in video_files}

    cfg_cols = st.columns([2.2, 1])
    with cfg_cols[0]:
        st.markdown(ui.card_open("Run Pipeline",
                                 "Select a match. The full analysis (keypoints → players → ball → segmentation → projection) will populate every tab."),
                    unsafe_allow_html=True)
        selected_name = st.selectbox("Match Video", options=list(video_options.keys()), index=0,
                                     label_visibility="visible")
        selected_path = video_options[selected_name]

        size_mb = Path(selected_path).stat().st_size / 1024 / 1024
        total_in_video = get_total_frames(selected_path)
        st.markdown(
            f"<div class='ps-card__sub' style='margin-top:6px;'>"
            f"<code style='background:var(--ps-bg-alt);padding:2px 8px;border-radius:6px;'>{Path(selected_path).name}</code>"
            f" · {size_mb:.0f} MB · {total_in_video:,} frames"
            f"</div>",
            unsafe_allow_html=True,
        )

        process_btn = st.button(
            "▶  Process Video",
            type="primary",
            use_container_width=True,
            disabled=not all(statuses.values()),
        )
        st.markdown(ui.card_close(), unsafe_allow_html=True)

    with cfg_cols[1]:
        st.markdown(
            """
            <div style="
                display:flex;
                flex-direction:column;
                gap:12px;
                margin-top:4px;
            ">

              <div style="display:flex;gap:10px;align-items:flex-start;">
                <div style="min-width:26px;height:26px;border-radius:50%;
                            background:var(--ps-accent-soft);
                            color:var(--ps-accent);
                            display:flex;align-items:center;justify-content:center;
                            font-weight:700;font-size:0.78rem;">1</div>
                <div>
                  <div style="color:var(--ps-text);font-weight:700;font-size:0.9rem;">
                    Pitch Segmentation
                  </div>
                  <div style="color:var(--ps-text-dim);font-size:0.82rem;line-height:1.45;">
                    Identifies the playable pitch region and separates it from the background.
                  </div>
                </div>
              </div>

              <div style="display:flex;gap:10px;align-items:flex-start;">
                <div style="min-width:26px;height:26px;border-radius:50%;
                            background:var(--ps-accent-soft);
                            color:var(--ps-accent);
                            display:flex;align-items:center;justify-content:center;
                            font-weight:700;font-size:0.78rem;">2</div>
                <div>
                  <div style="color:var(--ps-text);font-weight:700;font-size:0.9rem;">
                    Keypoint Detection
                  </div>
                  <div style="color:var(--ps-text-dim);font-size:0.82rem;line-height:1.45;">
                    Detects pitch landmarks and estimates the homography transform.
                  </div>
                </div>
              </div>

              <div style="display:flex;gap:10px;align-items:flex-start;">
                <div style="min-width:26px;height:26px;border-radius:50%;
                            background:var(--ps-accent-soft);
                            color:var(--ps-accent);
                            display:flex;align-items:center;justify-content:center;
                            font-weight:700;font-size:0.78rem;">3</div>
                <div>
                  <div style="color:var(--ps-text);font-weight:700;font-size:0.9rem;">
                    Player Tracking
                  </div>
                  <div style="color:var(--ps-text-dim);font-size:0.82rem;line-height:1.45;">
                    Tracks players with ByteTrack and groups teams using kit colours.
                  </div>
                </div>
              </div>

              <div style="display:flex;gap:10px;align-items:flex-start;">
                <div style="min-width:26px;height:26px;border-radius:50%;
                            background:var(--ps-accent-soft);
                            color:var(--ps-accent);
                            display:flex;align-items:center;justify-content:center;
                            font-weight:700;font-size:0.78rem;">4</div>
                <div>
                  <div style="color:var(--ps-text);font-weight:700;font-size:0.9rem;">
                    Ball Tracking
                  </div>
                  <div style="color:var(--ps-text-dim);font-size:0.82rem;line-height:1.45;">
                    Tracks the ball frame-by-frame and builds a smooth movement trail.
                  </div>
                </div>
              </div>

              <div style="display:flex;gap:10px;align-items:flex-start;">
                <div style="min-width:26px;height:26px;border-radius:50%;
                            background:var(--ps-accent-soft);
                            color:var(--ps-accent);
                            display:flex;align-items:center;justify-content:center;
                            font-weight:700;font-size:0.78rem;">5</div>
                <div>
                  <div style="color:var(--ps-text);font-weight:700;font-size:0.9rem;">
                    Top-down Projection
                  </div>
                  <div style="color:var(--ps-text-dim);font-size:0.82rem;line-height:1.45;">
                    Projects player and ball positions onto a tactical 2D pitch view.
                  </div>
                </div>
              </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(ui.card_close(), unsafe_allow_html=True)
    # Run pipeline ------------------------------------------------------------
    if process_btn:
        st.session_state.analytics_data = []
        st.session_state.game_data = []
        st.session_state.stabilizer_report = None
        st.session_state.ball_owner_map = None
        st.session_state.processing_done = False

        team_calibration = None
        if team_calibration_text.strip():
            try:
                team_calibration = json.loads(team_calibration_text)
            except Exception as exc:
                st.error(f"Invalid calibration JSON: {exc}")
                st.stop()

        safe_stem = "".join(c if c.isalnum() or c in " _-" else "_" for c in Path(selected_path).stem)
        output_dir = OUTPUT_BASE / f"processed_{safe_stem}"
        output_dir.mkdir(parents=True, exist_ok=True)
        st.session_state.last_output_dir = str(output_dir)
        st.session_state.last_video_name = selected_name

        total_frames = total_in_video
        if max_frames > 0:
            total_frames = min(total_frames, max_frames)

        # Stash the source FPS so the zone analytics / scrubber can use it
        # even though session_state doesn't keep the file handle.
        try:
            from frame_scrubber import get_video_meta as _gvm
            _meta = _gvm(selected_path)
            if _meta is not None:
                st.session_state.scrubber_fps = float(_meta[0])
        except Exception:
            pass

        # Ring progress placeholder (re-rendered every frame)
        ring_slot = st.empty()
        ring_slot.markdown(
            ui.ring_html(
                pct=0.0, label="Starting…",
                sublabel=f"Initialising pipeline on <code>{Path(selected_path).name}</code>",
                stat_pairs=[
                    ("Frames", f"0 / {total_frames:,}"),
                    ("Phase", "Starting"),
                    ("Homography", "—"),
                    ("Status", "Queued"),
                ],
            ),
            unsafe_allow_html=True,
        )

        try:
            pipeline = KeypointPipeline(
                keypoint_model_path=MODEL_PATHS["keypoint"],
                player_model_path=MODEL_PATHS["player"],
                seg_model_path=MODEL_PATHS["seg"],
                ball_model_path=MODEL_PATHS["ball"],
                enable_team_colors=enable_team_colors,
                flip_projection_x=flip_projection_x,
                flip_projection_y=flip_projection_y,
            )

            processed = 0
            ball_count = 0
            stabilizer_report = None
            for result in pipeline.process_video(
                source_video_path=selected_path,
                output_dir=str(output_dir),
                start_frame=0,
                max_frames=max_frames if max_frames > 0 else None,
                team_calibration=team_calibration,
            ):
                phase = result.get("phase", "render")
                if phase != "render":
                    phase_label = "Analysing" if phase == "analysis" else "Stabilising Teams"
                    phase_sub = "First pass: model inference and feature extraction" if phase == "analysis" else "Resolving stable identities, teams, and roles"
                    pct = float(result.get("progress_pct", 0.0))
                    phase_count = int(result.get("processed_count", processed))
                    stabilizer_report = result.get("stabilizer_report") or stabilizer_report
                    if phase_count % 2 == 0 or phase != "analysis":
                        ring_slot.markdown(
                            ui.ring_html(
                                pct=pct,
                                label=phase_label,
                                sublabel=phase_sub,
                                stat_pairs=[
                                    ("Frames", f"{phase_count:,} / {total_frames:,}"),
                                    ("Homography", "—"),
                                    ("Phase", phase_label),
                                    ("Status", "Running"),
                                ],
                            ),
                            unsafe_allow_html=True,
                        )
                    continue

                processed += 1
                has_ball = len(result.get("ball_xyxy", [])) > 0
                if has_ball:
                    ball_count += 1
                stabilizer_report = result.get("stabilizer_report") or stabilizer_report

                segs = result.get("processed_segments", [])
                if segs:
                    st.session_state.analytics_data.append({
                        "frame_idx": processed,
                        "segments": [
                            {"class_name": s.get("class_name"),
                             "confidence": float(s.get("confidence", 0.0))}
                            for s in segs
                        ],
                    })

                team_info = result.get("team_info")
                team_ids = team_info.get("team_ids") if team_info else None
                role_ids = team_info.get("role_ids") if team_info else None
                identity_ids = team_info.get("identity_ids") if team_info else None
                track_ids = result.get("track_ids", np.empty((0,), dtype=np.int32))
                track_quality = team_info.get("track_quality") if team_info else None
                team1_bgr = team_info.get("team1_bgr") if team_info else None
                team2_bgr = team_info.get("team2_bgr") if team_info else None
                st.session_state.game_data.append(
                    KeypointPipeline._build_game_data_entry(result, processed_count=processed)
                )

                pct = float(result.get("progress_pct", (processed / max(total_frames, 1)) * 100))
                h_mode = result.get("H_info", {}).get("mode", "N/A")

                # Update ring every 2 frames to avoid bottleneck
                if processed % 2 == 0 or processed == total_frames:
                    ring_slot.markdown(
                        ui.ring_html(
                            pct=pct,
                            label="Rendering",
                            sublabel=f"Final labels · frame <b>{processed:,}</b> of <b>{total_frames:,}</b> · "
                                     f"{'⚽ ball in view' if has_ball else 'no ball this frame'}",
                            stat_pairs=[
                                ("Frames", f"{processed:,} / {total_frames:,}"),
                                ("Phase", "Rendering"),
                                ("Homography", str(h_mode)),
                                ("Status", "Running"),
                            ],
                        ),
                        unsafe_allow_html=True,
                    )

            # Done ----------------------------------------------------------
            ring_slot.markdown(
                ui.ring_html(
                    pct=100.0, label="Complete",
                    sublabel=f"✅ Processed <b>{processed:,}</b> frames · "
                             f"Ball detected in <b>{ball_count:,}</b> ({(ball_count/max(processed,1)*100):.0f}%).",
                    stat_pairs=[
                        ("Frames", f"{processed:,} / {total_frames:,}"),
                        ("Phase", "Complete"),
                        ("Status", "Done"),
                        ("Homography", "Final"),
                    ],
                ),
                unsafe_allow_html=True,
            )
            st.session_state.processing_done = True
            st.session_state.stabilizer_report = stabilizer_report
            st.success("Processing finished — open **Match Centre**, **Pitch Analysis**, or **Outputs**.")
            if stabilizer_report:
                validation = stabilizer_report.get("validation", {})
                status = "passed" if validation.get("ok", False) else "failed"
                st.info(
                    f"Team stabilizer {status}: "
                    f"{stabilizer_report.get('identity_count', 0)} identities, "
                    f"{len(stabilizer_report.get('linked_fragments', []))} linked fragments, "
                    f"{len(stabilizer_report.get('goalkeeper_identities', []))} GKs, "
                    f"{len(stabilizer_report.get('referee_identities', []))} referees."
                )
            st.balloons()
        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.exception(e)

    elif not st.session_state.processing_done:
        st.markdown(
            """
            <div class='ps-card' style='text-align:center;padding:36px 24px;'>
              <div style='font-size:3rem;margin-bottom:10px;'>📊</div>
              <h3 style='margin:0 0 8px;'>Ready to analyse</h3>
              <p style='color:var(--ps-text-dim);max-width:560px;margin:0 auto;'>
                Pick a match above and hit <b>Process Video</b>. The ring progress will tick up live,
                then the Match Centre and Pitch Analysis tabs populate with interactive Plotly charts.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — MATCH CENTRE
# ═════════════════════════════════════════════════════════════════════════════
with tab_match:
    if not st.session_state.game_data:
        st.markdown(
            """
            <div class='ps-card' style='text-align:center;padding:40px 24px;'>
              <h3>No match loaded yet</h3>
              <p style='color:var(--ps-text-dim);'>Run the pipeline on the <b>Processing</b> tab first.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        game_data = st.session_state.game_data
        possession = GameAnalyzer.compute_possession(game_data, "Team 1", "Team 2")
        formation = GameAnalyzer.compute_formation(game_data)
        territory = GameAnalyzer.compute_territory(game_data)
        stats = GameAnalyzer.compute_match_stats(game_data)
        fps_val = float(st.session_state.get("scrubber_fps", 30.0))
        xg = GameAnalyzer.compute_expected_goals(game_data, fps=fps_val)
        win = GameAnalyzer.compute_win_probability_from_xg(xg)

        # Override the team1/team2 colours in the active palette with the
        # BGR centroids detected by the team_analyzer. The values are
        # EMA-blended across the match, so they reflect each team's
        # representative jersey colour rather than a single-frame outlier.
        detected_palette = dict(palette)
        t1_bgr = GameAnalyzer.dominant_team_bgr(game_data, team=0)
        t2_bgr = GameAnalyzer.dominant_team_bgr(game_data, team=1)
        # Boost separation when the detected jerseys are too similar so the
        # hero bar / donut / radar stay readable.
        t1_bgr, t2_bgr = ensure_contrasting_bgr(t1_bgr, t2_bgr)
        if t1_bgr is not None:
            detected_palette["team1"] = GameAnalyzer.bgr_to_hex(t1_bgr)
        if t2_bgr is not None:
            detected_palette["team2"] = GameAnalyzer.bgr_to_hex(t2_bgr)

        # Inject CSS variables so the hero bar / legend / cards use the
        # detected team colours for the duration of this render.
        st.markdown(
            f"<style>:root {{"
            f"  --ps-team1: {detected_palette['team1']};"
            f"  --ps-team2: {detected_palette['team2']};"
            f"}}</style>",
            unsafe_allow_html=True,
        )

        t1_pct = possession["team1_possession_pct"]
        t2_pct = possession["team2_possession_pct"]

        # Hero possession bar
        st.markdown(
            ui.hero_possession("Team 1", "Team 2", t1_pct, t2_pct,
                               possession["team1_frames"], possession["team2_frames"]),
            unsafe_allow_html=True,
        )

        # Charts -- top row
        row1 = st.columns(2)
        with row1[0]:
            st.markdown(ui.card_open("Possession Distribution",
                                     "Donut breakdown · nearest-player share"),
                        unsafe_allow_html=True)
            fig = ch.build_possession_donut(detected_palette, t1_pct, t2_pct)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(ui.card_close(), unsafe_allow_html=True)

        with row1[1]:
            st.markdown(ui.card_open("Possession Timeline",
                                     "Rolling 30-frame window · momentum"),
                        unsafe_allow_html=True)
            fig = ch.build_possession_timeline(detected_palette, game_data, window=30)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(ui.card_close(), unsafe_allow_html=True)

        # Charts -- bottom row
        row2 = st.columns(2)
        with row2[0]:
            st.markdown(ui.card_open("Team DNA Radar",
                                     "5-axis tactical profile (0 – 100)"),
                        unsafe_allow_html=True)
            fig = ch.build_team_radar(detected_palette, formation, stats, possession)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(ui.card_close(), unsafe_allow_html=True)

        with row2[1]:
            st.markdown(ui.card_open("Territory Control",
                                     "Zone dominance — hover any cell for details",
                                     chip=f"{territory['team1_total_presence'] + territory['team2_total_presence']:,} player-frames"),
                        unsafe_allow_html=True)
            fig = ch.build_territory_grid(detected_palette, territory)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(ui.card_close(), unsafe_allow_html=True)

        # Expected goals and xG-only win expectancy ---------------------------
        total_xg_events = xg.get("team1_event_count", 0) + xg.get("team2_event_count", 0)
        scoreline_text = ""
        if win.get("top_scorelines"):
            scoreline_text = " · likely scores " + ", ".join(
                f"{s['team1_goals']}-{s['team2_goals']} ({s['prob_pct']:.1f}%)"
                for s in win.get("top_scorelines", [])[:3]
            )
        st.markdown(
            ui.card_open(
                "Expected Goals And Win Expectancy",
                "Tracking-proxy estimate from ball/player positions; no explicit shot labels.",
                chip=f"{total_xg_events} events",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(ui.kpi_grid([
            ("Team 1 xG", f"{xg.get('team1_xg', 0.0):.2f}",
             f"{xg.get('team1_event_count', 0)} chance events"),
            ("Team 2 xG", f"{xg.get('team2_xg', 0.0):.2f}",
             f"{xg.get('team2_event_count', 0)} chance events"),
            ("Team 1 Win", f"{win.get('team1_win_pct', 0.0):.1f}%",
             "xG-only Poisson"),
            ("Draw", f"{win.get('draw_pct', 0.0):.1f}%",
             "xG-only Poisson"),
            ("Team 2 Win", f"{win.get('team2_win_pct', 0.0):.1f}%",
             "xG-only Poisson"),
        ]), unsafe_allow_html=True)
        st.markdown(
            f"<div class='ps-card__sub'>Method: {xg.get('model_version', 'tracking_proxy')}"
            f"{scoreline_text}</div>",
            unsafe_allow_html=True,
        )
        warnings = [w for w in (xg.get("warnings") or []) if "No explicit shot events" not in str(w)]
        if warnings:
            st.markdown(
                f"<div class='ps-card__sub'>Caveat: {warnings[0]}</div>",
                unsafe_allow_html=True,
            )
        st.markdown(ui.card_close(), unsafe_allow_html=True)

        xg_cols = st.columns(2)
        with xg_cols[0]:
            st.markdown(ui.card_open("Cumulative xG Timeline",
                                     "Chance-quality accumulation over processed frames"),
                        unsafe_allow_html=True)
            fig = ch.build_xg_timeline(detected_palette, xg)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(ui.card_close(), unsafe_allow_html=True)
        with xg_cols[1]:
            st.markdown(ui.card_open("xG Chance Map",
                                     "Marker size reflects estimated chance quality"),
                        unsafe_allow_html=True)
            fig = ch.build_xg_chance_map(detected_palette, xg)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(ui.card_close(), unsafe_allow_html=True)

        # Attacking direction -------------------------------------------------
        direction_info = GameAnalyzer.infer_attacking_direction(game_data) or {}
        if direction_info.get("team1_attacks"):
            t1_dir = direction_info["team1_attacks"]
            t2_dir = direction_info["team2_attacks"]
            source = direction_info["source"]
            conf_pct = int(round(direction_info.get("confidence", 0.0) * 100))
            if source == "gk":
                t1_box = direction_info.get("team1_gk_box")
                t2_box = direction_info.get("team2_gk_box")
                t1_frac = direction_info.get("team1_gk_box_frac")
                t2_frac = direction_info.get("team2_gk_box_frac")
                caption = (f"Inferred from goalkeeper position · "
                           f"GK X = T1 {direction_info.get('team1_gk_x', 'n/a')} m, "
                           f"T2 {direction_info.get('team2_gk_x', 'n/a')} m · "
                           f"box side = T1 {t1_box or 'n/a'} ({int(round((t1_frac or 0) * 100))}%), "
                           f"T2 {t2_box or 'n/a'} ({int(round((t2_frac or 0) * 100))}%) · "
                           f"confidence {conf_pct}%")
            elif source == "mean_x":
                caption = (f"Inferred from team centroid (no GK detected) · "
                           f"mean X = T1 {direction_info.get('team1_mean_x', 'n/a')} m, "
                           f"T2 {direction_info.get('team2_mean_x', 'n/a')} m · "
                           f"confidence {conf_pct}%")
            else:
                caption = "Direction could not be inferred"
            st.markdown(ui.card_open("Attacking Direction",
                                     "Each team's attacking axis inferred from player positions",
                                     chip=caption),
                        unsafe_allow_html=True)
            fig = ch.build_attacking_direction_diagram(
                detected_palette, direction_info,
                detected_palette["team1"], detected_palette["team2"],
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f"<div class='ps-card__sub'>"
                f"Team 1 attacks <b>{'right →' if t1_dir == 'right' else '← left'}</b> · "
                f"Team 2 attacks <b>{'right →' if t2_dir == 'right' else '← left'}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(ui.card_close(), unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — PITCH ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab_pitch:
    has_game = bool(st.session_state.game_data)
    has_seg = bool(st.session_state.analytics_data)
    if not has_game and not has_seg:
        st.markdown(
            """
            <div class='ps-card' style='text-align:center;padding:40px 24px;'>
              <h3>No pitch data yet</h3>
              <p style='color:var(--ps-text-dim);'>Process a video to view interactive heatmaps and pitch analytics.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Same detected-colour override as the Match Centre tab so the
        # heatmaps / scatter / KPIs use the live team colours.
        detected_palette = dict(palette)
        t1_bgr = GameAnalyzer.dominant_team_bgr(st.session_state.game_data, team=0)
        t2_bgr = GameAnalyzer.dominant_team_bgr(st.session_state.game_data, team=1)
        t1_bgr, t2_bgr = ensure_contrasting_bgr(t1_bgr, t2_bgr)
        if t1_bgr is not None:
            detected_palette["team1"] = GameAnalyzer.bgr_to_hex(t1_bgr)
        if t2_bgr is not None:
            detected_palette["team2"] = GameAnalyzer.bgr_to_hex(t2_bgr)

        # Density heatmaps -----------------------------------------------------
        if has_game:
            heat_summary = GameAnalyzer.compute_heatmaps(st.session_state.game_data)
            heat_cols = st.columns(2)
            with heat_cols[0]:
                st.markdown(ui.card_open("Team 1 Density Heatmap",
                                         "Hover any cell — sample count, pitch third, lateral band",
                                         chip=f"{heat_summary['team1_count']:,} samples"),
                            unsafe_allow_html=True)
                fig = ch.build_density_heatmap(detected_palette, st.session_state.game_data,
                                               team_id=0, name="Team 1")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown(ui.card_close(), unsafe_allow_html=True)
            with heat_cols[1]:
                st.markdown(ui.card_open("Team 2 Density Heatmap",
                                         "Hover any cell — sample count, pitch third, lateral band",
                                         chip=f"{heat_summary['team2_count']:,} samples"),
                            unsafe_allow_html=True)
                fig = ch.build_density_heatmap(detected_palette, st.session_state.game_data,
                                               team_id=1, name="Team 2")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown(ui.card_close(), unsafe_allow_html=True)

            # Combined positioning ---------------------------------------------
            st.markdown(ui.card_open("Combined Positioning & Ball Trail",
                                     "Both teams sampled across the match + the ball trajectory"),
                        unsafe_allow_html=True)
            fig = ch.build_formation_scatter(detected_palette, st.session_state.game_data)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(ui.card_close(), unsafe_allow_html=True)

            # Formation KPI cards ----------------------------------------------
            formation = GameAnalyzer.compute_formation(st.session_state.game_data)
            form_kpi = ui.kpi_grid([
                ("T1 Avg Spread",        f"{formation['team1_avg_spread']:.1f} m",  "team compactness"),
                ("T2 Avg Spread",        f"{formation['team2_avg_spread']:.1f} m",  "team compactness"),
            ])
            st.markdown(form_kpi, unsafe_allow_html=True)

        # Segmentation analytics ----------------------------------------------
        if has_seg:
            analytics = build_seg_analytics(st.session_state.analytics_data)

            seg_kpi = ui.kpi_grid([
                ("Frames w/ Segments",   f"{analytics['frames_with_seg']:,}",  "with pitch regions"),
                ("Segmentation Coverage", f"{analytics['coverage_pct']:.1f}%",  "of all frames"),
                ("Region Detections",    f"{analytics['total_detections']:,}",  "across all regions"),
            ])
            st.markdown(seg_kpi, unsafe_allow_html=True)

            seg_cols = st.columns(2)
            with seg_cols[0]:
                st.markdown(ui.card_open("Region Detection Pie",
                                         "Pitch zones recognised by the segmentation model"),
                            unsafe_allow_html=True)
                fig = ch.build_region_pie(palette, analytics["class_frequency"])
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown(ui.card_close(), unsafe_allow_html=True)
            with seg_cols[1]:
                st.markdown(ui.card_open("Region Detection Counts",
                                         "Per-region detection volume"),
                            unsafe_allow_html=True)
                fig = ch.build_region_bar(palette, analytics["class_frequency"])
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown(ui.card_close(), unsafe_allow_html=True)

            # --- Deeper pitch-zones analytics ------------------------------
            zone_summary = GameAnalyzer.compute_zone_summary(
                st.session_state.analytics_data,
                fps=float(st.session_state.get("scrubber_fps", 30.0)),
            )
            zone_timeline = GameAnalyzer.compute_zone_timeline(
                st.session_state.analytics_data, window=100,
            )

            most_label = SEG_CLASS_LABELS.get(
                zone_summary.get("most_detected") or "",
                zone_summary.get("most_detected") or "—",
            )
            zone_kpi = ui.kpi_grid([
                ("Most-Detected Region", most_label,
                 f"{zone_summary.get('most_detected_count', 0):,} detections"),
                ("Half Field — Left",
                 f"{zone_summary.get('half_field_left_pct', 0.0):.1f}%",
                 "of half-field frames"),
                ("Half Field — Right",
                 f"{zone_summary.get('half_field_right_pct', 0.0):.1f}%",
                 "of half-field frames"),
                ("Zone Coverage",
                 f"{analytics['coverage_pct']:.1f}%",
                 "frames with any region"),
            ])
            st.markdown(zone_kpi, unsafe_allow_html=True)

            zone_cols = st.columns(2)
            with zone_cols[0]:
                st.markdown(ui.card_open("Time in Zone",
                                         "Estimated seconds per pitch region"),
                            unsafe_allow_html=True)
                fig = ch.build_zone_time_bar(palette, zone_summary)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown(ui.card_close(), unsafe_allow_html=True)
            with zone_cols[1]:
                st.markdown(ui.card_open("Region Detection Timeline",
                                         "Rolling 100-frame window — detections per region"),
                            unsafe_allow_html=True)
                fig = ch.build_zone_timeline(palette, zone_timeline)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown(ui.card_close(), unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — PLAYER ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
with tab_players:
    if not st.session_state.game_data:
        st.markdown(
            """
            <div class='ps-card' style='text-align:center;padding:40px 24px;'>
              <h3>No player data yet</h3>
              <p style='color:var(--ps-text-dim);'>Process a video to view per-player and tactical analytics.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        gd = st.session_state.game_data
        fps_val = float(st.session_state.get("scrubber_fps", 30.0))

        # Same detected-colour override as the other tabs so the new charts
        # use the live team colours.
        detected_palette = dict(palette)
        t1_bgr = GameAnalyzer.dominant_team_bgr(gd, team=0)
        t2_bgr = GameAnalyzer.dominant_team_bgr(gd, team=1)
        t1_bgr, t2_bgr = ensure_contrasting_bgr(t1_bgr, t2_bgr)
        if t1_bgr is not None:
            detected_palette["team1"] = GameAnalyzer.bgr_to_hex(t1_bgr)
        if t2_bgr is not None:
            detected_palette["team2"] = GameAnalyzer.bgr_to_hex(t2_bgr)
        st.markdown(
            f"<style>:root {{"
            f"  --ps-team1: {detected_palette['team1']};"
            f"  --ps-team2: {detected_palette['team2']};"
            f"}}</style>",
            unsafe_allow_html=True,
        )

        profiles = GameAnalyzer.compute_player_profiles(gd, fps=fps_val)
        profile_rows = profiles.get("profiles", [])
        network = GameAnalyzer.compute_passing_network(gd)
        pressing = GameAnalyzer.compute_pressing_timeline(gd, window=30)
        setpieces = GameAnalyzer.compute_set_pieces(gd, fps=fps_val)
        halves = _half_possession_summary(gd)

        # --- KPI strip -------------------------------------------------------
        total_passes = (network.get("total_passes_team1", 0)
                        + network.get("total_passes_team2", 0))
        total_dist = sum(float(p.get("distance_m", 0.0)) for p in profile_rows)
        total_setpieces = setpieces.get("total", 0)
        kpi = ui.kpi_grid([
            ("Tracked Players", f"{len(profile_rows)}",
             "with team assignment"),
            ("Distance Covered", f"{total_dist / 1000:.1f} km",
             "cumulative across all tracked players"),
            ("Passes Detected", f"{total_passes:,}",
             f"T1 {network.get('total_passes_team1', 0)} · "
             f"T2 {network.get('total_passes_team2', 0)}"),
            ("Set Pieces",      f"{total_setpieces}",
             f"corners {setpieces['counts'].get('corner', 0)} · "
             f"goal kicks {setpieces['counts'].get('goal_kick', 0)} · "
             f"FK-danger {setpieces['counts'].get('free_kick_dangerous', 0)}"),
        ])
        st.markdown(kpi, unsafe_allow_html=True)

        # --- Passing networks by team (edges colored by pitch third) --------
        st.markdown("#### Passing Networks By Team")
        net_cols = st.columns(2)
        for col, team_key, team_label, team_color in [
            (net_cols[0], "team1", "Team 1", detected_palette["team1"]),
            (net_cols[1], "team2", "Team 2", detected_palette["team2"]),
        ]:
            with col:
                st.markdown(
                    ui.card_open(
                        f"{team_label} Passing Network",
                        "Directed player→player passes; edge color encodes the pitch third where each pass occurred",
                        chip="color = pitch third",
                    ),
                    unsafe_allow_html=True,
                )
                fig = ch.build_passing_network_by_thirds(
                    detected_palette, network.get(team_key, {}),
                    team_label, team_color,
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown(ui.card_close(), unsafe_allow_html=True)

        # --- Player profile regions ----------------------------------------
        st.markdown("#### Player Profiles By Team And Pitch Third")
        dist_cols = st.columns(2)
        for col, team_id, team_label in [
            (dist_cols[0], 0, "Team 1"),
            (dist_cols[1], 1, "Team 2"),
        ]:
            with col:
                st.markdown(ui.card_open(
                    f"{team_label} Distance By Region",
                    "Distance weighted by each player's time in Defensive, Middle, and Attacking thirds"),
                    unsafe_allow_html=True)
                fig = ch.build_player_region_distance_bar(
                    detected_palette, profile_rows, team_id, team_label,
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown(ui.card_close(), unsafe_allow_html=True)

        # --- Pressing by pitch third ----------------------------------------
        press_cols = st.columns(2)
        for col, team_id, team_label in [
            (press_cols[0], 0, "Team 1"),
            (press_cols[1], 1, "Team 2"),
        ]:
            with col:
                st.markdown(ui.card_open(
                    f"{team_label} Pressing By Region",
                    "Rolling nearest-opponent distance split by the ball's pitch third"),
                    unsafe_allow_html=True)
                fig = ch.build_team_pressing_by_region_timeline(
                    detected_palette, pressing, gd, team_id, team_label,
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown(ui.card_close(), unsafe_allow_html=True)

        # --- Halves comparison ---------------------------------------------
        st.markdown(ui.card_open("First Half vs Second Half Possession",
                                 f"Split at frame {halves.get('split_frame', 0):,}"),
                    unsafe_allow_html=True)
        fp = halves.get("first", {})
        sp = halves.get("second", {})
        st.markdown(ui.kpi_grid([
            ("1H Team 1", f"{fp.get('team1_possession_pct', 0):.0f}%",
             f"{halves.get('first_frames', 0):,} frames"),
            ("1H Team 2", f"{fp.get('team2_possession_pct', 0):.0f}%",
             f"{halves.get('first_frames', 0):,} frames"),
            ("2H Team 1", f"{sp.get('team1_possession_pct', 0):.0f}%",
             f"{halves.get('second_frames', 0):,} frames"),
            ("2H Team 2", f"{sp.get('team2_possession_pct', 0):.0f}%",
             f"{halves.get('second_frames', 0):,} frames"),
        ]), unsafe_allow_html=True)
        st.markdown(ui.card_close(), unsafe_allow_html=True)

        # --- Set-piece summary ----------------------------------------------
        sp_cols = st.columns(4)
        sp_kpis = [
            ("Corners",            setpieces["counts"].get("corner", 0),
             "ball-stationary near flag"),
            ("Goal Kicks",         setpieces["counts"].get("goal_kick", 0),
             "ball-stationary in 6-yard box"),
            ("Dangerous Free Kicks", setpieces["counts"].get("free_kick_dangerous", 0),
             "ball-stationary in opp box"),
            ("Other Stoppages",    setpieces["counts"].get("other", 0),
             "ball-stationary elsewhere"),
        ]
        for c, (label, val, hint) in zip(sp_cols, sp_kpis):
            with c:
                st.markdown(ui.kpi_card(label, f"{val}", hint),
                            unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — OUTPUTS
# ═════════════════════════════════════════════════════════════════════════════
with tab_videos:



    out_dir_str = st.session_state.last_output_dir
    if not out_dir_str:
        st.markdown(
            """
            <div class='ps-card' style='text-align:center;padding:40px 24px;'>
              <h3>No outputs yet</h3>
              <p style='color:var(--ps-text-dim);'>Process a video to see the generated outputs here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        out_dir = Path(out_dir_str)
        outputs = get_output_videos(out_dir)
        if not outputs:
            st.warning("No output videos were generated.")
        else:
            st.markdown(
                f"<p class='ps-card__sub' style='margin-bottom:12px;'>"
                f"Saved to <code style='background:var(--ps-bg-alt);padding:2px 8px;border-radius:6px;'>{out_dir}</code></p>",
                unsafe_allow_html=True,
            )
            for i in range(0, len(outputs), 2):
                cols = st.columns(2)
                for j in range(2):
                    idx = i + j
                    if idx >= len(outputs):
                        continue
                    path, label, desc = outputs[idx]
                    with cols[j]:
                        st.markdown(ui.card_open(label, desc), unsafe_allow_html=True)
                        st.video(str(path))
                        st.markdown(ui.card_close(), unsafe_allow_html=True)

            # ─── Frame Inspector ────────────────────────────────────────────
            min_total, fps, frame_rows = get_scrubber_bounds(out_dir)
            if min_total > 0 and frame_rows:
                # Clamp the slider into the valid range whenever the source
                # match changes (or first load).
                if st.session_state.scrubber_frame_idx > min_total:
                    st.session_state.scrubber_frame_idx = min_total
                if st.session_state.scrubber_frame_idx < 1:
                    st.session_state.scrubber_frame_idx = 1

                st.markdown(
                    ui.card_open("Frame Inspector",
                                 "Pick a frame — preview all available output videos at that "
                                 "timestamp, with match context.",
                                 chip=f"{min_total:,} frames · {fps:.1f} fps"),
                    unsafe_allow_html=True,
                )

                nav_cols = st.columns([1, 1, 1, 1, 4])
                with nav_cols[0]:
                    if st.button("⟪ -10", use_container_width=True,
                                 key="scrub_back10"):
                        st.session_state.scrubber_frame_idx = max(
                            1, st.session_state.scrubber_frame_idx - 10)
                        st.rerun()
                with nav_cols[1]:
                    if st.button("◀ -1", use_container_width=True,
                                 key="scrub_back1"):
                        st.session_state.scrubber_frame_idx = max(
                            1, st.session_state.scrubber_frame_idx - 1)
                        st.rerun()
                with nav_cols[2]:
                    if st.button("+1 ▶", use_container_width=True,
                                 key="scrub_fwd1"):
                        st.session_state.scrubber_frame_idx = min(
                            min_total, st.session_state.scrubber_frame_idx + 1)
                        st.rerun()
                with nav_cols[3]:
                    if st.button("+10 ⟫", use_container_width=True,
                                 key="scrub_fwd10"):
                        st.session_state.scrubber_frame_idx = min(
                            min_total, st.session_state.scrubber_frame_idx + 10)
                        st.rerun()
                with nav_cols[4]:
                    slider_val = st.slider(
                        "Frame",
                        min_value=1, max_value=int(min_total),
                        value=int(st.session_state.scrubber_frame_idx),
                        step=1, key="scrubber_frame_idx_slider",
                        label_visibility="collapsed",
                    )
                    if slider_val != st.session_state.scrubber_frame_idx:
                        st.session_state.scrubber_frame_idx = int(slider_val)

                # Context line: frame # / total, timecode, ball position, possession
                idx_now = int(st.session_state.scrubber_frame_idx)
                seconds = (idx_now - 1) / max(fps, 1e-6)
                m, s = int(seconds // 60), seconds - int(seconds // 60) * 60
                tc = f"{m:02d}:{s:05.2f}"
                ball_txt = "no ball"
                poss_txt = ""
                entry = find_closest_entry(st.session_state.game_data, idx_now)
                if entry is not None:
                    bp = entry.get("ball_position")
                    if bp is not None:
                        bp = np.asarray(bp, dtype=float).reshape(-1)
                        if bp.shape[0] >= 2:
                            ball_txt = f"ball pitch ({bp[0]:.1f}, {bp[1]:.1f})"
                    # Per-frame possession comes from the SAME canonical-team
                    # + bbox-overlap + sticky-carry logic the Overview tab
                    # uses (cached in session_state), so the two never
                    # disagree. Falls back to "" (loose ball / no data).
                    owner_map = _ball_owner_by_frame(st.session_state.game_data)
                    owner = owner_map.get(idx_now)
                    if owner is not None:
                        poss_txt = f" · possession: Team {owner + 1}"
                st.markdown(
                    f"<div class='ps-card__sub' style='margin:4px 0 10px;'>"
                    f"Frame <b>{idx_now:,}</b> / {min_total:,} · "
                    f"<b>{tc}</b> @ {fps:.1f}fps · {ball_txt}{poss_txt}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Regions in view (segmentation chips)
                segs = regions_for_frame(st.session_state.analytics_data, idx_now)
                if segs:
                    chips = " ".join(
                        f'<span class="ps-chip">'
                        f'{SEG_CLASS_LABELS.get(s.get("class_name", ""), s.get("class_name", ""))}'
                        f' · {float(s.get("confidence", 0)):.2f}</span>'
                        for s in segs[:5]
                    )
                    st.markdown(
                        f"<div style='display:flex;flex-wrap:wrap;gap:6px;"
                        f"margin-bottom:10px;'>"
                        f"<span class='ps-card__sub' style='margin-right:4px;'>Regions in view:</span>"
                        f"{chips}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='ps-card__sub' style='margin-bottom:10px;'>"
                        "Regions in view: <i>No pitch regions detected at this timestamp.</i></div>",
                        unsafe_allow_html=True,
                    )

                # Preview grid for displayed outputs
                preview_cols = st.columns(len(frame_rows))
                for col, (path, label, _total) in zip(preview_cols, frame_rows):
                    with col:
                        bgr = seek_to_frame(str(path), idx_now)
                        if bgr is None:
                            st.warning(f"Could not open {path.name}")
                            continue
                        st.image(
                            frame_to_rgb(bgr),
                            caption=label,
                        )

                # Frame-length sanity check
                distinct_totals = {t for (_p, _l, t) in frame_rows}
                if len(distinct_totals) > 1:
                    st.warning(
                        "Output videos have mismatched frame counts: "
                        + ", ".join(f"{Path(p).name}={t}" for p, _l, t in frame_rows)
                        + ". Scrubber is clamped to the shortest."
                    )

                st.markdown(ui.card_close(), unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='ps-footer'>PitchSense · CV pipeline · streamlit + plotly · "
    f"theme: <b>{st.session_state.theme}</b></div>",
    unsafe_allow_html=True,
)
