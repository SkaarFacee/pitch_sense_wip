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
import ui_theme as ui
import charts as ch


# ─── Configuration ────────────────────────────────────────────────────────────
TEST_DATA_DIR = _PROJECT_ROOT / "data" / "matches"
OUTPUT_BASE = _PROJECT_ROOT / "output"

MODEL_PATHS = {
    "keypoint": str(_PROJECT_ROOT / "models" / "keypoint_model" / "26n_pipeline" / "no_aug" / "weights" / "best.pt"),
    "player":   str(_PROJECT_ROOT / "models" / "player_model" / "best.pt"),
    "seg":      str(_PROJECT_ROOT / "models" / "segmentation" / "best.pt"),
    "ball":     str(_PROJECT_ROOT / "models" / "ball_model" / "yolo26_best.pt"),
}

SUPPORTED_EXTENSIONS = (".webm", ".mp4", ".avi", ".mov", ".mkv")

OUTPUT_VIDEOS = [
    ("final_draft.mp4",          "Final Draft", "Original + PiP top-down pitch map"),
    ("annotated_video.mp4",      "Annotated",   "Keypoints, team bboxes, ball"),
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


def check_models() -> dict[str, bool]:
    return {name: Path(path).exists() for name, path in MODEL_PATHS.items()}


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

    # Model status
    st.markdown("<div style='font-weight:600;font-size:0.82rem;margin-bottom:6px;color:var(--ps-text-dim);'>SYSTEM STATUS</div>", unsafe_allow_html=True)
    statuses = check_models()
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
    flip_projection_y = st.checkbox("Flip projection (Y — short axis)", value=True,
                                    help="Mirror the short axis if the ball/players appear on the wrong side of the top-down pitch canvas.")


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
        {"All models loaded" if all(statuses.values()) else "Models missing"}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_processing, tab_match, tab_pitch, tab_videos = st.tabs(
    ["Processing", "Match Centre", "Pitch Analysis", "Outputs"]
)






# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PROCESSING
# ═════════════════════════════════════════════════════════════════════════════
with tab_processing:
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
                    Ball Detection
                  </div>
                  <div style="color:var(--ps-text-dim);font-size:0.82rem;line-height:1.45;">
                    Detects the ball frame-by-frame and builds a smooth movement trail.
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
        st.session_state.processing_done = False

        safe_stem = "".join(c if c.isalnum() or c in " _-" else "_" for c in Path(selected_path).stem)
        output_dir = OUTPUT_BASE / f"processed_{safe_stem}"
        output_dir.mkdir(parents=True, exist_ok=True)
        st.session_state.last_output_dir = str(output_dir)
        st.session_state.last_video_name = selected_name

        total_frames = total_in_video
        if max_frames > 0:
            total_frames = min(total_frames, max_frames)

        # Ring progress placeholder (re-rendered every frame)
        ring_slot = st.empty()
        ring_slot.markdown(
            ui.ring_html(
                pct=0.0, label="Starting…",
                sublabel=f"Initialising pipeline on <code>{Path(selected_path).name}</code>",
                stat_pairs=[
                    ("Frames", f"0 / {total_frames:,}"),
                    ("Ball Detections", "0"),
                    ("Players/Frame", "—"),
                    ("Homography", "—"),
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
            for result in pipeline.process_video(
                source_video_path=selected_path,
                output_dir=str(output_dir),
                start_frame=0,
                max_frames=max_frames if max_frames > 0 else None,
            ):
                processed += 1
                has_ball = len(result.get("ball_xyxy", [])) > 0
                if has_ball:
                    ball_count += 1

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
                track_ids = result.get("track_ids", np.empty((0,), dtype=np.int32))
                track_quality = team_info.get("track_quality") if team_info else None
                team1_bgr = team_info.get("team1_bgr") if team_info else None
                team2_bgr = team_info.get("team2_bgr") if team_info else None
                st.session_state.game_data.append({
                    "frame_idx": processed,
                    "player_positions": result.get("player_pitch_pts", np.empty((0, 2))),
                    "team_ids": team_ids,
                    "track_ids": track_ids,
                    "track_quality": track_quality,
                    "team1_bgr": team1_bgr,
                    "team2_bgr": team2_bgr,
                    "ball_position": result.get("ball_pitch_pt"),
                    "player_conf": result.get("player_conf", np.empty((0,))),
                })

                pct = (processed / max(total_frames, 1)) * 100
                h_mode = result.get("H_info", {}).get("mode", "N/A")
                n_players = int(len(result.get("player_pitch_pts", [])))

                # Update ring every 2 frames to avoid bottleneck
                if processed % 2 == 0 or processed == total_frames:
                    ring_slot.markdown(
                        ui.ring_html(
                            pct=pct,
                            label="Processing",
                            sublabel=f"Frame <b>{processed:,}</b> of <b>{total_frames:,}</b> · "
                                     f"{'⚽ ball in view' if has_ball else 'no ball this frame'}",
                            stat_pairs=[
                                ("Frames", f"{processed:,} / {total_frames:,}"),
                                ("Ball Detections", f"{ball_count:,}"),
                                ("Players/Frame", f"{n_players}"),
                                ("Homography", str(h_mode)),
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
                        ("Ball Detections", f"{ball_count:,}"),
                        ("Players/Frame", "—"),
                        ("Status", "Done"),
                    ],
                ),
                unsafe_allow_html=True,
            )
            st.session_state.processing_done = True
            st.success("Processing finished — open **Match Centre**, **Pitch Analysis**, or **Outputs**.")
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

        # Override the team1/team2 colours in the active palette with the
        # BGR centroids detected by the team_analyzer. The values are
        # EMA-blended across the match, so they reflect each team's
        # representative jersey colour rather than a single-frame outlier.
        detected_palette = dict(palette)
        t1_bgr = GameAnalyzer.dominant_team_bgr(game_data, team=0)
        t2_bgr = GameAnalyzer.dominant_team_bgr(game_data, team=1)
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

        # KPI row
        avg_p_total = stats["avg_players_total"]
        ball_pct = stats["ball_detection_rate"]
        kpi_html = ui.kpi_grid([
            ("Total Frames",       f"{stats['total_frames']:,}",          "frames analysed"),
            ("Ball Detection",     f"{ball_pct:.1f}%",                    f"{stats['ball_detection_frames']:,} frames"),
            ("Players / Frame",    f"{avg_p_total:.1f}",                  f"T1 {stats['avg_players_team1']:.1f} · T2 {stats['avg_players_team2']:.1f}"),
            ("Avg Spread",         f"{stats['avg_player_spread']} m",     "distance from team centre"),
            ("Ball Progression",   f"{stats['ball_progression_m']} m",    "total ball travel"),
            ("Possession Lead",    f"{abs(t1_pct - t2_pct):.1f}%",        ("Team 1 leads" if t1_pct >= t2_pct else "Team 2 leads")),
        ])
        st.markdown(kpi_html, unsafe_allow_html=True)

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
                                     "6-axis tactical profile (0 – 100)"),
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
            t1c = formation["team1_avg_center"] or [0, 0]
            t2c = formation["team2_avg_center"] or [0, 0]
            form_kpi = ui.kpi_grid([
                ("T1 Avg Position",      f"({t1c[0]:.1f}, {t1c[1]:.1f}) m", "centre of mass"),
                ("T1 Avg Spread",        f"{formation['team1_avg_spread']:.1f} m",  "team compactness"),
                ("T1 Deepest Player",    f"{formation['team1_defensive_depth']:.1f} m", "avg min-X position"),
                ("T2 Avg Position",      f"({t2c[0]:.1f}, {t2c[1]:.1f}) m", "centre of mass"),
                ("T2 Avg Spread",        f"{formation['team2_avg_spread']:.1f} m",  "team compactness"),
                ("T2 Deepest Player",    f"{formation['team2_defensive_depth']:.1f} m", "avg min-X position"),
            ])
            st.markdown(form_kpi, unsafe_allow_html=True)

        # Segmentation analytics ----------------------------------------------
        if has_seg:
            analytics = build_seg_analytics(st.session_state.analytics_data)

            seg_kpi = ui.kpi_grid([
                ("Total Frames",         f"{analytics['total_frames']:,}",     "processed"),
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


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — OUTPUTS
# ═════════════════════════════════════════════════════════════════════════════
with tab_videos:




    out_dir_str = st.session_state.last_output_dir
    if not out_dir_str:
        st.markdown(
            """
            <div class='ps-card' style='text-align:center;padding:40px 24px;'>
              <h3>No outputs yet</h3>
              <p style='color:var(--ps-text-dim);'>Process a video to see the five generated outputs here.</p>
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


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='ps-footer'>PitchSense · CV pipeline · streamlit + plotly · "
    f"theme: <b>{st.session_state.theme}</b></div>",
    unsafe_allow_html=True,
)
