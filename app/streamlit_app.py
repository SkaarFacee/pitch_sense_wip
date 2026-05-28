"""
PitchSense Streamlit App — Interactive video processing with ball detection,
pitch segmentation analytics, and game dynamics analysis.

Tabs:
    1. Processing  — Select video, run pipeline, view outputs
    2. Analytics   — Pitch segmentation region analysis from processed video
    3. Game Analysis — Possession, heatmaps, formation, territory & match stats

Usage:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
from collections import Counter

# Ensure the project root is on sys.path so app modules can be imported
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import cv2
import numpy as np
from app.constants import (
    PLAYER_CONF,
    SEG_CONF,
    MAX_FRAMES,
    PROCESS_EVERY_N_FRAMES,
)
from app.keypoint_pipeline import KeypointPipeline
from app.game_analyzer import GameAnalyzer

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PitchSense - Ball Detection & Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEST_DATA_DIR = _PROJECT_ROOT / "data" / "matches"
OUTPUT_BASE = _PROJECT_ROOT / "output"

MODEL_PATHS = {
    "keypoint": str(_PROJECT_ROOT / "models" / "keypoint_model" / "26n_pipeline" / "no_aug" / "weights" / "best.pt"),
    "player": str(_PROJECT_ROOT / "models" / "player_model" / "best.pt"),
    "seg": str(_PROJECT_ROOT / "models" / "segmentation" / "best.pt"),
    "ball": str(_PROJECT_ROOT / "models" / "ball_model" / "yolo_11_best.pt"),
}

SUPPORTED_EXTENSIONS = (".webm", ".mp4", ".avi", ".mov", ".mkv")

OUTPUT_VIDEOS = [
    ("final_draft.mp4", "🎬 Final Draft (Main + Pitch PIP)"),
    ("annotated_video.mp4", "🎯 Annotated (Keypoints + Team Bboxes + Ball)"),
    ("deep_analysis.mp4", "🔬 Deep Analysis (Segmentation Overlay + Ball)"),
    ("full_pitch_debug_map.mp4", "🗺️ Full Pitch Map (Top-Down View + Ball Trail)"),
    ("keypoint_annotations.mp4", "🔑 Keypoint Annotations (Keypoints on Original)"),
]

# Segmentation class display names and colors (BGR)
SEG_CLASS_INFO = {
    "18Yard": {"label": "Penalty Area (18yd)", "color": (255, 0, 0)},
    "18Yard Circle": {"label": "Penalty Arc", "color": (0, 255, 0)},
    "5Yard": {"label": "Goal Area (5yd)", "color": (0, 0, 255)},
    "Half Central Circle": {"label": "Center Circle", "color": (255, 255, 0)},
    "Half Field": {"label": "Half Field", "color": (255, 0, 255)},
}

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "analytics_data" not in st.session_state:
    st.session_state.analytics_data = None  # Will hold seg data after processing
if "game_data" not in st.session_state:
    st.session_state.game_data = None      # Will hold player/ball tracking data
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_video_files() -> list:
    """Return sorted list of supported video files in TEST_DATA_DIR."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(TEST_DATA_DIR.glob(f"*{ext}"))
    return sorted(files)


def get_total_frames(video_path: str) -> int:
    """Get total frame count for a video file."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def get_output_videos(output_dir: Path) -> list:
    """Return list of (path, display_name) tuples for generated videos."""
    results = []
    for filename, display_name in OUTPUT_VIDEOS:
        video_path = output_dir / filename
        if video_path.exists():
            results.append((str(video_path), display_name))
    return results


def check_models() -> dict:
    """Verify all model paths exist and return status dict."""
    status = {}
    for name, path in MODEL_PATHS.items():
        status[name] = Path(path).exists()
    return status


def build_seg_analytics(analytics_data: list) -> dict:
    class_counter = Counter()
    per_frame = {}
    frames_with_seg = 0
    for entry in analytics_data:
        frame_idx = entry["frame_idx"]
        segments = entry.get("segments", [])
        if segments:
            frames_with_seg += 1
        frame_classes = []
        for seg in segments:
            cn = seg.get("class_name", "unknown")
            frame_classes.append(cn)
            class_counter[cn] += 1
        per_frame[frame_idx] = frame_classes
    return {"class_frequency": dict(class_counter), "frames_with_seg": frames_with_seg,
            "total_frames": len(analytics_data), "per_frame_classes": per_frame}


# ---------------------------------------------------------------------------
# Sidebar — Configuration (shared across all tabs)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # Model status
    st.subheader("Model Status")
    model_status = check_models()
    for name, exists in model_status.items():
        emoji = "✅" if exists else "❌"
        st.markdown(f"{emoji} **{name.capitalize()}**: {'Loaded' if exists else 'Missing'}")

    st.markdown("---")

    # Processing options
    st.subheader("Processing Options")
    max_frames = st.number_input(
        "Max frames to process (0 = all)",
        min_value=0,
        max_value=10000,
        value=0,
        step=100,
        help="Limit processing to N frames for faster testing. 0 = process entire video.",
    )
    process_every_n = st.number_input(
        "Process every N frames",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        help="Process every Nth frame. Higher values = faster but less smooth video.",
    )
    enable_team_colors = st.checkbox("Enable team color analysis", value=True)

    st.markdown("---")
    st.markdown(
        "<small>Built with Streamlit + YOLO + OpenCV</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# UI — Header
# ---------------------------------------------------------------------------
st.title("⚽ PitchSense — Ball Detection & Analysis")
st.markdown("---")

# ---------------------------------------------------------------------------
# Main area — Tabs
# ---------------------------------------------------------------------------
tab_processing, tab_analytics, tab_game = st.tabs(
    ["🎬 Processing", "📊 Pitch Analytics", "🎮 Game Analysis"]
)

# ===========================================================================
# TAB 1: PROCESSING (existing functionality)
# ===========================================================================
with tab_processing:
    col1, col2 = st.columns([2, 1])

    with col1:
        video_files = get_video_files()
        if not video_files:
            st.warning(f"No video files found in `{TEST_DATA_DIR}`.")
            st.stop()

        video_options = {f.name: str(f) for f in video_files}
        selected_name = st.selectbox(
            "📁 Select a video to process",
            options=list(video_options.keys()),
            index=0,
        )
        selected_path = video_options[selected_name]

    with col2:
        st.markdown("#### &nbsp;")  # vertical spacing
        process_btn = st.button(
            "▶️ Process Video",
            type="primary",
            use_container_width=True,
            disabled=not all(model_status.values()),
        )

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    if process_btn:
        # Reset analytics + game data for this run
        st.session_state.analytics_data = []
        st.session_state.game_data = []
        st.session_state.processing_done = False

        # Determine output directory: named after input video stem
        video_stem = Path(selected_path).stem
        # Sanitize filename for directory name (remove special chars)
        safe_stem = "".join(c if c.isalnum() or c in " _-" else "_" for c in video_stem)
        output_dir = OUTPUT_BASE / f"processed_{safe_stem}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get total frames for progress
        total_frames = get_total_frames(selected_path)
        if max_frames > 0:
            total_frames = min(total_frames, max_frames)
        actual_process_every = process_every_n

        st.markdown("---")
        st.subheader("📊 Processing Progress")

        # Progress bar and status
        progress_bar = st.progress(0, text="Initializing pipeline...")
        status_placeholder = st.empty()
        status_placeholder.info(f"⏳ Processing `{selected_name}` — 0 / {total_frames} frames")

        # Side-by-side status metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            frame_metric = st.empty()
            frame_metric.metric("Frames Processed", "0")
        with metric_col2:
            ball_metric = st.empty()
            ball_metric.metric("Ball Detected", "—")
        with metric_col3:
            player_metric = st.empty()
            player_metric.metric("Players/Frame", "—")
        with metric_col4:
            h_metric = st.empty()
            h_metric.metric("Homography", "—")

        # Initialize pipeline
        try:
            pipeline = KeypointPipeline(
                keypoint_model_path=MODEL_PATHS["keypoint"],
                player_model_path=MODEL_PATHS["player"],
                seg_model_path=MODEL_PATHS["seg"],
                ball_model_path=MODEL_PATHS["ball"],
                enable_team_colors=enable_team_colors,
            )

            processed_count = 0
            ball_count_total = 0

            # Process video with progress tracking
            for result in pipeline.process_video(
                source_video_path=selected_path,
                output_dir=str(output_dir),
                start_frame=0,
                max_frames=max_frames if max_frames > 0 else None,
                process_every_n=actual_process_every,
            ):
                processed_count += 1
                has_ball = len(result.get('ball_xyxy', [])) > 0
                if has_ball:
                    ball_count_total += 1

                # --- Collect segmentation data for analytics tab ---
                segs = result.get('processed_segments', [])
                if segs:
                    st.session_state.analytics_data.append({
                        "frame_idx": processed_count,
                        "segments": segs,
                    })

                # --- Collect game tracking data for game analysis tab ---
                team_info = result.get('team_info')
                team_ids = team_info.get('team_ids') if team_info else None

                st.session_state.game_data.append({
                    "frame_idx": processed_count,
                    "player_positions": result.get('player_pitch_pts', np.empty((0, 2))),
                    "team_ids": team_ids,
                    "ball_position": result.get('ball_pitch_pt'),
                    "player_conf": result.get('player_conf', np.empty((0,))),
                })

                # Calculate progress percentage
                pct = min(processed_count / max(total_frames, 1), 1.0)

                # Update progress bar
                progress_bar.progress(
                    pct,
                    text=f"Frame {processed_count} / {total_frames} ({int(pct * 100)}%)",
                )

                # Update metrics every frame
                frame_metric.metric("Frames Processed", str(processed_count))
                ball_metric.metric(
                    "Ball Detected",
                    f"✅ Yes (x{ball_count_total})" if has_ball else "❌ No",
                    delta="Detected" if has_ball else None,
                )

                n_players = len(result.get('player_pitch_pts', []))
                player_metric.metric("Players/Frame", str(n_players))

                h_mode = result.get('H_info', {}).get('mode', 'N/A')
                h_metric.metric("Homography", str(h_mode))

                # Update status text
                status_placeholder.info(
                    f"⏳ Processing frame {processed_count} / {total_frames} "
                    f"({int(pct * 100)}%) — "
                    f"{'⚽ Ball!' if has_ball else 'No ball'} | "
                    f"{n_players} players"
                )

            # Processing complete
            progress_bar.progress(1.0, text="✅ Processing complete!")
            status_placeholder.success(
                f"✅ Processing complete! Processed {processed_count} frames "
                f"in {int(processed_count / max(total_frames, 1) * 100)}% of video. "
                f"Ball detected in {ball_count_total} frames."
            )
            st.session_state.processing_done = True

            st.balloons()

        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            st.exception(e)
            st.stop()

        # ------------------------------------------------------------------
        # Display generated videos
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("🎬 Generated Videos")

        output_videos = get_output_videos(output_dir)

        if not output_videos:
            st.warning("No output videos were generated.")
            st.stop()

        # Display videos in a 2x2 grid
        for i in range(0, len(output_videos), 2):
            row_cols = st.columns(2)
            for j in range(2):
                idx = i + j
                if idx < len(output_videos):
                    video_path, display_name = output_videos[idx]
                    with row_cols[j]:
                        st.markdown(f"**{display_name}**")
                        st.video(video_path)

        # Also show the directory path
        st.markdown(f"📁 **Output directory**: `{output_dir}/`")

    # ------------------------------------------------------------------
    # Landing info (before processing)
    # ------------------------------------------------------------------
    elif not process_btn and not st.session_state.processing_done:
        st.markdown(
            """
            ### 🚀 Ready to analyze

            Select a video from the dropdown and click **Process Video** to run the
            full analysis pipeline:

            - 🔑 **Keypoint detection** → Homography matrix for pitch registration
            - 👤 **Player detection** → ByteTrack tracking + team color analysis
            - ⚽ **Ball detection** → Dedicated YOLO ball model with trajectory trail
            - 🎯 **Segmentation** → Pitch region segmentation overlay
            - 🗺️ **Top-down pitch** → Projected player + ball positions with trail

            **Output videos** will appear here once processing is complete.
            """,
            unsafe_allow_html=True,
        )

        # Show available matches prominently
        video_files = get_video_files()
        if video_files:
            st.markdown("#### 📁 Available Matches")
            match_cards = st.columns(min(len(video_files), 3))
            for idx, vf in enumerate(video_files):
                col = match_cards[idx % 3]
                if idx > 0 and idx % 3 == 0:
                    match_cards = st.columns(min(len(video_files) - idx, 3))
                    col = match_cards[0]
                with col:
                    size_mb = vf.stat().st_size / (1024 * 1024)
                    # Extract team names from filename (remove special chars)
                    name = vf.stem
                    # Clean up common patterns
                    display_name = name.replace("FULL MATCH ", "").replace("｜", " vs ").replace("|", " vs ").strip()
                    # Truncate if too long
                    if len(display_name) > 40:
                        display_name = display_name[:37] + "..."
                    st.markdown(
                        f"""
                        <div style="border:1px solid #ddd; border-radius:10px; padding:12px; margin-bottom:12px;
                                    background: #f8f9fa; text-align:center;">
                            <div style="font-size:2rem;">⚽</div>
                            <div style="font-weight:600; font-size:0.95rem; margin:6px 0;">{display_name}</div>
                            <div style="font-size:0.8rem; color:#666;">{size_mb:.1f} MB</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ===========================================================================
# TAB 2: ANALYTICS — Pitch Segmentation Analysis
# ===========================================================================
with tab_analytics:
    st.subheader("📊 Pitch Segmentation Analytics")

    if st.session_state.analytics_data is None or len(st.session_state.analytics_data) == 0:
        st.info(
            "ℹ️ No segmentation data available yet. "
            "Process a video in the **Processing** tab first, then return here for analytics."
        )
    else:
        analytics = build_seg_analytics(st.session_state.analytics_data)

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Frames (processed)", analytics["total_frames"])
        with col2:
            st.metric("Frames with Segments", analytics["frames_with_seg"])
        with col3:
            pct_seg = (analytics["frames_with_seg"] / max(analytics["total_frames"], 1)) * 100
            st.metric("Segmentation Coverage", f"{pct_seg:.1f}%")
        with col4:
            total_detections = sum(analytics["class_frequency"].values())
            st.metric("Total Region Detections", total_detections)

        st.markdown("---")

        # --- Region Detection Frequency ---
        st.markdown("### 🧩 Region Detection Frequency")

        freq = analytics["class_frequency"]
        if freq:
            sorted_classes = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            class_names = [SEG_CLASS_INFO.get(c, {}).get("label", c) for c, _ in sorted_classes]
            counts = [v for _, v in sorted_classes]

            chart_data = {"Region": class_names, "Detections": counts}
            st.bar_chart(chart_data, x="Region", y="Detections", use_container_width=True)

            total_det = sum(counts)
            st.markdown("**Detection breakdown:**")
            rows = []
            for (cls_name, count), display_name in zip(sorted_classes, class_names):
                pct_cls = (count / max(total_det, 1)) * 100
                rows.append(f"- **{display_name}** (`{cls_name}`): {count} detections ({pct_cls:.1f}%)")
            st.markdown("\n".join(rows))
        else:
            st.warning("No pitch regions were detected in the processed frames.")

        st.markdown("---")

        with st.expander("📋 Raw Frame-by-Frame Data"):
            per_frame = analytics["per_frame_classes"]
            st.json({str(k): v for k, v in per_frame.items()})

# ===========================================================================
# TAB 3: GAME ANALYSIS — Possession, Heatmaps, Formation, Territory, Stats
# ===========================================================================
with tab_game:
    st.subheader("🎮 Game Analysis — From Pipeline Tracking Data")

    if st.session_state.game_data is None or len(st.session_state.game_data) == 0:
        st.info(
            "ℹ️ No game tracking data available yet. "
            "Process a video in the **Processing** tab first, then return here for analysis."
        )
    else:
        game_data = st.session_state.game_data

        # Get team labels from the team colors detected in processing
        # (We don't have user-input names, so use defaults)
        team1_label = "Team 1"
        team2_label = "Team 2"

        # ---- SECTION 1: POSSESSION ----
        st.markdown("### ⚽ Ball Possession")
        possession = GameAnalyzer.compute_possession(game_data, team1_label, team2_label)

        if possession["total_ball_frames"] == 0:
            st.warning("Ball was not detected in any frames — possession cannot be calculated.")
        else:
            pos_col1, pos_col2, pos_col3 = st.columns([2, 1, 2])
            with pos_col1:
                t1_pct = possession["team1_possession_pct"]
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 1.2rem; border-radius: 10px;
                                background: linear-gradient(135deg, #1a73e8, #0d47a1); color: white;">
                        <h4 style="margin:0;">🔵 {possession['team1_label']}</h4>
                        <p style="font-size: 2.5rem; font-weight: bold; margin: 0.3rem 0;">{t1_pct}%</p>
                        <p style="margin:0; font-size:0.85rem;">{possession['team1_frames']} frames</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with pos_col2:
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 1.2rem; border-radius: 10px;
                                background: #6c757d; color: white;">
                        <h4 style="margin:0;">🤝</h4>
                        <p style="font-size: 2.5rem; font-weight: bold; margin: 0.3rem 0;">
                            {100 - t1_pct - possession['team2_possession_pct']:.0f}%
                        </p>
                        <p style="margin:0; font-size:0.85rem;">contested</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with pos_col3:
                t2_pct = possession["team2_possession_pct"]
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 1.2rem; border-radius: 10px;
                                background: linear-gradient(135deg, #d32f2f, #b71c1c); color: white;">
                        <h4 style="margin:0;">🔴 {possession['team2_label']}</h4>
                        <p style="font-size: 2.5rem; font-weight: bold; margin: 0.3rem 0;">{t2_pct}%</p>
                        <p style="margin:0; font-size:0.85rem;">{possession['team2_frames']} frames</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(f"Based on {possession['total_ball_frames']} frames where ball was detected. "
                       f"Possession = which team's players were closest to the ball (GKs excluded).")
            st.progress(t1_pct / 100, text=f"Team 1 {t1_pct}% — Team 2 {t2_pct}%")

        st.markdown("---")

        # ---- SECTION 2: PITCH HEATMAPS ----
        st.markdown("### 🔥 Player Density Heatmaps")

        heatmaps = GameAnalyzer.compute_heatmaps(game_data)

        heat_col1, heat_col2 = st.columns(2)
        with heat_col1:
            if heatmaps["team1_count"] > 0:
                fig1 = GameAnalyzer.draw_pitch_heatmap(
                    heatmaps["team1_heatmap"],
                    heatmaps["x_edges"],
                    heatmaps["y_edges"],
                    f"Team 1 Density ({heatmaps['team1_count']} samples)",
                    team_color=(0, 0, 255),
                    cmap="Blues",
                )
                st.pyplot(fig1)
            else:
                st.info("No Team 1 positions recorded.")

        with heat_col2:
            if heatmaps["team2_count"] > 0:
                fig2 = GameAnalyzer.draw_pitch_heatmap(
                    heatmaps["team2_heatmap"],
                    heatmaps["x_edges"],
                    heatmaps["y_edges"],
                    f"Team 2 Density ({heatmaps['team2_count']} samples)",
                    team_color=(255, 0, 0),
                    cmap="Reds",
                )
                st.pyplot(fig2)
            else:
                st.info("No Team 2 positions recorded.")

        st.markdown("---")

        # ---- SECTION 3: FORMATION & POSITIONING ----
        st.markdown("### 📐 Formation & Positioning")

        formation = GameAnalyzer.compute_formation(game_data)

        form_col1, form_col2, form_col3, form_col4 = st.columns(4)
        with form_col1:
            if formation["team1_avg_center"]:
                st.metric("Team 1 Avg Position",
                          f"({formation['team1_avg_center'][0]:.1f}, {formation['team1_avg_center'][1]:.1f}) m")
        with form_col2:
            st.metric("Team 1 Spread", f"{formation['team1_avg_spread']:.1f} m")
        with form_col3:
            if formation["team2_avg_center"]:
                st.metric("Team 2 Avg Position",
                          f"({formation['team2_avg_center'][0]:.1f}, {formation['team2_avg_center'][1]:.1f}) m")
        with form_col4:
            st.metric("Team 2 Spread", f"{formation['team2_avg_spread']:.1f} m")

        # Defensive depth
        depth_col1, depth_col2 = st.columns(2)
        with depth_col1:
            st.metric("Team 1 Defensive Depth (avg min X)",
                      f"{formation['team1_defensive_depth']:.1f} m",
                      help="Average of each frame's deepest (minimum X) player position. Lower = more defensive.")
        with depth_col2:
            st.metric("Team 2 Defensive Depth (avg min X)",
                      f"{formation['team2_defensive_depth']:.1f} m",
                      help="Average of each frame's deepest (minimum X) player position. Lower = more defensive.")

        # Formation scatter plot
        if formation["frames_with_players"] > 0:
            st.markdown("**Player Positioning Scatter** (sample of frames)")
            scatter_fig = GameAnalyzer.draw_formation_scatter(
                game_data,
                team1_color=(0.2, 0.4, 0.9),
                team2_color=(0.9, 0.2, 0.2),
                team1_label=team1_label,
                team2_label=team2_label,
            )
            st.pyplot(scatter_fig)

        st.markdown("---")

        # ---- SECTION 4: TERRITORY CONTROL ----
        st.markdown("### 🗺️ Territory Control (9-Zone Grid)")

        territory = GameAnalyzer.compute_territory(game_data)
        zone_grid = territory["zone_grid"]

        # Build a visual 3x3 grid with streamlit columns
        for row in range(3):
            tcols = st.columns(3)
            for col in range(3):
                zone = zone_grid[row][col]
                with tcols[col]:
                    dominant = zone["dominant_team"]
                    t1_pct_z = zone["team1_pct"]
                    t2_pct_z = zone["team2_pct"]

                    if dominant == 0:
                        bg = "linear-gradient(135deg, #1a73e8, #0d47a1)"
                        emoji = "🔵"
                    elif dominant == 1:
                        bg = "linear-gradient(135deg, #d32f2f, #b71c1c)"
                        emoji = "🔴"
                    else:
                        bg = "#6c757d"
                        emoji = "⬜"

                    st.markdown(
                        f"""
                        <div style="text-align: center; padding: 0.8rem; border-radius: 8px;
                                    background: {bg}; color: white; margin-bottom: 0.5rem;">
                            <strong>{zone['zone_name']}</strong><br>
                            T1: {t1_pct_z}% — T2: {t2_pct_z}%
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Summary table
        st.markdown("**Zone Control Summary:**")
        summary_rows = []
        for row in range(3):
            for col in range(3):
                zone = zone_grid[row][col]
                dom = "Team 1" if zone["dominant_team"] == 0 else ("Team 2" if zone["dominant_team"] == 1 else "Neutral")
                summary_rows.append(
                    f"| {zone['zone_name']} | {zone['team1_pct']}% | {zone['team2_pct']}% | {dom} |"
                )

        st.markdown(
            "| Zone | Team 1 % | Team 2 % | Dominant |\n"
            "|------|----------|----------|----------|\n"
            + "\n".join(summary_rows)
        )

        st.caption(
            f"Total player-zone occurrences: Team 1 = {territory['team1_total_presence']}, "
            f"Team 2 = {territory['team2_total_presence']}"
        )

        st.markdown("---")

        # ---- SECTION 5: MATCH STATS DASHBOARD ----
        st.markdown("### 📊 Key Match Stats")

        stats = GameAnalyzer.compute_match_stats(game_data)

        stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
        with stat_col1:
            st.metric("Total Frames", stats["total_frames"])
        with stat_col2:
            st.metric("Ball Detection Rate", f"{stats['ball_detection_rate']}%")
        with stat_col3:
            st.metric("Avg Players/Frame", stats["avg_players_total"])
        with stat_col4:
            st.metric("Avg Team 1", stats["avg_players_team1"])
        with stat_col5:
            st.metric("Avg Team 2", stats["avg_players_team2"])

        stat_col6, stat_col7 = st.columns(2)
        with stat_col6:
            st.metric("Avg Player Spread", f"{stats['avg_player_spread']} m",
                      help="Average distance of players from their team's center of mass.")
        with stat_col7:
            st.metric("Ball Progression", f"{stats['ball_progression_m']} m",
                      help="Approximate total distance the ball traveled across the pitch.")

        st.markdown("---")
        st.caption(
            "Analysis derived from player tracking (ByteTrack + homography projection), "
            "team color clustering (K-means), and ball detection data."
        )