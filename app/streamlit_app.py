"""
PitchSense Streamlit App — Interactive video processing with ball detection.

Features:
    - Select a test video from data/test_data/
    - Process through the full KeypointPipeline (keypoint → homography → players
      → team colors → ball detection → segmentation)
    - Real-time progress bar showing frame-by-frame processing
    - Playback of all 4 generated output videos

Usage:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so app modules can be imported
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import cv2
from app.constants import (
    PLAYER_CONF,
    SEG_CONF,
    MAX_FRAMES,
    PROCESS_EVERY_N_FRAMES,
)
from app.keypoint_pipeline import KeypointPipeline

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
TEST_DATA_DIR = _PROJECT_ROOT / "data" / "test_data"
OUTPUT_BASE = _PROJECT_ROOT / "output"

MODEL_PATHS = {
    "keypoint": str(_PROJECT_ROOT / "models" / "keypoint_model" / "26n_pipeline" / "no_aug" / "weights" / "best.pt"),
    "player": str(_PROJECT_ROOT / "models" / "player_model" / "best.pt"),
    "seg": str(_PROJECT_ROOT / "models" / "segmentation" / "best.pt"),
    "ball": str(_PROJECT_ROOT / "models" / "ball_model" / "yolo_11_best.pt"),
}

SUPPORTED_EXTENSIONS = (".webm", ".mp4", ".avi", ".mov")

OUTPUT_VIDEOS = [
    ("final_draft.mp4", "🎬 Final Draft (Main + Pitch PIP)"),
    ("annotated_video.mp4", "🎯 Annotated (Keypoints + Team Bboxes + Ball)"),
    ("deep_analysis.mp4", "🔬 Deep Analysis (Segmentation Overlay + Ball)"),
    ("full_pitch_debug_map.mp4", "🗺️ Full Pitch Map (Top-Down View + Ball Trail)"),
]

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


# ---------------------------------------------------------------------------
# UI — Header
# ---------------------------------------------------------------------------
st.title("⚽ PitchSense — Ball Detection & Analysis")
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar — Configuration
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
# Main area — Video selection & processing
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
if process_btn:
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

        st.balloons()

    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        st.exception(e)
        st.stop()

    # -------------------------------------------------------------------
    # Display generated videos
    # -------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Landing info (before processing)
# ---------------------------------------------------------------------------
else:
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

    # Show available test videos
    video_files = get_video_files()
    if video_files:
        st.markdown("#### 📁 Available test videos")
        for vf in video_files:
            size_mb = vf.stat().st_size / (1024 * 1024)
            st.markdown(f"- `{vf.name}` ({size_mb:.1f} MB)")