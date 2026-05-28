# PitchSense ⚽

**PitchSense** is a football (soccer) video analysis pipeline that processes tactical/broadcast footage to produce:

- **Keypoint detection** → pitch registration via homography
- **Player detection & tracking** → ByteTrack + team color clustering
- **Ball detection** → dedicated YOLO model + trajectory trail
- **Pitch segmentation** → region overlay (penalty areas, center circle, etc.)
- **Top-down pitch map** → projected player + ball positions with trail
- **Game analytics** → possession, heatmaps, formation, territory control, match stats

---

## Requirements

- Python 3.10+
- [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO)
- OpenCV, NumPy, Matplotlib, Streamlit, scikit-learn

```bash
pip install ultralytics opencv-python numpy matplotlib streamlit scikit-learn
```

---

## Models

Place YOLO `.pt` weights in the following paths (configured in [`app/constants.py`](app/constants.py) and [`app/streamlit_app.py`](app/streamlit_app.py)):

| Model | Expected Path |
|-------|--------------|
| Keypoint (YOLO-Pose) | [`models/keypoint_model/26n_pipeline/no_aug/weights/best.pt`](models/keypoint_model/26n_pipeline/no_aug/weights/best.pt) |
| Player detection | [`models/player_model/best.pt`](models/player_model/best.pt) |
| Segmentation | [`models/segmentation/best.pt`](models/segmentation/best.pt) |
| Ball detection | [`models/ball_model/yolo_11_best.pt`](models/ball_model/yolo_11_best.pt) |

To use different paths, edit the [`MODEL_PATHS`](app/streamlit_app.py:52) dict in [`app/streamlit_app.py`](app/streamlit_app.py).

---

## Usage

### Streamlit App (recommended)

```bash
streamlit run app/streamlit_app.py
```

Opens a browser UI with three tabs:

1. **🎬 Processing** — Select a video, configure options, run the full pipeline, view 5 output videos
2. **📊 Pitch Analytics** — Region detection frequency from segmentation data
3. **🎮 Game Analysis** — Possession, heatmaps, formation scatter, territory control, match stats

### Processing Options (sidebar)

| Option | Default | Description |
|--------|---------|-------------|
| Max frames | 0 (all) | Limit processing to N frames for testing |
| Process every N frames | 1 | Higher = faster but less smooth |
| Enable team colors | ✅ | Run jersey color clustering per frame |

---

## Pipeline Architecture

All source files are in [`app/`](app/). The pipeline flows through these modules:

```
[Video Frame]
    │
    ├── segmentation.Segmentor ─────────► pitch region masks + overlay
    │
    ├── keypoint_service.KeypointHomographyComputer
    │       └── YOLO-Pose → 29 keypoints → filter → DLT → homography H
    │
    ├── player_service.PlayerDetector
    │       └── YOLO detection + ByteTrack → bboxes → bottom-center → project via H
    │       └── team_analyzer.TeamColorAnalyzer → K-means on HSV jersey colors
    │
    ├── ball_service.BallDetector
    │       └── YOLO ball model → bbox → bottom-center → project via H → trajectory
    │
    ├── pitch.PitchArtist
    │       └── Draw top-down pitch canvas with players, ball, trail, legend
    │
    └── keypoint_pipeline.KeypointPipeline.process_frame()
            └── Orchestrates all of the above → returns dict with 5 output frames
```

### Output Videos

| File | Content |
|------|---------|
| `final_draft.mp4` | Original frame + PiP top-down pitch map |
| `annotated_video.mp4` | Keypoints + team bboxes + ball bbox |
| `deep_analysis.mp4` | Segmentation overlay + team bboxes + ball |
| `full_pitch_debug_map.mp4` | Top-down pitch view with players + ball + trajectory |
| `keypoint_annotations.mp4` | Keypoint skeleton on original frame |

---

## File Overview

| File | Lines | Purpose |
|------|-------|---------|
| [`streamlit_app.py`](app/streamlit_app.py) | 744 | Streamlit UI (Processing, Analytics, Game Analysis tabs) |
| [`keypoint_pipeline.py`](app/keypoint_pipeline.py) | 261 | Core pipeline orchestrator per-frame + video writer |
| [`game_analyzer.py`](app/game_analyzer.py) | 290 | Possession, heatmaps, formation, territory, match stats |
| [`team_analyzer.py`](app/team_analyzer.py) | 613 | K-means HSV color clustering for team assignment |
| [`keypoint_service.py`](app/keypoint_service.py) | 149 | YOLO-Pose → homography (DLT + EMA smoothing) |
| [`pitch.py`](app/pitch.py) | 306 | Top-down pitch canvas drawing (lines, players, ball, trail) |
| [`ball_service.py`](app/ball_service.py) | 126 | Ball detection + pitch projection |
| [`segmentation.py`](app/segmentation.py) | 100 | YOLO-seg → quad extraction for pitch regions |
| [`seg_helpers.py`](app/seg_helpers.py) | 125 | Canvas coordinate mapping for segmentation regions |
| [`player_service.py`](app/player_service.py) | 89 | YOLO player detection + ByteTrack + projection |
| [`constants.py`](app/constants.py) | 80 | All configurable constants (thresholds, geometry, colors) |
| [`director.py`](app/director.py) | 11 | H.264 video writer factory |

---

## Configuration

All tunable parameters are in [`app/constants.py`](app/constants.py):

| Constant | Default | Description |
|----------|---------|-------------|
| `SEG_CONF` | 0.8 | Segmentation confidence threshold |
| `PLAYER_CONF` | 0.25 | Player detection confidence |
| `KEYPOINT_CONF` | 0.3 | Keypoint confidence threshold |
| `BALL_CONF` | 0.25 | Ball detection confidence |
| `SMOOTHING_ALPHA` | 0.4 | EMA factor for homography smoothing |
| `H_STABILITY_THRESHOLD` | 0.15 | Max relative change to accept new homography |
| `BALL_TRAIL_LENGTH` | 50 | Number of past ball positions for trajectory |
| `TEAM_N_CLUSTERS` | 2 | Number of teams for K-means clustering |

---

## Data Structure

Place input videos in [`data/matches/`](data/matches/). Supported formats: `.webm`, `.mp4`, `.avi`, `.mov`, `.mkv`.

Outputs are saved to [`output/processed_{video_name}/`](output/) with 5 video files per run.

### Per-frame game data dict

Collected during processing for the Game Analysis tab:

```python
{
    "frame_idx": int,
    "player_positions": np.ndarray  # (N, 2) pitch-coordinates in meters
    "team_ids": np.ndarray          # (N,) int: 0=Team1, 1=Team2, -1=GK, -2=Ref
    "ball_position": np.ndarray    # (2,) or None — ball pitch-coordinate
    "player_conf": np.ndarray      # (N,) detection confidences
}
```

---

## Analytics Outputs

**Pitch Analytics tab:**
- Total frames, frames with segments, coverage %
- Region detection frequency bar chart + breakdown
- Raw frame-by-frame JSON export

**Game Analysis tab:**
- Ball possession % (proximity-based, GKs excluded)
- Player density heatmaps (per team on pitch outline)
- Formation scatter + defensive depth metrics
- 9-zone territory control grid
- Match stats (detection rates, player counts, spread, ball progression)

---

## Notes

- ByteTrack tracking is built into Ultralytics `model.track()` — no separate tracker installation needed
- For best results, use tactical (high-angle) camera footage
- Homography EMA smoothing reduces jitter; adjust `SMOOTHING_ALPHA` in [`constants.py`](app/constants.py) if needed