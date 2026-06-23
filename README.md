# PitchSense ⚽

**PitchSense** turns ordinary football match footage into tactical telemetry: player and ball coordinates projected onto a real pitch, stabilized team identities, possession and spatial analytics, and synchronized review videos without GPS vests, sensor rigs, or manual tagging.

For a university techfest, PitchSense is both a product pitch and a hiring signal. It solves a real sports-analysis access problem for teams that cannot afford proprietary tracking systems, while demonstrating an end-to-end applied AI system across model inference, geometric computer vision, tracking, analytics, and product UX.

---

## Techfest Pitch

**Core pitch:** professional-style match intelligence from one normal video file.

| Audience | Pitch | What to show |
|----------|-------|--------------|
| **Judges** | PitchSense makes advanced football telemetry accessible to schools, clubs, and student teams by replacing expensive hardware with computer vision. | Standard video input, four-model CV stack, homography to a 105 m x 68 m pitch, top-down replay, possession, territory, and pass analytics. |
| **Prospective employers** | This is not just a model demo; it is a full applied-AI product with modular services, sequence-level stabilization, interactive analytics, and a coach-facing dashboard. | `KeypointPipeline`, two-pass processing, ByteTrack integration, team stabilizer, Plotly/Streamlit UI, clean data contracts, and documented architecture. |
| **Other students** | PitchSense is a hackable sports-AI platform that shows how raw pixels become useful tactical insight. | Clear Python modules, visual outputs, Mermaid docs, extensible analytics, and readable pipeline stages. |
| **End users** | Upload a match and get coaching insight: who controlled the ball, where each team played, how compact they were, and which moments deserve review. | Possession timeline, density heatmaps, formation scatter, passing networks, pressing by region, set pieces, and frame inspector. |

---

## What to Emphasize in a Demo

| Signal | Existing feature to emphasize | Why it matters |
|--------|------------------------------|----------------|
| Market viability | No-special-hardware workflow from standard `.mp4`, `.webm`, `.avi`, `.mov`, or `.mkv` footage | Lowers adoption cost for clubs, schools, and analysts. |
| Technical merit | Four-model perception stack: keypoint YOLO-Pose, player detection, dedicated ball detection, and pitch segmentation | Shows breadth across detection, pose/keypoints, segmentation, and small-object tracking. |
| Computer vision depth | 29 pitch landmarks mapped through DLT homography with EMA smoothing | Converts broadcast perspective into real pitch coordinates rather than only drawing boxes. |
| Robustness | ByteTrack plus full-sequence team/role stabilization | Reduces identity/team flicker and proves attention to temporal consistency. |
| Sports intelligence | Possession, heatmaps, formation, territory, passing networks, pressing, set pieces, distance, and speed | Turns detections into decisions coaches understand. |
| Product polish | Streamlit dashboard with Match Centre, Pitch Analysis, Player Analytics, Outputs, and frame inspector | Makes the system demoable and usable beyond notebooks. |
| Recruiting signal | Modular services, documented architecture, explicit configuration, and generated artifacts | Demonstrates software engineering judgment, not just ML experimentation. |

---

## Features That Would Strengthen Appeal

| Priority | New feature | Market and hiring impact |
|----------|-------------|--------------------------|
| 1 | Surface advanced analytics already available in code paths, including xT heatmaps, Voronoi pitch control, defensive-line timelines, and possession-chain histograms | Converts hidden technical work into visible demo value. |
| 2 | Exportable coach report as PDF/HTML with key charts, pass maps, set pieces, and review frames | Makes PitchSense easier to share with coaches and judges after a live demo. |
| 3 | CSV/JSON telemetry exports for player tracks, ball positions, passes, possession, and profiles | Appeals to analysts, researchers, and employers looking for clean data products. |
| 4 | Confidence and evaluation dashboard for ball detection rate, homography quality, track quality, and pass confidence | Shows production thinking, trust calibration, and measurable model quality. |
| 5 | Human-in-the-loop correction for team colors, projection flips, player relabeling, and pass confirmation | Makes the tool more reliable in real club workflows. |
| 6 | Automatic highlight clips for set pieces, turnovers, long possession chains, and high-press moments | Increases end-user value and creates memorable techfest demos. |
| 7 | Reproducible one-command demo with pinned dependencies, sample footage, and optional Docker packaging | Strengthens hiring signal by proving deployment and handoff readiness. |

---

## Current Capabilities

PitchSense processes tactical or broadcast footage to produce:

- **Keypoint detection** -> pitch registration via homography
- **Player detection and tracking** -> ByteTrack plus full-sequence team stabilization
- **Ball detection** -> dedicated YOLO model plus trajectory trail
- **Pitch segmentation** -> region overlay for penalty areas, center circle, and pitch halves
- **Top-down pitch map** -> projected player and ball positions with trail
- **Game analytics** -> possession, heatmaps, formation, territory control, pass networks, pressing, set pieces, match stats

---

## Requirements

- Python 3.10+
- [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO)
- OpenCV, NumPy, Matplotlib, Streamlit, scikit-learn, Plotly

```bash
pip install ultralytics opencv-python numpy matplotlib streamlit scikit-learn plotly
```

---

## Models

Place YOLO `.pt` weights in the following paths (configured in [`app/constants.py`](app/constants.py) and [`app/streamlit_app.py`](app/streamlit_app.py)):

| Model | Expected Path |
|-------|--------------|
| Keypoint (YOLO-Pose) | [`models/keypoint_model/26n_pipeline/no_aug/weights/best.pt`](models/keypoint_model/26n_pipeline/no_aug/weights/best.pt) |
| Player detection | [`models/player_model/best.pt`](models/player_model/best.pt) |
| Segmentation | [`models/segmentation/best.pt`](models/segmentation/best.pt) |
| Ball detection | [`models/ball_model/yolo26_best.pt`](models/ball_model/yolo26_best.pt) |

To use different paths, edit the `MODEL_PATHS` dict in [`app/streamlit_app.py`](app/streamlit_app.py).

---

## Usage

### Streamlit App (recommended)

```bash
streamlit run app/streamlit_app.py
```

Opens a browser UI with five tabs:

1. **🎬 Processing** — Select a video, configure options, and run the full pipeline
2. **📊 Match Centre** — Possession, momentum timeline, team DNA radar, territory control, attacking direction
3. **🗺️ Pitch Analysis** — Heatmaps, formation scatter, ball trail, segmentation regions, zone timelines
4. **👤 Player Analytics** — Passing networks, pressing by region, distance/speed by third, set pieces
5. **🎥 Outputs** — Generated videos plus frame-by-frame inspector

### Processing Options (sidebar)

| Option | Default | Description |
|--------|---------|-------------|
| Max frames | 0 (all) | Limit processing to N frames for testing |
| Team colour clustering | Enabled | Run jersey colour analysis and team assignment |
| Flip projection X | Off | Mirror the long pitch axis if the camera orientation is reversed |
| Flip projection Y | On | Mirror the short pitch axis if players appear on the wrong side |
| Advanced team calibration | Empty JSON | Optional seed colours and track/identity overrides |

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
    │       └── team_analyzer.TeamColorAnalyzer → robust kit feature extraction
    │       └── team_stabilizer.TeamSequenceStabilizer → stable identity/team/role labels
    │
    ├── ball_service.BallDetector
    │       └── YOLO ball model → bbox → bottom-center → project via H → trajectory
    │
    ├── pitch.PitchArtist
    │       └── Draw top-down pitch canvas with players, ball, trail, legend
    │
    └── keypoint_pipeline.KeypointPipeline.process_video()
            └── Pass 1: inference + compact metadata
            └── Pass 2: stabilized rendering + 5 output videos
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

| File | Purpose |
|------|---------|
| [`streamlit_app.py`](app/streamlit_app.py) | Streamlit UI (Processing, Match Centre, Pitch Analysis, Player Analytics, Outputs) |
| [`keypoint_pipeline.py`](app/keypoint_pipeline.py) | Core pipeline orchestrator, two-pass video processing, rendering, video writers |
| [`team_stabilizer.py`](app/team_stabilizer.py) | Sequence-level identity linking, stable team membership, role assignment, diagnostics |
| [`game_analyzer.py`](app/game_analyzer.py) | Possession, heatmaps, formation, territory, match stats, player profiles |
| [`team_analyzer.py`](app/team_analyzer.py) | Robust jersey/shorts feature extraction and best-effort online fallback |
| [`keypoint_service.py`](app/keypoint_service.py) | YOLO-Pose → homography (DLT + EMA smoothing) |
| [`pitch.py`](app/pitch.py) | Top-down pitch canvas drawing (lines, players, ball, trail) |
| [`ball_service.py`](app/ball_service.py) | Ball detection + pitch projection |
| [`segmentation.py`](app/segmentation.py) | YOLO-seg → quad extraction for pitch regions |
| [`seg_helpers.py`](app/seg_helpers.py) | Canvas coordinate mapping for segmentation regions |
| [`player_service.py`](app/player_service.py) | YOLO player detection + ByteTrack + projection |
| [`constants.py`](app/constants.py) | Configurable thresholds, geometry, colors, team/role schema constants |
| [`director.py`](app/director.py) | H.264 video writer factory |

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

Collected during processing for the analytics dashboard tabs:

```python
{
    "frame_idx": int,
    "player_positions": np.ndarray  # (N, 2) pitch-coordinates in meters
    "track_ids": np.ndarray         # (N,) raw ByteTrack IDs
    "identity_ids": np.ndarray      # (N,) canonical full-sequence identity IDs
    "team_ids": np.ndarray          # (N,) int: 0=Team1, 1=Team2, -1=NO_TEAM
    "role_ids": np.ndarray          # (N,) int: 0=Outfield, 1=GK, 2=Ref, -1=Unknown
    "ball_position": np.ndarray    # (2,) or None — ball pitch-coordinate
    "player_conf": np.ndarray      # (N,) detection confidences
}
```

`team_ids` now represents actual team membership only. Goalkeepers keep their team membership (`0` or `1`) and are identified by `role_ids == 1`. Referees/unknowns use `team_ids == -1` and `role_ids == 2` or `-1`.

Full-video processing uses a two-pass stabilizer. The zero-switch guarantee applies to `KeypointPipeline.process_video()`, where each canonical `identity_id` receives one stable team assignment for the whole sequence. `process_frame()` remains available as best-effort online behavior.

---

## Analytics Outputs

**Match Centre tab:**
- Ball possession % using bbox overlap and sticky carry-forward logic
- Rolling possession timeline and possession donut
- Team DNA radar, attacking direction, and 9-zone territory control

**Pitch Analysis tab:**
- Player density heatmaps per team on a pitch outline
- Formation scatter, average spread, and ball trajectory trail
- Region detection frequency charts and pitch-zone timelines

**Player Analytics tab:**
- Passing networks by team and pitch third
- Pressing intensity by pitch region
- Per-player distance, top speed, dominant third, and set-piece counts

**Outputs tab:**
- Generated analysis videos
- Frame inspector with synchronized previews, ball location, possession, and region tags

---

## Notes

- ByteTrack tracking is built into Ultralytics `model.track()` — no separate tracker installation needed
- For best results, use tactical (high-angle) camera footage
- Homography EMA smoothing reduces jitter; adjust `SMOOTHING_ALPHA` in [`constants.py`](app/constants.py) if needed
