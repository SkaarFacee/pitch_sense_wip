# Keypoint-Based Homography Pipeline Plan

## Overview
Replace the segmentation-based homography pipeline with a **keypoint-based** approach for computing the perspective transform. The segmentation model is repurposed for "deep analysis" visual overlays.

## Pipeline Flow

```
Video Frame
    │
    ├──► Keypoint Model (YOLO-Pose, 29 kpts)
    │       │
    │       ├── kpt_id → image (x, y)
    │       └── kpt_name → pitch (x, y)  [PitchKeypointMapper]
    │           │
    │           └── cv2.findHomography(RANSAC) → H matrix
    │
    ├──► Player Model (YOLO-Detection)
    │       │
    │       └── Player boxes → bottom-center (x, y)
    │           │
    │           └── cv2.perspectiveTransform(bottom_center, H) → pitch coords
    │
    ├──► Segmentation Model (YOLO-Seg)
    │       │
    │       └── Pitch region masks → overlay on original frame (deep analysis)
    │
    └──► Output:
            ├── Annotated video (original frame + segmentation overlay + keypoints)
            └── Top-down pitch canvas (players projected)
```

## Files to Create/Modify

### 1. ENHANCE `app/keypoint_service.py`
Add two new classes:

**A) `PitchKeypointMapper`** — Static mapping from keypoint ID 0-28 to real-world pitch coordinates (meters).

- Maps each of the 29 keypoint names to `(x_pitch, y_pitch)` using constants from `constants.py`
- Method: `get_pitch_coords(kpt_id: int) -> (float, float) | None`
- Method: `get_keypoint_name(kpt_id: int) -> str`

**B) `KeypointHomographyComputer`** — Runs inference, filters, and computes H.

- `__init__(model_path, conf_threshold=0.5, visibility_threshold=0.5)`
- `compute_homography(frame, last_H=None) -> (H, info_dict)`:
  1. Run YOLO pose inference on frame
  2. Extract keypoints from result (`result.keypoints`)
  3. Filter: keep keypoints with confidence >= threshold AND visibility >= visibility_threshold
  4. Map filtered keypoints to pitch coordinates via `PitchKeypointMapper`
  5. Collect image_points[] and pitch_points[] arrays
  6. If >= 4 correspondences: `cv2.findHomography(image_pts, pitch_pts, cv2.RANSAC, 3.0)`
  7. If < 4 but > 0: fallback to last known H (if provided and REUSE_LAST_HOMOGRAPHY=True)
  8. Return (H, info_dict) where info_dict has: used_keypoints, mode, inliers, total

### 2. CREATE `app/keypoint_pipeline.py`
Orchestrator module that runs the full pipeline on a video.

```
class KeypointPipeline:
    def __init__(self, keypoint_model_path, player_model_path, seg_model_path):
        self.keypoint_comp = KeypointHomographyComputer(keypoint_model_path)
        self.player_detector = PlayerDetector(player_model_path)
        self.segmentor = Segmentor(seg_model_path)  # reuse from seg_helpers
        self.last_H = None
```

Key method: `process_frame(frame) -> dict`
- Returns: {H, player_pitch_points, seg_overlay_frame, keypoints_used, debug_info}

Helper methods:
- `create_annotated_frame(frame, keypoints_used, H_info)` — draw keypoints on frame
- `create_pitch_canvas(player_pitch_points, seg_data)` — top-down view with players + optional seg overlay
- `create_deep_analysis_frame(frame, seg_result)` — original frame with seg masks overlaid

### 3. CREATE NEW NOTEBOOK (e.g., `app/keypoint_demo.ipynb`)
Demonstration notebook that:
- Loads all models
- Runs the pipeline on a video
- Shows side-by-side outputs: annotated frame + pitch canvas + deep analysis overlay

### 4. UPDATE `app/del.ipynb` (or mark as deprecated)
Update to reference the new pipeline module.

## Detailed Keypoint-to-Pitch Mapping

| ID | Name | Pitch X | Pitch Y | Derived From |
|-----|------|---------|---------|--------------|
| 0 | sideline_top_left | 0 | 0 | Top-left corner |
| 1 | big_rect_left_top_pt1 | 0 | 13.84 | (0, PENALTY_Y_TOP) |
| 2 | big_rect_left_top_pt2 | 16.5 | 13.84 | (LEFT_PENALTY_X, PENALTY_Y_TOP) |
| 3 | big_rect_left_bottom_pt1 | 0 | 54.16 | (0, PENALTY_Y_BOTTOM) |
| 4 | big_rect_left_bottom_pt2 | 16.5 | 54.16 | (LEFT_PENALTY_X, PENALTY_Y_BOTTOM) |
| 5 | small_rect_left_top_pt1 | 0 | 24.84 | (0, GOAL_AREA_Y_TOP) |
| 6 | small_rect_left_top_pt2 | 5.5 | 24.84 | (LEFT_GOAL_AREA_X, GOAL_AREA_Y_TOP) |
| 7 | small_rect_left_bottom_pt1 | 0 | 43.16 | (0, GOAL_AREA_Y_BOTTOM) |
| 8 | small_rect_left_bottom_pt2 | 5.5 | 43.16 | (LEFT_GOAL_AREA_X, GOAL_AREA_Y_BOTTOM) |
| 9 | sideline_bottom_left | 0 | 68 | (0, PITCH_WIDTH) |
| 10 | left_semicircle_right | 20.15 | 34 | (SPOT_X + ARC_RADIUS, CENTER_Y) |
| 11 | center_line_top | 52.5 | 0 | (CENTER_X, 0) |
| 12 | center_line_bottom | 52.5 | 68 | (CENTER_X, PITCH_WIDTH) |
| 13 | center_circle_top | 52.5 | 24.85 | (CENTER_X, CENTER_Y - RADIUS) |
| 14 | center_circle_bottom | 52.5 | 43.15 | (CENTER_X, CENTER_Y + RADIUS) |
| 15 | field_center | 52.5 | 34 | (CENTER_X, CENTER_Y) |
| 16 | sideline_top_right | 105 | 0 | (PITCH_LENGTH, 0) |
| 17 | big_rect_right_top_pt1 | 88.5 | 13.84 | (RIGHT_PENALTY_X, PENALTY_Y_TOP) |
| 18 | big_rect_right_top_pt2 | 105 | 13.84 | (PITCH_LENGTH, PENALTY_Y_TOP) |
| 19 | big_rect_right_bottom_pt1 | 88.5 | 54.16 | (RIGHT_PENALTY_X, PENALTY_Y_BOTTOM) |
| 20 | big_rect_right_bottom_pt2 | 105 | 54.16 | (PITCH_LENGTH, PENALTY_Y_BOTTOM) |
| 21 | small_rect_right_top_pt1 | 99.5 | 24.84 | (RIGHT_GOAL_AREA_X, GOAL_AREA_Y_TOP) |
| 22 | small_rect_right_top_pt2 | 105 | 24.84 | (PITCH_LENGTH, GOAL_AREA_Y_TOP) |
| 23 | small_rect_right_bottom_pt1 | 99.5 | 43.16 | (RIGHT_GOAL_AREA_X, GOAL_AREA_Y_BOTTOM) |
| 24 | small_rect_right_bottom_pt2 | 105 | 43.16 | (PITCH_LENGTH, GOAL_AREA_Y_BOTTOM) |
| 25 | sideline_bottom_right | 105 | 68 | (PITCH_LENGTH, PITCH_WIDTH) |
| 26 | right_semicircle_left | 84.85 | 34 | (RIGHT_SPOT_X - ARC_RADIUS, CENTER_Y) |
| 27 | center_circle_left | 43.35 | 34 | (CENTER_X - RADIUS, CENTER_Y) |
| 28 | center_circle_right | 61.65 | 34 | (CENTER_X + RADIUS, CENTER_Y) |

## Reuse Strategy
- If fewer than 4 valid keypoint correspondences are available for a frame, fall back to `last_H_matrix` (controlled by `constants.REUSE_LAST_HOMOGRAPHY`)
- The `info_dict` tracks which keypoints were used, confidence scores, inlier counts

## Segmentation "Deep Analysis"
- Run segmentation model on the frame
- Overlay the segmentation masks on the original frame (transparent colored polygons per class)
- Optionally, also overlay on the transformed pitch canvas for comparison
- This provides visual confirmation of pitch region detection quality

## Dependencies
- `ultralytics` (YOLO)
- `opencv-python` (cv2)
- `numpy`
- `matplotlib` (notebook visualization)
- All existing modules reused: `constants.py`, `pitch.py`, `player_service.py`, `segmentation.py`, `director.py`
