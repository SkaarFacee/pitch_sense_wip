# Plan: Improving Center-Pitch Keypoint Quality

## Problem Analysis

The user reports that center-pitch keypoints are not accurate. From the demo notebook output, center keypoints (IDs 13, 27, 28) have very low visibility/confidence values (~0.1) compared to sideline/penalty area keypoints (~0.6-0.9), yet they still pass through the filter because `KEYPOINT_VISIBILITY` = 0.05 is extremely permissive.

## Key Findings

### 1. Two separate thresholds exist in [`constants.py`](app/constants.py:5-6):
- **`KEYPOINT_CONF = 0.25`**: Controls the YOLO **detection** confidence threshold (passed to `model.predict(conf=...)`). Only affects whether a person detection is accepted at all.
- **`KEYPOINT_VISIBILITY = 0.05`**: Controls the per-keypoint filter. With a threshold of 0.05, virtually ALL keypoints pass through, including very low-confidence ones.

### 2. Center keypoints (IDs 11, 12, 13, 14, 15) are excluded from homography computation:
In [`KeypointHomographyComputer.__init__`](app/keypoint_service.py:178), center-line and center-circle keypoints `{11, 12, 13, 14, 15}` are excluded by default, leaving only `27` (center_circle_left) and `28` (center_circle_right) from the center area.

### 3. The third value in keypoint data is mislabeled:
The `_extract_keypoints` method calls the third value "visibility" (treating it as COCO 0/1/2 flag), but in YOLO11 pose models, this is actually a **confidence score** in [0,1]. The code references it as `visibility` throughout, but the low values suggest the model is simply not confident about center-circle keypoints.

### 4. Current center keypoints from demo output:
| ID | Name | Pitch (x,y) | Visibility/Confidence |
|----|------|------------|----------------------|
| 13 | center_circle_top | (52.50, 24.85) | **0.1** |
| 27 | center_circle_left | (43.35, 34.00) | **0.1** |
| 28 | center_circle_right | (61.65, 34.00) | **0.1** |

---

## Proposed Solution

### Option A: Increase `KEYPOINT_VISIBILITY` threshold
- **What**: Raise from 0.05 to 0.3-0.5 to filter out low-confidence keypoints
- **Trade-off**: Center keypoints (conf=0.1) would be entirely filtered out, leaving NO center-pitch constraints for the homography
- **Risk**: Without center keypoints, the homography relies only on sideline/penalty area keypoints, which could make center-pitch projection less accurate
- **Verdict**: Not recommended unless RANSAC can compensate

### Option B: Keep low visibility threshold but add a **per-keypoint confidence filter** (Recommended)
- **What**: Add a new constant like `KEYPOINT_MIN_CONF = 0.3` that filters individual keypoints by their confidence score (the third value), separate from the YOLO detection confidence
- **Why**: This makes the code clearer and the threshold more explicit. The center keypoints (0.1) get filtered, but RANSAC with 8+ high-quality keypoints from other regions can still produce a good homography
- **Risk**: Need to ensure RANSAC still has enough spatially diverse keypoints (at least 4 non-collinear)

### Option C: **Re-include center keypoints 11, 12, 13, 14, 15 selectively**
- **What**: Remove some of these from `exclude_kpt_ids` so they CAN be used when confidence is high enough
- **Why**: If the model occasionally predicts these well (conf > 0.5), they would add valuable center-pitch constraints
- **Risk**: Collinear keypoints (all on x=52.5 line) can make homography degenerate

### Option D: Increase `KEYPOINT_CONF` (detection confidence)
- **What**: Raise from 0.25 to 0.5
- **Trade-off**: This affects the YOLO detection level, not individual keypoints. May not significantly affect center keypoint quality
- **Verdict**: Unlikely to help directly

---

## Recommendation

Implement **Options A + C combined**:

1. **Increase `KEYPOINT_VISIBILITY` from 0.05 to 0.3** — Filters out the lowest-confidence keypoints that are likely mislocalized
2. **Re-add center keypoints 13 and 14** (center_circle_top/bottom) to the allowed set — These provide vertical constraints at center x=52.5
3. **Keep 11, 12, 15 excluded** — These are perfectly collinear on x=52.5 and would cause degenerate homography
4. **Add a `KEYPOINT_MIN_CONF` constant** (e.g., 0.3) that explicitly controls the per-keypoint confidence threshold, with clearer naming

### Alternative simpler approach (if only "increase conf score"):
Simply **increase `KEYPOINT_VISIBILITY` from 0.05 to 0.3**. This directly implements what the user suggested — raising the confidence threshold to filter out low-quality keypoints.

---

## Required Code Changes

### File: [`app/constants.py`](app/constants.py)
- Rename `KEYPOINT_VISIBILITY` → `KEYPOINT_MIN_CONF` (or keep both)
- Increase from 0.05 to 0.3 (experimentally determined)

### File: [`app/keypoint_service.py`](app/keypoint_service.py)
- In [`KeypointHomographyComputer.__init__`](app/keypoint_service.py:153): Update parameter name and default
- In [`compute_homography`](app/keypoint_service.py:289): The visibility filter at line 355 already works — just need to change the threshold
- Update [`_extract_keypoints`](app/keypoint_service.py:191): Rename variable references from "visibility" to "confidence" for clarity

### File: [`app/keypoint_pipeline.py`](app/keypoint_pipeline.py)
- No changes needed — it reads from `constants.py` which will be updated

---

## Testing Plan

1. Run the demo notebook on a single test frame before and after changes
2. Compare:
   - Number of keypoints used
   - Inlier count/ratio
   - Visual quality of homography (do projected keypoints align with pitch lines?)
   - Player projection accuracy on the top-down pitch canvas
3. Process a full video and compare stability across frames

---

## Diagram: Current vs Proposed Keypoint Flow

```mermaid
flowchart TD
    subgraph Current
        A1[YOLO-Pose Inference] --> B1[conf_threshold=0.25]
        B1 --> C1[vis_threshold=0.05]
        C1 --> D1[exclude: 11,12,13,14,15]
        D1 --> E1[All 10+ kpts including\ncenter kpts with conf=0.1]
    end

    subgraph Proposed
        A2[YOLO-Pose Inference] --> B2[conf_threshold=0.25]
        B2 --> C2[min_conf threshold=0.3]
        C2 --> D2[exclude: only 11,12,15\nkeep 13,14 for center constraints]
        D2 --> E2[Only 6-8 high-quality kpts\ncenter kpts with conf>=0.3]
    end