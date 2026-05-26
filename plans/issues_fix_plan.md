# Issues Fix Plan

## Overview

Two issues need to be addressed:

1. **Team Analyzer includes referees** — causing wrong player clustering
2. **Keypoint model hallucinating a 5th point** — only top 4 should be used

---

## Issue 1: Referees Included in Team Color Analysis

### Root Cause

In [`app/player_service.py`](app/player_service.py:8), the `PlayerDetector` has:

```python
self.remove_names = {"ball", "other"}
```

This only filters out "ball" and "other" class detections. The player model (YOLO) likely also detects referees as a separate class (often labeled "referee" or "person"), but since there is no mechanism to filter referees, they are passed through as players. These referee detections then flow into [`TeamColorAnalyzer.assign_team_colors()`](app/keypoint_pipeline.py:154) in [`app/keypoint_pipeline.py`](app/keypoint_pipeline.py:154), where they are clustered into one of the two teams — corrupting the team assignment.

**Evidence**: The notebook output shows "NOTHING REMOVED" printed for every frame, meaning no detections are ever filtered. Some frames show >22 "players" (e.g., 24 players at frame 59), confirming referees are included.

### Solution

**Primary approach**: Add `"referee"` to the `remove_names` set in [`PlayerDetector.__init__`](app/player_service.py:7)

```python
self.remove_names = {"ball", "other", "referee"}
```

This assumes the player model has a "referee" class. This is the cleanest solution.

**Alternative if model doesn't have a "referee" class**: Use post-team-assignment outlier detection. After K-means clustering assigns each player to Team 0 or 1, check each player's distance to both team centroids. If a player's color is far from BOTH teams (using a more sensitive threshold than the goalkeeper detection), mark them as referee (team_id = -1 or a new value like -2).

**Recommended implementation**:
1. First try adding `"referee"` to `remove_names`
2. If the player model doesn't have "referee" as a class, implement a referee detection mechanism in the `TeamColorAnalyzer` that uses a pairwise distance check against both team centroids AND also checks bounding box properties (position on field, aspect ratio) as heuristics

---

## Issue 2: Keypoint Model Predicts a Hallucinated 5th Point

### Root Cause

In [`KeypointHomographyComputer.compute_homography()`](app/keypoint_service.py:306), ALL keypoint candidates that pass the confidence filter ([`KEYPOINT_MIN_CONF=0.3`](app/constants.py:6)) and segmentation validation are used. When exactly 5 candidates pass, all 5 are fed into [`cv2.findHomography(RANSAC)`](app/keypoint_service.py:416). However, one of these 5 corresponds to a pitch feature that isn't actually visible in the frame — the model hallucinated it. Even RANSAC may fail to reject it because the hallucinated point is geometrically consistent with the other 4 points by coincidence.

**Evidence from notebook**: Cell 15 output shows "Keypoints used: 5, Inliers: 5/5" — ALL 5 are inliers, meaning the hallucinated point wasn't filtered by RANSAC. The center circle keypoints have low confidence (0.4-0.6), and the 5th keypoint is likely an additional center-circle or collinear point.

### Solution

**Primary approach**: When selecting candidates, use only the top 4 keypoints ranked by confidence score. This ensures exactly 4 high-quality correspondences are used for the DLT computation, which is the minimum required for a homography.

In [`compute_homography()`](app/keypoint_service.py:306), after collecting all valid candidates, sort by confidence and keep only the top 4:

```python
# Sort candidates by confidence descending and take top 4
candidates.sort(key=lambda c: c['confidence'], reverse=True)
candidates = candidates[:4]
```

**Additional improvements**:
1. **Stricter confidence filter**: Increase [`KEYPOINT_MIN_CONF`](app/constants.py:6) from 0.3 to 0.4 or 0.5 to filter out low-quality predictions
2. **Confidence-weighted RANSAC**: Use the confidence scores to weight correspondences in RANSAC
3. **Spatial validation**: After sorting by confidence, check that the top 4 keypoints are not all collinear (which would make the homography degenerate). The current [`exclude_kpt_ids`](app/keypoint_service.py:192) already excludes center line keypoints (11, 12, 15), but the remaining 4 could still be degenerate if they're all on the center circle.

---

## Implementation Plan

### Step 1: Fix referee filtering in player_service.py
- Add `"referee"` to `self.remove_names` set
- If model doesn't have a referee class, implement color-based referee detection in `team_analyzer.py`

### Step 2: Fix keypoint selection in keypoint_service.py
- After collecting all valid keypoint candidates, sort by confidence descending
- Keep only the top 4 candidates for homography computation
- Ensure the 4 selected keypoints are geometrically diverse (not all collinear)

### Step 3: Review and adjust confidence thresholds
- Consider increasing `KEYPOINT_MIN_CONF` from 0.3 to improve quality
- Update `constants.py` if threshold changes are needed