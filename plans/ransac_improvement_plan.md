# RANSAC Improvement Plan

## Problem
- `top_k=4`: Selected points can be collinear or suboptimal
- No `top_k` (all points): Low-quality keypoints degrade the homography

## Solution: RANSAC with ALL good keypoints + minimum inlier check

### Approach
1. **Remove `top_k` limit** — Use ALL keypoints that pass the low visibility threshold (0.05)
2. **RANSAC with all correspondences** — `cv2.findHomography` is specifically designed to handle outliers
3. **Minimum inlier threshold** — Reject solutions with < 4 inliers (degenerate) or inlier ratio < 20%
4. **Spatial diversity check** — After RANSAC, verify the inliers span the pitch (not all in one corner)
5. **Fallback** — If RANSAC fails or produces degenerate H, fall back to `last_H`

### Why this works
RANSAC randomly samples 4-point subsets and finds the one that maximizes inliers. Even with 20 good and 10 bad keypoints, RANSAC will preferentially sample from the 20 good ones. The inlier check then ensures the result is trustworthy.

### Code Changes
In `app/keypoint_service.py`:
- Remove `top_k` parameter entirely
- Use all keypoints passing the visibility filter
- After RANSAC, check `inliers >= 4 AND inlier_ratio >= 0.2`
- Add a spatial diversity heuristic: inlier bounding box must cover at least 30% of pitch width AND height