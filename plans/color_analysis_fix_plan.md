# Color Analysis Accuracy Fix Plan

## Overview

The user has identified two issues:
1. **Color analysis accuracy** — referees are being clustered into teams, potentially corrupting team centroids
2. **Referee color** — currently white (255,255,255), should be black (0,0,0)

The team colors currently showing as **blue** and **light pink** in the legend and debug map should remain unchanged.

---

## Root Cause Analysis

### Issue 1: Referees Not Properly Detected

The player model (YOLO) **does not have a "referee" class** (confirmed in `app/constants.py:26`). Referees are detected as generic "player" detections and flow into the `TeamColorAnalyzer`. While [`_cluster_teams()`](app/team_analyzer.py:276) has post-clustering referee detection, it has two gaps:

**Gap A**: Low-saturation check only catches black/white/gray uniforms. Referees in colored uniforms (e.g., yellow, green, red) have saturation > `REF_SATURATION_THRESHOLD=40` and pass through.

**Gap B**: The outlier check (`dist_to_own > team_mean + REF_DIST_THRESHOLD * team_std`) may fail if a referee's color is coincidentally close to a team centroid, or if the team cluster itself is noisy.

**Gap C**: The cached assignment method [`_assign_to_nearest_team()`](app/team_analyzer.py:359) — used between cache refreshes — **only has the low-saturation check** (lines 392-397). It completely lacks the outlier distance check. This means once a referee is assigned to a team during clustering, they stay in that team for N frames until the next refresh.

**Evidence from notebook output** (frame 300):
```
Teams: Team1=6, Team2=13, GK=0
```
19 total "players" but 0 referees detected — in a real match with 22 players + officials, some of these 19 must be referees that aren't being flagged.

### Issue 2: Referee Color

In [`team_analyzer.py:51`](app/team_analyzer.py:51):
```python
REF_COLOR = (255, 255, 255)  # White
```
Should be black `(0, 0, 0)`.

---

## Proposed Changes

### Change 1: `REF_COLOR` → Black

**File**: [`app/team_analyzer.py`](app/team_analyzer.py:51)

Change `REF_COLOR` from `(255, 255, 255)` to `(0, 0, 0)`.

This is a single-line change.

### Change 2: Filter Referees Before K-Means Clustering

**File**: [`app/team_analyzer.py`](app/team_analyzer.py:276)

Currently:
```
All players → K-means clustering → Outlier/GK/Ref detection → Final labels
```

Proposed:
```
All players → Pre-filter low-saturation (potential referees) → K-means on remaining players → GK detection → Re-check pre-filtered players for ref/player
```

**Why**: By removing low-saturation players (potential referees in black/white/gray) BEFORE clustering, the team centroids won't be contaminated by referee colors.

For colored-uniform referees that pass the saturation check, the post-clustering outlier detection remains as a second line of defense.

### Change 3: Add Outlier Check to `_assign_to_nearest_team()`

**File**: [`app/team_analyzer.py`](app/team_analyzer.py:359)

Currently, the cached assignment method only does a low-saturation check for referee detection (lines 392-397). It should also include:
- Distance check against both team centroids
- If a player is far from both centroids (beyond `REF_DIST_THRESHOLD * std`), flag as referee

### Change 4: Relax Bright Mask for Better Color Extraction

**File**: [`app/team_analyzer.py`](app/team_analyzer.py:212)

Current:
```python
bright_mask = cv2.inRange(hsv_roi, np.array([0, 0, 230]), np.array([180, 80, 255]))
```

This filters out pixels with V > 230 AND S < 80. Light-colored jerseys (like the current light pink) have high V and moderate S values, and could have some pixels incorrectly filtered. Reduce the V lower bound to avoid filtering jersey pixels.

Alternatively, this mask may not need changing if the current team colors are already correct. This can be evaluated after other fixes.

### Change 5: Display Referee Count in Notebook Output

**File**: [`app/keypoint_demo.ipynb`](app/keypoint_demo.ipynb:278-282)

Add referee count to the output:
```python
n_ref = int((team_info['team_ids'] == -2).sum())
print(f"\n🎨 Teams: Team1={n_team1}, Team2={n_team2}, GK={n_gk}, Referee={n_ref}")
```

---

## Files to Modify

| File | Changes |
|------|---------|
| [`app/team_analyzer.py`](app/team_analyzer.py:51) | Line 51: `REF_COLOR = (0, 0, 0)` # Black |
| [`app/team_analyzer.py`](app/team_analyzer.py:276) | Add pre-filtering of low-saturation players before K-means clustering |
| [`app/team_analyzer.py`](app/team_analyzer.py:359) | Add outlier distance check to `_assign_to_nearest_team()` |
| [`app/team_analyzer.py`](app/team_analyzer.py:212) | Relax bright mask if needed |
| [`app/keypoint_demo.ipynb`](app/keypoint_demo.ipynb:278-282) | Add referee count to notebook output |
| [`app/constants.py`](app/constants.py:28) | Consider adjusting `REF_SATURATION_THRESHOLD` for better referee detection |

---

## Verification

1. After changes, run pipeline on a frame that contains referees in colored uniforms
2. Verify referees show as **black** dots on the debug map
3. Verify team colors in legend remain **blue** and **light pink**
4. Verify referee count > 0 in notebook output
