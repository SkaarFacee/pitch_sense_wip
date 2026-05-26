
# SETTINGS
SEG_CONF = 0.8
PLAYER_CONF = 0.25
KEYPOINT_CONF = 0.3
KEYPOINT_MIN_CONF = 0.3
MAX_FRAMES = None
PROCESS_EVERY_N_FRAMES = 1
REUSE_LAST_HOMOGRAPHY = True

# Homography temporal smoothing — reduces frame-to-frame jitter
SMOOTHING_ALPHA = 0.4       # EMA factor (0=no update, 1=instant). Lower = smoother.
H_STABILITY_THRESHOLD = 0.15  # Max relative change (Frobenius norm) to accept new H

# Team color analysis
TEAM_N_CLUSTERS = 2           # Number of teams to cluster (usually 2)
TEAM_JERSEY_Y_START = 0.12   # Top of jersey crop as fraction of bbox height (12% from top, avoids head)
TEAM_JERSEY_Y_END = 0.50     # Bottom of jersey crop as fraction of bbox height (50% from top, avoids shorts)
TEAM_JERSEY_X_START = 0.15   # Left crop as fraction of bbox width (15% from left, avoids arms)
TEAM_JERSEY_X_END = 0.85     # Right crop as fraction of bbox width (85% from left)
GREEN_HSV_LOWER = (30, 30, 30)    # Lower HSV bound to exclude pitch green (wider range)
GREEN_HSV_UPPER = (90, 255, 255)  # Upper HSV bound for green mask
COLOR_CACHE_REFRESH_N = 5     # Recompute team centroids every N frames (frequent refresh)
GK_COLOR_DIST_THRESHOLD = 2.5  # Std-dev multiplier to flag goalkeeper colors (more permissive)

# Referee detection — color-based (player model has no "referee" class)
REF_DIST_THRESHOLD = 2.0       # Std-dev multiplier to flag referee as outlier from both teams
REF_SATURATION_THRESHOLD = 40  # Max saturation value to flag as referee (black/white/gray)


# PITCH GEOMETRY
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
CENTER_X = PITCH_LENGTH / 2.0
CENTER_Y = PITCH_WIDTH / 2.0
CENTER_CIRCLE_RADIUS = 9.15
PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_WIDTH = 40.32
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_WIDTH = 18.32
PENALTY_SPOT_DISTANCE = 11.0
PENALTY_ARC_RADIUS = 9.15
LEFT_PENALTY_X = PENALTY_AREA_DEPTH
RIGHT_PENALTY_X = PITCH_LENGTH - PENALTY_AREA_DEPTH
LEFT_GOAL_AREA_X = GOAL_AREA_DEPTH
RIGHT_GOAL_AREA_X = PITCH_LENGTH - GOAL_AREA_DEPTH
PENALTY_Y_TOP = (PITCH_WIDTH - PENALTY_AREA_WIDTH) / 2.0
PENALTY_Y_BOTTOM = (PITCH_WIDTH + PENALTY_AREA_WIDTH) / 2.0
GOAL_AREA_Y_TOP = (PITCH_WIDTH - GOAL_AREA_WIDTH) / 2.0
GOAL_AREA_Y_BOTTOM = (PITCH_WIDTH + GOAL_AREA_WIDTH) / 2.0
LEFT_PENALTY_SPOT_X = PENALTY_SPOT_DISTANCE
RIGHT_PENALTY_SPOT_X = PITCH_LENGTH - PENALTY_SPOT_DISTANCE

# CANVAS SETTINGS

DRAW_SCALE = 14 # Convert pixels into a bigger visual dot(s)
BORDER = 80

CANVAS_W = int(PITCH_LENGTH * DRAW_SCALE + 2 * BORDER)
CANVAS_H = int(PITCH_WIDTH * DRAW_SCALE + 2 * BORDER)


# PITCH COLORS 
GREEN=(34, 139, 34)
WHITE=(255, 255, 255)
WHITE_LINE_THICKNESS=2

# SegmentationPriority 
SEGMENTATION_PRIORITY = {
    "18Yard": 140,
    "5Yard": 170,
    "Half Central Circle": 200,
    "18Yard Circle": 100,
    "Half Field": 20,
}