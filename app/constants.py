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
COLOR_CACHE_REFRESH_N = 5     # Recompute team centroids every N frames (periodic refresh)
GK_COLOR_DIST_THRESHOLD = 2.5  # Std-dev multiplier to flag goalkeeper colors (more permissive)

# Referee detection — color-based (player model has no "referee" class)
REF_DIST_THRESHOLD = 2.0       # Std-dev multiplier to flag referee as outlier from both teams
REF_SATURATION_THRESHOLD = 40  # Max saturation value to flag as referee (black/white/gray)

# Track-aware clustering (robustness for moving players + referees)
TRACK_HISTORY_LEN = 30           # Per-track deque length for running-median color
TRACK_MIN_FRAMES_TO_CLUSTER = 5  # Min frames a track must be seen before it contributes to re-clustering
TRACK_MIN_FRAMES_FOR_TEAM = 4    # Min frames before a sticky team assignment is trusted
TRACK_STICKY_DIST = 25.0         # HSV dist below which we trust a track's existing team assignment
TRACK_RELABEL_DIST = 50.0        # HSV dist below which we re-evaluate team id (vs treating as occlusion)
STALE_TRACK_FRAMES = 90          # Drop tracks not seen for this many frames
WARMUP_FRAMES = 15               # Accumulate tracks for this many frames before initial K-means
CENTROID_EMA_ALPHA = 0.2         # EMA factor for periodic centroid updates (lower = more stable)
GK_MIN_FRAMES = 6                # Frames a track must persist as outlier before being labelled GK
REF_MIN_FRAMES = 6               # Frames a track must persist as low-sat / outlier before being labelled REF

# --- Tier 1: robust color extraction ---------------------------------------
TEAM_JERSEY_Y_END_EXTENDED = 0.55   # Optional extended band for shorts signal (off by default)
ILLUMINATION_NORMALIZE = True       # Gray-world per-crop illumination normalization
ADAPTIVE_JERSEY_BAND = True         # Tighten/widen Y_END based on bbox aspect ratio
SHORTS_BAND_Y_START = 0.45
SHORTS_BAND_Y_END = 0.62
EMA_PER_TRACK_ALPHA = 0.3           # EMA factor for per-track color smoothing (lower = stickier)
INVALID_PIXEL_MIN = 30              # Tighten the invalid-pixel fallback when fewer than this many valid pixels

# --- Tier 1.4: multi-cue referee detection ---------------------------------
REF_SAT_HIST_FRACTION = 0.40        # Fraction of observations with s < 30 → referee
REF_SAT_HIST_THRESHOLD = 30         # "very low saturation" cap for histogram peak
REF_HUE_MULTIMODAL_FRAC = 0.15      # Fraction threshold per hue mode to count as multi-modal
REF_HUE_MULTIMODAL_MODES = 3        # At least this many modes above the threshold → referee
REF_BINS = 18                       # Histogram bins for saturation / hue analysis
TOUCHLINE_MARGIN_M = 2.0            # Within this many meters of the touchline → linesman prior
GK_PENALTY_BOX_MIN_FRAC = 0.5       # Fraction of last N frames a GK must spend in the box

# --- Tier 2: smarter clustering core ---------------------------------------
USE_GMM = False                     # Use Gaussian Mixture Model instead of KMeans when True
GMM_COVARIANCE_TYPE = "full"        # "full" | "diag" | "spherical"
GMM_MIN_PROB_FOR_TEAM = 0.6         # Below this probability → uncertain (REF/GK)
RE_CLUSTER_DRIFT_THRESHOLD = 8.0    # Mean per-track HSV shift (avg over all tracks) that triggers immediate re-cluster
TRACK_QUALITY_EMA_ALPHA = 0.2       # EMA factor for per-track quality in team_analyzer
TRACK_QUALITY_LABEL_FLIP_PENALTY = 0.5  # Multiplier on label_flip_rate in track_quality formula
TRACK_HISTORY_SHORT_TERM = 10       # Window for label-flip rate computation (last N votes)

# --- Tier 3: pose & shape priors -------------------------------------------
USE_POSE_AWARE_CROP = False         # Requires a pose model — currently no-op (Tier 3.1 dropped)

# --- Similar-team disambiguation -------------------------------------------
# When the two team centroids in HSV (jersey only) are closer than this,
# treat the split as unreliable and fall back to a pitch-position-aware
# split (different teams usually start in different halves).
SIMILAR_TEAM_CENTROID_DIST = 25.0
# Top-K fraction of saturated pixels used to compute the per-track jersey
# feature (more discriminative than naive median when stripes are involved).
JERSEY_TOPK_FRACTION = 0.20
# Weight of the shorts-band feature in the multi-band distance metric
# (0 = ignore shorts, 1 = shorts and jersey contribute equally).
SHORTS_FEATURE_WEIGHT = 0.6


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

# Ball detection
BALL_CONF = 0.25                  # Confidence threshold for ball detection
BALL_TRAIL_LENGTH = 50            # Number of past positions for ball trajectory trail
BALL_BBOX_COLOR = (0, 255, 0)     # Green for ball bbox on annotated frame
BALL_DOT_COLOR = (0, 255, 255)    # Yellow for ball dot on pitch canvas

# SegmentationPriority
SEGMENTATION_PRIORITY = {
    "18Yard": 140,
    "5Yard": 170,
    "Half Central Circle": 200,
    "18Yard Circle": 100,
    "Half Field": 20,
}