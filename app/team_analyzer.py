"""TeamColorAnalyzer — Track-aware jersey-colour clustering with multi-tier robustness.

Tiers implemented (all gated behind flags in `constants.py`):
* Tier 1.1 — Gray-world illumination normalization on the jersey crop before
  HSV sampling.
* Tier 1.2 — Adaptive jersey band that tightens Y_END for tall bboxes and
  widens X band for small bboxes (handles jumping / sliding / camera tilt).
* Tier 1.3 — Per-track EMA (`track_ema_bgr`) layered on top of the running
  median. Median stays the outlier-resistant fallback; EMA is what
  centroid-distance comparisons use — smooths motion blur without losing
  the ability to reject bad frames.
* Tier 1.4 — Multi-cue referee detection: low-saturation histogram peak
  AND hue multi-modality check. Spatial touchline prior is also wired in
  when an `H` (homography) is supplied — projected touchline occupancy
  flags linesmen.
* Tier 2.1 — Optional Gaussian Mixture Model (`USE_GMM = True`) with soft
  probabilities. Hard `team_ids` preserved for backwards compatibility.
* Tier 2.2 — Drift-triggered re-clustering: if mean per-track shift vs the
  current centroids exceeds `RE_CLUSTER_DRIFT_THRESHOLD`, re-cluster
  immediately (handles half-time swaps / replays with reversed direction).
* Tier 2.3 — `track_quality` field in the result dict (per detection in
  [0, 1]). Down-weight low-quality tracks in possession / heatmap
  / territory analytics.
* Tier 3.2 — Pitch-position prior for GK: when `H` and the bbox are
  available, a track only keeps its GK label if > `GK_PENALTY_BOX_MIN_FRAC`
  of the last 30 frames are inside a penalty area.
* Tier 3.3 — Two-stage jersey → shorts disambiguation for uncertain tracks.
"""
from collections import deque
import numpy as np
import cv2
from sklearn.cluster import KMeans
from constants import (
    TEAM_N_CLUSTERS, TEAM_JERSEY_Y_START, TEAM_JERSEY_Y_END, TEAM_JERSEY_X_START, TEAM_JERSEY_X_END,
    GREEN_HSV_LOWER, GREEN_HSV_UPPER, REF_SATURATION_THRESHOLD,
    COLOR_CACHE_REFRESH_N, TRACK_HISTORY_LEN, TRACK_MIN_FRAMES_TO_CLUSTER,
    TRACK_MIN_FRAMES_FOR_TEAM, TRACK_STICKY_DIST, TRACK_RELABEL_DIST,
    STALE_TRACK_FRAMES, WARMUP_FRAMES, CENTROID_EMA_ALPHA,
    GK_MIN_FRAMES, REF_MIN_FRAMES,
    # Tier 1
    ILLUMINATION_NORMALIZE, ADAPTIVE_JERSEY_BAND, EMA_PER_TRACK_ALPHA,
    INVALID_PIXEL_MIN, SHORTS_BAND_Y_START, SHORTS_BAND_Y_END,
    # Tier 1.4
    REF_SAT_HIST_FRACTION, REF_SAT_HIST_THRESHOLD, REF_HUE_MULTIMODAL_FRAC,
    REF_HUE_MULTIMODAL_MODES, REF_BINS, TOUCHLINE_MARGIN_M,
    GK_PENALTY_BOX_MIN_FRAC, GK_BOX_MIN_FRAMES, GK_GOAL_AREA_MIN_FRAC,
    GK_LEFTMOST_X_MARGIN, GK_RIGHTMOST_X_MARGIN, GK_EDGE_MIN_FRAMES,
    # Tier 1.5
    JERSEY_BAND_FRONT_Y_START, JERSEY_BAND_FRONT_Y_END,
    JERSEY_BAND_BACK_Y_START, JERSEY_BAND_BACK_Y_END,
    JERSEY_BAND_BACK_WEIGHT, SIDE_PANEL_X_FRAC,
    JERSEY_BACK_TOPK_FRACTION, COLD_START_DIST_GAP, COLD_START_FRAMES,
    # Tier 1.6
    REF_OUTLIER_MIN_DIST_RATIO, REF_PEAK_HUE_FRAC, REF_TOUCHLINE_MIN_FRAC,
    REF_MIN_FRAMES_FAST,
    # Tier 2
    USE_GMM, GMM_COVARIANCE_TYPE, GMM_MIN_PROB_FOR_TEAM,
    RE_CLUSTER_DRIFT_THRESHOLD, TRACK_QUALITY_EMA_ALPHA,
    TRACK_QUALITY_LABEL_FLIP_PENALTY, TRACK_HISTORY_SHORT_TERM,
    # Similar-team disambiguation
    SIMILAR_TEAM_CENTROID_DIST, JERSEY_TOPK_FRACTION, SHORTS_FEATURE_WEIGHT,
    # Tier 3.4
    GK_DEFENSIVE_THIRD_MIN_FRAC, GK_DEMOTE_IN_BOX_FRAC,
    GK_DEMOTE_DEF_FRAC, GK_DEMOTE_CONSEC_FRAMES,
    GK_MAX_OUTFIELDERS_PER_TEAM,
)

# Penalty + goal-area geometry (must mirror constants.PENALTY_* / GOAL_AREA_* values)
from constants import (
    PITCH_LENGTH, PITCH_WIDTH, CENTER_X, CENTER_Y,
    LEFT_PENALTY_X, RIGHT_PENALTY_X,
    PENALTY_Y_TOP, PENALTY_Y_BOTTOM,
    LEFT_GOAL_AREA_X, RIGHT_GOAL_AREA_X,
    GOAL_AREA_Y_TOP, GOAL_AREA_Y_BOTTOM,
)

# How many frames a track must have been seen with a team label before
# the per-frame decision locks in (no more per-frame re-voting — the
# track's running median drives the label from then on).
TEAM_LOCK_MIN_FRAMES = 8
# Once a track has been seen for TEAM_HARD_LOCK_FRAMES frames AND
# has a TEAM0/TEAM1 label, that label is HARD-LOCKED. Once locked, no
# per-frame vote can change the label — only the position-based priors
# (GK) can demote. This is the ByteTrack ID stickiness guarantee.
TEAM_HARD_LOCK_FRAMES = 5
# Consecutive frames on the same TEAM0/TEAM1 label that trigger an
# early `team_locked = True` even before TEAM_HARD_LOCK_FRAMES elapses.
TEAM_CONSEC_STREAK_LOCK = 3

# Spatial-continuity team lock: minimum IoU between a current detection
# and a confident prev-frame TEAM0/TEAM1 box for the current (possibly
# new) track to inherit the prev box's team identity. Symmetric (mutual
# 1-to-1 best match) so converging/ambiguous boxes are rejected.
SPATIAL_CONTINUITY_IOU = 0.5

# How many position samples are required before the pitch-position
# fallback engages for similar-colored teams.
POSITION_MIN_SAMPLES = 5


def _safe_import_gmm():
    """GMM is optional. Import lazily so KMeans-only callers don't pay the cost."""
    try:
        from sklearn.mixture import GaussianMixture
        return GaussianMixture
    except Exception:
        return None


class TeamColorAnalyzer:
    DEFAULT_TEAM_COLORS = [(255, 0, 0), (0, 0, 255)]
    GK_COLOR = (0, 255, 255)
    REF_COLOR = (0, 0, 0)

    # Class labels
    TEAM0 = 0
    TEAM1 = 1
    GK = -1
    REF = -2

    REF_BBOX_AREA = 60.0 * 120.0

    def __init__(self, n_clusters: int = TEAM_N_CLUSTERS):
        self.n_clusters = n_clusters
        self.team_centroids_bgr_feat = None
        self.team_centroids_bgr = None
        self.initialized = False
        self.gmm_model = None  # populated when USE_GMM=True

        self.tracks: dict[int, dict] = {}
        self.frame_count = 0
        self._last_centroids_hsv = None  # for drift-trigger re-cluster
        self._last_H = None  # latest homography (for Tier 1.6 touchline check)

        # Spatial-continuity team lock (anti-jitter for ByteTrack ID
        # switches). Each entry: {'bbox': np.ndarray(4,), 'team_id': int}.
        # Only confident/locked TEAM0/TEAM1 detections are recorded, and
        # only TEAM0/TEAM1 prev entries are eligible for inheritance.
        self._prev_frame_detections: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def assign_team_colors(self, frame: np.ndarray, player_xyxy: np.ndarray,
                           player_conf: np.ndarray, track_ids: np.ndarray = None,
                           H: np.ndarray = None,
                           player_pitch_pts: np.ndarray = None) -> dict:
        """Assign per-detection team labels.

        Optional `H` (3x3 homography) enables Tier 1.4 touchline check and
        Tier 3.2 GK-penalty-box prior. Both are best-effort and silently
        no-op if `H` is missing.

        Optional `player_pitch_pts` (Nx2 projected pitch coordinates
        aligned with `player_xyxy`) is used by the position-based GK
        decision — specifically the leftmost/rightmost safety net. If
        missing, the safety net silently no-ops; the rest of the
        position-based logic still works off `_record_pitch_positions`.
        """
        if len(player_xyxy) == 0:
            # Clear the prev-frame buffer so a one-frame gap (e.g. pitch
            # mask dropping everyone) doesn't match a stale box next frame.
            self._prev_frame_detections = []
            self._cleanup_stale_tracks()
            return self._empty_result()

        # Tier 1.1: apply gray-world illumination normalization to the whole
        # frame ONCE before per-crop extraction. Per-crop gray-world is wrong
        # because each crop is dominated by jersey color — it normalizes the
        # very signal we want to read.
        if ILLUMINATION_NORMALIZE:
            frame = self._illumination_normalize(frame)

        # 1) Per-detection color extraction (Tier 1.2 adaptive band) — 4D feature
        per_det_feature = []
        per_det_bgr = []
        per_det_weight = []
        for i, bbox in enumerate(player_xyxy):
            feature, bgr_c = self._extract_dominant_color(frame, bbox)
            per_det_feature.append(feature)
            per_det_bgr.append(bgr_c)
            area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            conf = float(player_conf[i]) if player_conf is not None and i < len(player_conf) else 1.0
            per_det_weight.append(conf * min(1.0, area / self.REF_BBOX_AREA))
        per_det_feature = np.array(per_det_feature, dtype=np.float32)
        per_det_bgr = np.array(per_det_bgr, dtype=np.float32)
        per_det_weight = np.array(per_det_weight, dtype=np.float32)

        # 2) Update per-track state (Tier 1.3 EMA layered on top of median)
        track_ids = self._normalize_track_ids(track_ids, len(player_xyxy))
        for i, tid in enumerate(track_ids):
            self._update_track(tid, per_det_feature[i], per_det_bgr[i], per_det_weight[i])

        # Record pitch positions for the position-aware tiebreaker
        if H is not None:
            self._record_pitch_positions(track_ids, player_xyxy, H)
            self._last_H = H
        else:
            self._last_H = None

        self.frame_count += 1

        # 3) Initialization
        if not self.initialized:
            if self.frame_count >= WARMUP_FRAMES or self._enough_tracks_to_init():
                self._initialize_from_tracks()

        # 4) Periodic + drift-triggered re-clustering (Tier 2.2)
        if self.initialized:
            if self.frame_count % COLOR_CACHE_REFRESH_N == 0:
                self._recluster_from_tracks()
            elif self._drift_exceeded():
                self._recluster_from_tracks(force=True)

        # 5) Decide this-frame team_id per detection
        team_ids = self._decide_per_frame_team_ids(
            track_ids, per_det_feature, frame, player_xyxy, H,
            player_pitch_pts=player_pitch_pts,
        )

        # 6) Stale-track cleanup
        self._cleanup_stale_tracks()

        if not self.initialized or team_ids is None or len(team_ids) == 0:
            return self._empty_result()

        result = self._build_result(team_ids, track_ids)

        # Record confident/locked TEAM0/TEAM1 detections as the prev-frame
        # source for next frame's spatial-continuity check. Only TEAM0/TEAM1
        # labels are inheritable, and only confident/locked tracks are
        # recorded so warm-up votes never seed a wrong inheritance.
        prev_dets: list[dict] = []
        for i in range(len(player_xyxy)):
            label = int(team_ids[i])
            if label not in (self.TEAM0, self.TEAM1):
                continue
            t = self.tracks.get(int(track_ids[i]))
            if t is None:
                continue
            if not (t.get('team_locked', False) or t['frames_seen'] >= TEAM_LOCK_MIN_FRAMES):
                continue
            prev_dets.append({
                'bbox': player_xyxy[i].copy(),
                'team_id': label,
            })
        self._prev_frame_detections = prev_dets

        return result

    # ------------------------------------------------------------------
    # Per-track bookkeeping
    # ------------------------------------------------------------------
    def _normalize_track_ids(self, track_ids, n):
        if track_ids is None or len(track_ids) == 0:
            return np.arange(n, dtype=np.int32)
        return np.asarray(track_ids, dtype=np.int32)

    def _new_track(self) -> dict:
        return {
            'obs': deque(maxlen=TRACK_HISTORY_LEN),
            'sat_obs': deque(maxlen=TRACK_HISTORY_LEN),     # Tier 1.4: per-obs BGR colorfulness proxy
            'hue_obs': deque(maxlen=TRACK_HISTORY_LEN),      # Tier 1.4: per-obs B channel (proxy)
            'track_bgr': None,          # (3,) jersey feature [B, G, R] — BGR space
            'track_feature': None,      # (6,) [B_jersey, G_jersey, R_jersey, B_shorts, G_shorts, R_shorts]
            'track_ema_bgr': None,                            # Tier 1.3 (BGR version)
            'team_id': None,
            'team_votes': {0: 0, 1: 0, self.GK: 0, self.REF: 0},
            'team_votes_history': deque(maxlen=TRACK_HISTORY_SHORT_TERM),  # Tier 2.3
            'frames_seen': 0,
            'last_seen_frame': 0,
            'low_sat_streak': 0,
            'gk_streak': 0,
            'ref_outlier_streak': 0,
            'quality': 1.0,                                   # Tier 2.3
            'shorts_bgr': None,                               # Tier 3.3 (BGR version)
            'pitch_positions': deque(maxlen=30),              # Tier 3.2: GK penalty-box prior
            'soft_team_probs': None,                          # Tier 2.1: P(team | track_feature)
            'gk_locked': False,                               # Tier 3.4: hysteresis
            'gk_demote_streak': 0,                            # Tier 3.4: consec frames meeting demote criteria
            'cold_start_team': None,                          # Tier 3: anti-vote owner-team
            'team_locked': False,                             # Tier 1.5-BGR: hard-lock by ByteTrack ID
            # Position-based GK (plan (b))
            'box_frames': 0,                                  # CONSECUTIVE frames in 18-yd box (resets on exit)
            'goal_area_frames': 0,                            # CONSECUTIVE frames in 6-yd box
            'leftmost_streak': 0,                             # consec frames as leftmost player
            'rightmost_streak': 0,                            # consec frames as rightmost player
            'gk_candidate_locked': False,                     # promoted via box / edge streak
            'team_consec_streak': 0,                          # consec frames on current TEAM0/TEAM1 label
        }

    def _update_track(self, tid: int, feature: np.ndarray, bgr_c: np.ndarray, weight: float):
        """feature is now 6D BGR: [B_j, G_j, R_j, B_s, G_s, R_s].

        Also computes HSV-based saturation/hue per-obs for the referee
        cues. HSV is computed only on the jersey region (small area,
        cheap) and stored as `sat_obs`/`hue_obs` for ref detection.
        """
        t = self.tracks.get(int(tid))
        if t is None:
            t = self._new_track()
            self.tracks[int(tid)] = t

        # Update BGR EMA (jersey only, 3D)
        if t['track_ema_bgr'] is None:
            t['track_ema_bgr'] = np.array([float(feature[0]), float(feature[1]), float(feature[2])], dtype=np.float32)
        else:
            a = EMA_PER_TRACK_ALPHA
            t['track_ema_bgr'] = a * np.array([float(feature[0]), float(feature[1]), float(feature[2])], dtype=np.float32) \
                                  + (1.0 - a) * t['track_ema_bgr']

        if weight < 0.08:
            t['last_seen_frame'] = self.frame_count
            t['frames_seen'] += 1
            return

        # Obs stores the full 6D BGR feature + weight
        t['obs'].append((float(feature[0]), float(feature[1]), float(feature[2]),
                         float(feature[3]), float(feature[4]), float(feature[5]),
                         float(weight)))

        # Compute HSV saturation/hue for the jersey BGR for ref cues.
        # HSV normalizes for brightness, so shadows don't drop saturation
        # — this is more robust than (max-min) in BGR.
        jersey_bgr = np.array([[
            [float(feature[0]), float(feature[1]), float(feature[2])]
        ]], dtype=np.uint8)
        try:
            jersey_hsv = cv2.cvtColor(jersey_bgr, cv2.COLOR_BGR2HSV)[0, 0]
            sat_value = float(jersey_hsv[1])
            hue_value = float(jersey_hsv[0])
        except Exception:
            sat_value = 0.0
            hue_value = 0.0
        t['sat_obs'].append(sat_value)
        t['hue_obs'].append(hue_value)
        t['last_seen_frame'] = self.frame_count
        t['frames_seen'] += 1

        # Update running medians (3D jersey + 6D full feature) from observation history
        if len(t['obs']) >= 3:
            obs_arr = np.array([[o[0], o[1], o[2], o[3], o[4], o[5]] for o in t['obs']], dtype=np.float32)
            t['track_feature'] = np.median(obs_arr, axis=0).astype(np.float32)
            t['track_bgr'] = t['track_feature'][:3].astype(np.float32)
            t['shorts_bgr'] = t['track_feature'][3:6].astype(np.float32)

        # Low-sat streak: track whose jersey has near-zero HSV saturation
        # for many frames → ref candidate (gray/black/white).
        if sat_value < REF_SATURATION_THRESHOLD:
            t['low_sat_streak'] += 1
        else:
            t['low_sat_streak'] = max(0, t['low_sat_streak'] - 1)

    def _enough_tracks_to_init(self) -> bool:
        stable = [t for t in self.tracks.values()
                  if t['frames_seen'] >= TRACK_MIN_FRAMES_TO_CLUSTER and t['track_bgr'] is not None]
        return len(stable) >= 2

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _initialize_from_tracks(self):
        stable_tracks = [t for t in self.tracks.values()
                         if t['frames_seen'] >= TRACK_MIN_FRAMES_TO_CLUSTER
                         and t['track_feature'] is not None]
        # Drop tracks that look like referees / GKs — they would pollute
        # the team centroids. A track is excluded if it has low HSV
        # saturation (gray/black/white ref), or its BGR color is a vivid
        # outlier (yellow/lime/cyan linesman).
        colored = []
        for t in stable_tracks:
            if t['track_feature'][1] < REF_SATURATION_THRESHOLD:
                continue  # gray/black/white → skip
            jersey_bgr = t['track_feature'][:3]
            colorfulness = float(max(jersey_bgr) - min(jersey_bgr))
            if colorfulness > 80.0 and self._is_vivid_outlier_candidate(t):
                continue  # vivid single-color shirt that doesn't match a team
            colored.append(t)
        if len(colored) < 2:
            colored = [t for t in stable_tracks
                       if t['track_feature'][1] >= REF_SATURATION_THRESHOLD]
        if len(colored) < 2:
            colored = stable_tracks
        if len(colored) < 2:
            return

        feat = np.array([t['track_feature'] for t in colored], dtype=np.float32)
        k = min(self.n_clusters, len(colored))
        centroids, gmm = self._fit_cluster(feat, k)
        if centroids is None:
            return

        sorted_c, sorted_bgr, gmm = self._finalise_centroids(
            centroids, gmm, track_pool=colored
        )

        self.team_centroids_bgr_feat = sorted_c
        self.team_centroids_bgr = sorted_bgr
        self.gmm_model = gmm
        self.initialized = True
        self._last_centroids_hsv = sorted_c.copy()

        # If centroids are too close (or identical), seed track labels by
        # pitch half — otherwise per-frame voting would leave every track
        # on whichever cluster KMeans happened to assign it to.
        centroids_close = (
            len(sorted_c) >= 2
            and self._feature_distance(sorted_c[0], sorted_c[1]) < SIMILAR_TEAM_CENTROID_DIST
        )
        if centroids_close and self._has_position_signal(colored):
            self._seed_labels_by_position(colored)

        for t in self.tracks.values():
            if t['track_bgr'] is None:
                continue
            label = self._initial_label_for_track(t)
            self._cast_team_vote(t, label)

    def _reconcile_position_vs_votes(self, track_pool):
        """Belt-and-braces: for any track whose team_id disagrees with
        its pitch position, snap team_id back to the position-based
        label and bump the vote for that team so the vote tally agrees.

        Catches the residual case where the per-frame vote drifted the
        label even though the track's median pitch position has been
        stable on one half all along.
        """
        if not any(len(t.get('pitch_positions', [])) >= POSITION_MIN_SAMPLES
                   for t in track_pool):
            return
        global_xs = []
        for t in track_pool:
            for (x, _y) in t.get('pitch_positions', []):
                global_xs.append(float(x))
        if not global_xs:
            return
        global_median = float(np.median(global_xs))

        for t in track_pool:
            xs = [x for (x, _y) in t.get('pitch_positions', [])]
            if len(xs) < POSITION_MIN_SAMPLES:
                continue
            med_x = float(np.median(xs))
            pos_label = self.TEAM0 if med_x < global_median else self.TEAM1
            if t.get('team_id') != pos_label and pos_label in (self.TEAM0, self.TEAM1):
                t['team_id'] = pos_label
                t['team_votes'][pos_label] = t['team_votes'].get(pos_label, 0) + 6
                # Lightly penalise the wrong team so resolve() agrees.
                other = self.TEAM1 if pos_label == self.TEAM0 else self.TEAM0
                t['team_votes'][other] = max(0, t['team_votes'].get(other, 0) - 2)

    def _seed_labels_by_position(self, track_pool):
        """Hard-assign each track to team 0 or 1 based on its median pitch X
        (left half → 0, right half → 1). Clears prior votes so this is the
        dominant signal for these tracks."""
        # Compute overall median X across all positioned tracks
        all_xs = []
        for t in track_pool:
            for (x, _y) in t.get('pitch_positions', []):
                all_xs.append(x)
        if not all_xs:
            return
        global_median = float(np.median(all_xs))

        for t in track_pool:
            xs = [x for (x, _y) in t.get('pitch_positions', [])]
            if not xs:
                continue
            med = float(np.median(xs))
            label = 0 if med < global_median else 1
            # Reset vote history so the position-based label dominates
            t['team_votes'] = {0: 0, 1: 0, self.GK: 0, self.REF: 0}
            t['team_votes_history'].clear()
            t['team_votes'][label] = 6
            t['team_id'] = label

    def _initial_label_for_track(self, t) -> int:
        # Referee cues first (Tier 1.4 + 1.6)
        if t['track_feature'] is not None and t['low_sat_streak'] >= 3:
            return self.REF
        if self._is_ref_candidate(t):
            return self.REF
        # Init-time fallback for vivid outlier tracks that haven't
        # accumulated the frames_seen >= 6 required by
        # _bgr_outlier_says_ref yet. Yellow / lime / cyan shirts that
        # are far from both team centroids are referees, not
        # outfielders — without this, init can mis-classify a vivid
        # yellow linesman as the closest team if the 6-frame threshold
        # hasn't elapsed.
        if (t['track_feature'] is not None
                and self.team_centroids_bgr_feat is not None
                and len(self.team_centroids_bgr_feat) >= 2):
            jersey_bgr = t['track_feature'][:3]
            colorfulness = float(max(jersey_bgr) - min(jersey_bgr))
            if colorfulness > 80.0:
                d0 = self._bgr_distance(jersey_bgr, self.team_centroids_bgr_feat[0][:3])
                d1 = self._bgr_distance(jersey_bgr, self.team_centroids_bgr_feat[1][:3])
                if min(d0, d1) > 40.0:
                    return self.REF
        if self.team_centroids_bgr_feat is None or len(self.team_centroids_bgr_feat) == 0:
            return self.GK
        dists = [self._feature_distance(t['track_feature'], c) for c in self.team_centroids_bgr_feat]
        return int(np.argmin(dists))

    # ------------------------------------------------------------------
    # Periodic + drift-triggered re-clustering
    # ------------------------------------------------------------------
    def _recluster_from_tracks(self, force: bool = False):
        stable_tracks = [t for t in self.tracks.values()
                         if t['frames_seen'] >= TRACK_MIN_FRAMES_TO_CLUSTER
                         and t['track_feature'] is not None]
        # Drop referee-like tracks so they don't pull centroids toward
        # yellow/cyan/lime.
        colored = []
        for t in stable_tracks:
            if t['track_feature'][1] < REF_SATURATION_THRESHOLD:
                continue
            jersey_bgr = t['track_feature'][:3]
            colorfulness = float(max(jersey_bgr) - min(jersey_bgr))
            # If a vivid-colored track is far from both existing centroids
            # it's a referee, not a player — exclude from clustering.
            if colorfulness > 80.0 and self.team_centroids_bgr_feat is not None \
                    and len(self.team_centroids_bgr_feat) >= 2:
                d0 = self._feature_distance(t['track_feature'], self.team_centroids_bgr_feat[0])
                d1 = self._feature_distance(t['track_feature'], self.team_centroids_bgr_feat[1])
                if min(d0, d1) > 50.0:
                    continue
            colored.append(t)
        if len(colored) < 2:
            return

        feat = np.array([t['track_feature'] for t in colored], dtype=np.float32)
        k = min(self.n_clusters, len(colored))
        centroids, gmm = self._fit_cluster(feat, k)
        if centroids is None:
            return

        if (self.team_centroids_bgr_feat is None
                or len(self.team_centroids_bgr_feat) != len(centroids)
                or force):
            sorted_c, sorted_bgr, gmm = self._finalise_centroids(
                centroids, gmm, track_pool=colored
            )
            self.team_centroids_bgr_feat = sorted_c
            self.team_centroids_bgr = sorted_bgr
            self.gmm_model = gmm
            self._last_centroids_hsv = sorted_c.copy()
        else:
            sorted_c, sorted_bgr, gmm = self._finalise_centroids(
                centroids, gmm, track_pool=colored
            )
            a = CENTROID_EMA_ALPHA
            self.team_centroids_bgr_feat = a * sorted_c + (1.0 - a) * self.team_centroids_bgr_feat
            self.team_centroids_bgr = a * sorted_bgr + (1.0 - a) * self.team_centroids_bgr

        # Re-seed by position if centroids are too close (catches the case
        # where lighting drift brings the clusters together after warm-up)
        if (len(self.team_centroids_bgr_feat) >= 2
                and self._feature_distance(self.team_centroids_bgr_feat[0], self.team_centroids_bgr_feat[1])
                < SIMILAR_TEAM_CENTROID_DIST
                and self._has_position_signal(colored)):
            self._seed_labels_by_position(colored)

        # Belt-and-braces: regardless of whether the centroids are close,
        # if any track currently has a different team_id than its pitch
        # position would predict, gently nudge it back to match. This
        # keeps per-track team_id aligned with the spatial prior so the
        # "same person switching between T1 and T2" issue can't recur
        # even if the per-frame vote momentarily drifted.
        self._reconcile_position_vs_votes(colored)

    def _finalise_centroids(self, centroids, gmm, track_pool):
        """Sort centroids by hue, convert to BGR for viz, and apply the
        position-based tiebreaker if the two centroids are too close in
        feature space (handles similar-colored teams like red+white vs
        blue+red)."""
        order = np.argsort(centroids[:, 0])
        sorted_c = centroids[order]
        sorted_bgr = self._centroids_to_bgr(sorted_c)
        if gmm is not None and hasattr(gmm, 'means_') and len(gmm.means_) == len(centroids):
            gmm.means_ = gmm.means_[order]
            if hasattr(gmm, 'covariances_') and len(gmm.covariances_) == len(centroids):
                gmm.covariances_ = gmm.covariances_[order]

        if len(sorted_c) >= 2 and len(track_pool) >= 2:
            d = self._feature_distance(sorted_c[0], sorted_c[1])
            if d < SIMILAR_TEAM_CENTROID_DIST and self._has_position_signal(track_pool):
                sorted_c, sorted_bgr = self._position_tiebreak(sorted_c, sorted_bgr, track_pool)
        return sorted_c, sorted_bgr, gmm

    def _has_position_signal(self, track_pool) -> bool:
        return any(len(t.get('pitch_positions', [])) > 0 for t in track_pool)

    def _position_tiebreak(self, sorted_c, sorted_bgr, track_pool):
        """When the two centroids are too close in color space, use the
        median pitch-X of the tracks assigned to each cluster to pick the
        canonical left/right ordering."""
        # Initial assignment by nearest centroid
        groups = {0: [], 1: []}
        for t in track_pool:
            if t['track_feature'] is None:
                continue
            d0 = self._feature_distance(t['track_feature'], sorted_c[0])
            d1 = self._feature_distance(t['track_feature'], sorted_c[1])
            groups[0 if d0 <= d1 else 1].append(t)

        if not groups[0] or not groups[1]:
            return sorted_c, sorted_bgr  # can't tiebreak, keep as is

        def _median_x(ts):
            xs = []
            for t in ts:
                for (x, _y) in t.get('pitch_positions', []):
                    xs.append(x)
            return float(np.median(xs)) if xs else None

        m0 = _median_x(groups[0])
        m1 = _median_x(groups[1])
        if m0 is None or m1 is None:
            return sorted_c, sorted_bgr

        # If group 1 is actually to the left of group 0, swap so team 0 = left half
        if m1 < m0:
            sorted_c = np.stack([sorted_c[1], sorted_c[0]])
            sorted_bgr = np.stack([sorted_bgr[1], sorted_bgr[0]])
        return sorted_c, sorted_bgr

    def _record_pitch_positions(self, track_ids, player_xyxy, H):
        if H is None or player_xyxy is None:
            return
        for i, tid in enumerate(track_ids):
            t = self.tracks.get(int(tid))
            if t is None:
                continue
            bb = player_xyxy[i]
            try:
                bx = 0.5 * (float(bb[0]) + float(bb[2]))
                by = float(bb[3])
                pt = np.array([[[bx, by]]], dtype=np.float32)
                proj = cv2.perspectiveTransform(pt, H)[0, 0]
                x, y = float(proj[0]), float(proj[1])
            except Exception:
                continue
            t['pitch_positions'].append((x, y))
            if len(t['pitch_positions']) > 30:
                t['pitch_positions'].popleft()

    def _drift_exceeded(self) -> bool:
        """Tier 2.2: mean per-track shift vs current centroids above threshold."""
        if self.team_centroids_bgr_feat is None:
            return False
        stable = [t for t in self.tracks.values()
                  if t['frames_seen'] >= TRACK_MIN_FRAMES_TO_CLUSTER
                  and t['track_ema_bgr'] is not None]
        if not stable:
            return False
        total, count = 0.0, 0
        for t in stable:
            ema = t['track_ema_bgr']
            d = min(self._bgr_distance(ema, c[:3]) for c in self.team_centroids_bgr_feat)
            total += d
            count += 1
        return (total / max(count, 1)) > RE_CLUSTER_DRIFT_THRESHOLD

    # ------------------------------------------------------------------
    # Cluster fitting (KMeans or GMM, gated on USE_GMM)
    # ------------------------------------------------------------------
    def _fit_cluster(self, feat: np.ndarray, k: int):
        """Returns (centroids (k, 6) sorted by hue, gmm_or_None)."""
        # Cluster on jersey only (first 3 dims)
        feat_j = feat[:, :3]
        if USE_GMM and k >= 1:
            GMM = _safe_import_gmm()
            if GMM is not None:
                try:
                    gmm = GMM(n_components=k, covariance_type=GMM_COVARIANCE_TYPE,
                              random_state=0, n_init=1, reg_covar=1e-3)
                    gmm.fit(feat_j)
                    c3 = gmm.means_.astype(np.float32)
                    c6 = np.zeros((k, 6), dtype=np.float32)
                    c6[:, :3] = c3
                    return c6, gmm
                except Exception:
                    pass
        try:
            km = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(feat_j)
        except Exception:
            return None, None
        
        c3 = km.cluster_centers_.astype(np.float32)
        c6 = np.zeros((k, 6), dtype=np.float32)
        c6[:, :3] = c3
        return c6, None

    @staticmethod
    def _centroids_to_bgr(centroids: np.ndarray) -> np.ndarray:
        """Convert centroid features (3D or 6D BGR) to BGR colours for visualization.
        Uses only the jersey (B, G, R) components."""
        j_bgr = centroids[:, :3]
        return j_bgr.astype(np.float32)

    # ------------------------------------------------------------------
    # Per-frame team-id decision
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_iou_matrix(cur_xyxy: np.ndarray, prev_boxes: np.ndarray) -> np.ndarray:
        """Vectorized IoU between every current box and every prev box.

        ``cur_xyxy``: (N, 4) float in xyxy. ``prev_boxes``: (M, 4) float.
        Returns an (N, M) float32 IoU matrix; zero where boxes do not
        overlap. Handles N==0 or M==0 by returning an empty matrix.
        """
        N = len(cur_xyxy)
        M = len(prev_boxes)
        if N == 0 or M == 0:
            return np.zeros((N, M), dtype=np.float32)
        cur = cur_xyxy.astype(np.float32)
        prev = prev_boxes.astype(np.float32)
        lt = np.maximum(cur[:, None, :2], prev[None, :, :2])  # (N, M, 2)
        rb = np.minimum(cur[:, None, 2:], prev[None, :, 2:])  # (N, M, 2)
        wh = np.clip(rb - lt, 0.0, None)
        inter = wh[..., 0] * wh[..., 1]                          # (N, M)
        cur_area = (cur[:, 2] - cur[:, 0]) * (cur[:, 3] - cur[:, 1])   # (N,)
        prev_area = (prev[:, 2] - prev[:, 0]) * (prev[:, 3] - prev[:, 1])  # (M,)
        union = cur_area[:, None] + prev_area[None, :] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-6), 0.0)
        return iou.astype(np.float32)

    def _spatial_continuity_overrides(self, track_ids, player_xyxy, team_ids):
        """Inherit a confident prev-frame TEAM0/TEAM1 label onto fresh /
        ID-switched tracks whose boxes uniquely overlap a prev box.

        Runs *before* any per-frame reassignment in
        ``_decide_per_frame_team_ids``. Only prev entries whose resolved
        team_id is TEAM0/TEAM1 are considered inheritable (never GK or
        REF — those are positional/situational). Only tracks that are
        NOT already ``team_locked`` are eligible (fresh / ID-switched
        tracks); a locked track is never overwritten.

        Matching uses a symmetric mutual 1-to-1 best match with IoU
        >= ``SPATIAL_CONTINUITY_IOU``: the current detection's best
        prev box AND that prev box's best current detection must be
        each other.

        For each accepted pair ``(i, prev)`` the inherited team is
        written onto ``team_ids[i]`` and the new track's ``team_id``,
        a strong vote is cast via ``_cast_team_vote`` (so the existing
        hard-lock path engages quickly), and ``i`` is added to the
        returned ``skip`` set so the rest of
        ``_decide_per_frame_team_ids`` does not reassign it this frame.

        Returns ``(team_ids, skip_set)``.
        """
        skip: set[int] = set()
        if len(self._prev_frame_detections) == 0 or len(player_xyxy) == 0:
            return team_ids, skip

        prev_boxes = np.array(
            [d['bbox'] for d in self._prev_frame_detections],
            dtype=np.float32,
        ).reshape(-1, 4)
        prev_team_ids = np.array(
            [d['team_id'] for d in self._prev_frame_detections],
            dtype=np.int32,
        )
        inheritable = (prev_team_ids == self.TEAM0) | (prev_team_ids == self.TEAM1)
        if not inheritable.any():
            return team_ids, skip

        prev_boxes_inh = prev_boxes[inheritable]
        prev_team_inh = prev_team_ids[inheritable]
        iou = self._compute_iou_matrix(player_xyxy, prev_boxes_inh)  # (N_cur, M_inh)
        best_prev_for_cur = iou.argmax(axis=1)                       # (N_cur,)
        best_cur_for_prev = iou.argmax(axis=0)                       # (M_inh,)
        best_iou_cur = iou.max(axis=1)                               # (N_cur,)

        thresh = SPATIAL_CONTINUITY_IOU
        for i in range(len(player_xyxy)):
            if best_iou_cur[i] < thresh:
                continue
            tid = int(track_ids[i])
            t = self.tracks.get(tid)
            if t is None:
                continue
            # Only fire for NOT-locked tracks (fresh / ID-switched).
            if t.get('team_locked', False):
                continue
            j = int(best_prev_for_cur[i])
            if int(best_cur_for_prev[j]) != i:
                continue  # not a mutual 1-to-1 best match
            if iou[i, j] < thresh:
                continue
            inherited = int(prev_team_inh[j])
            team_ids[i] = inherited
            t['team_id'] = inherited
            self._cast_team_vote(t, inherited)
            t['quality'] = self._compute_track_quality(t)
            skip.add(i)
        return team_ids, skip

    def _decide_per_frame_team_ids(self, track_ids, per_det_feature, frame, player_xyxy, H,
                                   player_pitch_pts=None):
        """Decide team_id per detection with strong stickiness (plan (a))
        and position-based GK promotion (plan (b)).

        (a) Fixes the SAME-track-flips-T1↔T2 jitter:
          * The "confident" gate no longer requires ``team_id`` to
            already be TEAM0/TEAM1 — only that the track has been seen
            for ``>= TEAM_LOCK_MIN_FRAMES`` and has a running median.
            This way a track whose label briefly landed on REF/GK during
            the warm-up still gets promoted into the sticky path the
            next time it crosses the frame threshold.
          * In the confident path the label is picked from the track's
            RUNNING MEDIAN (or pitch position when centroids are
            close) — NEVER from the current single-frame observation.
          * A 3-frame streak on the same TEAM0/TEAM1 label sets
            ``team_locked = True`` early (before
            ``TEAM_HARD_LOCK_FRAMES`` elapses).

        (b) Rewrites the GK rule to be position-based:
          * After the sticky-label branch resolves a TEAM0/TEAM1 label,
            ``_decide_gk_by_position`` is consulted. A track that has
            spent ``GK_BOX_MIN_FRAMES`` consecutive frames inside
            either 18-yard or 6-yard box is promoted to GK.
          * A track that is the leftmost or rightmost player on the
            pitch (within margin of the nearest penalty spot) for
            ``GK_EDGE_MIN_FRAMES`` frames is also promoted — this
            catches sweeper-keepers and GKs caught a step outside the
            box.
          * Colour-based GK detection (the old
            ``GK_COLOR_DIST_THRESHOLD`` rule) is REMOVED.
        """
        team_ids = np.full(len(track_ids), self.GK, dtype=np.int32)
        centroids_close = self._centroids_too_close()

        # Spatial-continuity team lock: before any per-frame reassignment,
        # inherit a confident prev-frame TEAM0/TEAM1 label onto fresh /
        # ID-switched tracks whose boxes uniquely overlap. Returns a
        # `skip` set of detection indices whose label has been fixed this
        # frame and must NOT be reassigned below.
        team_ids, skip_set = self._spatial_continuity_overrides(
            track_ids, player_xyxy, team_ids,
        )

        # Build the all-track pitch list once per frame so the leftmost /
        # rightmost safety net has a stable set of comparisons. Each
        # entry is (track_id_int, pitch_x, pitch_y) for tracks with at
        # least one pitch position recorded.
        all_track_pitches = []
        for tid, t in self.tracks.items():
            if t['pitch_positions']:
                xs = [p[0] for p in t['pitch_positions']]
                ys = [p[1] for p in t['pitch_positions']]
                all_track_pitches.append((int(tid), float(np.median(xs)), float(np.median(ys))))

        for i, tid in enumerate(track_ids):
            # Spatial-continuity-matched detections: team_id already set
            # and votes already cast by `_spatial_continuity_overrides`.
            # Skip all per-frame reassignment for them this frame.
            if i in skip_set:
                continue

            t = self.tracks.get(int(tid))
            if t is None or t['track_feature'] is None:
                continue

            obs_feature = per_det_feature[i]

            # Tier 1.5-BGR: ByteTrack hard-lock (TEAM0/TEAM1).
            #   - TEAM0/TEAM1 locks are SOLID: only the GK position prior
            #     can demote them. This is what fixes the per-frame
            #     "jitter" the user observed — once a track is identified
            #     as team-0 or team-1, it stays that way for its entire
            #     life (or until the position prior says it's actually a GK).
            if t.get('team_locked', False) and t['team_id'] in (self.TEAM0, self.TEAM1):
                locked_label = t['team_id']
                # GK position prior can still override a locked team → GK,
                # but ONLY via _decide_gk_by_position which requires
                # GK_BOX_MIN_FRAMES CONSECUTIVE frames in the box (the
                # counters reset the moment the player steps out).
                #
                # We deliberately do NOT call _apply_gk_defensive_prior
                # here — its demote-criteria checks operate on the FULL
                # position history and can set gk_locked=True for a
                # track that visited the box earlier and is now in
                # midfield, which would then return GK and clear the
                # team lock (causing T1↔T2 jitter).
                if player_xyxy is not None:
                    ppt = player_pitch_pts[i] if (player_pitch_pts is not None
                                                  and i < len(player_pitch_pts)) else None
                    held_gk = self._decide_gk_by_position(
                        t, player_xyxy[i], ppt, H, all_track_pitches,
                        locked_label,
                    )
                    if held_gk == self.GK:
                        team_ids[i] = self.GK
                        t['team_id'] = self.GK
                        t['team_locked'] = False
                        t['quality'] = self._compute_track_quality(t)
                        continue
                team_ids[i] = locked_label
                t['team_id'] = locked_label
                t['quality'] = self._compute_track_quality(t)
                continue
            if t.get('team_locked', False) and t['team_id'] == self.REF:
                team_ids[i] = t['team_id']
                t['quality'] = self._compute_track_quality(t)
                continue

            # Plan (a) "confident" gate. The previous version required
            # ``team_id in (TEAM0, TEAM1)`` which permanently excluded
            # tracks whose warm-up label was REF/GK/None — those tracks
            # were stuck in the per-frame-vote branch forever. Now the
            # gate fires as soon as the track has enough samples to
            # trust its running median.
            confident = (
                t['frames_seen'] >= TEAM_LOCK_MIN_FRAMES
                and t['track_feature'] is not None
            )

            if confident:
                # Tier: stickiness. Pick the label from the track's
                # own running median (or pitch position when centroids
                # are too close) — never the noisy single-frame obs.
                if centroids_close and self._has_enough_positions(t):
                    label = self._position_based_label(t)
                else:
                    label = self._sticky_label_for_track(t)

                # Plan (b): position-based GK promotion. Even if colour
                # says this is a TEAM0 player, if they're standing in
                # the box for GK_BOX_MIN_FRAMES frames (or are the
                # leftmost/rightmost sweeper-keeper), they're the GK.
                if player_xyxy is not None:
                    ppt = player_pitch_pts[i] if (player_pitch_pts is not None
                                                  and i < len(player_pitch_pts)) else None
                    label = self._decide_gk_by_position(
                        t, player_xyxy[i], ppt, H, all_track_pitches,
                        label,
                    )

                # Vote tally for the resolved label (drives downstream
                # canonical-team resolution).
                self._cast_team_vote(t, label)

                # Plan (a) early-lock: 3 consecutive frames on the
                # same TEAM0/TEAM1 label → team_locked = True. Cheap
                # because it reads only the short-term history deque.
                self._maybe_early_lock_by_streak(t)

                # Tier 1.5-BGR: HARD-LOCK by ByteTrack ID. Once the
                # track has been seen for TEAM_HARD_LOCK_FRAMES frames,
                # the CURRENT team_id is locked forever (only the GK
                # position prior can demote).
                if t['frames_seen'] >= TEAM_HARD_LOCK_FRAMES and t['team_id'] is not None:
                    t['team_locked'] = True

                team_ids[i] = label
                t['team_id'] = label
                t['quality'] = self._compute_track_quality(t)
                continue

            # Not yet confident → vote per-frame until the track settles.
            label = self._vote_for_label(t, obs_feature)

            # Tier 3: cold-start anti-vote. If the obs is much closer to
            # the OPPOSING centroid than to the track's running median,
            # don't let it seed a wrong-team label.
            if (t['frames_seen'] < COLD_START_FRAMES
                    and self.team_centroids_bgr_feat is not None
                    and len(self.team_centroids_bgr_feat) == 2
                    and t['track_feature'] is not None
                    and label in (self.TEAM0, self.TEAM1)):
                obs_to_track = self._feature_distance(obs_feature, t['track_feature'])
                d0 = self._feature_distance(obs_feature, self.team_centroids_bgr_feat[0])
                d1 = self._feature_distance(obs_feature, self.team_centroids_bgr_feat[1])
                d_self = d0 if label == self.TEAM0 else d1
                d_other = d1 if label == self.TEAM0 else d0
                if d_other + COLD_START_DIST_GAP < d_self and obs_to_track > TRACK_RELABEL_DIST * 0.6:
                    if t['team_id'] in (self.TEAM0, self.TEAM1):
                        label = t['team_id']
                    self._cast_team_vote(t, label)
                else:
                    self._cast_team_vote(t, label)
            else:
                self._cast_team_vote(t, label)

            # If we have enough frames to pick a real team, lock it.
            if (t['frames_seen'] >= TEAM_LOCK_MIN_FRAMES
                    and label in (self.TEAM0, self.TEAM1)):
                if centroids_close and self._has_enough_positions(t):
                    label = self._position_based_label(t)
                else:
                    label = self._sticky_label_for_track(t)

            resolved = label

            # Plan (b): position-based GK promotion also runs during
            # the warm-up window — a track that's been in the box
            # from frame 1 should still pick up GK status.
            if player_xyxy is not None:
                ppt = player_pitch_pts[i] if (player_pitch_pts is not None
                                              and i < len(player_pitch_pts)) else None
                resolved = self._decide_gk_by_position(
                    t, player_xyxy[i], ppt, H, all_track_pitches,
                    resolved,
                )

            # Tier 1.5-BGR: HARD-LOCK by ByteTrack ID.
            if t['frames_seen'] >= TEAM_HARD_LOCK_FRAMES and t['team_id'] is not None:
                t['team_locked'] = True

            team_ids[i] = resolved
            t['team_id'] = team_ids[i]

            # Update per-track quality (Tier 2.3) using short-term flip rate
            t['quality'] = self._compute_track_quality(t)
        return team_ids

    def _centroids_too_close(self) -> bool:
        """True iff the two team centroids are within
        ``SIMILAR_TEAM_CENTROID_DIST`` in feature space — colour alone
        is unreliable for team assignment and pitch position should
        dominate."""
        if self.team_centroids_bgr_feat is None or len(self.team_centroids_bgr_feat) < 2:
            return False
        return bool(
            self._feature_distance(
                self.team_centroids_bgr_feat[0], self.team_centroids_bgr_feat[1],
            ) < SIMILAR_TEAM_CENTROID_DIST
        )

    @staticmethod
    def _has_enough_positions(t) -> bool:
        return len(t.get('pitch_positions', [])) >= POSITION_MIN_SAMPLES

    def _sticky_label_for_track(self, t) -> int:
        """Decide the track's team using its running median feature
        (``track_feature``) instead of the current frame's observation.
        Stable across momentary noise.
        """
        # Referee cues first (Tier 1.4 + 1.6 OR-conditions)
        if self._is_ref_candidate(t):
            return self.REF
        # Low-saturation streak requires a longer streak than the
        # general REF_MIN_FRAMES threshold to avoid false positives on
        # white-shorts outfielders with brief shadow periods.
        if t['low_sat_streak'] >= REF_MIN_FRAMES * 2:
            return self.REF

        feat = t['track_feature']
        if self.team_centroids_bgr_feat is None or len(self.team_centroids_bgr_feat) == 0:
            return self.GK
        dists = [self._feature_distance(feat, c) for c in self.team_centroids_bgr_feat]
        return int(np.argmin(dists))

    def _position_based_label(self, t) -> int:
        """Assign team based on the track's median pitch-X.

        Teams start on opposite halves at kickoff and each team
        defends its own goal, so median X is a strong, stable signal —
        much more reliable than colour when the two jerseys are
        similar (e.g. red vs orange).
        """
        if not self._has_enough_positions(t):
            return self._sticky_label_for_track(t)
        xs = [float(x) for (x, _y) in t['pitch_positions']]
        median_x = float(np.median(xs))
        return self.TEAM0 if median_x < CENTER_X else self.TEAM1

    def _maybe_early_lock_by_streak(self, t) -> None:
        """Plan (a): promote `team_locked = True` as soon as the track has
        been on the same TEAM0/TEAM1 label for TEAM_CONSEC_STREAK_LOCK
        consecutive frames (cheaper than waiting for TEAM_HARD_LOCK_FRAMES).

        The streak counter is maintained against the running majority of
        `team_votes_history` (the last N per-frame labels) rather than
        per-frame observations, so a single bad-frame vote doesn't break
        the lock prematurely.
        """
        if t.get('team_locked', False):
            return
        if t['team_id'] not in (self.TEAM0, self.TEAM1):
            return
        history = list(t['team_votes_history'])
        if len(history) < TEAM_CONSEC_STREAK_LOCK:
            return
        tail = history[-TEAM_CONSEC_STREAK_LOCK:]
        if all(x == t['team_id'] for x in tail):
            t['team_locked'] = True

    def _decide_gk_by_position(self, t, player_xyxy, player_pitch_pts,
                               H, all_track_pitches, current_label) -> int:
        """Plan (b): position-driven GK decision.

        Returns GK if any position-based criterion fires, otherwise
        returns ``current_label`` (the label already resolved by the
        sticky-label branch — typically a TEAM0/TEAM1 or REF label).
        This avoids the stale-``t['team_id']`` trap where the track's
        ``team_id`` was set to GK during the warm-up window and would
        leak through after the team label was resolved.

        The position-based criteria are checked in priority order:

          1. Inside either 18-yard (penalty) box — bump box_frames.
          2. Inside either 6-yard / 5-yard (goal area) box — bump
             goal_area_frames.
          3. Combined box+goal-area count >= GK_BOX_MIN_FRAMES → GK
             AND set ``gk_candidate_locked = True``.
          4. Leftmost or rightmost player on the pitch AND pitch-X
             within margin of the nearest penalty spot — bump
             leftmost/rightmost streak.
          5. Either streak >= GK_EDGE_MIN_FRAMES → GK AND lock.

        ``all_track_pitches`` is a list[(track_id_int, pitch_x, pitch_y)]
        built by the caller so we know the leftmost / rightmost
        currently-on-pitch player.
        """
        # Need a current frame pitch position. Prefer the externally
        # provided `player_pitch_pts[i]`; fall back to projecting the
        # bbox with H. If NEITHER is available, this function is a
        # no-op — return the label the caller already resolved.
        if player_xyxy is None:
            return current_label if current_label is not None else self.GK
        px = None
        py = None
        try:
            if player_pitch_pts is not None:
                px = float(player_pitch_pts[0])
                py = float(player_pitch_pts[1])
            elif H is not None:
                bx = 0.5 * (float(player_xyxy[0]) + float(player_xyxy[2]))
                by = float(player_xyxy[3])
                pt = np.array([[[bx, by]]], dtype=np.float32)
                proj = cv2.perspectiveTransform(pt, H)[0, 0]
                px, py = float(proj[0]), float(proj[1])
        except Exception:
            px = None
            py = None
        if px is None or py is None:
            return current_label if current_label is not None else self.GK 

        in_left_pen = (px <= LEFT_PENALTY_X) and (PENALTY_Y_TOP <= py <= PENALTY_Y_BOTTOM)
        in_right_pen = (px >= RIGHT_PENALTY_X) and (PENALTY_Y_TOP <= py <= PENALTY_Y_BOTTOM)
        in_left_goal = (px <= LEFT_GOAL_AREA_X) and (GOAL_AREA_Y_TOP <= py <= GOAL_AREA_Y_BOTTOM)
        in_right_goal = (px >= RIGHT_GOAL_AREA_X) and (GOAL_AREA_Y_TOP <= py <= GOAL_AREA_Y_BOTTOM)

        # The 6-yd box is geometrically inside the 18-yd box, so a
        # player in the 6-yd strip is ALWAYS counted as "in the 18-yd
        # box" too. We use a single `box_frames` consecutive counter
        # that increments while the player is in either box and
        # resets the moment they step out. This is what prevents a
        # track that was in the box 4 frames ago (and is now in
        # midfield) from being promoted to GK — and crucially prevents
        # the locked-team→GK demote path in _apply_gk_defensive_prior
        # from firing on a track whose box visit was already over.
        in_box_now = in_left_pen or in_right_pen or in_left_goal or in_right_goal
        if in_box_now:
            t['box_frames'] += 1
            t['goal_area_frames'] += 1
        else:
            t['box_frames'] = 0
            t['goal_area_frames'] = 0

        # Hysteresis: require GK_BOX_MIN_FRAMES CONSECUTIVE frames
        # inside either box before promoting.
        if (t['box_frames'] + t['goal_area_frames']) >= GK_BOX_MIN_FRAMES:
            t['gk_candidate_locked'] = True
            return self.GK

        # Leftmost / rightmost safety net (sweeper-keepers).
        if all_track_pitches and len(all_track_pitches) >= 2:
            xs = [p[1] for p in all_track_pitches]
            leftmost_x = min(xs)
            rightmost_x = max(xs)

            is_leftmost = abs(px - leftmost_x) < 1e-3
            is_rightmost = abs(px - rightmost_x) < 1e-3

            median_xs = [p[1] for p in t['pitch_positions']] if t['pitch_positions'] else []
            if median_xs:
                med_x = float(np.median(median_xs))
            else:
                med_x = px

            # A leftmost track whose median X is within margin of
            # LEFT_PENALTY_X is a GK candidate. (Symmetric for right.)
            left_gk_zone = (px <= LEFT_PENALTY_X + GK_LEFTMOST_X_MARGIN)
            right_gk_zone = (px >= RIGHT_PENALTY_X - GK_RIGHTMOST_X_MARGIN)

            if is_leftmost and left_gk_zone:
                t['leftmost_streak'] += 1
                t['rightmost_streak'] = 0
            elif is_rightmost and right_gk_zone:
                t['rightmost_streak'] += 1
                t['leftmost_streak'] = 0
            else:
                # Decay slowly so a single occlusion doesn't kill the streak
                t['leftmost_streak'] = max(0, t['leftmost_streak'] - 1)
                t['rightmost_streak'] = max(0, t['rightmost_streak'] - 1)

            if (t['leftmost_streak'] >= GK_EDGE_MIN_FRAMES
                    or t['rightmost_streak'] >= GK_EDGE_MIN_FRAMES):
                t['gk_candidate_locked'] = True
                return self.GK

        # No GK criterion fired — keep the caller's resolved label.
        return current_label if current_label is not None else self.GK

    def _vote_for_label(self, t, obs_feature):
        """Pick a label (team 0/1, GK, REF) to cast a vote for this frame.

        The colour-based GK rule has been removed: a jersey colour that
        is mildly different from the team centroids is no longer enough
        to flag the track as a goalkeeper. GK assignment is purely
        position-driven (see ``_decide_gk_by_position``). The referee
        cues remain because the user instruction was explicit:
        "anyone who does not have the colors similar to t1 and t2
        would be a refree".
        """
        # Tier 1.4a / 1.6: any referee cue
        if self._is_ref_candidate(t):
            return self.REF
        # Early vivid-outlier ref check (mirrors _initial_label_for_track).
        # _bgr_outlier_says_ref requires frames_seen >= 6, so during the
        # warm-up window a vivid yellow/lime/cyan shirt that's far from
        # both team centroids would otherwise be mis-voted as the
        # nearest team.
        if (self.team_centroids_bgr_feat is not None
                and len(self.team_centroids_bgr_feat) >= 2):
            jersey_bgr = np.array([
                float(obs_feature[0]), float(obs_feature[1]), float(obs_feature[2]),
            ], dtype=np.float32)
            colorfulness = float(max(jersey_bgr) - min(jersey_bgr))
            if colorfulness > 80.0:
                d0 = self._bgr_distance(jersey_bgr, self.team_centroids_bgr_feat[0][:3])
                d1 = self._bgr_distance(jersey_bgr, self.team_centroids_bgr_feat[1][:3])
                if min(d0, d1) > 40.0:
                    return self.REF
        # Legacy low-sat referee heuristic: jersey HSV saturation below threshold
        bgr_only = np.array([[
            [float(obs_feature[0]), float(obs_feature[1]), float(obs_feature[2])]
        ]], dtype=np.uint8)
        try:
            obs_hsv = cv2.cvtColor(bgr_only, cv2.COLOR_BGR2HSV)[0, 0]
            obs_s = float(obs_hsv[1])
        except Exception:
            obs_s = 255.0
        if obs_s < REF_SATURATION_THRESHOLD:
            return self.REF

        if self.team_centroids_bgr_feat is None or len(self.team_centroids_bgr_feat) == 0:
            return self.GK

        dists = [self._feature_distance(obs_feature, c) for c in self.team_centroids_bgr_feat]
        md = float(min(dists))
        mean_d = float(np.mean(dists))
        std_d = float(np.std(dists))

        # Strong outlier from BOTH teams → referee
        if md > mean_d * 1.8 + REF_SATURATION_THRESHOLD * 0.05:
            return self.REF

        # Tier 3.4: per-team "only one GK" prior — kept as a SAFETY NET for
        # the warm-up window only. If a team already has many outfielders,
        # any further mildly-outlier track is promoted to GK (matches the
        # legacy behaviour during the first 12 frames; after that the
        # position-based GK logic takes over).
        nearest_team = int(np.argmin(dists))
        if (t['frames_seen'] >= 10
                and md > mean_d + 2.0 * (std_d + 1e-6)
                and self._outfielder_count(nearest_team) >= GK_MAX_OUTFIELDERS_PER_TEAM):
            return self.GK

        return int(np.argmin(dists))

    def _outfielder_count(self, team: int) -> int:
        """Count tracks currently labelled as the given team that are NOT
        GKs (used by the per-team GK cap)."""
        return sum(1 for t in self.tracks.values()
                   if t.get('team_id') == team
                   and t.get('gk_streak', 0) < GK_MIN_FRAMES)

    def _cast_team_vote(self, t, label):
        """Accumulate a vote for ``label`` without decaying prior votes.

        Per-track team_id was previously flipping whenever the per-frame
        vote oscillated between two similar centroids (a single noisy
        frame was enough to erode the leader's margin). The accumulator
        below never decreases prior votes, so a track correctly labelled
        on the first ~30 frames will stay correctly labelled forever
        unless the obs is DRAMATICALLY different from its track median
        (handled by the sticky-label branch in ``_decide_per_frame_team_ids``).
        """
        t['team_votes'][label] = t['team_votes'].get(label, 0) + 1
        t['team_votes_history'].append(int(label))

        if label == self.REF:
            # Tier 1.6: faster REF lock when multiple cues agree
            if self._multi_cue_ref_agreement(t) >= 2 and t['ref_outlier_streak'] + 1 >= REF_MIN_FRAMES_FAST:
                t['ref_outlier_streak'] = max(t['ref_outlier_streak'] + 1, REF_MIN_FRAMES_FAST)
            else:
                t['ref_outlier_streak'] += 1
            t['gk_streak'] = max(0, t['gk_streak'] - 1)
        elif label == self.GK:
            t['gk_streak'] += 1
            t['ref_outlier_streak'] = max(0, t['ref_outlier_streak'] - 1)
        else:
            t['gk_streak'] = max(0, t['gk_streak'] - 1)
            t['ref_outlier_streak'] = max(0, t['ref_outlier_streak'] - 1)

    def _resolve_track_team(self, t) -> int:
        if t['frames_seen'] < TRACK_MIN_FRAMES_FOR_TEAM:
            return self._fallback_label(t)

        if t['low_sat_streak'] >= REF_MIN_FRAMES:
            return self.REF

        if t['gk_streak'] >= GK_MIN_FRAMES:
            return self.GK

        votes = t['team_votes']
        best = max(votes.items(), key=lambda kv: kv[1])
        return int(best[0])

    def _fallback_label(self, t):
        if t['track_feature'] is None:
            return self.GK
        # Low HSV saturation jersey → REF
        bgr_only = np.array([[
            [float(t['track_feature'][0]), float(t['track_feature'][1]), float(t['track_feature'][2])]
        ]], dtype=np.uint8)
        try:
            hsv = cv2.cvtColor(bgr_only, cv2.COLOR_BGR2HSV)[0, 0]
            s = float(hsv[1])
        except Exception:
            s = 255.0
        if s < REF_SATURATION_THRESHOLD:
            return self.REF
        if self.team_centroids_bgr_feat is None or len(self.team_centroids_bgr_feat) == 0:
            return self.GK
        dists = [self._feature_distance(t['track_feature'], c) for c in self.team_centroids_bgr_feat]
        return int(np.argmin(dists))

    # ------------------------------------------------------------------
    # Tier 1.4 / 1.6: referee detection cues
    # ------------------------------------------------------------------
    def _saturation_histogram_says_ref(self, t) -> bool:
        """True if > REF_SAT_HIST_FRACTION of the track's observations have
        low colorfulness (max-min < REF_SAT_HIST_THRESHOLD) → referee
        / black-white / grey."""
        if len(t['sat_obs']) < 8:
            return False
        sats = np.array(t['sat_obs'], dtype=np.float32)
        return float(np.mean(sats < REF_SAT_HIST_THRESHOLD)) >= REF_SAT_HIST_FRACTION

    def _hue_multimodal_says_ref(self, t) -> bool:
        """True if the track's HSV-hue histogram has >= REF_HUE_MULTIMODAL_MODES
        modes above REF_HUE_MULTIMODAL_FRAC mass (mixed colors → referee).

        HSV hue is the natural signal here: it captures dominant color
        direction independent of brightness and is bounded 0-180.
        """
        if len(t['hue_obs']) < 8:
            return False
        hues = np.array(t['hue_obs'], dtype=np.float32)
        hist, _ = np.histogram(hues, bins=REF_BINS, range=(0, 180))
        if hist.sum() == 0:
            return False
        frac = hist / hist.sum()
        above = frac >= REF_HUE_MULTIMODAL_FRAC
        smoothed = np.convolve(above.astype(np.int32), np.ones(3, dtype=np.int32), mode='same')
        n_modes = int(np.sum(smoothed >= 2))
        return n_modes >= REF_HUE_MULTIMODAL_MODES

    def _outlier_score_says_ref(self, t) -> bool:
        """Tier 1.6: strong outlier from BOTH team centroids in 6D BGR.

        min(track_feature, centroid) / mean(track_feature, centroid)
        ratio >= REF_OUTLIER_MIN_DIST_RATIO AND min_dist > 30.

        Catches single-color ref shirts (yellow, cyan, lime) that don't
        match either team's centroid in BGR feature space.
        """
        if self.team_centroids_bgr_feat is None or len(self.team_centroids_bgr_feat) < 2:
            return False
        if t['track_feature'] is None or t['frames_seen'] < 4:
            return False
        d0 = self._feature_distance(t['track_feature'], self.team_centroids_bgr_feat[0])
        d1 = self._feature_distance(t['track_feature'], self.team_centroids_bgr_feat[1])
        min_d = min(d0, d1)
        mean_d = (d0 + d1) / 2.0
        if mean_d <= 1e-6:
            return False
        ratio = min_d / mean_d
        return ratio >= REF_OUTLIER_MIN_DIST_RATIO and min_d > 30.0

    def _peak_hue_says_ref(self, t) -> bool:
        """Tier 1.6: peaked single-hue ref shirt (uni-color).

        Builds an 18-bin HSV-hue histogram; if a single bin contains
        >= REF_PEAK_HUE_FRAC of observations AND median saturation is
        moderate (ref shirts are vivid but not deep colors) AND that
        peak hue is far from both team centroids → ref.
        """
        if self.team_centroids_bgr_feat is None or len(self.team_centroids_bgr_feat) < 2:
            return False
        if len(t['hue_obs']) < 8 or len(t['sat_obs']) < 8:
            return False
        med_s = float(np.median(list(t['sat_obs'])))
        if not (30.0 < med_s < 130.0):
            return False
        hues = np.array(t['hue_obs'], dtype=np.float32)
        hist, edges = np.histogram(hues, bins=REF_BINS, range=(0, 180))
        if hist.sum() == 0:
            return False
        peak_idx = int(np.argmax(hist))
        peak_frac = float(hist[peak_idx] / hist.sum())
        if peak_frac < REF_PEAK_HUE_FRAC:
            return False
        peak_hue = float((edges[peak_idx] + edges[peak_idx + 1]) / 2.0)
        # Distance from peak HSV-hue to each team's jersey BGR centroid (jersey only).
        # Convert peak HSV → BGR for comparison with BGR centroids.
        hsv_sample = np.array([[[int(peak_hue), int(max(med_s, 80)), 200]]], dtype=np.uint8)
        bgr_sample = cv2.cvtColor(hsv_sample, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
        d0 = self._bgr_distance(bgr_sample, self.team_centroids_bgr_feat[0][:3])
        d1 = self._bgr_distance(bgr_sample, self.team_centroids_bgr_feat[1][:3])
        min_d = min(d0, d1)
        return min_d > 35.0

    def _touchline_says_ref(self, t) -> bool:
        """Tier 1.6: linesman prior.

        A track whose median pitch-Y is within TOUCHLINE_MARGIN_M of the
        pitch boundary for > REF_TOUCHLINE_MIN_FRAC of its observations
        AND has been seen for >= 20 frames AND its color is not a strong
        match to either team centroid → linesman.
        """
        if self._last_H is None:
            return False
        if self.team_centroids_bgr_feat is None or len(self.team_centroids_bgr_feat) < 2:
            return False
        positions = list(t.get('pitch_positions', []))
        if len(positions) < 20:
            return False
        if t['track_feature'] is None:
            return False
        on_touchline = 0
        for (x, y) in positions:
            if y <= TOUCHLINE_MARGIN_M or y >= PITCH_WIDTH - TOUCHLINE_MARGIN_M:
                on_touchline += 1
        if on_touchline / len(positions) < REF_TOUCHLINE_MIN_FRAC:
            return False
        # Color must not be a strong match to either team.
        d0 = self._feature_distance(t['track_feature'], self.team_centroids_bgr_feat[0])
        d1 = self._feature_distance(t['track_feature'], self.team_centroids_bgr_feat[1])
        mean_d = (d0 + d1) / 2.0
        if mean_d < 12.0:
            return False
        return True

    def _is_vivid_outlier_candidate(self, t) -> bool:
        """Used during initialization BEFORE centroids exist: returns True
        if a track's jersey color is vivid AND clearly outside the
        dominant color cluster of all seen tracks.

        Computed as: jersey's BGR distance to the median jersey color
        of all stable tracks > 1.5× the typical inter-track distance.
        Falls back to False when there are too few tracks to compare.
        """
        if len(self.tracks) < 4:
            return False
        # Median jersey BGR across all stable tracks
        jerseys = []
        for tr in self.tracks.values():
            if (tr['frames_seen'] >= TRACK_MIN_FRAMES_TO_CLUSTER
                    and tr['track_feature'] is not None):
                jerseys.append(tr['track_feature'][:3])
        if len(jerseys) < 4:
            return False
        jerseys_arr = np.array(jerseys, dtype=np.float32)
        median_color = np.median(jerseys_arr, axis=0)
        # Typical distance from the median
        distances = [np.linalg.norm(j - median_color) for j in jerseys_arr]
        typical_d = float(np.median(distances))
        if typical_d < 1e-3:
            return False
        # This track's distance from the median
        this_d = float(np.linalg.norm(t['track_feature'][:3] - median_color))
        return this_d > 2.0 * typical_d and this_d > 60.0

    def _is_ref_candidate(self, t) -> bool:
        """OR of all referee cues (Tier 1.4 + 1.6).

        Yellow linesmen are caught by `_bgr_outlier_says_ref` (the BGR
        distance to the nearest centroid is much larger than to the
        farthest — i.e. this is a vivid single-color shirt far from
        both team colors).
        """
        if self._saturation_histogram_says_ref(t):
            return True
        if self._hue_multimodal_says_ref(t):
            return True
        if self._outlier_score_says_ref(t):
            return True
        if self._peak_hue_says_ref(t):
            return True
        if self._touchline_says_ref(t):
            return True
        if self._bgr_outlier_says_ref(t):
            return True
        return False

    def _bgr_outlier_says_ref(self, t) -> bool:
        """BGR-based vivid-color outlier check (Tier 1.6).

        A vivid-colored track (max-min > 80) whose jersey BGR is far
        from BOTH team centroids in absolute distance is a referee in
        a single-color shirt (yellow linesman, lime GK, cyan sleeves).

        The threshold is absolute (min_dist > 40) rather than ratio-
        based because yellow/lime are often HSV-near one team (yellow
        is near red) but BGR-far from both (yellow has G=230 while
        red and blue have G=0).
        """
        if self.team_centroids_bgr_feat is None or len(self.team_centroids_bgr_feat) < 2:
            return False
        if t['track_feature'] is None or t['frames_seen'] < 6:
            return False
        jersey_bgr = t['track_feature'][:3]
        colorfulness = float(max(jersey_bgr) - min(jersey_bgr))
        if colorfulness < 80.0:
            return False  # not vivid enough
        # Use ONLY the jersey BGR for the distance check (shorts band
        # is often missing for refs at the touchline).
        d0 = self._bgr_distance(jersey_bgr, self.team_centroids_bgr_feat[0][:3])
        d1 = self._bgr_distance(jersey_bgr, self.team_centroids_bgr_feat[1][:3])
        return min(d0, d1) > 40.0

    def _multi_cue_ref_agreement(self, t) -> int:
        """Count how many independent referee cues agree (for fast lock-in)."""
        n = 0
        if self._saturation_histogram_says_ref(t):
            n += 1
        if self._hue_multimodal_says_ref(t):
            n += 1
        if self._outlier_score_says_ref(t):
            n += 1
        if self._peak_hue_says_ref(t):
            n += 1
        return n

    # ------------------------------------------------------------------
    # Tier 2.3: per-track quality
    # ------------------------------------------------------------------
    def _compute_track_quality(self, t) -> float:
        """Quality in [0, 1]: 1 = stable team label, low label-flip rate.
        Uses exponential moving average over short-term vote history.
        """
        history = list(t['team_votes_history'])
        if not history:
            return 1.0

        # Flip rate over short-term window
        n = len(history)
        if n < 2:
            flip_rate = 0.0
        else:
            flips = sum(1 for a, b in zip(history[:-1], history[1:]) if a != b)
            flip_rate = flips / (n - 1)

        # Vote entropy (label distribution)
        from collections import Counter
        c = Counter(history)
        total = sum(c.values())
        if total == 0:
            return 1.0
        entropy = 0.0
        for v in c.values():
            p = v / total
            if p > 0:
                entropy -= p * np.log2(p)
        # Normalise by max entropy (log2(num_classes_seen))
        n_classes_seen = len(c)
        max_entropy = np.log2(max(2, n_classes_seen))
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        base = 1.0 - flip_rate * TRACK_QUALITY_LABEL_FLIP_PENALTY * 2.0
        quality = base * (1.0 - 0.5 * norm_entropy)
        return float(max(0.0, min(1.0, quality)))

    # ------------------------------------------------------------------
    # Tier 3.2 / 3.4: GK penalty-box + defensive-half prior
    # ------------------------------------------------------------------
    def _apply_gk_defensive_prior(self, t, bbox_xyxy, H, current_label):
        """Plan (a) extension: when a track is ``team_locked`` to a
        TEAM0/TEAM1 label AND currently in the box, only demote it to
        GK if the box-fraction criteria have held for ``>= GK_BOX_MIN_FRAMES``
        consecutive frames (the ``box_frames`` and ``goal_area_frames``
        counters maintained by ``_decide_gk_by_position``).

        Legacy behaviour for the demote path (GK → team) is preserved
        unchanged.
        """
        if H is None or bbox_xyxy is None:
            return current_label

        # Plan (a) promote-locked-team-to-GK hysteresis: a locked
        # team→GK demotion needs GK_BOX_MIN_FRAMES consecutive frames
        # in the box. We trust the `box_frames` / `goal_area_frames`
        # counters which are bumped monotonically by _decide_gk_by_position.
        if (t.get('team_locked', False)
                and current_label in (self.TEAM0, self.TEAM1)
                and (t['box_frames'] + t['goal_area_frames']) >= GK_BOX_MIN_FRAMES):
            t['gk_candidate_locked'] = True
            return self.GK

        # Project bbox bottom-center to pitch coords
        try:
            bx = 0.5 * (float(bbox_xyxy[0]) + float(bbox_xyxy[2]))
            by = float(bbox_xyxy[3])
            pt = np.array([[[bx, by]]], dtype=np.float32)
            proj = cv2.perspectiveTransform(pt, H)[0, 0]
            x, y = float(proj[0]), float(proj[1])
        except Exception:
            return current_label
        # NOTE: we DO NOT append to pitch_positions here — _record_pitch_positions
        # already does that exactly once per frame in assign_team_colors. The
        # previous implementation double-appended, inflating the deque and
        # distorting the in-box fraction calculation.
        positions = list(t['pitch_positions'])
        if len(positions) < 10:
            return current_label

        in_box = 0
        for (px, py) in positions:
            left_box = (px <= LEFT_PENALTY_X) and (PENALTY_Y_TOP <= py <= PENALTY_Y_BOTTOM)
            right_box = (px >= RIGHT_PENALTY_X) and (PENALTY_Y_TOP <= py <= PENALTY_Y_BOTTOM)
            if left_box or right_box:
                in_box += 1
        in_box_frac = in_box / len(positions)

        # Defensive third: pitch X in either end-third (we don't know which
        # side this team defends at this point — use the track's own median
        # X to pick "its" defensive end). If track is on the left half,
        # the defensive third is X < PITCH_LENGTH/3; if on the right half,
        # the defensive third is X > 2*PITCH_LENGTH/3. Either satisfies
        # the prior (GKs in either end are eligible).
        third = PITCH_LENGTH / 3.0
        in_def_third = 0
        for (px, _py) in positions:
            if px <= third or px >= PITCH_LENGTH - third:
                in_def_third += 1
        def_third_frac = in_def_third / len(positions)

        # Hold GK if either the box or the defensive-third prior is met.
        holds_box = in_box_frac >= GK_PENALTY_BOX_MIN_FRAC
        holds_def = def_third_frac >= GK_DEFENSIVE_THIRD_MIN_FRAC
        if holds_box or holds_def:
            t['gk_demote_streak'] = 0
            # Lock the GK label once we have enough evidence.
            if t['gk_streak'] >= GK_MIN_FRAMES:
                t['gk_locked'] = True
            # If the track is gk_locked, the position prior KEEPS it as
            # GK even if the current color-based label drifted to a team.
            if t.get('gk_locked', False):
                return self.GK
            return current_label

        # Demote criteria: in_box < 20% AND def_third < 40%.
        if in_box_frac < GK_DEMOTE_IN_BOX_FRAC and def_third_frac < GK_DEMOTE_DEF_FRAC:
            t['gk_demote_streak'] += 1
        else:
            t['gk_demote_streak'] = 0

        # Tier 3.4 hysteresis: a locked GK requires the demote criteria
        # to hold for GK_DEMOTE_CONSEC_FRAMES consecutive frames before
        # being demoted.
        if t['gk_locked'] and t['gk_demote_streak'] < GK_DEMOTE_CONSEC_FRAMES:
            return self.GK

        return self._fallback_label(t)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def _cleanup_stale_tracks(self):
        cutoff = self.frame_count - STALE_TRACK_FRAMES
        stale = [tid for tid, t in self.tracks.items() if t['last_seen_frame'] < cutoff]
        for tid in stale:
            self.tracks.pop(tid, None)

    # ------------------------------------------------------------------
    # Color extraction
    # ------------------------------------------------------------------
    def _extract_dominant_color(self, frame: np.ndarray, bbox: np.ndarray):
        """Return (feature_6d, bgr_3d) where feature_6d is
        [B_jersey, G_jersey, R_jersey, B_shorts, G_shorts, R_shorts].

        BGR (not HSV) — direct Euclidean distance in color space, no
        circular-hue wrap-around, no saturation-vs-brightness confusion.
        ByteTrack IDs drive stickiness; the BGR feature is what the
        per-track running median is computed over.

        Jersey feature (Tier 1.5) is a blend of:
          * a FRONT band (Y=0.12-0.42, top-K by brightness/colorfulness)
          * a BACK-SAFE band (Y=0.42-0.55, top-K with side panels only
            so the central jersey number is masked OUT) — orientation-
            invariant identity (number is on the BACK, not the side
            strips of the torso, and below the chest band is mostly
            waistband / sponsor-free).
        The blend weight JERSEY_BAND_BACK_WEIGHT (default 0.7) keeps the
        front band responsive during the first 10 frames then converges
        to the back band for steady-state identity.
        """
        return self._extract_robust_jersey_color(frame, bbox)

    def _extract_robust_jersey_color(self, frame: np.ndarray, bbox: np.ndarray):
        x1, y1, x2, y2 = max(0, int(bbox[0])), max(0, int(bbox[1])), \
                          min(frame.shape[1] - 1, int(bbox[2])), min(frame.shape[0] - 1, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return np.zeros(6, dtype=np.float32), np.array([128, 128, 128], dtype=np.float32)

        h, w = y2 - y1, x2 - x1

        # --- Front band: chest / sponsor area ---------------------------
        front_y1 = y1 + int(h * JERSEY_BAND_FRONT_Y_START)
        front_y2 = y1 + int(h * JERSEY_BAND_FRONT_Y_END)
        x_l = x1 + int(w * TEAM_JERSEY_X_START)
        x_r = x1 + int(w * TEAM_JERSEY_X_END)
        front_crop = frame[front_y1:front_y2, x_l:x_r]
        jb_f, jg_f, jr_f = self._topk_jersey_bgr(front_crop, JERSEY_TOPK_FRACTION)

        # --- Back-safe band: waistband / lower jersey, side panels only
        back_y1 = y1 + int(h * JERSEY_BAND_BACK_Y_START)
        back_y2 = y1 + int(h * JERSEY_BAND_BACK_Y_END)
        # Mask out the central (1 - 2*SIDE_PANEL_X_FRAC) fraction so the
        # number zone is excluded. The kept strips are the two side panels.
        strip_w = int(w * SIDE_PANEL_X_FRAC)
        left_x1 = x1
        left_x2 = x1 + strip_w
        right_x1 = x2 - strip_w
        right_x2 = x2
        back_crop = np.concatenate(
            [frame[back_y1:back_y2, left_x1:left_x2],
             frame[back_y1:back_y2, right_x1:right_x2]],
            axis=1,
        ) if strip_w > 0 else frame[back_y1:back_y2, x_l:x_r]
        jb_b, jg_b, jr_b = self._topk_jersey_bgr(back_crop, JERSEY_BACK_TOPK_FRACTION)

        # Blend. Front still contributes for warm-up.
        w_back = JERSEY_BAND_BACK_WEIGHT
        jb = w_back * jb_b + (1.0 - w_back) * jb_f
        jg = w_back * jg_b + (1.0 - w_back) * jg_f
        jr = w_back * jr_b + (1.0 - w_back) * jr_f

        # Shorts band — secondary feature (Tier 3.3)
        shorts_bgr = self._extract_shorts_color(frame, bbox)
        if shorts_bgr is None:
            sb, sg, sr = 0.0, 0.0, 0.0
        else:
            sb, sg, sr = float(shorts_bgr[0]), float(shorts_bgr[1]), float(shorts_bgr[2])

        feature = np.array([jb, jg, jr, sb, sg, sr], dtype=np.float32)
        vis = np.array([jb, jg, jr], dtype=np.float32)
        return feature, vis

    @staticmethod
    def _topk_jersey_bgr(crop_bgr: np.ndarray, topk_fraction: float):
        """Apply green/dark/bright masks in BGR space, then return
        (B, G, R) of the top-K most-colorful valid pixels (sorted by
        max-channel minus min-channel — proxy for saturation/vividness).

        Falls back to raw pixels when the mask leaves fewer than
        INVALID_PIXEL_MIN valid pixels.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return 0.0, 0.0, 0.0
        bgr = crop_bgr.astype(np.int16)  # avoid uint8 overflow in subtract
        # Green mask in BGR: G dominant, B and R lower
        b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
        green = (g > 60) & (g > r + 15) & (g > b + 15)
        dark = (bgr.max(axis=2) < 30)
        bright = (bgr.min(axis=2) > 220)
        valid_mask = ~(green | dark | bright)
        valid = bgr[valid_mask]
        if len(valid) < INVALID_PIXEL_MIN:
            valid = bgr.reshape(-1, 3)
        if len(valid) == 0:
            return 0.0, 0.0, 0.0
        # Colorfulness proxy: max - min per pixel
        colorfulness = valid.max(axis=1) - valid.min(axis=1)
        k = max(8, int(len(valid) * topk_fraction))
        order = np.argsort(colorfulness)
        topk = valid[order[-k:]] if len(order) >= k else valid
        return float(np.mean(topk[:, 0])), float(np.mean(topk[:, 1])), float(np.mean(topk[:, 2]))

    @staticmethod
    def _illumination_normalize(frame_bgr: np.ndarray) -> np.ndarray:
        """Gray-world illumination normalization applied to the WHOLE frame.
        Scales each channel so its mean equals the global mean — removes
        warm/cool lighting drift across the field without destroying the
        per-crop jersey color signal (which per-crop gray-world would do).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr
        try:
            means = cv2.mean(frame_bgr)[:3]
            gray = sum(means) / 3.0
            eps = 1e-6
            scale = gray / (np.array(means) + eps)
            scale = np.clip(scale, 0.7, 1.4)  # conservative — don't amplify noise
            normalized = np.clip(frame_bgr.astype(np.float32) * scale, 0, 255).astype(np.uint8)
            return normalized
        except Exception:
            return frame_bgr

    # ------------------------------------------------------------------
    # Tier 3.3: two-stage jersey+shorts disambiguation
    # ------------------------------------------------------------------
    def _extract_shorts_color(self, frame: np.ndarray, bbox: np.ndarray):
        """Sample shorts band; used to disambiguate refs from outfielders
        whose jersey color is uncertain. Returns BGR (3,) or None."""
        x1, y1, x2, y2 = max(0, int(bbox[0])), max(0, int(bbox[1])), \
                          min(frame.shape[1] - 1, int(bbox[2])), min(frame.shape[0] - 1, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None
        h, w = y2 - y1, x2 - x1
        crop = frame[y1 + int(h * SHORTS_BAND_Y_START):y1 + int(h * SHORTS_BAND_Y_END),
                     x1 + int(w * TEAM_JERSEY_X_START):x1 + int(w * TEAM_JERSEY_X_END)]
        if crop.size == 0:
            return None
        bgr = crop.astype(np.int16)
        b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
        # Green mask in BGR
        green = (g > 60) & (g > r + 15) & (g > b + 15)
        valid = bgr[~green]
        if len(valid) < 10:
            return None
        return np.array([float(np.median(valid[:, 0])), float(np.median(valid[:, 1])),
                         float(np.median(valid[:, 2]))], dtype=np.float32)

    # ------------------------------------------------------------------
    # Distance helpers — BGR only (no HSV, no circular hue).
    # ------------------------------------------------------------------
    def _bgr_distance(self, bgr1, bgr2):
        """3D BGR Euclidean distance. Used by EMA drift check."""
        d = np.array([float(bgr1[i]) - float(bgr2[i]) for i in range(3)], dtype=np.float32)
        return float(np.sqrt(np.sum(d * d)))

    def _feature_distance(self, f1, f2):
        """6D feature distance: jersey (B, G, R) + shorts (B, G, R).
        Shorts component only contributes when BOTH observations have a
        non-default shorts feature (i.e. shorts band was sampled, indicated
        by R channel > 5 in raw pixel space — same heuristic as before)."""
        d_j = self._bgr_distance(f1[:3], f2[:3])
        # Shorts contribution: only if either observation has a real shorts sample.
        # The shorts feature is the median BGR; if all channels are near zero
        # the shorts band was not sampled (e.g. very dark crop).
        s1_max = float(max(f1[3], f1[4], f1[5]))
        s2_max = float(max(f2[3], f2[4], f2[5]))
        if s1_max > 5.0 and s2_max > 5.0:
            d_s = self._bgr_distance(f1[3:], f2[3:])
            return float(np.sqrt(d_j * d_j + (SHORTS_FEATURE_WEIGHT * d_s) ** 2))
        return d_j

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def _build_result(self, team_ids, track_ids) -> dict:
        centroids = self.team_centroids_bgr
        team_colors = []
        track_quality = np.empty(len(team_ids), dtype=np.float32)
        soft_team_probs = np.zeros((len(team_ids), 2), dtype=np.float32)
        for i, label in enumerate(team_ids):
            t = self.tracks.get(int(track_ids[i]))
            q = float(t['quality']) if t is not None else 1.0
            track_quality[i] = q

            # Tier 2.1: GMM soft probabilities
            if USE_GMM and self.gmm_model is not None and t is not None and t['track_feature'] is not None:
                try:
                    probs = self.gmm_model.predict_proba(t['track_feature'][:3].reshape(1, -1))[0]
                    # Reorder so [0] = team0 (lowest hue), [1] = team1
                    order = np.argsort(self.gmm_model.means_[:, 0])
                    inv = np.argsort(order)
                    probs = probs[inv]
                    soft_team_probs[i] = probs
                    if t is not None:
                        t['soft_team_probs'] = probs
                except Exception:
                    pass

            if label == self.REF:
                team_colors.append(self.REF_COLOR)
            elif label == self.GK:
                team_colors.append(self.GK_COLOR)
            elif label == self.TEAM0:
                team_colors.append(tuple(map(int, centroids[0])) if centroids is not None else self.DEFAULT_TEAM_COLORS[0])
            else:
                team_colors.append(tuple(map(int, centroids[1])) if centroids is not None and len(centroids) > 1 else self.DEFAULT_TEAM_COLORS[1])

        t1 = tuple(map(int, centroids[0])) if centroids is not None else self.DEFAULT_TEAM_COLORS[0]
        t2 = tuple(map(int, centroids[1])) if centroids is not None and len(centroids) > 1 else self.DEFAULT_TEAM_COLORS[1]
        return {'team_ids': np.array(team_ids, dtype=np.int32), 'team_colors': team_colors,
                'team1_bgr': t1, 'team2_bgr': t2,
                'track_quality': track_quality,
                'soft_team_probs': soft_team_probs}

    def _empty_result(self) -> dict:
        return {'team_ids': np.empty((0,), dtype=np.int32), 'team_colors': [],
                'team1_bgr': self.DEFAULT_TEAM_COLORS[0], 'team2_bgr': self.DEFAULT_TEAM_COLORS[1],
                'track_quality': np.empty((0,), dtype=np.float32),
                'soft_team_probs': np.empty((0, 2), dtype=np.float32)}

    def reset(self):
        self.team_centroids_bgr_feat = self.team_centroids_bgr = None
        self.initialized = False
        self.gmm_model = None
        self.tracks.clear()
        self.frame_count = 0
        self._last_centroids_hsv = None
        self._prev_frame_detections = []
