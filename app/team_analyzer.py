"""TeamColorAnalyzer — Track-aware jersey-colour clustering with multi-tier robustness.

Tiers implemented (all gated behind flags in `constants.py`):
* Tier 1.1 — Gray-world illumination normalization on the jersey crop before
  HSV sampling.
* Tier 1.2 — Adaptive jersey band that tightens Y_END for tall bboxes and
  widens X band for small bboxes (handles jumping / sliding / camera tilt).
* Tier 1.3 — Per-track EMA (`track_ema_hsv`) layered on top of the running
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
    GREEN_HSV_LOWER, GREEN_HSV_UPPER, GK_COLOR_DIST_THRESHOLD, REF_SATURATION_THRESHOLD,
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
    GK_PENALTY_BOX_MIN_FRAC,
    # Tier 2
    USE_GMM, GMM_COVARIANCE_TYPE, GMM_MIN_PROB_FOR_TEAM,
    RE_CLUSTER_DRIFT_THRESHOLD, TRACK_QUALITY_EMA_ALPHA,
    TRACK_QUALITY_LABEL_FLIP_PENALTY, TRACK_HISTORY_SHORT_TERM,
    # Similar-team disambiguation
    SIMILAR_TEAM_CENTROID_DIST, JERSEY_TOPK_FRACTION, SHORTS_FEATURE_WEIGHT,
)

# Penalty area geometry (must mirror constants.PENALTY_* values for Tier 3.2)
from constants import (
    PITCH_LENGTH, PITCH_WIDTH, LEFT_PENALTY_X, RIGHT_PENALTY_X,
    PENALTY_Y_TOP, PENALTY_Y_BOTTOM,
)


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
        self.team_centroids_hsv = None
        self.team_centroids_bgr = None
        self.initialized = False
        self.gmm_model = None  # populated when USE_GMM=True

        self.tracks: dict[int, dict] = {}
        self.frame_count = 0
        self._last_centroids_hsv = None  # for drift-trigger re-cluster

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def assign_team_colors(self, frame: np.ndarray, player_xyxy: np.ndarray,
                           player_conf: np.ndarray, track_ids: np.ndarray = None,
                           H: np.ndarray = None) -> dict:
        """Assign per-detection team labels.

        Optional `H` (3x3 homography) enables Tier 1.4 touchline check and
        Tier 3.2 GK-penalty-box prior. Both are best-effort and silently
        no-op if `H` is missing.
        """
        if len(player_xyxy) == 0:
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
        team_ids = self._decide_per_frame_team_ids(track_ids, per_det_feature, frame, player_xyxy, H)

        # 6) Stale-track cleanup
        self._cleanup_stale_tracks()

        if not self.initialized or team_ids is None or len(team_ids) == 0:
            return self._empty_result()

        return self._build_result(team_ids)

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
            'sat_obs': deque(maxlen=TRACK_HISTORY_LEN),     # Tier 1.4: per-obs saturation
            'hue_obs': deque(maxlen=TRACK_HISTORY_LEN),      # Tier 1.4: per-obs hue
            'track_hsv': None,          # (2,) jersey feature [h, s] — for back-compat
            'track_feature': None,      # (4,) [h_jersey, s_jersey, h_shorts, s_shorts] for clustering
            'track_ema_hsv': None,                            # Tier 1.3
            'team_id': None,
            'team_votes': {0: 0, 1: 0, self.GK: 0, self.REF: 0},
            'team_votes_history': deque(maxlen=TRACK_HISTORY_SHORT_TERM),  # Tier 2.3
            'frames_seen': 0,
            'last_seen_frame': 0,
            'low_sat_streak': 0,
            'gk_streak': 0,
            'ref_outlier_streak': 0,
            'quality': 1.0,                                   # Tier 2.3
            'shorts_hsv': None,                               # Tier 3.3
            'pitch_positions': deque(maxlen=30),              # Tier 3.2: GK penalty-box prior
            'soft_team_probs': None,                          # Tier 2.1: P(team | track_hsv)
        }

    def _update_track(self, tid: int, feature: np.ndarray, bgr_c: np.ndarray, weight: float):
        """feature is now 4D: [h_jersey, s_jersey, h_shorts, s_shorts]."""
        t = self.tracks.get(int(tid))
        if t is None:
            t = self._new_track()
            self.tracks[int(tid)] = t

        # Update EMA regardless of weight (still useful for smoothing)
        if t['track_ema_hsv'] is None:
            t['track_ema_hsv'] = np.array([float(feature[0]), float(feature[1])], dtype=np.float32)
        else:
            a = EMA_PER_TRACK_ALPHA
            t['track_ema_hsv'] = a * np.array([float(feature[0]), float(feature[1])], dtype=np.float32) \
                                  + (1.0 - a) * t['track_ema_hsv']

        if weight < 0.08:
            t['last_seen_frame'] = self.frame_count
            t['frames_seen'] += 1
            return

        t['obs'].append((float(feature[0]), float(feature[1]), float(feature[2]), float(feature[3]), float(weight)))
        t['sat_obs'].append(float(feature[1]))
        t['hue_obs'].append(float(feature[0]))
        t['last_seen_frame'] = self.frame_count
        t['frames_seen'] += 1

        # Update running medians (2D jersey + 4D full feature) from observation history
        if len(t['obs']) >= 3:
            obs_arr = np.array([[o[0], o[1], o[2], o[3]] for o in t['obs']], dtype=np.float32)
            t['track_feature'] = np.median(obs_arr, axis=0).astype(np.float32)
            t['track_hsv'] = t['track_feature'][:2].astype(np.float32)
            t['shorts_hsv'] = t['track_feature'][2:4].astype(np.float32)

        if feature[1] < REF_SATURATION_THRESHOLD:
            t['low_sat_streak'] += 1
        else:
            t['low_sat_streak'] = max(0, t['low_sat_streak'] - 1)

    def _enough_tracks_to_init(self) -> bool:
        stable = [t for t in self.tracks.values()
                  if t['frames_seen'] >= TRACK_MIN_FRAMES_TO_CLUSTER and t['track_hsv'] is not None]
        return len(stable) >= 2

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _initialize_from_tracks(self):
        stable_tracks = [t for t in self.tracks.values()
                         if t['frames_seen'] >= TRACK_MIN_FRAMES_TO_CLUSTER
                         and t['track_feature'] is not None]
        # Drop tracks that have no shorts observation (default 0,0) for
        # clustering — the shorts component carries 0 weight for them
        # anyway, but excluding them keeps the matrix well-conditioned.
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

        self.team_centroids_hsv = sorted_c
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
            if t['track_hsv'] is None:
                continue
            label = self._initial_label_for_track(t)
            self._cast_team_vote(t, label)

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
        if t['track_feature'][1] < REF_SATURATION_THRESHOLD and t['low_sat_streak'] >= 3:
            return self.REF
        if self.team_centroids_hsv is None or len(self.team_centroids_hsv) == 0:
            return self.GK
        dists = [self._feature_distance(t['track_feature'], c) for c in self.team_centroids_hsv]
        md = float(min(dists))
        if md > np.mean(dists) + GK_COLOR_DIST_THRESHOLD * (np.std(dists) + 1e-6):
            return self.GK
        return int(np.argmin(dists))

    # ------------------------------------------------------------------
    # Periodic + drift-triggered re-clustering
    # ------------------------------------------------------------------
    def _recluster_from_tracks(self, force: bool = False):
        stable_tracks = [t for t in self.tracks.values()
                         if t['frames_seen'] >= TRACK_MIN_FRAMES_TO_CLUSTER
                         and t['track_feature'] is not None]
        colored = [t for t in stable_tracks
                   if t['track_feature'][1] >= REF_SATURATION_THRESHOLD]
        if len(colored) < 2:
            return

        feat = np.array([t['track_feature'] for t in colored], dtype=np.float32)
        k = min(self.n_clusters, len(colored))
        centroids, gmm = self._fit_cluster(feat, k)
        if centroids is None:
            return

        if (self.team_centroids_hsv is None
                or len(self.team_centroids_hsv) != len(centroids)
                or force):
            sorted_c, sorted_bgr, gmm = self._finalise_centroids(
                centroids, gmm, track_pool=colored
            )
            self.team_centroids_hsv = sorted_c
            self.team_centroids_bgr = sorted_bgr
            self.gmm_model = gmm
            self._last_centroids_hsv = sorted_c.copy()
        else:
            sorted_c, sorted_bgr, gmm = self._finalise_centroids(
                centroids, gmm, track_pool=colored
            )
            a = CENTROID_EMA_ALPHA
            self.team_centroids_hsv = a * sorted_c + (1.0 - a) * self.team_centroids_hsv
            self.team_centroids_bgr = a * sorted_bgr + (1.0 - a) * self.team_centroids_bgr

        # Re-seed by position if centroids are too close (catches the case
        # where lighting drift brings the clusters together after warm-up)
        if (len(self.team_centroids_hsv) >= 2
                and self._feature_distance(self.team_centroids_hsv[0], self.team_centroids_hsv[1])
                < SIMILAR_TEAM_CENTROID_DIST
                and self._has_position_signal(colored)):
            self._seed_labels_by_position(colored)

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
        if self.team_centroids_hsv is None:
            return False
        stable = [t for t in self.tracks.values()
                  if t['frames_seen'] >= TRACK_MIN_FRAMES_TO_CLUSTER
                  and t['track_ema_hsv'] is not None]
        if not stable:
            return False
        total, count = 0.0, 0
        for t in stable:
            ema = t['track_ema_hsv']
            d = min(self._hsv_distance(ema, c[:2]) for c in self.team_centroids_hsv)
            total += d
            count += 1
        return (total / max(count, 1)) > RE_CLUSTER_DRIFT_THRESHOLD

    # ------------------------------------------------------------------
    # Cluster fitting (KMeans or GMM, gated on USE_GMM)
    # ------------------------------------------------------------------
    def _fit_cluster(self, feat: np.ndarray, k: int):
        """Returns (centroids (k, 2) sorted by hue, gmm_or_None)."""
        if USE_GMM and k >= 1:
            GMM = _safe_import_gmm()
            if GMM is not None:
                try:
                    gmm = GMM(n_components=k, covariance_type=GMM_COVARIANCE_TYPE,
                              random_state=0, n_init=1, reg_covar=1e-3)
                    gmm.fit(feat)
                    centroids = gmm.means_.astype(np.float32)
                    return centroids, gmm
                except Exception:
                    pass
        try:
            km = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(feat)
        except Exception:
            return None, None
        return km.cluster_centers_.astype(np.float32), None

    @staticmethod
    def _centroids_to_bgr(centroids_hsv: np.ndarray) -> np.ndarray:
        """Convert centroid features (2D or 4D) to BGR colours for visualization.
        Uses only the jersey (h, s) components for the BGR sample."""
        j_h = centroids_hsv[:, 0]
        j_s = centroids_hsv[:, 1]
        return np.array([
            cv2.cvtColor(np.uint8([[[int(h), int(max(s, 80)), 200]]]),
                         cv2.COLOR_HSV2BGR)[0, 0]
            for h, s in zip(j_h, j_s)
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Per-frame team-id decision
    # ------------------------------------------------------------------
    def _decide_per_frame_team_ids(self, track_ids, per_det_feature, frame, player_xyxy, H):
        team_ids = np.full(len(track_ids), self.GK, dtype=np.int32)
        for i, tid in enumerate(track_ids):
            t = self.tracks.get(int(tid))
            if t is None or t['track_feature'] is None:
                continue

            obs_feature = per_det_feature[i]
            track_feature = t['track_feature']
            obs_dist_to_track = self._feature_distance(obs_feature, track_feature)

            if t['frames_seen'] >= TRACK_MIN_FRAMES_FOR_TEAM and t['team_id'] is not None:
                if obs_dist_to_track > TRACK_RELABEL_DIST:
                    team_ids[i] = t['team_id']
                    continue

            label = self._vote_for_label(t, obs_feature)
            self._cast_team_vote(t, label)

            resolved = self._resolve_track_team(t)

            # Tier 3.2: GK penalty-box prior (only when H + bbox available)
            if H is not None and player_xyxy is not None and resolved == self.GK:
                resolved = self._apply_gk_penalty_prior(t, player_xyxy[i], H, resolved)

            team_ids[i] = resolved
            t['team_id'] = team_ids[i]

            # Update per-track quality (Tier 2.3) using short-term flip rate
            t['quality'] = self._compute_track_quality(t)
        return team_ids

    def _vote_for_label(self, t, obs_feature):
        """Pick a label (team 0/1, GK, REF) to cast a vote for this frame."""
        # Tier 1.4a: saturation-histogram referee check
        if self._saturation_histogram_says_ref(t):
            return self.REF
        # Tier 1.4b: hue multi-modality check
        if self._hue_multimodal_says_ref(t):
            return self.REF
        # Legacy low-sat referee heuristic (jersey band only)
        if obs_feature[1] < REF_SATURATION_THRESHOLD:
            return self.REF

        if self.team_centroids_hsv is None or len(self.team_centroids_hsv) == 0:
            return self.GK

        dists = [self._feature_distance(obs_feature, c) for c in self.team_centroids_hsv]
        md = float(min(dists))
        mean_d = float(np.mean(dists))
        std_d = float(np.std(dists))

        # Strong outlier from BOTH teams → referee
        if md > mean_d + REF_SATURATION_THRESHOLD * 0.05 + mean_d * 0.8:
            return self.REF
        # Mild outlier from nearest team → GK candidate
        if md > mean_d + GK_COLOR_DIST_THRESHOLD * (std_d + 1e-6):
            return self.GK
        return int(np.argmin(dists))

    def _cast_team_vote(self, t, label):
        for k in list(t['team_votes'].keys()):
            t['team_votes'][k] = max(0, int(t['team_votes'][k]) - 1)
        t['team_votes'][label] = t['team_votes'].get(label, 0) + 4
        t['team_votes_history'].append(int(label))

        if label == self.REF:
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
        if t['track_feature'][1] < REF_SATURATION_THRESHOLD:
            return self.REF
        if self.team_centroids_hsv is None or len(self.team_centroids_hsv) == 0:
            return self.GK
        dists = [self._feature_distance(t['track_feature'], c) for c in self.team_centroids_hsv]
        return int(np.argmin(dists))

    # ------------------------------------------------------------------
    # Tier 1.4: referee detection cues
    # ------------------------------------------------------------------
    def _saturation_histogram_says_ref(self, t) -> bool:
        """True if > REF_SAT_HIST_FRACTION of the track's observations have
        saturation below REF_SAT_HIST_THRESHOLD (referee / black-white)."""
        if len(t['sat_obs']) < 8:
            return False
        sats = np.array(t['sat_obs'], dtype=np.float32)
        return float(np.mean(sats < REF_SAT_HIST_THRESHOLD)) >= REF_SAT_HIST_FRACTION

    def _hue_multimodal_says_ref(self, t) -> bool:
        """True if the track's hue histogram has >= REF_HUE_MULTIMODAL_MODES
        modes above REF_HUE_MULTIMODAL_FRAC mass (mixed colors → referee)."""
        if len(t['hue_obs']) < 8:
            return False
        hues = np.array(t['hue_obs'], dtype=np.float32)
        hist, _ = np.histogram(hues, bins=REF_BINS, range=(0, 180))
        if hist.sum() == 0:
            return False
        frac = hist / hist.sum()
        # Count contiguous peaks (above threshold), not noise
        above = frac >= REF_HUE_MULTIMODAL_FRAC
        # Smooth out single-bin noise
        smoothed = np.convolve(above.astype(np.int32), np.ones(3, dtype=np.int32), mode='same')
        n_modes = int(np.sum(smoothed >= 2))
        return n_modes >= REF_HUE_MULTIMODAL_MODES

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
    # Tier 3.2: GK penalty-box prior
    # ------------------------------------------------------------------
    def _apply_gk_penalty_prior(self, t, bbox_xyxy, H, current_label):
        """If a track is labelled GK but doesn't actually stay in either
        penalty area, flip back to nearest-centroid team."""
        if H is None or bbox_xyxy is None:
            return current_label
        # Project bbox bottom-center to pitch coords
        try:
            bx = 0.5 * (float(bbox_xyxy[0]) + float(bbox_xyxy[2]))
            by = float(bbox_xyxy[3])
            pt = np.array([[[bx, by]]], dtype=np.float32)
            proj = cv2.perspectiveTransform(pt, H)[0, 0]
            x, y = float(proj[0]), float(proj[1])
        except Exception:
            return current_label
        t['pitch_positions'].append((x, y))
        if len(t['pitch_positions']) < 10:
            return current_label
        in_box = 0
        for (px, py) in t['pitch_positions']:
            left_box = (px <= LEFT_PENALTY_X) and (PENALTY_Y_TOP <= py <= PENALTY_Y_BOTTOM)
            right_box = (px >= RIGHT_PENALTY_X) and (PENALTY_Y_TOP <= py <= PENALTY_Y_BOTTOM)
            if left_box or right_box:
                in_box += 1
        frac = in_box / len(t['pitch_positions'])
        if frac < GK_PENALTY_BOX_MIN_FRAC:
            # Demote to nearest-centroid team
            return self._fallback_label(t)
        return current_label

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
        """Return (feature_4d, bgr_3d) where feature_4d is
        [h_jersey, s_jersey, h_shorts, s_shorts].

        Jersey feature uses the TOP-K most-saturated pixels after the green /
        dark / bright masks, so vivid jersey pixels dominate over white
        stripes or pitch noise.
        """
        x1, y1, x2, y2 = max(0, int(bbox[0])), max(0, int(bbox[1])), \
                          min(frame.shape[1] - 1, int(bbox[2])), min(frame.shape[0] - 1, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return np.zeros(4, dtype=np.float32), np.array([128, 128, 128], dtype=np.float32)

        h, w = y2 - y1, x2 - x1
        # Tier 1.2: adaptive jersey band
        if ADAPTIVE_JERSEY_BAND:
            aspect = w / max(h, 1)
            area = h * w
            y_end = TEAM_JERSEY_Y_END
            if aspect < 0.45:
                y_end = max(TEAM_JERSEY_Y_START + 0.15, y_end - 0.05)
            elif area < 0.5 * self.REF_BBOX_AREA and aspect < 0.7:
                y_end = min(0.6, y_end + 0.05)
            y_start, y_end_used = TEAM_JERSEY_Y_START, y_end
        else:
            y_start, y_end_used = TEAM_JERSEY_Y_START, TEAM_JERSEY_Y_END

        jersey_crop = frame[y1 + int(h * y_start):y1 + int(h * y_end_used),
                            x1 + int(w * TEAM_JERSEY_X_START):x1 + int(w * TEAM_JERSEY_X_END)]
        if jersey_crop.size == 0:
            return np.zeros(4, dtype=np.float32), np.array([128, 128, 128], dtype=np.float32)

        jersey_hsv = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(jersey_hsv, np.array(GREEN_HSV_LOWER, dtype=np.uint8), np.array(GREEN_HSV_UPPER, dtype=np.uint8))
        dark = cv2.inRange(jersey_hsv, np.array([0, 0, 0]), np.array([180, 255, 30]))
        bright = cv2.inRange(jersey_hsv, np.array([0, 0, 230]), np.array([180, 40, 255]))
        valid = jersey_hsv[cv2.bitwise_and(cv2.bitwise_not(green), cv2.bitwise_and(cv2.bitwise_not(dark), cv2.bitwise_not(bright))) > 0]
        if len(valid) < INVALID_PIXEL_MIN:
            valid = jersey_hsv.reshape(-1, 3)

        # Top-K saturated pixels — discriminative when teams have stripes
        # (e.g., red+white vs blue+red) where the median washes out.
        k = max(8, int(len(valid) * JERSEY_TOPK_FRACTION))
        order = np.argsort(valid[:, 1])  # sort by saturation ascending
        topk = valid[order[-k:]] if len(order) >= k else valid
        jh = float(np.mean(topk[:, 0]))
        js = float(np.mean(topk[:, 1]))
        jv = float(np.mean(topk[:, 2]))

        # Shorts band — secondary feature (Tier 3.3)
        shorts_hsv = self._extract_shorts_color(frame, bbox)
        if shorts_hsv is None:
            sh, ss = 0.0, 0.0
        else:
            sh, ss = float(shorts_hsv[0]), float(shorts_hsv[1])

        feature = np.array([jh, js, sh, ss], dtype=np.float32)
        vis = cv2.cvtColor(np.uint8([[[int(jh), int(max(js, 80)), int(max(jv, 80))]]]),
                           cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
        return feature, vis

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
        whose jersey color is uncertain."""
        x1, y1, x2, y2 = max(0, int(bbox[0])), max(0, int(bbox[1])), \
                          min(frame.shape[1] - 1, int(bbox[2])), min(frame.shape[0] - 1, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None
        h, w = y2 - y1, x2 - x1
        crop = frame[y1 + int(h * SHORTS_BAND_Y_START):y1 + int(h * SHORTS_BAND_Y_END),
                     x1 + int(w * TEAM_JERSEY_X_START):x1 + int(w * TEAM_JERSEY_X_END)]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array(GREEN_HSV_LOWER, dtype=np.uint8), np.array(GREEN_HSV_UPPER, dtype=np.uint8))
        valid = hsv[cv2.bitwise_not(green) > 0]
        if len(valid) < 10:
            return None
        return np.array([float(np.median(valid[:, 0])), float(np.median(valid[:, 1]))], dtype=np.float32)

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------
    def _hsv_distance(self, hsv1, hsv2):
        """2D HSV (jersey) distance — used by EMA drift check and tests."""
        dh = min(abs(float(hsv1[0]) - float(hsv2[0])), 180 - abs(float(hsv1[0]) - float(hsv2[0])))
        return np.sqrt(dh * dh + (float(hsv1[1]) - float(hsv2[1])) ** 2)

    def _feature_distance(self, f1, f2):
        """4D feature distance: jersey (h, s) + shorts (h, s).
        Shorts component only contributes when BOTH observations have a
        non-default shorts feature (sat > 5)."""
        dh_j = min(abs(float(f1[0]) - float(f2[0])), 180 - abs(float(f1[0]) - float(f2[0])))
        ds_j = float(f1[1]) - float(f2[1])
        d_j = np.sqrt(dh_j * dh_j + ds_j * ds_j)
        s1 = float(f1[3])
        s2 = float(f2[3])
        if s1 > 5.0 and s2 > 5.0:
            dh_s = min(abs(float(f1[2]) - float(f2[2])), 180 - abs(float(f1[2]) - float(f2[2])))
            ds_s = s1 - s2
            d_s = np.sqrt(dh_s * dh_s + ds_s * ds_s)
            return np.sqrt(d_j * d_j + (SHORTS_FEATURE_WEIGHT * d_s) ** 2)
        return d_j

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def _build_result(self, team_ids) -> dict:
        centroids = self.team_centroids_bgr
        team_colors = []
        track_quality = np.empty(len(team_ids), dtype=np.float32)
        soft_team_probs = np.zeros((len(team_ids), 2), dtype=np.float32)
        for i, tid in enumerate(team_ids):
            t = self.tracks.get(int(tid))
            q = float(t['quality']) if t is not None else 1.0
            track_quality[i] = q

            # Tier 2.1: GMM soft probabilities (use 4D feature for predictions)
            if USE_GMM and self.gmm_model is not None and t is not None and t['track_feature'] is not None:
                try:
                    probs = self.gmm_model.predict_proba(t['track_feature'].reshape(1, -1))[0]
                    # Reorder so [0] = team0 (lowest hue), [1] = team1
                    order = np.argsort(self.gmm_model.means_[:, 0])
                    inv = np.argsort(order)
                    probs = probs[inv]
                    soft_team_probs[i] = probs
                    if t is not None:
                        t['soft_team_probs'] = probs
                except Exception:
                    pass

            if tid == self.REF:
                team_colors.append(self.REF_COLOR)
            elif tid == self.GK:
                team_colors.append(self.GK_COLOR)
            elif tid == self.TEAM0:
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
        self.team_centroids_hsv = self.team_centroids_bgr = None
        self.initialized = False
        self.gmm_model = None
        self.tracks.clear()
        self.frame_count = 0
        self._last_centroids_hsv = None
