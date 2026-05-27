"""
TeamColorAnalyzer — Extract dominant jersey colors from player detections,
cluster players into teams via K-means on HSV hue-saturation, and assign
team colors consistently across video frames.

Improvements over v1:
 - Crops a middle band of the bounding box (avoids heads + shorts)
 - Uses HSV hue+saturation for lighting-invariant clustering
 - Uses median hue for robust per-player dominant color
 - Frequent centroid refresh (every N frames)
 - Wider green pitch mask for better background exclusion
"""

import numpy as np
import cv2
from collections import deque, Counter
from sklearn.cluster import KMeans
from constants import (
    TEAM_N_CLUSTERS,
    TEAM_JERSEY_Y_START,
    TEAM_JERSEY_Y_END,
    TEAM_JERSEY_X_START,
    TEAM_JERSEY_X_END,
    GREEN_HSV_LOWER,
    GREEN_HSV_UPPER,
    COLOR_CACHE_REFRESH_N,
    GK_COLOR_DIST_THRESHOLD,
    REF_DIST_THRESHOLD,
    REF_SATURATION_THRESHOLD,
)


class TeamColorAnalyzer:
    """
    Analyzes player bounding boxes to extract team jersey colors and
    assign players to teams.

    Maintains cached team color centroids across frames for consistent
    team assignment throughout a video.
    """

    # Default BGR colors for team visualization
    DEFAULT_TEAM_COLORS = [
        (255, 0, 0),     # Team 1: Blue
        (0, 0, 255),     # Team 2: Red
    ]

    # BGR color for goalkeeper dots
    GK_COLOR = (0, 255, 255)  # Yellow

    # BGR color for referee/unknown
    REF_COLOR = (0, 0, 0)  # Black

    # Maximum frames of team assignment history to keep per track
    MAX_HISTORY_LENGTH = 20
    
    # Minimum majority ratio required to change a track's team assignment.
    # Higher = more stable (less flickering), but slower to adapt to real changes.
    MAJORITY_THRESHOLD = 0.65
    
    # How much extra weight to give the most recent assignment.
    # Helps stable tracks resist noise from occasional misclassifications.
    RECENCY_WEIGHT = 1.5

    def __init__(self, n_clusters: int = TEAM_N_CLUSTERS, lock_centroids: bool = True):
        """
        Args:
            n_clusters: Number of teams to cluster (typically 2).
            lock_centroids: If True, team centroids are computed ONCE on first frame and
                            then locked for the entire video. Prevents color flickering
                            caused by periodic K-means re-clustering.
        """
        self.n_clusters = n_clusters
        self.team_centroids_hsv = None   # Cached HSV centroids (n_clusters, 2) — hue+sat only
        self.team_centroids_bgr = None   # Cached BGR centroids for visualization
        self.frame_counter = 0
        self.initialized = False
        self.lock_centroids = lock_centroids

        # Per-track team assignment history for cross-frame consistency
        self.track_team_history = {}      # track_id -> deque of recent team IDs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign_team_colors(
        self,
        frame: np.ndarray,
        player_xyxy: np.ndarray,
        player_conf: np.ndarray,
        track_ids: np.ndarray = None,
    ) -> dict:
        """
        Assign team colors to all detected players.

        Args:
            frame: BGR image (H, W, 3).
            player_xyxy: Player bounding boxes (N, 4) in [x1, y1, x2, y2] format.
            player_conf: Confidence scores (N,).
            track_ids: Optional (N,) int array of ByteTrack track IDs for
                       cross-frame player identity. Enables per-track majority
                       voting for stable team assignment.

        Returns:
            dict with keys:
                'team_ids':    (N,) int array: 0=Team1, 1=Team2, -1=GK, -2=Referee
                'team_colors': (N, 3) BGR color tuple for each player
                'team1_bgr':   (3,) BGR — Team 1 representative color
                'team2_bgr':   (3,) BGR — Team 2 representative color
        """
        self.frame_counter += 1

        if len(player_xyxy) == 0:
            return self._empty_result()

        # ---- Step 1: Extract dominant HSV color per player ----
        player_hsv = []      # (N, 2) — [hue, saturation]
        player_bgr = []      # (N, 3) — BGR for visualization
        for bbox in player_xyxy:
            hsv_color, bgr_color = self._extract_dominant_color(frame, bbox)
            player_hsv.append(hsv_color)
            player_bgr.append(bgr_color)

        player_hsv = np.array(player_hsv, dtype=np.float32)   # (N, 2)
        player_bgr = np.array(player_bgr, dtype=np.float32)   # (N, 3)

        # ---- Step 2: Cluster or re-assign ----
        # When lock_centroids is True, only cluster once (on first frame).
        # Otherwise, re-cluster periodically based on COLOR_CACHE_REFRESH_N.
        if not self.initialized:
            team_ids, centroids_hsv, centroids_bgr = self._cluster_teams(
                player_hsv, player_bgr
            )
            if centroids_bgr is not None:
                self.team_centroids_hsv = centroids_hsv
                self.team_centroids_bgr = centroids_bgr
                self.initialized = True
        elif self.lock_centroids:
            # Centroids are locked — always use nearest-centroid assignment
            team_ids = self._assign_to_nearest_team(player_hsv)
        elif self.frame_counter % COLOR_CACHE_REFRESH_N == 0:
            # Periodic re-clustering (only when lock_centroids=False)
            team_ids, centroids_hsv, centroids_bgr = self._cluster_teams(
                player_hsv, player_bgr
            )
            if centroids_bgr is not None:
                self.team_centroids_hsv = centroids_hsv
                self.team_centroids_bgr = centroids_bgr
        else:
            team_ids = self._assign_to_nearest_team(player_hsv)

        if not self.initialized or team_ids is None:
            return self._empty_result()

        # ---- Step 2b: Per-track majority voting for stable team assignment ----
        # Apply ByteTrack per-player history to smooth out frame-to-frame
        # team label noise and detect K-means centroid label swaps.
        if track_ids is not None and len(track_ids) > 0 and len(track_ids) == len(team_ids):
            team_ids = self._apply_track_voting(track_ids, team_ids)

        # ---- Step 3: Build per-player color list ----
        centroids = self.team_centroids_bgr
        team_colors = []
        for tid in team_ids:
            if tid == -2:
                team_colors.append(self.REF_COLOR)  # referee → black
            elif tid == -1:
                team_colors.append(self.GK_COLOR)   # goalkeeper → yellow
            elif tid == 0:
                if centroids is not None:
                    team_colors.append(tuple(map(int, centroids[0])))
                else:
                    team_colors.append(self.DEFAULT_TEAM_COLORS[0])
            else:
                if centroids is not None and len(centroids) > 1:
                    team_colors.append(tuple(map(int, centroids[1])))
                else:
                    team_colors.append(self.DEFAULT_TEAM_COLORS[1])

        if centroids is not None:
            team1_bgr = tuple(map(int, centroids[0]))
            team2_bgr = tuple(map(int, centroids[1])) if len(centroids) > 1 else self.DEFAULT_TEAM_COLORS[1]
        else:
            team1_bgr = self.DEFAULT_TEAM_COLORS[0]
            team2_bgr = self.DEFAULT_TEAM_COLORS[1]

        return {
            'team_ids': np.array(team_ids, dtype=np.int32),
            'team_colors': team_colors,
            'team1_bgr': team1_bgr,
            'team2_bgr': team2_bgr,
        }

    # ------------------------------------------------------------------
    # Per-track team voting (ByteTrack integration)
    # ------------------------------------------------------------------

    def _apply_track_voting(
        self,
        track_ids: np.ndarray,
        team_ids_raw: np.ndarray,
    ) -> np.ndarray:
        """
        Apply per-track RECENCY-WEIGHTED majority voting for stable team
        assignments across frames.

        Maintains a deque of recent team assignments per track_id. Uses a
        weighted vote where the most recent assignment gets extra weight to
        provide a "sticky" bias. Requires a clear majority (>MAJORITY_THRESHOLD)
        to flip a track's team. If no clear majority, keeps the previous
        assignment.

        Also detects global label swaps (K-means centroid label flip) and
        corrects all assignments when >60% of tracks simultaneously flip.

        Args:
            track_ids: (N,) int — ByteTrack track IDs per detection.
            team_ids_raw: (N,) int — Raw team IDs from K-means/nearest-centroid.

        Returns:
            (N,) int — Smoothed team IDs after per-track voting + swap correction.
        """
        n = len(team_ids_raw)
        stable_ids = np.copy(team_ids_raw).astype(np.int32)

        # ---- Update per-track history ----
        for i, tid in enumerate(track_ids):
            tid = int(tid)
            raw_team = int(team_ids_raw[i])

            if tid not in self.track_team_history:
                self.track_team_history[tid] = deque(maxlen=self.MAX_HISTORY_LENGTH)

            self.track_team_history[tid].append(raw_team)

        # ---- Apply recency-weighted majority voting ----
        for i, tid in enumerate(track_ids):
            tid = int(tid)
            history = list(self.track_team_history[tid])

            if len(history) >= 3:
                # Only consider valid team assignments (0 or 1) for voting
                valid_assignments = [t for t in history if t >= 0]
                if len(valid_assignments) >= 2:
                    # Weighted voting: most recent assignment gets RECENCY_WEIGHT
                    weights = []
                    team_values = []
                    for pos, t in enumerate(valid_assignments):
                        team_values.append(t)
                        if pos == len(valid_assignments) - 1:
                            weights.append(self.RECENCY_WEIGHT)  # Extra weight to latest
                        else:
                            weights.append(1.0)

                    # Count weighted votes for each team
                    vote_count = {0: 0.0, 1: 0.0}
                    for team, w in zip(team_values, weights):
                        if team in vote_count:
                            vote_count[team] += w

                    total_weight = sum(weights)
                    best_team = max(vote_count, key=vote_count.get)
                    best_ratio = vote_count[best_team] / total_weight

                    # Only flip if clear majority. Otherwise keep previous assignment.
                    if best_ratio >= self.MAJORITY_THRESHOLD:
                        stable_ids[i] = best_team
                    else:
                        # No clear majority: keep the most common historical assignment
                        # or the raw assignment if history is split
                        if len(valid_assignments) >= 5:
                            stable_ids[i] = max(vote_count, key=vote_count.get)
                        # else keep raw assignment (already in stable_ids)

        # ---- Global label swap detection ----
        # If most tracks simultaneously flip teams, it means K-means centroids
        # swapped their label mapping (Team 0 <-> Team 1).
        tracks_with_history = [
            int(tid) for tid in track_ids
            if int(tid) in self.track_team_history
            and len(self.track_team_history[int(tid)]) >= 4
        ]

        if len(tracks_with_history) >= 4:
            flips = 0
            for tid in tracks_with_history:
                history = list(self.track_team_history[tid])
                # Filter to valid team assignments only
                valid = [t for t in history if t >= 0]
                if len(valid) < 4:
                    continue
                half = len(valid) // 2
                first_half = valid[:half]
                second_half = valid[half:]
                if first_half and second_half:
                    first_team = Counter(first_half).most_common(1)[0][0]
                    second_team = Counter(second_half).most_common(1)[0][0]
                    if first_team != second_team:
                        flips += 1

            # If >60% of tracked players flipped teams, correct globally
            if flips > len(tracks_with_history) * 0.6:
                # Flip all team assignments
                for i in range(n):
                    if stable_ids[i] >= 0:  # Don't flip GK (-1) or Ref (-2)
                        stable_ids[i] = 1 - stable_ids[i]
                # Also correct the stored history to reflect the swap
                for tid in tracks_with_history:
                    corrected = deque(
                        maxlen=self.MAX_HISTORY_LENGTH
                    )
                    for t in self.track_team_history[tid]:
                        corrected.append(1 - t if t >= 0 else t)
                    self.track_team_history[tid] = corrected

        return stable_ids

    # ------------------------------------------------------------------
    # Color extraction
    # ------------------------------------------------------------------

    def _extract_dominant_color(
        self, frame: np.ndarray, bbox: np.ndarray
    ):
        """
        Extract the dominant jersey color from a player's chest region.

        Crops a middle band of the bounding box (avoids head and shorts),
        converts to HSV, masks out green/bright/dark pixels, and returns
        the median hue+saturation as the dominant color.

        Args:
            frame: BGR image.
            bbox: [x1, y1, x2, y2].

        Returns:
            (hsv_color, bgr_color) where:
                hsv_color: (2,) — [median_hue, median_saturation] in HSV space
                bgr_color: (3,) — corresponding BGR color
        """
        x1, y1, x2, y2 = map(int, bbox)
        # Clamp to frame dimensions
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)

        if x2 <= x1 or y2 <= y1:
            return np.array([0, 0], dtype=np.float32), np.array([128, 128, 128], dtype=np.float32)

        h = y2 - y1
        w = x2 - x1

        # Crop a middle band of the bbox where the jersey/chest is
        crop_y1 = y1 + int(h * TEAM_JERSEY_Y_START)
        crop_y2 = y1 + int(h * TEAM_JERSEY_Y_END)
        crop_x1 = x1 + int(w * TEAM_JERSEY_X_START)
        crop_x2 = x1 + int(w * TEAM_JERSEY_X_END)

        if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
            return np.array([0, 0], dtype=np.float32), np.array([128, 128, 128], dtype=np.float32)

        jersey_roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if jersey_roi.size == 0:
            return np.array([0, 0], dtype=np.float32), np.array([128, 128, 128], dtype=np.float32)

        # Convert to HSV
        hsv_roi = cv2.cvtColor(jersey_roi, cv2.COLOR_BGR2HSV)

        # Mask out green pitch pixels (wider range to be more aggressive)
        lower_green = np.array(GREEN_HSV_LOWER, dtype=np.uint8)
        upper_green = np.array(GREEN_HSV_UPPER, dtype=np.uint8)
        green_mask = cv2.inRange(hsv_roi, lower_green, upper_green)

        # Mask out very dark (shadows) and very bright (white lines, sky)
        dark_mask = cv2.inRange(hsv_roi, np.array([0, 0, 0]), np.array([180, 255, 30]))
        # Bright mask: only filter very low-saturation bright pixels (white lines, sky)
        # Use S < 40 to avoid filtering light-colored jerseys (e.g., light pink)
        bright_mask = cv2.inRange(hsv_roi, np.array([0, 0, 230]), np.array([180, 40, 255]))

        # Keep pixels that are NOT green AND NOT dark AND NOT bright
        valid_mask = cv2.bitwise_not(green_mask)
        valid_mask = cv2.bitwise_and(valid_mask, cv2.bitwise_not(dark_mask))
        valid_mask = cv2.bitwise_and(valid_mask, cv2.bitwise_not(bright_mask))

        valid_hsv = hsv_roi[valid_mask > 0]

        if len(valid_hsv) < 15:
            # Too few valid pixels — fall back to whole ROI median
            valid_hsv = hsv_roi.reshape(-1, 3)

        # Use median hue and saturation for robustness (outlier-resistant)
        median_h = float(np.median(valid_hsv[:, 0]))
        median_s = float(np.median(valid_hsv[:, 1]))
        median_v = float(np.median(valid_hsv[:, 2]))

        # Also use the most common hue (mode-ish: fine histogram)
        hist_h = np.bincount(np.clip(valid_hsv[:, 0].astype(np.int32), 0, 179))
        mode_h = float(np.argmax(hist_h))

        # Prefer the mode hue if the median and mode are close; otherwise use median
        if abs(mode_h - median_h) < 20:
            final_h = mode_h
        else:
            final_h = median_h

        # Reconstruct HSV → BGR for visualization
        vis_hsv = np.uint8([[[final_h, median_s, max(median_v, 80)]]])
        vis_bgr = cv2.cvtColor(vis_hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)

        return np.array([final_h, median_s], dtype=np.float32), vis_bgr

    # ------------------------------------------------------------------
    # Team clustering (on HSV hue+saturation)
    # ------------------------------------------------------------------

    @staticmethod
    def _hsv_distance(hsv1, hsv2):
        """
        Compute distance between two HSV values, handling circular hue.
        Hue is in [0, 179] and wraps around (0=red, 179=red).
        """
        h1, s1 = hsv1
        h2, s2 = hsv2
        # Circular hue difference
        dh = min(abs(h1 - h2), 180 - abs(h1 - h2))
        # Weight saturation difference
        ds = abs(s1 - s2)
        # Return Euclidean distance in hue-sat space with circular hue
        return np.sqrt(dh * dh + ds * ds)

    @staticmethod
    def _hsv_dist_matrix(centroids, points):
        """Compute distance matrix between centroids and points using circular hue."""
        n_centroids = len(centroids)
        n_points = len(points)
        dists = np.zeros((n_points, n_centroids), dtype=np.float32)
        for i in range(n_points):
            for j in range(n_centroids):
                dists[i, j] = TeamColorAnalyzer._hsv_distance(points[i], centroids[j])
        return dists

    def _cluster_teams(self, player_hsv, player_bgr):
        """
        Cluster players into teams using K-means on HSV (hue, saturation).
        Uses circular hue distance for accurate color grouping.

        Improvement: Pre-filter low-saturation players (potential referees
        in black/white/gray) before K-means clustering so they don't
        contaminate the team centroids.

        Args:
            player_hsv: (N, 2) — [hue, saturation] per player.
            player_bgr: (N, 3) — BGR colors for visualization.

        Returns:
            (team_ids, centroids_hsv, centroids_bgr)
        """
        n_players = len(player_hsv)
        if n_players < 2:
            return None, None, None

        # ---- Pre-filter: identify low-saturation players (potential referees) ----
        # Remove them from clustering input so they don't pull team centroids
        low_sat_mask = player_hsv[:, 1] < REF_SATURATION_THRESHOLD
        colored_idx = np.where(~low_sat_mask)[0]  # indices of colored players

        if len(colored_idx) < 2:
            # Not enough colored players to cluster — fall back to original method
            colored_idx = np.arange(n_players)
            low_sat_mask = np.zeros(n_players, dtype=bool)

        colored_hsv = player_hsv[colored_idx]  # (M, 2) — only colored players

        # ---- K-means clustering on colored players only ----
        k = min(self.n_clusters, len(colored_hsv))
        try:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
            kmeans.fit(colored_hsv)
            labels = kmeans.labels_
            centroids_hsv = kmeans.cluster_centers_  # (k, 2)
        except Exception:
            return None, None, None

        # Sort centroids by hue for consistent team ordering
        hue_order = np.argsort(centroids_hsv[:, 0])
        remap = {old: new for new, old in enumerate(hue_order)}
        colored_team_ids = np.array([remap[l] for l in labels], dtype=np.int32)
        sorted_centroids_hsv = centroids_hsv[hue_order]

        # Build corresponding BGR centroids for visualization
        sorted_centroids_bgr = []
        for i in range(k):
            h, s = sorted_centroids_hsv[i]
            vis_hsv = np.uint8([[[int(h), int(max(s, 80)), 200]]])
            vis_bgr = cv2.cvtColor(vis_hsv, cv2.COLOR_HSV2BGR)[0, 0]
            sorted_centroids_bgr.append(vis_bgr)
        sorted_centroids_bgr = np.array(sorted_centroids_bgr, dtype=np.float32)

        # ---- Build full team_ids array ----
        # Initialize all as referee, then fill in colored players
        team_ids = np.full(n_players, -2, dtype=np.int32)
        for pos, idx_in_full in enumerate(colored_idx):
            team_ids[idx_in_full] = colored_team_ids[pos]

        # ---- Goalkeeper detection on colored players ----
        for i in colored_idx:
            dists = [self._hsv_distance(player_hsv[i], sorted_centroids_hsv[t])
                     for t in range(k)]
            min_dist = min(dists)
            mean_dist = np.mean(dists)
            std_dist = np.std(dists) + 1e-6
            if min_dist > mean_dist + GK_COLOR_DIST_THRESHOLD * std_dist:
                team_ids[i] = -1

        # --- Referee detection on clustered (colored) players ---
        # Check if any colored player is an outlier from their assigned team
        for i in colored_idx:
            if team_ids[i] == -1:
                continue  # already GK, don't override

            t = team_ids[i]
            dist_to_own = self._hsv_distance(player_hsv[i], sorted_centroids_hsv[t])
            # Only compute within-cluster stats if we have enough players in this team
            team_mask = (team_ids == t)
            if team_mask.sum() > 2:
                team_dists = np.array([
                    self._hsv_distance(player_hsv[j], sorted_centroids_hsv[t])
                    for j in range(n_players) if team_ids[j] == t
                ])
                team_mean = float(np.mean(team_dists))
                team_std = float(np.std(team_dists)) + 1e-6
                if dist_to_own > team_mean + REF_DIST_THRESHOLD * team_std:
                    team_ids[i] = -2

        return team_ids, sorted_centroids_hsv, sorted_centroids_bgr

    def _assign_to_nearest_team(self, player_hsv):
        """
        Assign each player to the nearest cached team centroid using
        circular hue distance.

        Includes referee detection via:
          1. Low saturation (achromatic: white/black/gray uniforms)
          2. Outlier distance: player is far from their assigned team centroid
             (mirrors the logic in _cluster_teams for cached frames)

        Args:
            player_hsv: (N, 2) — [hue, saturation] per player.

        Returns:
            (N,) int array: 0=Team1, 1=Team2, -1=GK, -2=Referee.
        """
        if self.team_centroids_hsv is None:
            return np.full(len(player_hsv), -1, dtype=np.int32)

        n_players = len(player_hsv)
        team_ids = np.zeros(n_players, dtype=np.int32)
        centroids = self.team_centroids_hsv

        for i in range(n_players):
            dists = [self._hsv_distance(player_hsv[i], centroids[t])
                     for t in range(len(centroids))]
            min_dist = min(dists)
            min_idx = int(np.argmin(dists))

            mean_dist = np.mean(dists)
            std_dist = np.std(dists) + 1e-6
            if min_dist > mean_dist + GK_COLOR_DIST_THRESHOLD * std_dist:
                team_ids[i] = -1
            else:
                team_ids[i] = min_idx

        # --- Referee detection for cached assignments ---
        # 1. Low-saturation check (white/black/gray referee uniforms)
        # 2. Outlier check: far from assigned team centroid
        for i in range(n_players):
            if team_ids[i] == -1:
                continue  # already GK

            sat = player_hsv[i, 1]

            # 1. Very low saturation → achromatic (white/black/gray) → referee
            if sat < REF_SATURATION_THRESHOLD:
                team_ids[i] = -2
                continue

            # 2. Outlier check: is this player unusually far from their team?
            t = team_ids[i]
            dist_to_own = self._hsv_distance(player_hsv[i], centroids[t])
            # Compute within-team stats
            team_mask = (team_ids == t)
            if team_mask.sum() > 2:
                team_dists = np.array([
                    self._hsv_distance(player_hsv[j], centroids[t])
                    for j in range(n_players) if team_ids[j] == t
                ])
                team_mean = float(np.mean(team_dists))
                team_std = float(np.std(team_dists)) + 1e-6
                if dist_to_own > team_mean + REF_DIST_THRESHOLD * team_std:
                    team_ids[i] = -2

        return team_ids

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_result(self) -> dict:
        """Return an empty result dict when no players are detected."""
        return {
            'team_ids': np.empty((0,), dtype=np.int32),
            'team_colors': [],
            'team1_bgr': self.DEFAULT_TEAM_COLORS[0],
            'team2_bgr': self.DEFAULT_TEAM_COLORS[1],
        }

    def reset(self):
        """Reset cached centroids and track history for a new video."""
        self.team_centroids_hsv = None
        self.team_centroids_bgr = None
        self.frame_counter = 0
        self.initialized = False
        self.track_team_history.clear()