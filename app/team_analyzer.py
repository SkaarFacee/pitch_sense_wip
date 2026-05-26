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

    def __init__(self, n_clusters: int = TEAM_N_CLUSTERS):
        """
        Args:
            n_clusters: Number of teams to cluster (typically 2).
        """
        self.n_clusters = n_clusters
        self.team_centroids_hsv = None   # Cached HSV centroids (n_clusters, 2) — hue+sat only
        self.team_centroids_bgr = None   # Cached BGR centroids for visualization
        self.frame_counter = 0
        self.initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign_team_colors(
        self,
        frame: np.ndarray,
        player_xyxy: np.ndarray,
        player_conf: np.ndarray,
    ) -> dict:
        """
        Assign team colors to all detected players.

        Args:
            frame: BGR image (H, W, 3).
            player_xyxy: Player bounding boxes (N, 4) in [x1, y1, x2, y2] format.
            player_conf: Confidence scores (N,).

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
        if (not self.initialized) or (self.frame_counter % COLOR_CACHE_REFRESH_N == 0):
            team_ids, centroids_hsv, centroids_bgr = self._cluster_teams(
                player_hsv, player_bgr
            )
            if centroids_bgr is not None:
                self.team_centroids_hsv = centroids_hsv
                self.team_centroids_bgr = centroids_bgr
                self.initialized = True
        else:
            team_ids = self._assign_to_nearest_team(player_hsv)

        if not self.initialized or team_ids is None:
            return self._empty_result()

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
        """Reset cached centroids for a new video."""
        self.team_centroids_hsv = None
        self.team_centroids_bgr = None
        self.frame_counter = 0
        self.initialized = False