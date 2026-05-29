"""TeamColorAnalyzer — K-means on HSV jersey colors to assign players to teams (sticky centroids)."""
import numpy as np
import cv2
from sklearn.cluster import KMeans
from constants import (
    TEAM_N_CLUSTERS, TEAM_JERSEY_Y_START, TEAM_JERSEY_Y_END, TEAM_JERSEY_X_START, TEAM_JERSEY_X_END,
    GREEN_HSV_LOWER, GREEN_HSV_UPPER, GK_COLOR_DIST_THRESHOLD, REF_SATURATION_THRESHOLD,
)


class TeamColorAnalyzer:
    DEFAULT_TEAM_COLORS = [(255, 0, 0), (0, 0, 255)]
    GK_COLOR = (0, 255, 255)
    REF_COLOR = (0, 0, 0)

    def __init__(self, n_clusters: int = TEAM_N_CLUSTERS):
        self.n_clusters = n_clusters
        self.team_centroids_hsv = None  # (n_clusters, 2) — hue+sat
        self.team_centroids_bgr = None  # (n_clusters, 3) — for viz
        self.initialized = False

    def assign_team_colors(self, frame: np.ndarray, player_xyxy: np.ndarray,
                           player_conf: np.ndarray, track_ids: np.ndarray = None) -> dict:
        if len(player_xyxy) == 0:
            return self._empty_result()

        # Extract dominant HSV color per player (chest region, masked)
        player_hsv, player_bgr = [], []
        for bbox in player_xyxy:
            hsv_c, bgr_c = self._extract_dominant_color(frame, bbox)
            player_hsv.append(hsv_c)
            player_bgr.append(bgr_c)
        player_hsv = np.array(player_hsv, dtype=np.float32)
        player_bgr = np.array(player_bgr, dtype=np.float32)

        # Cluster once on first frame, then use nearest-centroid thereafter
        if not self.initialized:
            team_ids, centroids_hsv, centroids_bgr = self._cluster_teams(player_hsv, player_bgr)
            if centroids_bgr is not None:
                self.team_centroids_hsv = centroids_hsv
                self.team_centroids_bgr = centroids_bgr
                self.initialized = True
        else:
            team_ids = self._assign_to_nearest_team(player_hsv)

        if not self.initialized or team_ids is None:
            return self._empty_result()

        centroids = self.team_centroids_bgr
        team_colors = []
        for tid in team_ids:
            if tid == -2:
                team_colors.append(self.REF_COLOR)
            elif tid == -1:
                team_colors.append(self.GK_COLOR)
            elif tid == 0:
                team_colors.append(tuple(map(int, centroids[0])) if centroids is not None else self.DEFAULT_TEAM_COLORS[0])
            else:
                team_colors.append(tuple(map(int, centroids[1])) if centroids is not None and len(centroids) > 1 else self.DEFAULT_TEAM_COLORS[1])

        t1 = tuple(map(int, centroids[0])) if centroids is not None else self.DEFAULT_TEAM_COLORS[0]
        t2 = tuple(map(int, centroids[1])) if centroids is not None and len(centroids) > 1 else self.DEFAULT_TEAM_COLORS[1]
        return {'team_ids': np.array(team_ids, dtype=np.int32), 'team_colors': team_colors, 'team1_bgr': t1, 'team2_bgr': t2}

    # Color extraction
    def _extract_dominant_color(self, frame: np.ndarray, bbox: np.ndarray):
        x1, y1, x2, y2 = max(0, int(bbox[0])), max(0, int(bbox[1])), min(frame.shape[1] - 1, int(bbox[2])), min(frame.shape[0] - 1, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return np.array([0, 0], dtype=np.float32), np.array([128, 128, 128], dtype=np.float32)

        h, w = y2 - y1, x2 - x1
        crop = frame[y1 + int(h * TEAM_JERSEY_Y_START):y1 + int(h * TEAM_JERSEY_Y_END),
                     x1 + int(w * TEAM_JERSEY_X_START):x1 + int(w * TEAM_JERSEY_X_END)]
        if crop.size == 0:
            return np.array([0, 0], dtype=np.float32), np.array([128, 128, 128], dtype=np.float32)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array(GREEN_HSV_LOWER, dtype=np.uint8), np.array(GREEN_HSV_UPPER, dtype=np.uint8))
        dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 30]))
        bright = cv2.inRange(hsv, np.array([0, 0, 230]), np.array([180, 40, 255]))
        valid = hsv[cv2.bitwise_and(cv2.bitwise_not(green), cv2.bitwise_and(cv2.bitwise_not(dark), cv2.bitwise_not(bright))) > 0]
        if len(valid) < 15:
            valid = hsv.reshape(-1, 3)

        median_h, median_s, median_v = float(np.median(valid[:, 0])), float(np.median(valid[:, 1])), float(np.median(valid[:, 2]))
        vis = cv2.cvtColor(np.uint8([[[median_h, median_s, max(median_v, 80)]]]), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
        return np.array([median_h, median_s], dtype=np.float32), vis


    # Team clustering via K-means on HSV

    def _hsv_distance(self, hsv1, hsv2):
        dh = min(abs(hsv1[0] - hsv2[0]), 180 - abs(hsv1[0] - hsv2[0]))
        return np.sqrt(dh * dh + (hsv1[1] - hsv2[1]) ** 2)

    def _cluster_teams(self, player_hsv, player_bgr):
        n = len(player_hsv)
        if n < 2:
            return None, None, None

        # Pre-filter low-saturation (referee) players so they don't pull centroids
        low_sat = (player_hsv[:, 1] < REF_SATURATION_THRESHOLD)
        colored_idx = np.where(~low_sat)[0]
        if len(colored_idx) < 2:
            colored_idx = np.arange(n)
            low_sat = np.zeros(n, dtype=bool)

        try:
            k = min(self.n_clusters, len(colored_idx))
            km = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(player_hsv[colored_idx])
            labels = km.labels_
            centroids = km.cluster_centers_
        except Exception:
            return None, None, None

        # Sort by the centroids for consistent team ordering
        order = np.argsort(centroids[:, 0])
        remap = {old: new for new, old in enumerate(order)}
        colored_tids = np.array([remap[l] for l in labels], dtype=np.int32)
        sorted_c = centroids[order]
        sorted_bgr = np.array([cv2.cvtColor(np.uint8([[[int(h), int(max(s, 80)), 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
                               for h, s in sorted_c], dtype=np.float32)

        # Build full team_ids: all default to referee, then fill colored players
        team_ids = np.full(n, -2, dtype=np.int32)
        for pos, idx in enumerate(colored_idx):
            team_ids[idx] = colored_tids[pos]

        # GK detection: outlier from both team centroids
        for i in colored_idx:
            dists = [self._hsv_distance(player_hsv[i], sorted_c[t]) for t in range(k)]
            if min(dists) > np.mean(dists) + GK_COLOR_DIST_THRESHOLD * (np.std(dists) + 1e-6):
                team_ids[i] = -1

        return team_ids, sorted_c, sorted_bgr

    def _assign_to_nearest_team(self, player_hsv):
        if self.team_centroids_hsv is None:
            return np.full(len(player_hsv), -1, dtype=np.int32)
        centroids = self.team_centroids_hsv
        team_ids = np.zeros(len(player_hsv), dtype=np.int32)
        for i in range(len(player_hsv)):
            dists = [self._hsv_distance(player_hsv[i], centroids[t]) for t in range(len(centroids))]
            md, mi = min(dists), int(np.argmin(dists))
            team_ids[i] = -1 if md > np.mean(dists) + GK_COLOR_DIST_THRESHOLD * (np.std(dists) + 1e-6) else mi
            if player_hsv[i, 1] < REF_SATURATION_THRESHOLD and team_ids[i] >= 0:
                team_ids[i] = -2  # low saturation → referee
        return team_ids

    def _empty_result(self) -> dict:
        return {'team_ids': np.empty((0,), dtype=np.int32), 'team_colors': [],
                'team1_bgr': self.DEFAULT_TEAM_COLORS[0], 'team2_bgr': self.DEFAULT_TEAM_COLORS[1]}

    def reset(self):
        self.team_centroids_hsv = self.team_centroids_bgr = None
        self.initialized = False