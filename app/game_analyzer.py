"""GameAnalyzer — match intelligence from pipeline tracking data (possession, heatmaps, formation, territory, stats).

Tier 0: track-aware analytics. A canonical "TrackRecord" is built once per
job by scanning all per-frame entries; possession / heatmaps / territory /
stats all consult this registry so a player who is briefly misclassified
for one frame cannot corrupt the season-level numbers.

Backwards compatible: if the per-frame entries lack `track_ids` (legacy
data produced before this upgrade) the methods fall back to the original
per-frame behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from constants import (
    PITCH_LENGTH, PITCH_WIDTH, CENTER_X, CENTER_Y, CENTER_CIRCLE_RADIUS,
    PENALTY_AREA_DEPTH, PENALTY_AREA_WIDTH, GOAL_AREA_DEPTH, GOAL_AREA_WIDTH,
    PENALTY_SPOT_DISTANCE, PENALTY_ARC_RADIUS,
    LEFT_PENALTY_X, RIGHT_PENALTY_X, LEFT_GOAL_AREA_X, RIGHT_GOAL_AREA_X,
    PENALTY_Y_TOP, PENALTY_Y_BOTTOM, GOAL_AREA_Y_TOP, GOAL_AREA_Y_BOTTOM,
    LEFT_PENALTY_SPOT_X, RIGHT_PENALTY_SPOT_X,
)

ZONE_X_EDGES = [0.0, 35.0, 70.0, PITCH_LENGTH]
ZONE_Y_EDGES = [0.0, PITCH_WIDTH / 3.0, 2.0 * PITCH_WIDTH / 3.0, PITCH_WIDTH]
ZONE_NAMES = [
    ["Def Left",  "Mid Left",  "Att Left"],
    ["Def Cent",  "Mid Cent",  "Att Cent"],
    ["Def Right", "Mid Right", "Att Right"],
]

# Class labels (kept in sync with team_analyzer.TeamColorAnalyzer)
TEAM0 = 0
TEAM1 = 1
GK = -1
REF = -2

# Smoothing factor for the registry's own EMA of per-track quality
TRACK_QUALITY_REGISTRY_EMA = 0.3

# Default ball-possession passing proxy: possession transition only counts
# as a "pass" if both tracks are on the SAME canonical team — otherwise
# it's an opposition turnover.
SAME_TEAM_PASS_MAX_DIST = 30.0  # meters; bail on giant jumps


@dataclass
class TrackRecord:
    """Canonical per-track summary built once from a full game_data list."""
    track_id: int
    team_votes: Dict[int, int] = field(default_factory=lambda: {TEAM0: 0, TEAM1: 0, GK: 0, REF: 0})
    positions: List[Tuple[float, float]] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)
    team_ids_seen: List[int] = field(default_factory=list)
    qualities: List[float] = field(default_factory=list)
    frames_seen: int = 0
    canonical_team: int = GK  # resolved after build
    canonical_quality: float = 0.0  # registry-aggregated quality in [0, 1]


@dataclass
class GameRegistry:
    """Pass-1 canonical identity layer. All downstream analytics consult it."""
    tracks: Dict[int, TrackRecord]
    has_track_ids: bool
    has_track_quality: bool

    def canonical_team(self, track_id: int) -> int:
        rec = self.tracks.get(track_id)
        return rec.canonical_team if rec is not None else GK

    def canonical_quality(self, track_id: int) -> float:
        rec = self.tracks.get(track_id)
        return rec.canonical_quality if rec is not None else 0.0

    def team_tracks(self, team: int) -> List[TrackRecord]:
        return [t for t in self.tracks.values() if t.canonical_team == team]


class GameAnalyzer:
    """All methods are static — pass the list of per-frame data dicts."""

    # ------------------------------------------------------------------
    # Tier 0: canonical track registry
    # ------------------------------------------------------------------
    @staticmethod
    def build_registry(game_data: List[dict]) -> GameRegistry:
        """Scan every per-frame entry and assemble one TrackRecord per ByteTrack id.

        Falls back gracefully:
        * If `track_ids` is absent from every entry → empty registry, downstream
          methods use legacy per-frame behaviour.
        * `track_quality` is optional — defaults to 1.0 per detection if absent.
        """
        tracks: Dict[int, TrackRecord] = {}
        has_track_ids = False
        has_track_quality = False

        for entry in game_data:
            tids = entry.get("track_ids")
            if tids is None:
                continue
            tids = np.asarray(tids)
            if tids.size == 0:
                continue
            has_track_ids = True

            team_ids_raw = entry.get("team_ids")
            if team_ids_raw is None:
                team_ids_raw = np.full(len(tids), GK, dtype=np.int32)
            else:
                team_ids_raw = np.asarray(team_ids_raw)
                if len(team_ids_raw) != len(tids):
                    team_ids_raw = np.full(len(tids), GK, dtype=np.int32)

            positions = entry.get("player_positions")
            if positions is None or len(positions) != len(tids):
                pos_iter = [(0.0, 0.0)] * len(tids)
            else:
                pos_iter = [tuple(map(float, p)) for p in np.asarray(positions)]

            qualities = entry.get("track_quality")
            if qualities is None:
                qual_iter = [1.0] * len(tids)
                q_present = False
            else:
                qualities = np.asarray(qualities, dtype=np.float32)
                if len(qualities) != len(tids):
                    qual_iter = [1.0] * len(tids)
                    q_present = False
                else:
                    qual_iter = [float(q) for q in qualities]
                    q_present = True
                    if not has_track_quality and any(0.0 <= q <= 1.0 for q in qual_iter):
                        has_track_quality = True

            frame_idx = int(entry.get("frame_idx", 0))
            for tid, team_label, pos, q in zip(tids, team_ids_raw, pos_iter, qual_iter):
                rec = tracks.get(int(tid))
                if rec is None:
                    rec = TrackRecord(track_id=int(tid))
                    tracks[int(tid)] = rec
                rec.frames_seen += 1
                rec.frame_indices.append(frame_idx)
                rec.positions.append(pos)
                rec.team_ids_seen.append(int(team_label))
                rec.qualities.append(q)
                rec.team_votes[int(team_label)] = rec.team_votes.get(int(team_label), 0) + 1

        # Resolve canonical team + quality for each record
        for rec in tracks.values():
            rec.canonical_team = GameAnalyzer._resolve_majority_team(rec.team_votes, rec.frames_seen)
            rec.canonical_quality = GameAnalyzer._aggregate_quality(rec.qualities)

        return GameRegistry(tracks=tracks, has_track_ids=has_track_ids, has_track_quality=has_track_quality)

    @staticmethod
    def _resolve_majority_team(votes: Dict[int, int], frames_seen: int) -> int:
        """Pick the team that a track was on for the majority of its lifetime.

        Falls back to GK if no team ever won a vote (degenerate / empty).
        """
        if frames_seen <= 0 or not votes:
            return GK
        # Filter to positive votes
        positive = {k: v for k, v in votes.items() if v > 0}
        if not positive:
            return GK
        return int(max(positive.items(), key=lambda kv: kv[1])[0])

    @staticmethod
    def _aggregate_quality(qualities: List[float]) -> float:
        if not qualities:
            return 0.0
        # EMA: penalises long streaks of low quality more than a few bad frames
        ema = qualities[0]
        for q in qualities[1:]:
            ema = (1.0 - TRACK_QUALITY_REGISTRY_EMA) * ema + TRACK_QUALITY_REGISTRY_EMA * q
        return float(max(0.0, min(1.0, ema)))

    # ------------------------------------------------------------------
    # Shared per-frame data helper
    # ------------------------------------------------------------------
    @staticmethod
    def _split_teams(entry):
        positions = entry.get("player_positions")
        team_ids = entry.get("team_ids")
        if positions is None or team_ids is None:
            return None, None, None, None
        team_ids = np.asarray(team_ids)
        positions = np.asarray(positions)
        valid = team_ids >= 0
        if not np.any(valid):
            return None, None, None, None
        valid_pos = positions[valid]
        valid_tid = team_ids[valid]
        t1 = valid_pos[valid_tid == 0]
        t2 = valid_pos[valid_tid == 1]
        return valid_pos, valid_tid, t1, t2

    # ------------------------------------------------------------------
    # 1. POSSESSION (track-aware nearest-to-ball)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_possession(game_data: List[dict], team1_label="Team 1", team2_label="Team 2") -> dict:
        registry = GameAnalyzer.build_registry(game_data)
        t1_frames = t2_frames = total_ball = 0

        for entry in game_data:
            ball = entry.get("ball_position")
            if ball is None:
                continue
            ball_arr = np.asarray(ball, dtype=np.float32).reshape(1, 2)

            winning_team = GameAnalyzer._nearest_team_to_ball(entry, ball_arr, registry)
            if winning_team is None:
                continue

            total_ball += 1
            if winning_team == TEAM0:
                t1_frames += 1
            elif winning_team == TEAM1:
                t2_frames += 1

        pct1 = round(t1_frames / max(total_ball, 1) * 100, 1)
        pct2 = round(t2_frames / max(total_ball, 1) * 100, 1)
        return {"team1_possession_pct": pct1, "team2_possession_pct": pct2,
                "team1_frames": t1_frames, "team2_frames": t2_frames,
                "total_ball_frames": total_ball, "team1_label": team1_label, "team2_label": team2_label}

    @staticmethod
    def _nearest_team_to_ball(entry: dict, ball_arr: np.ndarray, registry: GameRegistry) -> Optional[int]:
        """Return the canonical team (TEAM0 / TEAM1) of the track nearest to the ball.

        Track-aware path uses track_ids + registry; legacy path keeps the
        original "mean-distance-per-team" semantics so existing callers do
        not silently change.
        """
        tids = entry.get("track_ids")
        positions = entry.get("player_positions")
        team_ids_raw = entry.get("team_ids")

        if (registry.has_track_ids and tids is not None and positions is not None
                and len(tids) == len(positions) and len(tids) > 0):
            best_tid = None
            best_dist = float("inf")
            for i, tid in enumerate(tids):
                rec = registry.tracks.get(int(tid))
                if rec is None:
                    continue
                if rec.canonical_team not in (TEAM0, TEAM1):
                    continue
                d = float(np.linalg.norm(np.asarray(positions[i], dtype=np.float32) - ball_arr[0]))
                if d < best_dist:
                    best_dist = d
                    best_tid = int(tid)
            if best_tid is not None:
                return registry.canonical_team(best_tid)

        # Legacy fallback: per-frame mean distance per team
        valid_pos, valid_tid, t1, t2 = GameAnalyzer._split_teams(entry)
        if valid_pos is None or (len(t1) == 0 and len(t2) == 0):
            return None
        dists = np.linalg.norm(valid_pos - ball_arr, axis=1)
        avg1 = float(np.mean(dists[valid_tid == 0])) if len(t1) > 0 else float("inf")
        avg2 = float(np.mean(dists[valid_tid == 1])) if len(t2) > 0 else float("inf")
        if avg1 <= avg2:
            return TEAM0
        return TEAM1

    # ------------------------------------------------------------------
    # 2. HEATMAPS (track-aware, quality-weighted)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_heatmaps(game_data: List[dict], bins: Tuple[int, int] = (21, 14)) -> dict:
        registry = GameAnalyzer.build_registry(game_data)
        t1_samples: List[Tuple[float, float, float]] = []
        t2_samples: List[Tuple[float, float, float]] = []

        if registry.has_track_ids:
            # One sample per track per frame, weighted by canonical_quality
            for entry in game_data:
                tids = entry.get("track_ids")
                positions = entry.get("player_positions")
                if tids is None or positions is None:
                    continue
                tids = np.asarray(tids)
                positions = np.asarray(positions)
                if len(tids) != len(positions) or len(tids) == 0:
                    continue
                frame_q = entry.get("track_quality")
                q_iter = (np.asarray(frame_q, dtype=np.float32)
                          if frame_q is not None and len(frame_q) == len(tids)
                          else None)
                for i, tid in enumerate(tids):
                    rec = registry.tracks.get(int(tid))
                    if rec is None:
                        continue
                    if rec.canonical_team not in (TEAM0, TEAM1):
                        continue
                    q = float(q_iter[i]) if q_iter is not None else rec.canonical_quality
                    if q <= 0:
                        continue
                    pt = positions[i]
                    if rec.canonical_team == TEAM0:
                        t1_samples.append((float(pt[0]), float(pt[1]), q))
                    else:
                        t2_samples.append((float(pt[0]), float(pt[1]), q))
        else:
            # Legacy per-frame aggregation, equal weight
            for entry in game_data:
                _, _, t1, t2 = GameAnalyzer._split_teams(entry)
                if t1 is not None and len(t1) > 0:
                    for pt in t1:
                        t1_samples.append((float(pt[0]), float(pt[1]), 1.0))
                if t2 is not None and len(t2) > 0:
                    for pt in t2:
                        t2_samples.append((float(pt[0]), float(pt[1]), 1.0))

        def _build(samples):
            if not samples:
                return np.empty((0, 2)), np.empty((0,)), np.zeros((bins[0], bins[1]))
            pts = np.array([(s[0], s[1]) for s in samples], dtype=np.float32)
            weights = np.array([s[2] for s in samples], dtype=np.float32)
            mask = ((pts[:, 0] >= -5) & (pts[:, 0] <= PITCH_LENGTH + 5)
                    & (pts[:, 1] >= -5) & (pts[:, 1] <= PITCH_WIDTH + 5))
            pts, weights = pts[mask], weights[mask]
            x_edges = np.linspace(0, PITCH_LENGTH, bins[0] + 1)
            y_edges = np.linspace(0, PITCH_WIDTH, bins[1] + 1)
            h, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=(x_edges, y_edges), weights=weights)
            return pts, weights, h

        t1_pts, _, h1 = _build(t1_samples)
        t2_pts, _, h2 = _build(t2_samples)
        x_edges = np.linspace(0, PITCH_LENGTH, bins[0] + 1)
        y_edges = np.linspace(0, PITCH_WIDTH, bins[1] + 1)
        return {"team1_heatmap": h1, "team2_heatmap": h2, "x_edges": x_edges, "y_edges": y_edges,
                "team1_count": len(t1_pts), "team2_count": len(t2_pts)}

    @staticmethod
    def draw_pitch_heatmap(heatmap: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray,
                           title: str, team_color: Tuple[int, int, int], cmap: str = "Reds") -> plt.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
        GameAnalyzer._draw_pitch_outline(ax)
        h = heatmap.copy()
        if h.max() > 0:
            h /= h.max()
        ax.imshow(h.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                  origin="lower", cmap=cmap, alpha=0.65, aspect="auto")
        ax.set_xlim(-2, PITCH_LENGTH + 2)
        ax.set_ylim(-2, PITCH_WIDTH + 2)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Pitch Length (m)")
        ax.set_ylabel("Pitch Width (m)")
        ax.set_aspect("equal")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 3. FORMATION
    # ------------------------------------------------------------------
    @staticmethod
    def compute_formation(game_data: List[dict]) -> dict:
        registry = GameAnalyzer.build_registry(game_data)
        if registry.has_track_ids:
            return GameAnalyzer._compute_formation_track_aware(game_data, registry)
        return GameAnalyzer._compute_formation_legacy(game_data)

    @staticmethod
    def _compute_formation_legacy(game_data: List[dict]) -> dict:
        t1_centers, t2_centers = [], []
        t1_spreads, t2_spreads = [], []
        t1_min_x, t2_min_x = [], []
        frames = 0
        for entry in game_data:
            _, _, t1, t2 = GameAnalyzer._split_teams(entry)
            if t1 is None and t2 is None:
                continue
            frames += 1
            if len(t1) > 0:
                c = np.mean(t1, axis=0)
                t1_centers.append(c)
                t1_spreads.append(np.mean(np.linalg.norm(t1 - c, axis=1)))
                t1_min_x.append(np.min(t1[:, 0]))
            if len(t2) > 0:
                c = np.mean(t2, axis=0)
                t2_centers.append(c)
                t2_spreads.append(np.mean(np.linalg.norm(t2 - c, axis=1)))
                t2_min_x.append(np.min(t2[:, 0]))
        return {"team1_centers": t1_centers, "team2_centers": t2_centers,
                "team1_spreads": t1_spreads, "team2_spreads": t2_spreads,
                "team1_avg_center": np.mean(t1_centers, axis=0).tolist() if t1_centers else None,
                "team2_avg_center": np.mean(t2_centers, axis=0).tolist() if t2_centers else None,
                "team1_avg_spread": float(np.mean(t1_spreads)) if t1_spreads else 0.0,
                "team2_avg_spread": float(np.mean(t2_spreads)) if t2_spreads else 0.0,
                "team1_defensive_depth": float(np.mean(t1_min_x)) if t1_min_x else 0.0,
                "team2_defensive_depth": float(np.mean(t2_min_x)) if t2_min_x else 0.0,
                "frames_with_players": frames}

    @staticmethod
    def _compute_formation_track_aware(game_data: List[dict], registry: GameRegistry) -> dict:
        t1_centers, t2_centers = [], []
        t1_spreads, t2_spreads = [], []
        t1_min_x, t2_min_x = [], []
        frames = 0
        for entry in game_data:
            tids = entry.get("track_ids")
            positions = entry.get("player_positions")
            if tids is None or positions is None or len(tids) == 0:
                continue
            tids = np.asarray(tids)
            positions = np.asarray(positions)
            t1_pts, t2_pts = [], []
            for i, tid in enumerate(tids):
                rec = registry.tracks.get(int(tid))
                if rec is None:
                    continue
                if rec.canonical_team == TEAM0:
                    t1_pts.append(positions[i])
                elif rec.canonical_team == TEAM1:
                    t2_pts.append(positions[i])
            if not t1_pts and not t2_pts:
                continue
            frames += 1
            if t1_pts:
                arr = np.array(t1_pts)
                c = arr.mean(axis=0)
                t1_centers.append(c)
                t1_spreads.append(float(np.mean(np.linalg.norm(arr - c, axis=1))))
                t1_min_x.append(float(np.min(arr[:, 0])))
            if t2_pts:
                arr = np.array(t2_pts)
                c = arr.mean(axis=0)
                t2_centers.append(c)
                t2_spreads.append(float(np.mean(np.linalg.norm(arr - c, axis=1))))
                t2_min_x.append(float(np.min(arr[:, 0])))
        return {"team1_centers": t1_centers, "team2_centers": t2_centers,
                "team1_spreads": t1_spreads, "team2_spreads": t2_spreads,
                "team1_avg_center": np.mean(t1_centers, axis=0).tolist() if t1_centers else None,
                "team2_avg_center": np.mean(t2_centers, axis=0).tolist() if t2_centers else None,
                "team1_avg_spread": float(np.mean(t1_spreads)) if t1_spreads else 0.0,
                "team2_avg_spread": float(np.mean(t2_spreads)) if t2_spreads else 0.0,
                "team1_defensive_depth": float(np.mean(t1_min_x)) if t1_min_x else 0.0,
                "team2_defensive_depth": float(np.mean(t2_min_x)) if t2_min_x else 0.0,
                "frames_with_players": frames}

    @staticmethod
    def draw_formation_scatter(game_data: List[dict], team1_color=(0.2, 0.4, 0.9),
                               team2_color=(0.9, 0.2, 0.2), team1_label="Team 1",
                               team2_label="Team 2", max_frames: int = 100) -> plt.Figure:
        registry = GameAnalyzer.build_registry(game_data)
        fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
        GameAnalyzer._draw_pitch_outline(ax)
        t1_pts, t2_pts = [], []
        step = max(1, len(game_data) // max_frames)
        for entry in game_data[::step]:
            if registry.has_track_ids:
                tids = entry.get("track_ids")
                positions = entry.get("player_positions")
                if tids is None or positions is None:
                    continue
                for i, tid in enumerate(np.asarray(tids)):
                    rec = registry.tracks.get(int(tid))
                    if rec is None:
                        continue
                    if rec.canonical_team == TEAM0:
                        t1_pts.append(positions[i])
                    elif rec.canonical_team == TEAM1:
                        t2_pts.append(positions[i])
            else:
                _, _, t1, t2 = GameAnalyzer._split_teams(entry)
                if t1 is not None and len(t1) > 0:
                    t1_pts.extend(t1)
                if t2 is not None and len(t2) > 0:
                    t2_pts.extend(t2)
        for pts, c, lbl in [(np.array(t1_pts) if t1_pts else np.empty((0, 2)), team1_color, team1_label),
                            (np.array(t2_pts) if t2_pts else np.empty((0, 2)), team2_color, team2_label)]:
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], c=[c], alpha=0.5, s=15, label=lbl, edgecolors="none")
        ax.set_xlim(-2, PITCH_LENGTH + 2)
        ax.set_ylim(-2, PITCH_WIDTH + 2)
        ax.set_title("Player Positioning Scatter", fontsize=14, fontweight="bold")
        ax.set_xlabel("Pitch Length (m)")
        ax.set_ylabel("Pitch Width (m)")
        ax.legend(loc="upper right", fontsize=10)
        ax.set_aspect("equal")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 4. TERRITORY (track-aware: vote per track per frame, divide by track-count)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_territory(game_data: List[dict]) -> dict:
        registry = GameAnalyzer.build_registry(game_data)
        counts = [[{"t1": 0, "t2": 0} for _ in range(3)] for _ in range(3)]

        if registry.has_track_ids:
            for entry in game_data:
                tids = entry.get("track_ids")
                positions = entry.get("player_positions")
                if tids is None or positions is None:
                    continue
                tids = np.asarray(tids)
                positions = np.asarray(positions)
                if len(tids) != len(positions) or len(tids) == 0:
                    continue
                for i, tid in enumerate(tids):
                    rec = registry.tracks.get(int(tid))
                    if rec is None:
                        continue
                    if rec.canonical_team not in (TEAM0, TEAM1):
                        continue
                    x, y = positions[i]
                    col = min(np.searchsorted(ZONE_X_EDGES[1:], x, side="right"), 2)
                    row = min(np.searchsorted(ZONE_Y_EDGES[1:], y, side="right"), 2)
                    if rec.canonical_team == TEAM0:
                        counts[row][col]["t1"] += 1
                    else:
                        counts[row][col]["t2"] += 1
        else:
            for entry in game_data:
                _, valid_tid, t1, t2 = GameAnalyzer._split_teams(entry)
                if valid_tid is None:
                    continue
                positions = np.asarray(entry["player_positions"])[np.asarray(entry["team_ids"]) >= 0]
                for i in range(len(positions)):
                    x, y = positions[i]
                    col = min(np.searchsorted(ZONE_X_EDGES[1:], x, side="right"), 2)
                    row = min(np.searchsorted(ZONE_Y_EDGES[1:], y, side="right"), 2)
                    if valid_tid[i] == 0:
                        counts[row][col]["t1"] += 1
                    elif valid_tid[i] == 1:
                        counts[row][col]["t2"] += 1

        zone_grid, t1_total, t2_total = [], 0, 0
        for row in range(3):
            zone_row = []
            for col in range(3):
                t1, t2 = counts[row][col]["t1"], counts[row][col]["t2"]
                total = t1 + t2
                t1_total += t1
                t2_total += t2
                dominant = -1 if total == 0 else (0 if t1 >= t2 else 1)
                t1_pct = round(t1 / max(total, 1) * 100, 1) if total > 0 else 0.0
                t2_pct = round(t2 / max(total, 1) * 100, 1) if total > 0 else 0.0
                zone_row.append({"zone_name": ZONE_NAMES[row][col], "team1_pct": t1_pct, "team2_pct": t2_pct,
                                 "team1_frames": t1, "team2_frames": t2, "total_frames": total, "dominant_team": dominant})
            zone_grid.append(zone_row)
        return {"zone_grid": zone_grid, "team1_total_presence": t1_total, "team2_total_presence": t2_total}

    # ------------------------------------------------------------------
    # 5. MATCH STATS (track-aware + per-track distance/time)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_match_stats(game_data: List[dict]) -> dict:
        total = len(game_data)
        ball_frames = sum(1 for e in game_data if e.get("ball_position") is not None)

        registry = GameAnalyzer.build_registry(game_data)

        if registry.has_track_ids:
            return GameAnalyzer._compute_match_stats_track_aware(
                game_data, registry, total=total, ball_frames=ball_frames
            )
        return GameAnalyzer._compute_match_stats_legacy(
            game_data, total=total, ball_frames=ball_frames
        )

    @staticmethod
    def _compute_match_stats_legacy(game_data: List[dict], total: int, ball_frames: int) -> dict:
        t1_counts, t2_counts, spreads = [], [], []
        ball_path = []
        for entry in game_data:
            _, valid_tid, _, _ = GameAnalyzer._split_teams(entry)
            if valid_tid is not None:
                t1_counts.append(int(np.sum(valid_tid == 0)))
                t2_counts.append(int(np.sum(valid_tid == 1)))
                positions = np.asarray(entry["player_positions"])[valid_tid]
                if len(positions) > 1:
                    c = np.mean(positions, axis=0)
                    spreads.append(float(np.mean(np.linalg.norm(positions - c, axis=1))))
            else:
                t1_counts.append(0)
                t2_counts.append(0)
            bp = entry.get("ball_position")
            if bp is not None:
                ball_path.append(np.asarray(bp))
        ball_prog = float(np.sum(np.linalg.norm(np.diff(ball_path, axis=0), axis=1))) if len(ball_path) > 1 else 0.0
        return {"total_frames": total, "ball_detection_frames": ball_frames,
                "ball_detection_rate": round(ball_frames / max(total, 1) * 100, 1),
                "avg_players_total": round(np.mean([a + b for a, b in zip(t1_counts, t2_counts)]), 1),
                "avg_players_team1": round(np.mean(t1_counts), 1),
                "avg_players_team2": round(np.mean(t2_counts), 1),
                "avg_player_spread": round(np.mean(spreads), 2) if spreads else 0.0,
                "ball_progression_m": round(ball_prog, 1)}

    @staticmethod
    def _compute_match_stats_track_aware(game_data: List[dict], registry: GameRegistry,
                                         total: int, ball_frames: int) -> dict:
        t1_counts, t2_counts, spreads = [], [], []
        ball_path = []

        # Distance / time-on-pitch per track (meters walked assuming pitch units are meters)
        per_track_distance: Dict[int, float] = {}
        per_track_frames: Dict[int, int] = {}

        for entry in game_data:
            tids = entry.get("track_ids")
            positions = entry.get("player_positions")
            if tids is None or positions is None or len(tids) == 0:
                t1_counts.append(0)
                t2_counts.append(0)
            else:
                tids = np.asarray(tids)
                positions = np.asarray(positions)
                t1_pts, t2_pts = [], []
                for i, tid in enumerate(tids):
                    rec = registry.tracks.get(int(tid))
                    if rec is None:
                        continue
                    if rec.canonical_team == TEAM0:
                        t1_pts.append(positions[i])
                    elif rec.canonical_team == TEAM1:
                        t2_pts.append(positions[i])
                t1_counts.append(len(t1_pts))
                t2_counts.append(len(t2_pts))
                if len(t1_pts) + len(t2_pts) > 1:
                    pts = np.array(t1_pts + t2_pts)
                    c = pts.mean(axis=0)
                    spreads.append(float(np.mean(np.linalg.norm(pts - c, axis=1))))

                # Per-track incremental distance: use the track's last-known pos
                for i, tid in enumerate(tids):
                    tid_i = int(tid)
                    rec = registry.tracks.get(tid_i)
                    if rec is None:
                        continue
                    prev = getattr(rec, "_last_pos", None)
                    pt = np.asarray(positions[i], dtype=np.float32)
                    if prev is not None:
                        d = float(np.linalg.norm(pt - prev))
                        if d < SAME_TEAM_PASS_MAX_DIST:  # reject teleports
                            per_track_distance[tid_i] = per_track_distance.get(tid_i, 0.0) + d
                    rec._last_pos = pt
                    per_track_frames[tid_i] = per_track_frames.get(tid_i, 0) + 1

            bp = entry.get("ball_position")
            if bp is not None:
                ball_path.append(np.asarray(bp))

        ball_prog = float(np.sum(np.linalg.norm(np.diff(ball_path, axis=0), axis=1))) if len(ball_path) > 1 else 0.0

        # Clean up scratch attribute so it doesn't leak into pickle / repr
        for rec in registry.tracks.values():
            rec.__dict__.pop("_last_pos", None)

        top_by_distance = sorted(per_track_distance.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top_by_distance = [{"track_id": int(t), "distance_m": round(d, 1)} for t, d in top_by_distance]

        # Team-level passing proxy: count ball-possession transitions
        # between tracks on opposing canonical teams.
        passes_attempted_t1, passes_attempted_t2 = GameAnalyzer._count_passes(game_data, registry)

        return {"total_frames": total, "ball_detection_frames": ball_frames,
                "ball_detection_rate": round(ball_frames / max(total, 1) * 100, 1),
                "avg_players_total": round(np.mean([a + b for a, b in zip(t1_counts, t2_counts)]), 1),
                "avg_players_team1": round(np.mean(t1_counts), 1),
                "avg_players_team2": round(np.mean(t2_counts), 1),
                "avg_player_spread": round(np.mean(spreads), 2) if spreads else 0.0,
                "ball_progression_m": round(ball_prog, 1),
                "top_5_by_distance": top_by_distance,
                "passes_attempted_team1": passes_attempted_t1,
                "passes_attempted_team2": passes_attempted_t2,
                "tracked_player_count": len(registry.tracks),
                "track_aware": True}

    @staticmethod
    def _count_passes(game_data: List[dict], registry: GameRegistry) -> Tuple[int, int]:
        """Proxy for passes: possession transitions where the new owner is on
        the same canonical team as the previous owner. Returns (t1, t2).
        """
        prev_owner_tid = None
        prev_owner_team = None
        t1, t2 = 0, 0
        for entry in game_data:
            ball = entry.get("ball_position")
            if ball is None:
                continue
            tids = entry.get("track_ids")
            positions = entry.get("player_positions")
            if tids is None or positions is None or len(tids) == 0:
                continue
            ball_arr = np.asarray(ball, dtype=np.float32)
            best_tid = None
            best_dist = float("inf")
            for i, tid in enumerate(tids):
                rec = registry.tracks.get(int(tid))
                if rec is None:
                    continue
                if rec.canonical_team not in (TEAM0, TEAM1):
                    continue
                d = float(np.linalg.norm(np.asarray(positions[i], dtype=np.float32) - ball_arr))
                if d < best_dist:
                    best_dist = d
                    best_tid = int(tid)
            if best_tid is None:
                continue
            new_team = registry.canonical_team(best_tid)
            if prev_owner_tid is not None and new_team == prev_owner_team and best_tid != prev_owner_tid:
                if new_team == TEAM0:
                    t1 += 1
                else:
                    t2 += 1
            prev_owner_tid = best_tid
            prev_owner_team = new_team
        return t1, t2

    # ------------------------------------------------------------------
    # Pitch Drawing Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _draw_pitch_outline(ax: plt.Axes) -> None:
        ax.plot([0, PITCH_LENGTH, PITCH_LENGTH, 0, 0], [0, 0, PITCH_WIDTH, PITCH_WIDTH, 0], color="black", linewidth=1.5)
        ax.plot([CENTER_X, CENTER_X], [0, PITCH_WIDTH], color="black", linewidth=1.0)
        circ = plt.Circle((CENTER_X, CENTER_Y), CENTER_CIRCLE_RADIUS, fill=False, color="black", linewidth=1.0)
        ax.add_patch(circ)
        ax.plot(CENTER_X, CENTER_Y, "ko", markersize=3)
        for pts in [
            ([0, LEFT_PENALTY_X, LEFT_PENALTY_X, 0], [PENALTY_Y_TOP, PENALTY_Y_TOP, PENALTY_Y_BOTTOM, PENALTY_Y_BOTTOM]),
            ([PITCH_LENGTH, RIGHT_PENALTY_X, RIGHT_PENALTY_X, PITCH_LENGTH], [PENALTY_Y_TOP, PENALTY_Y_TOP, PENALTY_Y_BOTTOM, PENALTY_Y_BOTTOM]),
            ([0, LEFT_GOAL_AREA_X, LEFT_GOAL_AREA_X, 0], [GOAL_AREA_Y_TOP, GOAL_AREA_Y_TOP, GOAL_AREA_Y_BOTTOM, GOAL_AREA_Y_BOTTOM]),
            ([PITCH_LENGTH, RIGHT_GOAL_AREA_X, RIGHT_GOAL_AREA_X, PITCH_LENGTH], [GOAL_AREA_Y_TOP, GOAL_AREA_Y_TOP, GOAL_AREA_Y_BOTTOM, GOAL_AREA_Y_BOTTOM]),
        ]:
            ax.plot(pts[0], pts[1], color="black", linewidth=1.0)
        ax.plot(LEFT_PENALTY_SPOT_X, CENTER_Y, "ko", markersize=3)
        ax.plot(RIGHT_PENALTY_SPOT_X, CENTER_Y, "ko", markersize=3)
        theta = np.arccos((LEFT_PENALTY_X - LEFT_PENALTY_SPOT_X) / PENALTY_ARC_RADIUS)
        for cx, a1, a2 in [(LEFT_PENALTY_SPOT_X, -theta, theta), (RIGHT_PENALTY_SPOT_X, np.pi - theta, np.pi + theta)]:
            ang = np.linspace(a1, a2, 20)
            ax.plot(cx + PENALTY_ARC_RADIUS * np.cos(ang), CENTER_Y + PENALTY_ARC_RADIUS * np.sin(ang), color="black", linewidth=1.0)
        ax.set_facecolor("#e8f5e9")

    # ------------------------------------------------------------------
    # Team colour extraction (for chart palette override)
    # ------------------------------------------------------------------
    @staticmethod
    def dominant_team_bgr(game_data: List[dict], team: int) -> Optional[Tuple[int, int, int]]:
        """Return the most common BGR colour for a team across all frames.

        Looks at every per-frame `team_info` entry (stored as
        `team1_bgr` / `team2_bgr` in `game_data`). Falls back to None when
        no frames have the colour recorded (legacy data or warmup only).
        """
        key = "team1_bgr" if team == 0 else "team2_bgr"
        counter: Counter = Counter()
        for entry in game_data:
            v = entry.get(key)
            if v is None:
                continue
            try:
                b, g, r = int(v[0]), int(v[1]), int(v[2])
            except Exception:
                continue
            counter[(b, g, r)] += 1
        if not counter:
            return None
        b, g, r = counter.most_common(1)[0][0]
        return (b, g, r)

    @staticmethod
    def bgr_to_hex(bgr: Tuple[int, int, int]) -> str:
        """Convert a BGR tuple (e.g. from OpenCV) to a #rrggbb hex string."""
        b, g, r = bgr
        return f"#{int(r) & 0xff:02x}{int(g) & 0xff:02x}{int(b) & 0xff:02x}"
