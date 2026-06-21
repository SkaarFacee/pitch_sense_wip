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
    GK_PENALTY_BOX_MIN_FRAC,
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

# Possession: fraction of a player's bbox size to pad the containment test
# when resolving the ball owner. Catches a ball at the player's feet whose
# center sits just outside (typically just below) the detected box.
BBOX_OWNER_PAD_FRAC = 0.15

# Possession: maximum number of CONSECUTIVE frames the ball can be missing
# before possession is released (set to None). Without this, a single
# detection followed by a long ball-loss stretch would carry one team's
# possession forward indefinitely and inflate its percentage.
MAX_BALL_LOST_CARRY_FRAMES = 30


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
        """Per-spec possession: possession only changes when the OTHER
        team's player bounding box overlaps with the ball bounding box.
        If no player bbox overlaps the ball this frame, possession is
        carried forward from the previous frame (sticky possession).
        """
        owners = GameAnalyzer.compute_ball_owner_per_frame(game_data)
        t1_frames = t2_frames = total_ball = 0
        for owner in owners:
            if owner is None:
                continue
            total_ball += 1
            if owner == TEAM0:
                t1_frames += 1
            elif owner == TEAM1:
                t2_frames += 1

        pct1 = round(t1_frames / max(total_ball, 1) * 100, 1)
        pct2 = round(t2_frames / max(total_ball, 1) * 100, 1)
        return {"team1_possession_pct": pct1, "team2_possession_pct": pct2,
                "team1_frames": t1_frames, "team2_frames": t2_frames,
                "total_ball_frames": total_ball, "team1_label": team1_label, "team2_label": team2_label}

    @staticmethod
    def _nearest_team_to_ball(entry: dict, ball_arr: np.ndarray, registry: GameRegistry) -> Optional[int]:
        """Return the canonical team (TEAM0 / TEAM1) currently in possession.

        New behaviour (per spec): possession is determined by which team's
        player bounding box OVERLAPS with the ball bounding box. The team
        of the player whose bbox contains the ball-bbox center (or, if no
        containment, the largest pixel intersection) is the owner. Per-
        frame only — callers that need carry-forward semantics should use
        ``compute_ball_owner_per_frame`` instead.

        Falls back to the legacy pitch-distance nearest-team logic when
        bbox data is missing (older runs without ``player_xyxy`` /
        ``ball_xyxy`` in the per-frame entries).
        """
        bbox_owner = GameAnalyzer._ball_owner_team_by_bbox(entry)
        if bbox_owner is not None:
            return bbox_owner

        # Legacy fallback: pitch-distance nearest
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

        valid_pos, valid_tid, t1, t2 = GameAnalyzer._split_teams(entry)
        if valid_pos is None or (len(t1) == 0 and len(t2) == 0):
            return None
        # Use the NEAREST (minimum-distance) player per team, not the mean.
        # A team spread across the pitch should not "lose" the ball just
        # because its average distance is large when one of its players is
        # right on the ball. This matches the track-aware branch above,
        # which also uses the single closest player.
        dists = np.linalg.norm(valid_pos - ball_arr, axis=1)
        min1 = float(np.min(dists[valid_tid == 0])) if len(t1) > 0 else float("inf")
        min2 = float(np.min(dists[valid_tid == 1])) if len(t2) > 0 else float("inf")
        if min1 <= min2:
            return TEAM0
        return TEAM1

    @staticmethod
    def _ball_owner_team_by_bbox(entry: dict) -> Optional[int]:
        """Per-frame ball-owner team resolved purely from BBOX OVERLAP.

        Returns TEAM0/TEAM1 if a player bbox overlaps the ball bbox,
        else None (the ball is loose / no detection this frame).

        Strategy (preference order):
          1. Player bbox that CONTAINS the ball-bbox center → strongest.
          2. Otherwise, the team whose player bbox has the largest pixel
             intersection with the ball bbox.
          3. If multiple players from the SAME team overlap, that team
             wins. Mixed-team overlaps are decided by the largest area.
        """
        ball_xyxy = entry.get("ball_xyxy")
        player_xyxy = entry.get("player_xyxy")
        team_ids = entry.get("team_ids")
        if ball_xyxy is None or len(ball_xyxy) == 0:
            return None
        if player_xyxy is None or len(player_xyxy) == 0:
            return None
        if team_ids is None or len(team_ids) == 0:
            return None
        team_ids = np.asarray(team_ids)
        n = min(len(player_xyxy), len(team_ids))
        bx1 = float(ball_xyxy[0, 0]); by1 = float(ball_xyxy[0, 1])
        bx2 = float(ball_xyxy[0, 2]); by2 = float(ball_xyxy[0, 3])
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        best_team: Optional[int] = None
        best_score = 0.0
        for i in range(n):
            t = int(team_ids[i])
            if t not in (TEAM0, TEAM1):
                continue
            px1 = float(player_xyxy[i, 0]); py1 = float(player_xyxy[i, 1])
            px2 = float(player_xyxy[i, 2]); py2 = float(player_xyxy[i, 3])
            # Containment → strongest signal, return immediately. Pad the
            # player box by a small fraction of its size so a ball at the
            # player's feet (its center sitting just outside the detected
            # box, typically just below it) still counts as possession
            # instead of falling through to the area tiebreak — which can
            # otherwise hand the ball to a barely-overlapping opponent.
            pad_x = (px2 - px1) * BBOX_OWNER_PAD_FRAC
            pad_y = (py2 - py1) * BBOX_OWNER_PAD_FRAC
            if (px1 - pad_x) <= cx <= (px2 + pad_x) and (py1 - pad_y) <= cy <= (py2 + pad_y):
                return t
            ix1 = max(bx1, px1); iy1 = max(by1, py1)
            ix2 = min(bx2, px2); iy2 = min(by2, py2)
            if ix2 > ix1 and iy2 > iy1:
                score = (ix2 - ix1) * (iy2 - iy1)
                if score > best_score:
                    best_score = score
                    best_team = t
        return best_team

    @staticmethod
    def compute_ball_owner_per_frame(game_data: List[dict]) -> List[Optional[int]]:
        """Per-frame ball-owner team (hybrid).

        Behaviour:
          * Possession STARTS like before — the first frame where the
            ball is detected, possession is seeded to the team of the
            player nearest to the ball in pitch space (track-aware
            nearest-pitch logic).
          * Once established, possession only CHANGES when the OTHER
            team's player bounding box overlaps the ball bounding box
            (bbox-overlap confirmation). If the nearest-pitch heuristic
            suggests a different team but bbox data is unavailable OR
            the bbox-overlap check is inconclusive (loose ball / no
            overlap), possession is retained by the current owner.
          * When bbox data is unavailable for the entire run (legacy
            data without ``player_xyxy``/``ball_xyxy``), the change
            gate is effectively bypassed and possession follows
            nearest-pitch exactly as before — no 0/0 frames.
          * Possession is carried forward across frames where the ball is
            not detected, but only for up to ``MAX_BALL_LOST_CARRY_FRAMES``
            consecutive frames; after that it is released (None) so a long
            ball-loss stretch doesn't inflate one team's possession.
        """
        registry = GameAnalyzer.build_registry(game_data)
        owners: List[Optional[int]] = []
        current: Optional[int] = None
        frames_since_ball = 0

        for entry in game_data:
            ball = entry.get("ball_position")
            if ball is None:
                # Carry possession forward across short ball-loss gaps, but
                # release it once the ball has been missing for too long so
                # one team's possession isn't inflated during a long stretch
                # with no ball detection.
                frames_since_ball += 1
                if frames_since_ball > MAX_BALL_LOST_CARRY_FRAMES:
                    current = None
                owners.append(current)
                continue
            frames_since_ball = 0
            ball_arr = np.asarray(ball, dtype=np.float32).reshape(1, 2)

            nearest = GameAnalyzer._nearest_team_to_ball(
                entry, ball_arr, registry,
            )
            if nearest is None:
                owners.append(current)
                continue

            has_bbox_data = (
                entry.get("ball_xyxy") is not None
                and len(entry.get("ball_xyxy")) > 0
                and entry.get("player_xyxy") is not None
                and len(entry.get("player_xyxy")) > 0
            )
            bbox_owner = GameAnalyzer._ball_owner_team_by_bbox(entry)

            if current is None:
                # Seed possession from nearest-pitch (legacy behaviour).
                current = nearest
            elif nearest != current:
                if has_bbox_data:
                    # Bbox data is available — only switch when the
                    # other team is confirmed by bbox overlap.
                    if bbox_owner is not None:
                        current = bbox_owner
                    # else: keep current (bbox inconclusive)
                else:
                    # No bbox data this frame — fall back to the
                    # nearest-pitch heuristic (legacy behaviour).
                    current = nearest

            owners.append(current)

        return owners

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
        direction_info = GameAnalyzer.infer_attacking_direction(game_data)
        if registry.has_track_ids:
            return GameAnalyzer._compute_formation_track_aware(
                game_data, registry, direction_info=direction_info,
            )
        return GameAnalyzer._compute_formation_legacy(
            game_data, direction_info=direction_info,
        )

    # Defensive depth depends on which direction each team is attacking.
    # If a team attacks "right" it defends the LEFT goal, so the deepest
    # outfield player is the one with the SMALLEST pitch-X (leftmost).
    # If a team attacks "left" it defends the RIGHT goal, so the deepest
    # outfield player is the one with the LARGEST pitch-X (rightmost).
    # When the direction can't be inferred we fall back to the historical
    # leftmost (min X) convention.
    @staticmethod
    def _depth_axis_for(team_dir: Optional[str]) -> str:
        return "min" if team_dir != "left" else "max"

    @staticmethod
    def _compute_formation_legacy(game_data: List[dict],
                                  direction_info: Optional[dict] = None) -> dict:
        t1_centers, t2_centers = [], []
        t1_spreads, t2_spreads = [], []
        t1_depth_x, t2_depth_x = [], []
        frames = 0
        t1_dir = (direction_info or {}).get("team1_attacks")
        t2_dir = (direction_info or {}).get("team2_attacks")
        t1_axis = GameAnalyzer._depth_axis_for(t1_dir)
        t2_axis = GameAnalyzer._depth_axis_for(t2_dir)
        for entry in game_data:
            _, _, t1, t2 = GameAnalyzer._split_teams(entry)
            if t1 is None and t2 is None:
                continue
            frames += 1
            if len(t1) > 0:
                c = np.mean(t1, axis=0)
                t1_centers.append(c)
                t1_spreads.append(np.mean(np.linalg.norm(t1 - c, axis=1)))
                t1_depth_x.append(float(np.min(t1[:, 0]) if t1_axis == "min"
                                         else np.max(t1[:, 0])))
            if len(t2) > 0:
                c = np.mean(t2, axis=0)
                t2_centers.append(c)
                t2_spreads.append(np.mean(np.linalg.norm(t2 - c, axis=1)))
                t2_depth_x.append(float(np.min(t2[:, 0]) if t2_axis == "min"
                                         else np.max(t2[:, 0])))
        return {"team1_centers": t1_centers, "team2_centers": t2_centers,
                "team1_spreads": t1_spreads, "team2_spreads": t2_spreads,
                "team1_avg_center": np.mean(t1_centers, axis=0).tolist() if t1_centers else None,
                "team2_avg_center": np.mean(t2_centers, axis=0).tolist() if t2_centers else None,
                "team1_avg_spread": float(np.mean(t1_spreads)) if t1_spreads else 0.0,
                "team2_avg_spread": float(np.mean(t2_spreads)) if t2_spreads else 0.0,
                "team1_defensive_depth": float(np.mean(t1_depth_x)) if t1_depth_x else 0.0,
                "team2_defensive_depth": float(np.mean(t2_depth_x)) if t2_depth_x else 0.0,
                "team1_defensive_axis": t1_axis,
                "team2_defensive_axis": t2_axis,
                "team1_attacks": t1_dir, "team2_attacks": t2_dir,
                "frames_with_players": frames}

    @staticmethod
    def _compute_formation_track_aware(game_data: List[dict], registry: GameRegistry,
                                       direction_info: Optional[dict] = None) -> dict:
        t1_centers, t2_centers = [], []
        t1_spreads, t2_spreads = [], []
        t1_depth_x, t2_depth_x = [], []
        frames = 0
        t1_dir = (direction_info or {}).get("team1_attacks")
        t2_dir = (direction_info or {}).get("team2_attacks")
        t1_axis = GameAnalyzer._depth_axis_for(t1_dir)
        t2_axis = GameAnalyzer._depth_axis_for(t2_dir)
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
                t1_depth_x.append(float(np.min(arr[:, 0]) if t1_axis == "min"
                                         else np.max(arr[:, 0])))
            if t2_pts:
                arr = np.array(t2_pts)
                c = arr.mean(axis=0)
                t2_centers.append(c)
                t2_spreads.append(float(np.mean(np.linalg.norm(arr - c, axis=1))))
                t2_depth_x.append(float(np.min(arr[:, 0]) if t2_axis == "min"
                                         else np.max(arr[:, 0])))
        return {"team1_centers": t1_centers, "team2_centers": t2_centers,
                "team1_spreads": t1_spreads, "team2_spreads": t2_spreads,
                "team1_avg_center": np.mean(t1_centers, axis=0).tolist() if t1_centers else None,
                "team2_avg_center": np.mean(t2_centers, axis=0).tolist() if t2_centers else None,
                "team1_avg_spread": float(np.mean(t1_spreads)) if t1_spreads else 0.0,
                "team2_avg_spread": float(np.mean(t2_spreads)) if t2_spreads else 0.0,
                "team1_defensive_depth": float(np.mean(t1_depth_x)) if t1_depth_x else 0.0,
                "team2_defensive_depth": float(np.mean(t2_depth_x)) if t2_depth_x else 0.0,
                "team1_defensive_axis": t1_axis,
                "team2_defensive_axis": t2_axis,
                "team1_attacks": t1_dir, "team2_attacks": t2_dir,
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
        """Count passes using the bbox-overlap ``pass_event`` recorded by
        the pipeline. Falls back to a nearest-pitch-distance proxy when no
        events are present (legacy data).

        Returns (t1, t2) pass counts.
        """
        t1, t2 = 0, 0
        for entry in game_data:
            ev = entry.get("pass_event")
            if ev is None:
                continue
            team = int(ev.get("team", -1))
            if team == TEAM0:
                t1 += 1
            elif team == TEAM1:
                t2 += 1
        if t1 + t2 > 0:
            return t1, t2

        # Legacy fallback: nearest-pitch-distance owner transitions where
        # the new owner is on the same canonical team as the previous.
        prev_owner_tid = None
        prev_owner_team = None
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

    # ------------------------------------------------------------------
    # 7. ZONE / SEGMENTATION ANALYTICS
    # ------------------------------------------------------------------
    @staticmethod
    def compute_zone_summary(analytics_data: List[dict], fps: float = 30.0) -> dict:
        """Aggregate ``analytics_data`` (per-frame segment entries) into
        time-in-zone statistics.

        Returns a dict with:
            region_time_s: {class_name: seconds}
            region_share_pct: {class_name: % of detections}
            most_detected: class_name
            most_detected_label: human label
            total_region_frames: int
            total_frames: int (from max frame_idx, not just entries present)
            half_field_left_pct: float  (0–100)
            half_field_right_pct: float (0–100)
            half_field_total: int
        """
        if not analytics_data:
            return {
                "region_time_s": {}, "region_share_pct": {},
                "most_detected": None, "most_detected_label": None,
                "total_region_frames": 0, "total_frames": 0,
                "half_field_left_pct": 0.0, "half_field_right_pct": 0.0,
                "half_field_total": 0,
            }

        from collections import Counter
        cls_counter: Counter = Counter()
        hf_left = hf_right = 0
        for entry in analytics_data:
            segs = entry.get("segments", [])
            for seg in segs:
                cn = seg.get("class_name", "")
                cls_counter[cn] += 1
                if cn == "Half Field":
                    side = seg.get("side_hint")
                    if side == "left":
                        hf_left += 1
                    elif side == "right":
                        hf_right += 1

        total = sum(cls_counter.values())
        share = {k: round((v / total) * 100, 1) for k, v in cls_counter.items()} if total else {}
        # seconds = (count / frames_with_data) * total_match_duration
        # We don't know total match frames here, so use count directly as
        # "detection count" and convert to seconds by treating each detection
        # as 1 frame.
        time_s = {k: round(v / max(float(fps), 1.0), 2) for k, v in cls_counter.items()}

        if cls_counter:
            most = cls_counter.most_common(1)[0]
            most_name, most_count = most
        else:
            most_name, most_count = None, 0

        total_frames = max(
            (int(e.get("frame_idx", 0)) for e in analytics_data), default=0
        )

        hf_total = hf_left + hf_right
        return {
            "region_time_s": time_s,
            "region_share_pct": share,
            "most_detected": most_name,
            "most_detected_label": most_name,
            "most_detected_count": int(most_count),
            "total_region_frames": int(total),
            "total_frames": int(total_frames),
            "half_field_left_pct": round((hf_left / hf_total) * 100, 1) if hf_total else 0.0,
            "half_field_right_pct": round((hf_right / hf_total) * 100, 1) if hf_total else 0.0,
            "half_field_total": int(hf_total),
        }

    @staticmethod
    def compute_zone_timeline(analytics_data: List[dict], window: int = 100) -> dict:
        """Build a rolling-window detection count timeline per region.

        Returns::

            {
              "x": [frame_idx_center, ...],
              "series": {class_name: [rolling_count, ...], ...},
              "region_names": [...],  # ordered, all regions seen
              "window": int,
            }

        ``frame_idx_center`` is the centre of each rolling window so the
        caller can render a sensible X axis without needing the original
        detection timestamps.
        """
        from collections import defaultdict
        if not analytics_data:
            return {"x": [], "series": {}, "region_names": [], "window": int(window)}

        # Build per-frame region presence (one frame may have multiple
        # regions; we count each once per frame).
        frames = sorted({int(e.get("frame_idx", 0)) for e in analytics_data})
        if not frames:
            return {"x": [], "series": {}, "region_names": [], "window": int(window)}

        fmin, fmax = frames[0], frames[-1]
        span = fmax - fmin + 1
        # Per-frame dict: frame_idx -> set of region names
        per_frame: dict[int, set] = defaultdict(set)
        region_set: set = set()
        for entry in analytics_data:
            fi = int(entry.get("frame_idx", 0))
            for seg in entry.get("segments", []):
                cn = seg.get("class_name", "")
                if cn:
                    per_frame[fi].add(cn)
                    region_set.add(cn)

        region_names = sorted(region_set)
        window = max(1, int(window))
        # Step windows every 10 frames (or window//4) to keep the chart light
        step = max(1, window // 4)
        x_vals = list(range(fmin, fmax + 1, step))
        if not x_vals or x_vals[-1] != fmax:
            x_vals.append(fmax)
        x_centers: list[int] = []
        series: dict[str, list[int]] = {r: [] for r in region_names}
        for x in x_vals:
            lo = max(fmin, x - window // 2)
            hi = min(fmax, x + window // 2)
            centre = (lo + hi) // 2
            x_centers.append(centre)
            for r in region_names:
                c = 0
                for fi in range(lo, hi + 1):
                    if r in per_frame.get(fi, ()):
                        c += 1
                series[r].append(c)
        return {
            "x": x_centers,
            "series": series,
            "region_names": region_names,
            "window": int(window),
        }

    # ------------------------------------------------------------------
    # 8. ATTACKING-DIRECTION INFERENCE
    # ------------------------------------------------------------------
    # A goalkeeper spends the majority of their tracked time inside their
    # own 18-yard box (penalty area). Use this as a spatial sanity check
    # to reject false-positive GK candidates — referees or outfielders
    # who briefly accumulated GK votes because of a colour-cluster
    # outlier — before they poison the attacking-direction inference.
    @staticmethod
    def _penalty_box_side(x: float, y: float) -> Optional[str]:
        """Return "left" / "right" if (x, y) is inside that goal's 18-yard box.

        Uses the same pitch geometry as the rest of the module
        (LEFT_PENALTY_X / RIGHT_PENALTY_X + PENALTY_Y_TOP/BOTTOM).
        """
        if not (0.0 <= y <= PITCH_WIDTH):
            return None
        if 0.0 <= x <= LEFT_PENALTY_X and PENALTY_Y_TOP <= y <= PENALTY_Y_BOTTOM:
            return "left"
        if RIGHT_PENALTY_X <= x <= PITCH_LENGTH and PENALTY_Y_TOP <= y <= PENALTY_Y_BOTTOM:
            return "right"
        return None

    @staticmethod
    def _gk_box_stats(record, min_box_frac: float = GK_PENALTY_BOX_MIN_FRAC
                      ) -> Optional[dict]:
        """Compute (side, fraction_inside) for a candidate GK track.

        ``fraction`` is the share of ALL tracked positions that fell inside
        the dominant box (not just the share inside any box). A real GK
        spends the majority of every match inside one 18-yard box — a
        referee or midfielder who only briefly visits the box has a low
        total-in-box ratio and is filtered out.

        Returns None when the track never enters any penalty box, or when
        the dominant box-side fraction is below ``min_box_frac``.
        The returned ``side`` is whichever box the track spent more time
        in (left or right).
        """
        left_n = right_n = 0
        for x, y in record.positions:
            side = GameAnalyzer._penalty_box_side(float(x), float(y))
            if side == "left":
                left_n += 1
            elif side == "right":
                right_n += 1
        total = left_n + right_n
        n_tracked = max(int(record.frames_seen), 1)
        if total <= 0:
            return None
        if left_n >= right_n:
            side, frac = "left", left_n / n_tracked
        else:
            side, frac = "right", right_n / n_tracked
        if frac < float(min_box_frac):
            return None
        return {"side": side, "fraction": float(frac),
                "in_box_frames": int(total),
                "track_frames": int(record.frames_seen)}

    @staticmethod
    def infer_attacking_direction(game_data: List[dict]) -> dict:
        """Heuristically infer each team's attacking direction.

        Strategy (preference order):
          1. Find the canonical goalkeeper for each team — must satisfy
             BOTH the team's GK-vote count AND a spatial test
             (≥ GK_PENALTY_BOX_MIN_FRAC of tracked positions inside one
             18-yard box). Refs and midfielders are filtered out. The
             team whose GK sits in the LEFT box attacks RIGHT, the team
             whose GK sits in the RIGHT box attacks LEFT.
          2. If both GKs qualify but land in the same box side, reject
             them (the "GK" candidates are likely both the same person or
             a confused ref) and fall back to mean team X.
          3. Fall back to mean X of every track on each team if no GK
             survives the filter — the team with the lower mean X
             defends the left.

        Returns a dict::

            {
              "team1_attacks": "left" | "right",
              "team2_attacks": "left" | "right",
              "source": "gk" | "mean_x",
              "confidence": float (0-1),
              "team1_gk_x": float | None,
              "team2_gk_x": float | None,
              "team1_gk_box": "left" | "right" | None,
              "team2_gk_box": "left" | "right" | None,
              "team1_gk_box_frac": float | None,
              "team2_gk_box_frac": float | None,
              "team1_mean_x": float | None,
              "team2_mean_x": float | None,
            }
        """
        empty = {
            "team1_attacks": None, "team2_attacks": None,
            "source": "none", "confidence": 0.0,
            "team1_gk_x": None, "team2_gk_x": None,
            "team1_gk_box": None, "team2_gk_box": None,
            "team1_gk_box_frac": None, "team2_gk_box_frac": None,
            "team1_mean_x": None, "team2_mean_x": None,
        }
        if not game_data:
            return empty

        registry = GameAnalyzer.build_registry(game_data)

        def _team_mean_x(team: int) -> Optional[float]:
            xs: list[float] = []
            for rec in registry.tracks.values():
                if rec.canonical_team != team:
                    continue
                for x, _y in rec.positions:
                    if -2 <= x <= PITCH_LENGTH + 2:
                        xs.append(float(x))
            if not xs:
                return None
            return float(np.mean(xs))

        def _team_gk(team: int) -> Optional[dict]:
            """Return {mean_x, votes, box_side, box_frac} for the best GK.

            A "GK" here is the team's track with the most GK votes whose
            tracked positions also spend at least GK_PENALTY_BOX_MIN_FRAC
            of their time inside ONE penalty box. Refs and outfielders
            don't pass the box test, so they're skipped.
            """
            best: Optional[dict] = None
            for rec in registry.tracks.values():
                if rec.canonical_team != team:
                    continue
                gk_votes = rec.team_votes.get(GK, 0)
                if gk_votes <= 0:
                    continue
                box = GameAnalyzer._gk_box_stats(rec)
                if box is None:
                    continue
                mean_x = float(np.mean(
                    [x for x, _y in rec.positions if -2 <= x <= PITCH_LENGTH + 2]
                )) if rec.positions else None
                if mean_x is None:
                    continue
                if best is None or gk_votes > best["votes"]:
                    best = {
                        "mean_x": mean_x,
                        "votes": int(gk_votes),
                        "box_side": box["side"],
                        "box_frac": box["fraction"],
                    }
            return best

        team1_gk = _team_gk(TEAM0)
        team2_gk = _team_gk(TEAM1)
        team1_mean = _team_mean_x(TEAM0)
        team2_mean = _team_mean_x(TEAM1)

        # Reject the GK path if both candidates ended up in the same box
        # (means at least one of them is mislabelled — likely a ref).
        gk_usable = (
            team1_gk is not None and team2_gk is not None
            and team1_gk["box_side"] != team2_gk["box_side"]
        )

        if gk_usable:
            t1 = team1_gk
            t2 = team2_gk
            t1_attacks = "right" if t1["box_side"] == "left" else "left"
            t2_attacks = "right" if t2["box_side"] == "left" else "left"
            expected_low = 0 if t1["mean_x"] < t2["mean_x"] else 1
            actual_low = 0 if t1["box_side"] == "left" else 1
            consistent = (expected_low == actual_low)
            gap = abs(t1["mean_x"] - t2["mean_x"]) / max(PITCH_LENGTH, 1.0)
            conf = float(min(1.0, gap * 4.0 + 0.5))
            if not consistent:
                conf *= 0.6
            return {
                "team1_attacks": t1_attacks, "team2_attacks": t2_attacks,
                "source": "gk", "confidence": round(conf, 2),
                "team1_gk_x": round(t1["mean_x"], 2),
                "team2_gk_x": round(t2["mean_x"], 2),
                "team1_gk_box": t1["box_side"],
                "team2_gk_box": t2["box_side"],
                "team1_gk_box_frac": round(t1["box_frac"], 2),
                "team2_gk_box_frac": round(t2["box_frac"], 2),
                "team1_gk_votes": t1["votes"], "team2_gk_votes": t2["votes"],
                "team1_mean_x": round(team1_mean, 2) if team1_mean is not None else None,
                "team2_mean_x": round(team2_mean, 2) if team2_mean is not None else None,
            }

        if team1_mean is not None and team2_mean is not None:
            t1_attacks = "right" if team1_mean < team2_mean else "left"
            t2_attacks = "left" if t1_attacks == "right" else "right"
            gap = abs(team1_mean - team2_mean) / max(PITCH_LENGTH, 1.0)
            conf = float(min(0.6, gap * 3.0 + 0.2))
            return {
                "team1_attacks": t1_attacks, "team2_attacks": t2_attacks,
                "source": "mean_x", "confidence": round(conf, 2),
                "team1_gk_x": None, "team2_gk_x": None,
                "team1_gk_box": (team1_gk or {}).get("box_side"),
                "team2_gk_box": (team2_gk or {}).get("box_side"),
                "team1_gk_box_frac": (team1_gk or {}).get("box_frac"),
                "team2_gk_box_frac": (team2_gk or {}).get("box_frac"),
                "team1_gk_votes": (team1_gk or {}).get("votes", 0) or 0,
                "team2_gk_votes": (team2_gk or {}).get("votes", 0) or 0,
                "team1_mean_x": round(team1_mean, 2),
                "team2_mean_x": round(team2_mean, 2),
            }

        return empty

    # ------------------------------------------------------------------
    # 9. DEEPER ANALYTICS — PLAYER, TACTICAL, AND PHASE-LEVEL METRICS
    # ------------------------------------------------------------------
    # The methods below intentionally use the same `registry`-aware design
    # as the rest of the file: they aggregate per-frame entries into
    # track-level summaries that are robust to single-frame mislabels.

    @staticmethod
    def compute_player_profiles(game_data: List[dict], fps: float = 30.0
                                ) -> dict:
        """Per-track summary: distance covered, frames seen, dominant
        position, time-in-thirds, top speed, etc.

        Returns::

            {
              "profiles": [{track_id, team, team_color, frames, distance_m,
                            avg_speed_m_s, top_speed_m_s, avg_x, avg_y,
                            time_in_thirds: [def, mid, att], dominant_third,
                            in_box_pct}, ...],
              "frames_with_track_ids": int,
            }
        """
        if not game_data:
            return {"profiles": [], "frames_with_track_ids": 0}
        registry = GameAnalyzer.build_registry(game_data)

        profiles: list[dict] = []
        for tid, rec in registry.tracks.items():
            if rec.canonical_team not in (TEAM0, TEAM1):
                continue
            if not rec.positions:
                continue

            xs = np.array([p[0] for p in rec.positions], dtype=np.float32)
            ys = np.array([p[1] for p in rec.positions], dtype=np.float32)
            # On-pitch filter (some projections can land slightly off-pitch
            # when homography is rough).
            mask = ((xs >= -2) & (xs <= PITCH_LENGTH + 2)
                    & (ys >= -2) & (ys <= PITCH_WIDTH + 2))
            if not np.any(mask):
                continue
            xs, ys = xs[mask], ys[mask]

            # Distance: sum of segment-to-segment deltas (cap per-segment
            # to reject homography-induced teleports).
            if len(xs) > 1:
                deltas = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
                deltas = deltas[deltas < 8.0]  # 8m teleport cap
                distance = float(deltas.sum())
            else:
                distance = 0.0

            # Speed estimates (m / s) — frames → seconds via fps.
            seconds_seen = max(float(rec.frames_seen) / max(float(fps), 1e-3), 1e-3)
            avg_speed = distance / seconds_seen
            if len(xs) > 1:
                deltas_full = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
                deltas_full = deltas_full[deltas_full < 8.0]
                top_speed = float(deltas_full.max() * fps) if len(deltas_full) else 0.0
            else:
                top_speed = 0.0

            # Time-in-thirds
            third_def = float(np.sum(xs < PITCH_LENGTH / 3))
            third_mid = float(np.sum((xs >= PITCH_LENGTH / 3) & (xs < 2 * PITCH_LENGTH / 3)))
            third_att = float(np.sum(xs >= 2 * PITCH_LENGTH / 3))
            thirds_total = max(third_def + third_mid + third_att, 1.0)
            time_in_thirds = [
                round(100.0 * third_def / thirds_total, 1),
                round(100.0 * third_mid / thirds_total, 1),
                round(100.0 * third_att / thirds_total, 1),
            ]
            dominant_idx = int(np.argmax([third_def, third_mid, third_att]))
            dominant_third = ["Defensive", "Middle", "Attacking"][dominant_idx]

            # Time inside their own 18-yard box
            in_box = 0
            for x, y in zip(xs, ys):
                if (0 <= x <= LEFT_PENALTY_X and PENALTY_Y_TOP <= y <= PENALTY_Y_BOTTOM) \
                        or (RIGHT_PENALTY_X <= x <= PITCH_LENGTH and PENALTY_Y_TOP <= y <= PENALTY_Y_BOTTOM):
                    in_box += 1
            in_box_pct = round(100.0 * in_box / max(len(xs), 1), 1)

            profiles.append({
                "track_id": int(tid),
                "team": int(rec.canonical_team),
                "frames": int(rec.frames_seen),
                "seconds_seen": round(seconds_seen, 1),
                "distance_m": round(distance, 1),
                "avg_speed_m_s": round(avg_speed, 2),
                "top_speed_m_s": round(top_speed, 2),
                "avg_x": round(float(xs.mean()), 1),
                "avg_y": round(float(ys.mean()), 1),
                "time_in_thirds_pct": time_in_thirds,
                "dominant_third": dominant_third,
                "in_box_pct": in_box_pct,
                "quality": round(float(rec.canonical_quality), 2),
            })

        profiles.sort(key=lambda p: p["distance_m"], reverse=True)
        return {
            "profiles": profiles,
            "frames_with_track_ids": sum(1 for e in game_data
                                         if e.get("track_ids") is not None
                                         and len(e["track_ids"]) > 0),
        }

    @staticmethod
    def compute_player_leaderboard(game_data: List[dict], top_n: int = 8
                                   ) -> dict:
        """Top-N players per team by distance covered (with MVPs).

        Returns::

            {team1: [profile, ...], team2: [profile, ...],
             mvp_team1: profile|None, mvp_team2: profile|None}
        """
        prof = GameAnalyzer.compute_player_profiles(game_data)
        t1 = sorted([p for p in prof["profiles"] if p["team"] == TEAM0],
                    key=lambda p: p["distance_m"], reverse=True)[:top_n]
        t2 = sorted([p for p in prof["profiles"] if p["team"] == TEAM1],
                    key=lambda p: p["distance_m"], reverse=True)[:top_n]
        return {
            "team1": t1,
            "team2": t2,
            "mvp_team1": t1[0] if t1 else None,
            "mvp_team2": t2[0] if t2 else None,
        }

    @staticmethod
    def compute_passing_network(game_data: List[dict]) -> dict:
        """Build a per-team directed passing graph.

        A pass is a possession transition where the new ball-owner is on the
        same canonical team as the previous owner (already computed in
        ``_count_passes``); we extend that to also remember which track was
        the previous owner so the UI can draw edges player→player.

        Returns::

            {
              "team1": {nodes: [(tid, x, y, ...)], edges: [(from, to, count)]},
              "team2": {...},
              "total_passes_team1": int,
              "total_passes_team2": int,
            }

        Each node carries the player's average pitch position (used for
        laying out the graph).
        """
        if not game_data:
            return {"team1": {"nodes": [], "edges": []},
                    "team2": {"nodes": [], "edges": []},
                    "total_passes_team1": 0, "total_passes_team2": 0}

        registry = GameAnalyzer.build_registry(game_data)

        # Prefer bbox-overlap pass events recorded by the pipeline (the
        # canonical source per spec). Fall back to nearest-pitch-distance
        # transitions for legacy data without ``pass_event``.
        edge_counts: dict[tuple[int, int], int] = {}
        used_bbox_events = False
        for entry in game_data:
            ev = entry.get("pass_event")
            if ev is None:
                continue
            used_bbox_events = True
            a = int(ev.get("from_tid"))
            b = int(ev.get("to_tid"))
            if a == b:
                continue
            edge_counts[(a, b)] = edge_counts.get((a, b), 0) + 1

        if not used_bbox_events:
            prev_owner_tid: Optional[int] = None
            prev_owner_team: Optional[int] = None
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
                    if rec is None or rec.canonical_team not in (TEAM0, TEAM1):
                        continue
                    d = float(np.linalg.norm(
                        np.asarray(positions[i], dtype=np.float32) - ball_arr))
                    if d < best_dist:
                        best_dist = d
                        best_tid = int(tid)
                if best_tid is None:
                    continue
                new_team = registry.canonical_team(best_tid)
                if (prev_owner_tid is not None
                        and new_team == prev_owner_team
                        and best_tid != prev_owner_tid):
                    edge_counts[(prev_owner_tid, best_tid)] = \
                        edge_counts.get((prev_owner_tid, best_tid), 0) + 1
                prev_owner_tid = best_tid
                prev_owner_team = new_team

        # Node layouts: only players who actually participated in a pass
        # (as ``from`` or ``to`` in some recorded edge), positioned at their
        # LAST on-pitch position rather than their average. This keeps the
        # graph focused on the players who moved the ball.
        def _nodes_for(team: int) -> list[dict]:
            # Track ids that touched a pass edge belonging to this team.
            participants: set[int] = set()
            for (a, b), c in edge_counts.items():
                if c <= 0:
                    continue
                ta = registry.canonical_team(a)
                tb = registry.canonical_team(b)
                if ta == team:
                    participants.add(a)
                if tb == team:
                    participants.add(b)

            nodes = []
            for tid, rec in registry.tracks.items():
                if rec.canonical_team != team or not rec.positions:
                    continue
                if tid not in participants:
                    continue
                xs = np.array([p[0] for p in rec.positions], dtype=np.float32)
                ys = np.array([p[1] for p in rec.positions], dtype=np.float32)
                mask = ((xs >= -2) & (xs <= PITCH_LENGTH + 2)
                        & (ys >= -2) & (ys <= PITCH_WIDTH + 2))
                if not np.any(mask):
                    continue
                # Last on-pitch position: walk the recorded positions
                # backwards and take the first that is on-pitch.
                last_x = last_y = None
                for px, py in reversed(rec.positions):
                    if -2 <= px <= PITCH_LENGTH + 2 and -2 <= py <= PITCH_WIDTH + 2:
                        last_x, last_y = float(px), float(py)
                        break
                if last_x is None:
                    last_x = float(xs[mask][-1])
                    last_y = float(ys[mask][-1])
                nodes.append({
                    "track_id": int(tid),
                    "x": last_x,
                    "y": last_y,
                    "frames": int(rec.frames_seen),
                })
            return nodes

        def _edges_for(team: int) -> list[dict]:
            edges = []
            for (a, b), c in edge_counts.items():
                ta = registry.canonical_team(a)
                tb = registry.canonical_team(b)
                if ta == team and tb == team:
                    edges.append({"from": a, "to": b, "count": int(c)})
            return edges

        t1_nodes = _nodes_for(TEAM0)
        t2_nodes = _nodes_for(TEAM1)
        t1_edges = _edges_for(TEAM0)
        t2_edges = _edges_for(TEAM1)
        return {
            "team1": {"nodes": t1_nodes, "edges": t1_edges},
            "team2": {"nodes": t2_nodes, "edges": t2_edges},
            "total_passes_team1": sum(e["count"] for e in t1_edges),
            "total_passes_team2": sum(e["count"] for e in t2_edges),
        }

    @staticmethod
    def compute_pressing_timeline(game_data: List[dict],
                                  window: int = 30) -> dict:
        """Mean nearest-opponent distance to the ball per frame.

        Lower values ⇒ more intense press. Two series, one per team.
        """
        if not game_data:
            return {"x": [], "team1": [], "team2": [], "window": int(window)}
        registry = GameAnalyzer.build_registry(game_data)

        buf_t1: list[Optional[float]] = []
        buf_t2: list[Optional[float]] = []
        series_t1: list[float] = []
        series_t2: list[float] = []
        x_axis: list[int] = []

        def _rolling(buf):
            valid = [v for v in buf if v is not None]
            return float(np.mean(valid)) if valid else float("nan")

        for entry in game_data:
            ball = entry.get("ball_position")
            tids = entry.get("track_ids")
            positions = entry.get("player_positions")
            fi = int(entry.get("frame_idx", len(x_axis) + 1))
            x_axis.append(fi)
            if ball is None or tids is None or positions is None or len(tids) == 0:
                buf_t1.append(None); buf_t2.append(None)
                if len(buf_t1) > window: buf_t1.pop(0)
                if len(buf_t2) > window: buf_t2.pop(0)
                series_t1.append(_rolling(buf_t1))
                series_t2.append(_rolling(buf_t2))
                continue
            ball_arr = np.asarray(ball, dtype=np.float32)
            tids = np.asarray(tids)
            positions = np.asarray(positions)
            # Identify the ball-owning team's nearest opponent and the
            # opponent team's nearest opponent distance to the ball.
            owner_team = GameAnalyzer._nearest_team_to_ball(entry, ball_arr, registry)
            t1_min = t2_min = None
            for i, tid in enumerate(tids):
                rec = registry.tracks.get(int(tid))
                if rec is None:
                    continue
                d = float(np.linalg.norm(positions[i] - ball_arr))
                if rec.canonical_team == TEAM0:
                    if t1_min is None or d < t1_min:
                        t1_min = d
                elif rec.canonical_team == TEAM1:
                    if t2_min is None or d < t2_min:
                        t2_min = d
            # Pressing intensity for a team = the OPPOSING team's nearest
            # distance to the ball (i.e. how close is the opponent pressing).
            if owner_team == TEAM0:
                buf_t1.append(None)         # team 1 has the ball — their own distance isn't "press"
                buf_t2.append(t2_min)       # team 2 pressing
            elif owner_team == TEAM1:
                buf_t1.append(t1_min)
                buf_t2.append(None)
            else:
                buf_t1.append(None); buf_t2.append(None)
            if len(buf_t1) > window: buf_t1.pop(0)
            if len(buf_t2) > window: buf_t2.pop(0)
            series_t1.append(_rolling(buf_t1))
            series_t2.append(_rolling(buf_t2))

        return {
            "x": x_axis,
            "team1": series_t1,
            "team2": series_t2,
            "window": int(window),
        }

    @staticmethod
    def compute_defensive_line_height(game_data: List[dict]) -> dict:
        """X of each team's deepest OUTFIELD player per frame.

        The "deepest" player is the outfielder nearest the goal that team
        DEFENDS — the smallest pitch-X when the team defends the left goal,
        the largest when it defends the right. The defending side is taken
        from ``infer_attacking_direction`` (robust GK-box inference with a
        mean-X fallback) so the series reflects the true back line rather
        than the squad average.

        Excludes the goalkeeper so the line reflects the back four / five.
        Uses the canonical team so a brief relabel doesn't shift the line.

        Returns::

            {x: [frame_idx,...], team1: [deepest_defender_x, ...],
             team2: [...]}
        """
        if not game_data:
            return {"x": [], "team1": [], "team2": []}
        registry = GameAnalyzer.build_registry(game_data)

        # Decide which goal each team defends (stable, whole-match estimate).
        direction = GameAnalyzer.infer_attacking_direction(game_data)
        # A team that ATTACKS right DEFENDS the left goal → deepest = min X.
        t1_attacks = direction.get("team1_attacks")
        t2_attacks = direction.get("team2_attacks")
        t1_defends_left = (t1_attacks == "right") if t1_attacks else True
        t2_defends_left = (t2_attacks == "right") if t2_attacks else (not t1_defends_left)

        # Pre-compute which tracks belong to each team and exclude their GK.
        gk_for_team: dict = {}
        for tid, rec in registry.tracks.items():
            if rec.canonical_team in (TEAM0, TEAM1) and rec.team_votes.get(GK, 0) > 0:
                # Pick the track with most GK votes per team
                cur = gk_for_team.get(("gk", rec.canonical_team))
                if cur is None or rec.team_votes[GK] > registry.tracks[cur].team_votes.get(GK, 0):
                    gk_for_team[("gk", rec.canonical_team)] = tid

        def _gk_track_id(team: int) -> Optional[int]:
            return gk_for_team.get(("gk", team))

        def _deepest(xs: list[float], defends_left: bool) -> Optional[float]:
            if not xs:
                return None
            return float(min(xs)) if defends_left else float(max(xs))

        x_axis: list[int] = []
        t1_series: list[Optional[float]] = []
        t2_series: list[Optional[float]] = []

        for entry in game_data:
            tids = entry.get("track_ids")
            positions = entry.get("player_positions")
            fi = int(entry.get("frame_idx", len(x_axis) + 1))
            x_axis.append(fi)
            if tids is None or positions is None or len(tids) == 0:
                t1_series.append(None); t2_series.append(None); continue
            tids = np.asarray(tids); positions = np.asarray(positions)
            t1_gk = _gk_track_id(TEAM0)
            t2_gk = _gk_track_id(TEAM1)
            t1_xs, t2_xs = [], []
            for i, tid in enumerate(tids):
                rec = registry.tracks.get(int(tid))
                if rec is None:
                    continue
                if rec.canonical_team == TEAM0 and int(tid) != t1_gk:
                    t1_xs.append(float(positions[i][0]))
                elif rec.canonical_team == TEAM1 and int(tid) != t2_gk:
                    t2_xs.append(float(positions[i][0]))
            t1_series.append(_deepest(t1_xs, t1_defends_left))
            t2_series.append(_deepest(t2_xs, t2_defends_left))

        return {"x": x_axis, "team1": t1_series, "team2": t2_series}

    @staticmethod
    def compute_set_pieces(game_data: List[dict], fps: float = 30.0,
                           still_window: int = 20,
                           still_radius_m: float = 1.5) -> dict:
        """Detect set-piece situations by finding ball-stationary episodes.

        A "stationary" episode is a run of >= ``still_window`` consecutive
        frames where the ball moves < ``still_radius_m``. The first frame
        of such a run is classified by its location:

          * corner area  (within 5 m of a corner flag) → "corner"
          * own 18-yard box → "goal_kick" (team that owns the box)
          * opp 18-yard box → "free_kick_dangerous"
          * elsewhere → "other"

        Returns::

            {events: [{frame_idx, type, x, y, team}],
             counts: {corner, goal_kick, free_kick_dangerous, other},
             by_team: {team1: {...}, team2: {...}}}
        """
        events: list[dict] = []
        if not game_data:
            return {"events": [], "counts": {}, "by_team": {}}

        team1_bgr = GameAnalyzer.dominant_team_bgr(game_data, team=TEAM0)
        # We classify the goal-kick ownership by which team's HALF the
        # event happened in (left half → that's Team 1's goal to defend
        # if Team 1 is on the LEFT, else Team 2's). Without an explicit
        # attacking direction here we just record the box side.

        # Build ball trajectory indexed by frame_idx.
        ball_pts: list[tuple[int, float, float]] = []
        for entry in game_data:
            bp = entry.get("ball_position")
            fi = int(entry.get("frame_idx", 0))
            if bp is None:
                ball_pts.append((fi, float("nan"), float("nan")))
                continue
            arr = np.asarray(bp, dtype=float).reshape(-1)
            if arr.shape[0] < 2:
                ball_pts.append((fi, float("nan"), float("nan")))
            else:
                ball_pts.append((fi, float(arr[0]), float(arr[1])))

        # Find stationary episodes.
        run_start = None
        run_xs, run_ys = [], []
        for i, (fi, x, y) in enumerate(ball_pts):
            if np.isnan(x) or np.isnan(y):
                if run_start is not None and len(run_xs) >= still_window:
                    _emit_set_piece(events, run_xs, run_ys, run_start)
                run_start = None
                run_xs, run_ys = [], []
                continue
            if run_start is None:
                run_start = fi
                run_xs, run_ys = [x], [y]
                continue
            # Check distance from rolling centre to add to the run.
            cx = float(np.mean(run_xs)); cy = float(np.mean(run_ys))
            if (x - cx) ** 2 + (y - cy) ** 2 < still_radius_m ** 2:
                run_xs.append(x); run_ys.append(y)
            else:
                if len(run_xs) >= still_window:
                    _emit_set_piece(events, run_xs, run_ys, run_start)
                run_start = fi
                run_xs, run_ys = [x], [y]
        if run_start is not None and len(run_xs) >= still_window:
            _emit_set_piece(events, run_xs, run_ys, run_start)

        # Aggregate
        from collections import Counter
        counts = Counter(e["type"] for e in events)
        return {
            "events": events,
            "counts": {
                "corner": int(counts.get("corner", 0)),
                "goal_kick": int(counts.get("goal_kick", 0)),
                "free_kick_dangerous": int(counts.get("free_kick_dangerous", 0)),
                "other": int(counts.get("other", 0)),
            },
            "total": len(events),
        }

    @staticmethod
    def compute_xt_heatmap(game_data: List[dict], bins: Tuple[int, int] = (18, 12)
                           ) -> dict:
        """Composite pitch-value heatmap.

        For each frame where the ball is owned by a team, we deposit a
        danger-weighted sample into the owning team's heatmap. The
        weight is a simple proxy for "how dangerous is this ball position":
            weight = base + danger

        where ``danger`` grows linearly from 0 at the team's own goal
        line to 1 at the opponent's goal line. The result is a per-team
        pitch-value surface that highlights territorial dominance.

        Returns::

            {team1_matrix, team2_matrix, x_edges, y_edges,
             team1_total_value, team2_total_value}
        """
        if not game_data:
            shape = (bins[0], bins[1])
            return {
                "team1_matrix": np.zeros(shape),
                "team2_matrix": np.zeros(shape),
                "x_edges": np.linspace(0, PITCH_LENGTH, bins[0] + 1),
                "y_edges": np.linspace(0, PITCH_WIDTH, bins[1] + 1),
                "team1_total_value": 0.0,
                "team2_total_value": 0.0,
            }
        registry = GameAnalyzer.build_registry(game_data)
        t1_pts: list[tuple[float, float, float]] = []
        t2_pts: list[tuple[float, float, float]] = []
        for entry in game_data:
            ball = entry.get("ball_position")
            if ball is None:
                continue
            arr = np.asarray(ball, dtype=float).reshape(-1)
            if arr.shape[0] < 2:
                continue
            x, y = float(arr[0]), float(arr[1])
            if not (-2 <= x <= PITCH_LENGTH + 2 and -2 <= y <= PITCH_WIDTH + 2):
                continue
            team = GameAnalyzer._nearest_team_to_ball(entry, arr.reshape(1, 2), registry)
            if team is None:
                continue
            # Danger is per-team: a ball at x=PITCH_LENGTH is dangerous for
            # the team defending the right goal, less so for the left.
            if team == TEAM0:
                danger = max(0.0, min(1.0, x / PITCH_LENGTH))
                weight = 1.0 + danger
                t1_pts.append((x, y, weight))
            elif team == TEAM1:
                danger = max(0.0, min(1.0, (PITCH_LENGTH - x) / PITCH_LENGTH))
                weight = 1.0 + danger
                t2_pts.append((x, y, weight))

        x_edges = np.linspace(0, PITCH_LENGTH, bins[0] + 1)
        y_edges = np.linspace(0, PITCH_WIDTH, bins[1] + 1)
        if t1_pts:
            arr1 = np.array(t1_pts)
            h1, _, _ = np.histogram2d(arr1[:, 0], arr1[:, 1],
                                       bins=(x_edges, y_edges),
                                       weights=arr1[:, 2])
        else:
            h1 = np.zeros((bins[0], bins[1]))
        if t2_pts:
            arr2 = np.array(t2_pts)
            h2, _, _ = np.histogram2d(arr2[:, 0], arr2[:, 1],
                                       bins=(x_edges, y_edges),
                                       weights=arr2[:, 2])
        else:
            h2 = np.zeros((bins[0], bins[1]))
        return {
            "team1_matrix": h1,
            "team2_matrix": h2,
            "x_edges": x_edges,
            "y_edges": y_edges,
            "team1_total_value": float(h1.sum()),
            "team2_total_value": float(h2.sum()),
        }

    @staticmethod
    def compute_half_comparison(game_data: List[dict]) -> dict:
        """First-half vs second-half comparison.

        Splits ``game_data`` by median frame_idx into two halves and runs
        possession + zone + value summaries on each. Returns a dict
        shaped for side-by-side KPIs.
        """
        if not game_data:
            return {"first": {}, "second": {}, "first_frames": 0, "second_frames": 0}
        frame_indices = [int(e.get("frame_idx", i))
                         for i, e in enumerate(game_data)]
        if not frame_indices:
            return {"first": {}, "second": {}, "first_frames": 0, "second_frames": 0}
        mid = (min(frame_indices) + max(frame_indices)) // 2
        first = [e for e, fi in zip(game_data, frame_indices) if fi <= mid]
        second = [e for e, fi in zip(game_data, frame_indices) if fi > mid]
        return {
            "first_frames": len(first),
            "second_frames": len(second),
            "split_frame": mid,
            "first": {
                "possession": GameAnalyzer.compute_possession(first),
                "ball_rate": GameAnalyzer.compute_match_stats(first).get(
                    "ball_detection_rate", 0.0),
            },
            "second": {
                "possession": GameAnalyzer.compute_possession(second),
                "ball_rate": GameAnalyzer.compute_match_stats(second).get(
                    "ball_detection_rate", 0.0),
            },
        }

    @staticmethod
    def compute_voronoi_control(game_data: List[dict], grid: int = 28
                                ) -> dict:
        """Per-cell pitch control: which team owns each cell on average.

        For each frame we compute a Voronoi-like assignment from every
        grid cell to its nearest player, then attribute the cell to the
        player's canonical team. Aggregate across frames.

        Returns::

            {matrix, x_edges, y_edges,
             team1_pct, team2_pct, contested_pct}

        ``matrix`` is signed in [-1, +1]: +1 = Team 1 controls the cell
        every frame, -1 = Team 2, 0 = contested.
        """
        shape = (grid, grid)
        if not game_data:
            return {
                "matrix": np.zeros(shape),
                "x_edges": np.linspace(0, PITCH_LENGTH, grid + 1),
                "y_edges": np.linspace(0, PITCH_WIDTH, grid + 1),
                "team1_pct": 0.0, "team2_pct": 0.0, "contested_pct": 0.0,
            }
        registry = GameAnalyzer.build_registry(game_data)
        x_edges = np.linspace(0, PITCH_LENGTH, grid + 1)
        y_edges = np.linspace(0, PITCH_WIDTH, grid + 1)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
        Xc, Yc = np.meshgrid(x_centers, y_centers)
        signed = np.zeros(shape, dtype=np.float32)
        n_frames = 0

        for entry in game_data:
            tids = entry.get("track_ids")
            positions = entry.get("player_positions")
            if tids is None or positions is None or len(tids) == 0:
                continue
            tids = np.asarray(tids); positions = np.asarray(positions)
            t1_best = np.full(shape, np.inf, dtype=np.float32)
            t2_best = np.full(shape, np.inf, dtype=np.float32)
            any_player = False
            for i, tid in enumerate(tids):
                rec = registry.tracks.get(int(tid))
                if rec is None:
                    continue
                px, py = float(positions[i][0]), float(positions[i][1])
                if not (-5 <= px <= PITCH_LENGTH + 5 and -5 <= py <= PITCH_WIDTH + 5):
                    continue
                d = np.sqrt((Xc - px) ** 2 + (Yc - py) ** 2)
                any_player = True
                if rec.canonical_team == TEAM0:
                    t1_best = np.minimum(t1_best, d)
                elif rec.canonical_team == TEAM1:
                    t2_best = np.minimum(t2_best, d)
            if not any_player:
                continue
            n_frames += 1
            closer_t1 = t1_best < t2_best
            closer_t2 = t2_best < t1_best
            signed += np.where(closer_t1, 1.0, 0.0)
            signed -= np.where(closer_t2, 1.0, 0.0)

        if n_frames > 0:
            signed /= float(n_frames)
        t1_cells = int(np.sum(signed > 0))
        t2_cells = int(np.sum(signed < 0))
        contested_cells = int(np.sum(signed == 0))
        total = max(t1_cells + t2_cells + contested_cells, 1)
        return {
            "matrix": signed,
            "x_edges": x_edges,
            "y_edges": y_edges,
            "team1_pct": round(100.0 * t1_cells / total, 1),
            "team2_pct": round(100.0 * t2_cells / total, 1),
            "contested_pct": round(100.0 * contested_cells / total, 1),
        }

    @staticmethod
    def compute_possession_chains(game_data: List[dict]) -> dict:
        """Longest possession chains per team + breakdown by zone.

        A "chain" is an unbroken sequence of frames where the ball-owner
        stays on the same team. Counts are returned as
        ``{chain_lengths: {length: count}, longest: int, total: int}``
        per team.
        """
        if not game_data:
            return {"team1": {}, "team2": {}}

        # Use the carry-forward bbox-overlap owners so chains are sticky:
        # possession only breaks when the OTHER team actually takes the
        # ball (i.e. the ball bbox overlaps the other team's bbox).
        owners = GameAnalyzer.compute_ball_owner_per_frame(game_data)

        def _chain_stats(team: int) -> dict:
            lengths: list[int] = []
            current = 0
            for owner in owners:
                if owner == team:
                    current += 1
                else:
                    if current > 0:
                        lengths.append(current)
                    current = 0
            if current > 0:
                lengths.append(current)
            if not lengths:
                return {"longest": 0, "total": 0, "mean": 0.0,
                        "median": 0, "histogram": {}}
            from collections import Counter
            bins = Counter()
            for ln in lengths:
                # Bucket lengths into 30-frame (~1s at 30fps) buckets.
                key = (ln // 30) * 30
                bins[key] += 1
            return {
                "longest": int(max(lengths)),
                "total": int(len(lengths)),
                "mean": round(float(np.mean(lengths)), 1),
                "median": int(np.median(lengths)),
                "histogram": {int(k): int(v) for k, v in sorted(bins.items())},
            }

        return {
            "team1": _chain_stats(TEAM0),
            "team2": _chain_stats(TEAM1),
        }


# ------------------------------------------------------------------
# Module-level helper for set-piece classification (used by compute_set_pieces)
# ------------------------------------------------------------------
def _emit_set_piece(events: list, xs: list, ys: list, frame_idx: int) -> None:
    """Append a set-piece event to ``events`` if the stationary episode
    is inside a flagged region (corner, goal area, or opp penalty box).
    """
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    # Corner regions (within 5m of either corner flag).
    corner_left = (cx <= 5.0 and cy <= 5.0)
    corner_right = (cx >= PITCH_LENGTH - 5.0 and cy <= 5.0)
    if corner_left or corner_right:
        events.append({
            "frame_idx": int(frame_idx), "type": "corner",
            "x": round(cx, 2), "y": round(cy, 2),
        })
        return
    # Goal kick: inside the own 6-yard box (extreme end).
    if cx <= LEFT_GOAL_AREA_X and GOAL_AREA_Y_TOP <= cy <= GOAL_AREA_Y_BOTTOM:
        events.append({
            "frame_idx": int(frame_idx), "type": "goal_kick",
            "x": round(cx, 2), "y": round(cy, 2),
        })
        return
    if cx >= RIGHT_GOAL_AREA_X and GOAL_AREA_Y_TOP <= cy <= GOAL_AREA_Y_BOTTOM:
        events.append({
            "frame_idx": int(frame_idx), "type": "goal_kick",
            "x": round(cx, 2), "y": round(cy, 2),
        })
        return
    # Dangerous free kick: inside opp 18-yard box.
    if (cx >= RIGHT_PENALTY_X and PENALTY_Y_TOP <= cy <= PENALTY_Y_BOTTOM):
        events.append({
            "frame_idx": int(frame_idx), "type": "free_kick_dangerous",
            "x": round(cx, 2), "y": round(cy, 2),
        })
        return
    if (cx <= LEFT_PENALTY_X and PENALTY_Y_TOP <= cy <= PENALTY_Y_BOTTOM):
        events.append({
            "frame_idx": int(frame_idx), "type": "free_kick_dangerous",
            "x": round(cx, 2), "y": round(cy, 2),
        })
        return
    events.append({
        "frame_idx": int(frame_idx), "type": "other",
        "x": round(cx, 2), "y": round(cy, 2),
    })

