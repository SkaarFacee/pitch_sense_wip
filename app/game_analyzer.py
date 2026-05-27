"""
GameAnalyzer — Derives match intelligence from pipeline tracking data.

Uses per-frame player positions, team assignments, and ball position to
compute:
  - Ball possession percentages
  - Pitch heatmaps (player density per team)
  - Formation & positioning (center of mass, spread, defensive depth)
  - Territory control (9-zone grid analysis)
  - Match stats summary

All positions are in pitch coordinates (meters) with the pitch dimensions:
  Length: 105m (x-axis: 0 → 105)
  Width:  68m  (y-axis: 0 → 68)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Pitch constants (same as constants.py)
# ---------------------------------------------------------------------------
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

# Zone definitions for territory control (9 zones)
# Columns: Defensive third (0-35m), Middle third (35-70m), Attacking third (70-105m)
ZONE_X_EDGES = [0.0, 35.0, 70.0, PITCH_LENGTH]
ZONE_Y_EDGES = [0.0, PITCH_WIDTH / 3.0, 2.0 * PITCH_WIDTH / 3.0, PITCH_WIDTH]
ZONE_NAMES = [
    ["Def Left",  "Mid Left",  "Att Left"],
    ["Def Cent",  "Mid Cent",  "Att Cent"],
    ["Def Right", "Mid Right", "Att Right"],
]


class GameAnalyzer:
    """
    Analyzes game data collected from the KeypointPipeline.
    All methods are static — pass the list of per-frame data dicts.
    """

    # ------------------------------------------------------------------
    # 1. POSSESSION ANALYSIS
    # ------------------------------------------------------------------
    @staticmethod
    def compute_possession(
        game_data: List[dict],
        team1_label: str = "Team 1",
        team2_label: str = "Team 2",
    ) -> dict:
        """
        Determine possession by ball proximity to each team's players.

        For each frame where ball is detected:
          - Exclude goalkeepers (team_id == -1)
          - Compute average distance from ball to Team 1 players
          - Compute average distance from ball to Team 2 players
          - Team with smaller average distance gets possession credit

        Args:
            game_data: List of per-frame dicts with 'player_positions',
                       'team_ids', 'ball_position'
            team1_label: Display name for Team 1
            team2_label: Display name for Team 2

        Returns:
            dict: {
                'team1_possession_pct': float,
                'team2_possession_pct': float,
                'team1_frames': int,
                'team2_frames': int,
                'total_ball_frames': int,
                'team1_label': str,
                'team2_label': str,
            }
        """
        team1_frames = 0
        team2_frames = 0
        total_ball_frames = 0

        for entry in game_data:
            ball = entry.get("ball_position")
            if ball is None:
                continue

            positions = entry.get("player_positions")
            team_ids = entry.get("team_ids")

            if positions is None or team_ids is None or len(positions) == 0:
                continue

            # Ensure team_ids is a numpy array
            team_ids = np.asarray(team_ids)
            positions = np.asarray(positions)

            # Filter out goalkeepers (team_id == -1) and referees (team_id == -2)
            valid_mask = (team_ids >= 0)
            if not np.any(valid_mask):
                continue

            valid_positions = positions[valid_mask]
            valid_team_ids = team_ids[valid_mask]

            # Split by team
            team1_mask = valid_team_ids == 0
            team2_mask = valid_team_ids == 1

            if not np.any(team1_mask) and not np.any(team2_mask):
                continue

            # Compute average distance from ball to each team
            ball_arr = np.asarray(ball, dtype=np.float32).reshape(1, 2)
            dists = np.linalg.norm(valid_positions - ball_arr, axis=1)

            avg_dist_team1 = np.mean(dists[team1_mask]) if np.any(team1_mask) else float("inf")
            avg_dist_team2 = np.mean(dists[team2_mask]) if np.any(team2_mask) else float("inf")

            total_ball_frames += 1
            if avg_dist_team1 <= avg_dist_team2:
                team1_frames += 1
            else:
                team2_frames += 1

        if total_ball_frames == 0:
            return {
                "team1_possession_pct": 0.0,
                "team2_possession_pct": 0.0,
                "team1_frames": 0,
                "team2_frames": 0,
                "total_ball_frames": 0,
                "team1_label": team1_label,
                "team2_label": team2_label,
            }

        team1_pct = round(team1_frames / total_ball_frames * 100, 1)
        team2_pct = round(team2_frames / total_ball_frames * 100, 1)

        return {
            "team1_possession_pct": team1_pct,
            "team2_possession_pct": team2_pct,
            "team1_frames": team1_frames,
            "team2_frames": team2_frames,
            "total_ball_frames": total_ball_frames,
            "team1_label": team1_label,
            "team2_label": team2_label,
        }

    # ------------------------------------------------------------------
    # 2. PITCH HEATMAPS
    # ------------------------------------------------------------------
    @staticmethod
    def compute_heatmaps(
        game_data: List[dict],
        bins: Tuple[int, int] = (21, 14),
    ) -> dict:
        """
        Build 2D player-density histograms per team across all frames.

        Args:
            game_data: List of per-frame dicts
            bins: (x_bins, y_bins) for the histogram grid

        Returns:
            dict: {
                'team1_heatmap': (H, W) np.ndarray,
                'team2_heatmap': (H, W) np.ndarray,
                'x_edges': np.ndarray,
                'y_edges': np.ndarray,
                'team1_count': int (total player samples),
                'team2_count': int (total player samples),
            }
        """
        team1_positions = []
        team2_positions = []

        for entry in game_data:
            positions = entry.get("player_positions")
            team_ids = entry.get("team_ids")
            if positions is None or team_ids is None:
                continue

            team_ids = np.asarray(team_ids)
            positions = np.asarray(positions)

            # Filter valid players (not GK, not referee)
            valid_mask = (team_ids >= 0)

            t1_mask = valid_mask & (team_ids == 0)
            t2_mask = valid_mask & (team_ids == 1)

            if np.any(t1_mask):
                team1_positions.append(positions[t1_mask])
            if np.any(t2_mask):
                team2_positions.append(positions[t2_mask])

        t1_all = np.vstack(team1_positions) if team1_positions else np.empty((0, 2))
        t2_all = np.vstack(team2_positions) if team2_positions else np.empty((0, 2))

        # Remove points clearly outside pitch bounds
        if len(t1_all) > 0:
            t1_all = t1_all[(t1_all[:, 0] >= -5) & (t1_all[:, 0] <= PITCH_LENGTH + 5) &
                            (t1_all[:, 1] >= -5) & (t1_all[:, 1] <= PITCH_WIDTH + 5)]
        if len(t2_all) > 0:
            t2_all = t2_all[(t2_all[:, 0] >= -5) & (t2_all[:, 0] <= PITCH_LENGTH + 5) &
                            (t2_all[:, 1] >= -5) & (t2_all[:, 1] <= PITCH_WIDTH + 5)]

        x_edges = np.linspace(0, PITCH_LENGTH, bins[0] + 1)
        y_edges = np.linspace(0, PITCH_WIDTH, bins[1] + 1)

        if len(t1_all) > 0:
            h1, _, _ = np.histogram2d(t1_all[:, 0], t1_all[:, 1], bins=(x_edges, y_edges))
        else:
            h1 = np.zeros((bins[0], bins[1]))

        if len(t2_all) > 0:
            h2, _, _ = np.histogram2d(t2_all[:, 0], t2_all[:, 1], bins=(x_edges, y_edges))
        else:
            h2 = np.zeros((bins[0], bins[1]))

        return {
            "team1_heatmap": h1,
            "team2_heatmap": h2,
            "x_edges": x_edges,
            "y_edges": y_edges,
            "team1_count": len(t1_all),
            "team2_count": len(t2_all),
        }

    @staticmethod
    def draw_pitch_heatmap(
        heatmap: np.ndarray,
        x_edges: np.ndarray,
        y_edges: np.ndarray,
        title: str,
        team_color: Tuple[int, int, int],
        cmap: str = "Reds",
    ) -> plt.Figure:
        """
        Draw a pitch soccer field with a heatmap overlay.

        Args:
            heatmap: (H, W) density array
            x_edges: Bin edges for x (pitch length)
            y_edges: Bin edges for y (pitch width)
            title: Plot title
            team_color: BGR tuple for accent
            cmap: Matplotlib colormap name

        Returns:
            matplotlib Figure ready for st.pyplot()
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

        # Draw pitch outline
        GameAnalyzer._draw_pitch_outline(ax)

        # Normalize heatmap to 0-1 for display
        h = heatmap.copy()
        if h.max() > 0:
            h = h / h.max()

        # Transpose for imshow (imshow expects rows=y, cols=x)
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
        ax.imshow(
            h.T,
            extent=extent,
            origin="lower",
            cmap=cmap,
            alpha=0.65,
            aspect="auto",
        )

        ax.set_xlim(-2, PITCH_LENGTH + 2)
        ax.set_ylim(-2, PITCH_WIDTH + 2)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Pitch Length (m)")
        ax.set_ylabel("Pitch Width (m)")
        ax.set_aspect("equal")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 3. FORMATION & POSITIONING
    # ------------------------------------------------------------------
    @staticmethod
    def compute_formation(game_data: List[dict]) -> dict:
        """
        Compute per-frame and aggregate formation metrics.

        Returns:
            dict: {
                'team1_centers': list of (mean_x, mean_y) per frame,
                'team2_centers': list of (mean_x, mean_y) per frame,
                'team1_spreads': list of float (mean dist from center) per frame,
                'team2_spreads': list of float,
                'team1_avg_center': (mean_x, mean_y),
                'team2_avg_center': (mean_x, mean_y),
                'team1_avg_spread': float,
                'team2_avg_spread': float,
                'team1_defensive_depth': float (avg min x),
                'team2_defensive_depth': float,
                'frames_with_players': int,
            }
        """
        t1_centers = []
        t2_centers = []
        t1_spreads = []
        t2_spreads = []
        t1_min_x = []  # defensive depth
        t2_min_x = []
        frames_with_players = 0

        for entry in game_data:
            positions = entry.get("player_positions")
            team_ids = entry.get("team_ids")
            if positions is None or team_ids is None or len(positions) == 0:
                continue

            team_ids = np.asarray(team_ids)
            positions = np.asarray(positions)
            valid_mask = team_ids >= 0
            if not np.any(valid_mask):
                continue

            valid_pos = positions[valid_mask]
            valid_tid = team_ids[valid_mask]

            t1_mask = valid_tid == 0
            t2_mask = valid_tid == 1

            if np.any(t1_mask):
                t1_pts = valid_pos[t1_mask]
                center = np.mean(t1_pts, axis=0)
                t1_centers.append(center)
                spread = np.mean(np.linalg.norm(t1_pts - center, axis=1))
                t1_spreads.append(spread)
                t1_min_x.append(np.min(t1_pts[:, 0]))

            if np.any(t2_mask):
                t2_pts = valid_pos[t2_mask]
                center = np.mean(t2_pts, axis=0)
                t2_centers.append(center)
                spread = np.mean(np.linalg.norm(t2_pts - center, axis=1))
                t2_spreads.append(spread)
                t2_min_x.append(np.min(t2_pts[:, 0]))

            if np.any(t1_mask) or np.any(t2_mask):
                frames_with_players += 1

        return {
            "team1_centers": t1_centers,
            "team2_centers": t2_centers,
            "team1_spreads": t1_spreads,
            "team2_spreads": t2_spreads,
            "team1_avg_center": np.mean(t1_centers, axis=0).tolist() if t1_centers else None,
            "team2_avg_center": np.mean(t2_centers, axis=0).tolist() if t2_centers else None,
            "team1_avg_spread": float(np.mean(t1_spreads)) if t1_spreads else 0.0,
            "team2_avg_spread": float(np.mean(t2_spreads)) if t2_spreads else 0.0,
            "team1_defensive_depth": float(np.mean(t1_min_x)) if t1_min_x else 0.0,
            "team2_defensive_depth": float(np.mean(t2_min_x)) if t2_min_x else 0.0,
            "frames_with_players": frames_with_players,
        }

    @staticmethod
    def draw_formation_scatter(
        game_data: List[dict],
        team1_color: Tuple[float, float, float] = (0.2, 0.4, 0.9),
        team2_color: Tuple[float, float, float] = (0.9, 0.2, 0.2),
        team1_label: str = "Team 1",
        team2_label: str = "Team 2",
        max_frames: int = 100,
    ) -> plt.Figure:
        """
        Scatter all player positions on a pitch, colored by team.
        Samples up to max_frames to avoid overplotting.

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
        GameAnalyzer._draw_pitch_outline(ax)

        # Collect sampled positions
        t1_pts = []
        t2_pts = []
        step = max(1, len(game_data) // max_frames)
        for entry in game_data[::step]:
            positions = entry.get("player_positions")
            team_ids = entry.get("team_ids")
            if positions is None or team_ids is None:
                continue
            team_ids = np.asarray(team_ids)
            positions = np.asarray(positions)
            valid = team_ids >= 0
            if not np.any(valid):
                continue
            t1_pts.extend(positions[valid & (team_ids == 0)])
            t2_pts.extend(positions[valid & (team_ids == 1)])

        t1_arr = np.array(t1_pts) if t1_pts else np.empty((0, 2))
        t2_arr = np.array(t2_pts) if t2_pts else np.empty((0, 2))

        if len(t1_arr) > 0:
            ax.scatter(t1_arr[:, 0], t1_arr[:, 1], c=[team1_color],
                       alpha=0.5, s=15, label=team1_label, edgecolors="none")
        if len(t2_arr) > 0:
            ax.scatter(t2_arr[:, 0], t2_arr[:, 1], c=[team2_color],
                       alpha=0.5, s=15, label=team2_label, edgecolors="none")

        ax.set_xlim(-2, PITCH_LENGTH + 2)
        ax.set_ylim(-2, PITCH_WIDTH + 2)
        ax.set_title("Player Positioning Scatter", fontsize=14, fontweight="bold")
        ax.set_xlabel("Pitch Length (m)")
        ax.set_ylabel("Pitch Width (m)")
        ax.legend(loc="upper right", fontsize=10)
        ax.set_aspect("equal")
        fig.tight_layout()
        return fig

    @staticmethod
    def draw_formation_center_chart(
        formation_data: dict,
        team1_label: str = "Team 1",
        team2_label: str = "Team 2",
    ) -> plt.Figure:
        """
        Line chart showing team center-of-mass x-position across frames.
        Higher x = more attacking, lower x = more defensive.

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(9, 4))

        t1_centers = formation_data.get("team1_centers", [])
        t2_centers = formation_data.get("team2_centers", [])

        if t1_centers:
            t1_x = [c[0] for c in t1_centers]
            ax.plot(t1_x, label=team1_label, alpha=0.8, linewidth=0.8)
        if t2_centers:
            t2_x = [c[0] for c in t2_centers]
            ax.plot(t2_x, label=team2_label, alpha=0.8, linewidth=0.8)

        # Draw pitch thirds reference lines
        ax.axhline(35.0, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.axhline(70.0, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.fill_between([0, len(t1_centers or t2_centers or [0])], 0, 35, alpha=0.05, color="blue")
        ax.fill_between([0, len(t1_centers or t2_centers or [0])], 70, 105, alpha=0.05, color="red")

        ax.set_ylim(0, PITCH_LENGTH)
        ax.set_ylabel("Attack → X Position on Pitch (m)")
        ax.set_xlabel("Frame")
        ax.set_title("Team Positioning (Center of Mass X) — Higher = More Attacking", fontsize=12)
        ax.legend(fontsize=10)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 4. TERRITORY CONTROL
    # ------------------------------------------------------------------
    @staticmethod
    def compute_territory(game_data: List[dict]) -> dict:
        """
        Divide pitch into 9 zones and count player presence per team.

        Returns:
            dict: {
                'zone_grid': 3×3 list of dicts:
                    { 'zone_name': str, 'team1_pct': float, 'team2_pct': float,
                      'team1_frames': int, 'team2_frames': int, 'total_frames': int,
                      'dominant_team': 0 | 1 | -1 (neutral) }
                'team1_total_presence': int (total player-zone occurrences)
                'team2_total_presence': int
            }
        """
        # Initialize zone counters: zone_grid[row][col]
        zone_counts = [[{"team1": 0, "team2": 0} for _ in range(3)] for _ in range(3)]

        for entry in game_data:
            positions = entry.get("player_positions")
            team_ids = entry.get("team_ids")
            if positions is None or team_ids is None:
                continue

            team_ids = np.asarray(team_ids)
            positions = np.asarray(positions)
            valid = team_ids >= 0
            if not np.any(valid):
                continue

            valid_pos = positions[valid]
            valid_tid = team_ids[valid]

            for i in range(len(valid_pos)):
                x, y = valid_pos[i]
                tid = valid_tid[i]

                # Determine zone
                col = np.searchsorted(ZONE_X_EDGES[1:], x, side="right")
                row = np.searchsorted(ZONE_Y_EDGES[1:], y, side="right")
                col = min(col, 2)
                row = min(row, 2)

                if tid == 0:
                    zone_counts[row][col]["team1"] += 1
                elif tid == 1:
                    zone_counts[row][col]["team2"] += 1

        # Build output
        zone_grid = []
        team1_total = 0
        team2_total = 0
        for row in range(3):
            zone_row = []
            for col in range(3):
                t1 = zone_counts[row][col]["team1"]
                t2 = zone_counts[row][col]["team2"]
                total = t1 + t2
                team1_total += t1
                team2_total += t2

                if total == 0:
                    dominant = -1
                    t1_pct = 0.0
                    t2_pct = 0.0
                else:
                    t1_pct = round(t1 / total * 100, 1)
                    t2_pct = round(t2 / total * 100, 1)
                    dominant = 0 if t1 >= t2 else 1

                zone_row.append({
                    "zone_name": ZONE_NAMES[row][col],
                    "team1_pct": t1_pct,
                    "team2_pct": t2_pct,
                    "team1_frames": t1,
                    "team2_frames": t2,
                    "total_frames": total,
                    "dominant_team": dominant,
                })
            zone_grid.append(zone_row)

        return {
            "zone_grid": zone_grid,
            "team1_total_presence": team1_total,
            "team2_total_presence": team2_total,
        }

    # ------------------------------------------------------------------
    # 5. MATCH STATS SUMMARY
    # ------------------------------------------------------------------
    @staticmethod
    def compute_match_stats(game_data: List[dict]) -> dict:
        """
        Compute aggregate match statistics.

        Returns:
            dict with keys:
                total_frames, ball_detection_rate, avg_players_total,
                avg_players_team1, avg_players_team2, avg_player_spread,
                ball_progression_m (approximate)
        """
        total_frames = len(game_data)
        ball_frames = sum(1 for e in game_data if e.get("ball_position") is not None)
        ball_detection_rate = (ball_frames / max(total_frames, 1)) * 100

        # Player counts per frame
        t1_counts = []
        t2_counts = []
        all_spreads = []

        for entry in game_data:
            positions = entry.get("player_positions")
            team_ids = entry.get("team_ids")
            if positions is None or team_ids is None or len(positions) == 0:
                t1_counts.append(0)
                t2_counts.append(0)
                continue

            team_ids = np.asarray(team_ids)
            positions = np.asarray(positions)
            valid = team_ids >= 0
            t1_counts.append(int(np.sum(valid & (team_ids == 0))))
            t2_counts.append(int(np.sum(valid & (team_ids == 1))))

            # Spread
            valid_pos = positions[valid]
            if len(valid_pos) > 1:
                center = np.mean(valid_pos, axis=0)
                spread = float(np.mean(np.linalg.norm(valid_pos - center, axis=1)))
                all_spreads.append(spread)

        # Ball progression: approximate total distance ball moved
        ball_path = []
        for entry in game_data:
            bp = entry.get("ball_position")
            if bp is not None:
                ball_path.append(np.asarray(bp))
        if len(ball_path) > 1:
            ball_progression = float(np.sum(np.linalg.norm(np.diff(ball_path, axis=0), axis=1)))
        else:
            ball_progression = 0.0

        return {
            "total_frames": total_frames,
            "ball_detection_frames": ball_frames,
            "ball_detection_rate": round(ball_detection_rate, 1),
            "avg_players_total": round(np.mean([a + b for a, b in zip(t1_counts, t2_counts)]), 1),
            "avg_players_team1": round(np.mean(t1_counts), 1),
            "avg_players_team2": round(np.mean(t2_counts), 1),
            "avg_player_spread": round(np.mean(all_spreads), 2) if all_spreads else 0.0,
            "ball_progression_m": round(ball_progression, 1),
        }

    # ------------------------------------------------------------------
    # Pitch Drawing Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _draw_pitch_outline(ax: plt.Axes) -> None:
        """Draw a soccer pitch outline on the given matplotlib Axes."""
        # Pitch boundary
        ax.plot([0, PITCH_LENGTH, PITCH_LENGTH, 0, 0],
                [0, 0, PITCH_WIDTH, PITCH_WIDTH, 0],
                color="black", linewidth=1.5)

        # Halfway line
        ax.plot([CENTER_X, CENTER_X], [0, PITCH_WIDTH], color="black", linewidth=1.0)

        # Center circle
        center_circle = plt.Circle((CENTER_X, CENTER_Y), CENTER_CIRCLE_RADIUS,
                                    fill=False, color="black", linewidth=1.0)
        ax.add_patch(center_circle)
        ax.plot(CENTER_X, CENTER_Y, "ko", markersize=3)

        # Left penalty area
        ax.plot([0, LEFT_PENALTY_X, LEFT_PENALTY_X, 0],
                [PENALTY_Y_TOP, PENALTY_Y_TOP, PENALTY_Y_BOTTOM, PENALTY_Y_BOTTOM],
                color="black", linewidth=1.0)

        # Right penalty area
        ax.plot([PITCH_LENGTH, RIGHT_PENALTY_X, RIGHT_PENALTY_X, PITCH_LENGTH],
                [PENALTY_Y_TOP, PENALTY_Y_TOP, PENALTY_Y_BOTTOM, PENALTY_Y_BOTTOM],
                color="black", linewidth=1.0)

        # Left goal area
        ax.plot([0, LEFT_GOAL_AREA_X, LEFT_GOAL_AREA_X, 0],
                [GOAL_AREA_Y_TOP, GOAL_AREA_Y_TOP, GOAL_AREA_Y_BOTTOM, GOAL_AREA_Y_BOTTOM],
                color="black", linewidth=1.0)

        # Right goal area
        ax.plot([PITCH_LENGTH, RIGHT_GOAL_AREA_X, RIGHT_GOAL_AREA_X, PITCH_LENGTH],
                [GOAL_AREA_Y_TOP, GOAL_AREA_Y_TOP, GOAL_AREA_Y_BOTTOM, GOAL_AREA_Y_BOTTOM],
                color="black", linewidth=1.0)

        # Penalty spots
        ax.plot(LEFT_PENALTY_SPOT_X, CENTER_Y, "ko", markersize=3)
        ax.plot(RIGHT_PENALTY_SPOT_X, CENTER_Y, "ko", markersize=3)

        # Penalty arcs (simplified as arc patches)
        left_arc = mpatches.Arc((LEFT_PENALTY_SPOT_X, CENTER_Y),
                                 PENALTY_ARC_RADIUS * 2, PENALTY_ARC_RADIUS * 2,
                                 angle=0, theta1=0, theta2=0,  # Invisible by default
                                 color="black", linewidth=1.0)
        # We approximate arcs with a few line segments
        theta = np.arccos((LEFT_PENALTY_X - LEFT_PENALTY_SPOT_X) / PENALTY_ARC_RADIUS)
        arc_angles = np.linspace(-theta, theta, 20)
        arc_x = LEFT_PENALTY_SPOT_X + PENALTY_ARC_RADIUS * np.cos(arc_angles)
        arc_y = CENTER_Y + PENALTY_ARC_RADIUS * np.sin(arc_angles)
        ax.plot(arc_x, arc_y, color="black", linewidth=1.0)

        arc_angles2 = np.linspace(np.pi - theta, np.pi + theta, 20)
        arc_x2 = RIGHT_PENALTY_SPOT_X + PENALTY_ARC_RADIUS * np.cos(arc_angles2)
        arc_y2 = CENTER_Y + PENALTY_ARC_RADIUS * np.sin(arc_angles2)
        ax.plot(arc_x2, arc_y2, color="black", linewidth=1.0)

        # Set pitch background
        ax.set_facecolor("#e8f5e9")  # Light green