"""GameAnalyzer — match intelligence from pipeline tracking data (possession, heatmaps, formation, territory, stats)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Tuple
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


class GameAnalyzer:
    """All methods are static — pass the list of per-frame data dicts."""

    # ------------------------------------------------------------------
    # Shared data helper
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
    # 1. POSSESSION
    # ------------------------------------------------------------------
    @staticmethod
    def compute_possession(game_data: List[dict], team1_label="Team 1", team2_label="Team 2") -> dict:
        t1_frames = t2_frames = total_ball = 0
        for entry in game_data:
            ball = entry.get("ball_position")
            if ball is None:
                continue
            valid_pos, valid_tid, t1, t2 = GameAnalyzer._split_teams(entry)
            if valid_pos is None or (len(t1) == 0 and len(t2) == 0):
                continue
            ball_arr = np.asarray(ball, dtype=np.float32).reshape(1, 2)
            dists = np.linalg.norm(valid_pos - ball_arr, axis=1)
            avg1 = np.mean(dists[valid_tid == 0]) if len(t1) > 0 else float("inf")
            avg2 = np.mean(dists[valid_tid == 1]) if len(t2) > 0 else float("inf")
            total_ball += 1
            t1_frames += avg1 <= avg2
            t2_frames += avg2 < avg1
        pct1 = round(t1_frames / max(total_ball, 1) * 100, 1)
        pct2 = round(t2_frames / max(total_ball, 1) * 100, 1)
        return {"team1_possession_pct": pct1, "team2_possession_pct": pct2,
                "team1_frames": t1_frames, "team2_frames": t2_frames,
                "total_ball_frames": total_ball, "team1_label": team1_label, "team2_label": team2_label}

    # ------------------------------------------------------------------
    # 2. HEATMAPS
    # ------------------------------------------------------------------
    @staticmethod
    def compute_heatmaps(game_data: List[dict], bins: Tuple[int, int] = (21, 14)) -> dict:
        t1_all, t2_all = [], []
        for entry in game_data:
            _, _, t1, t2 = GameAnalyzer._split_teams(entry)
            if t1 is not None and len(t1) > 0:
                t1_all.append(t1)
            if t2 is not None and len(t2) > 0:
                t2_all.append(t2)
        t1_all = np.vstack(t1_all) if t1_all else np.empty((0, 2))
        t2_all = np.vstack(t2_all) if t2_all else np.empty((0, 2))
        if len(t1_all) > 0:
            mask = (t1_all[:, 0] >= -5) & (t1_all[:, 0] <= PITCH_LENGTH + 5) & (t1_all[:, 1] >= -5) & (t1_all[:, 1] <= PITCH_WIDTH + 5)
            t1_all = t1_all[mask]
        if len(t2_all) > 0:
            mask = (t2_all[:, 0] >= -5) & (t2_all[:, 0] <= PITCH_LENGTH + 5) & (t2_all[:, 1] >= -5) & (t2_all[:, 1] <= PITCH_WIDTH + 5)
            t2_all = t2_all[mask]
        x_edges = np.linspace(0, PITCH_LENGTH, bins[0] + 1)
        y_edges = np.linspace(0, PITCH_WIDTH, bins[1] + 1)
        h1 = np.histogram2d(t1_all[:, 0], t1_all[:, 1], bins=(x_edges, y_edges))[0] if len(t1_all) > 0 else np.zeros((bins[0], bins[1]))
        h2 = np.histogram2d(t2_all[:, 0], t2_all[:, 1], bins=(x_edges, y_edges))[0] if len(t2_all) > 0 else np.zeros((bins[0], bins[1]))
        return {"team1_heatmap": h1, "team2_heatmap": h2, "x_edges": x_edges, "y_edges": y_edges,
                "team1_count": len(t1_all), "team2_count": len(t2_all)}

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
    def draw_formation_scatter(game_data: List[dict], team1_color=(0.2, 0.4, 0.9),
                               team2_color=(0.9, 0.2, 0.2), team1_label="Team 1",
                               team2_label="Team 2", max_frames: int = 100) -> plt.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
        GameAnalyzer._draw_pitch_outline(ax)
        t1_pts, t2_pts = [], []
        step = max(1, len(game_data) // max_frames)
        for entry in game_data[::step]:
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
    # 4. TERRITORY
    # ------------------------------------------------------------------
    @staticmethod
    def compute_territory(game_data: List[dict]) -> dict:
        counts = [[{"t1": 0, "t2": 0} for _ in range(3)] for _ in range(3)]
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
    # 5. MATCH STATS
    # ------------------------------------------------------------------
    @staticmethod
    def compute_match_stats(game_data: List[dict]) -> dict:
        total = len(game_data)
        ball_frames = sum(1 for e in game_data if e.get("ball_position") is not None)
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