"""
charts.py — Plotly figure builders for the PitchSense Streamlit dashboard.

All builders accept the theme palette (dict) so dark/light modes look consistent.

Charts produced:
    • build_possession_donut       — donut % chart for nearest-player possession
    • build_possession_timeline    — area chart of rolling possession
    • build_team_radar             — 5-axis tactical DNA radar
    • build_territory_grid         — 3×3 dominance heatmap (with hover details)
    • build_density_heatmap        — Player density heatmap on a pitch (interactive tooltips)
    • build_formation_scatter      — Combined positioning scatter on a pitch
    • build_region_pie             — Pie of pitch region detection frequency
    • build_region_bar             — Horizontal bar of region detection counts
    • build_zone_time_bar          — Horizontal bar of seconds in each pitch region
    • build_zone_timeline          — Multi-line timeline of region detections
    • build_attacking_direction_diagram — Two-arrow pitch diagram showing attack direction
    • build_passing_network        — Directed player→player pass graph on a pitch
    • build_player_region_distance_bar — Team distance split by pitch third
    • build_player_region_top_speed_bar — Team top speed split by pitch third
    • build_team_pressing_by_region_timeline — Team pressing split by pitch third
"""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from typing import Optional

from constants import (
    PITCH_LENGTH, PITCH_WIDTH, CENTER_X, CENTER_Y, CENTER_CIRCLE_RADIUS,
    PENALTY_ARC_RADIUS, LEFT_PENALTY_X, RIGHT_PENALTY_X,
    LEFT_GOAL_AREA_X, RIGHT_GOAL_AREA_X,
    PENALTY_Y_TOP, PENALTY_Y_BOTTOM,
    GOAL_AREA_Y_TOP, GOAL_AREA_Y_BOTTOM,
    LEFT_PENALTY_SPOT_X, RIGHT_PENALTY_SPOT_X,
)


PITCH_THIRD_REGIONS = [
    {"key": "defensive", "short": "Defensive", "label": "Defensive Third",
     "x_min": 0.0, "x_max": PITCH_LENGTH / 3.0},
    {"key": "middle", "short": "Middle", "label": "Middle Third",
     "x_min": PITCH_LENGTH / 3.0, "x_max": 2.0 * PITCH_LENGTH / 3.0},
    {"key": "attacking", "short": "Attacking", "label": "Attacking Third",
     "x_min": 2.0 * PITCH_LENGTH / 3.0, "x_max": PITCH_LENGTH},
]
_THIRD_SHORTS = [r["short"] for r in PITCH_THIRD_REGIONS]


# ─── Layout helpers ──────────────────────────────────────────────────────────
def _base_layout(palette: dict, height: int = 360, *, title: Optional[str] = None) -> dict:
    # Always pass an explicit title dict — `title=None` can be serialised to
    # JS undefined which Plotly then renders as the literal text "undefined"
    # in the chart header.
    title_text = title or ""
    return dict(
        height=height,
        margin=dict(l=10, r=10, t=30 if title else 10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=palette["text"], family="Inter, -apple-system, sans-serif"),
        title=dict(text=title_text, x=0.02, font=dict(size=14, color=palette["text"])),

        # NOTE: legend.title.text="" explicitly prevents Plotly from rendering
        # the literal word "undefined" as the legend title for traces (e.g. Pie,
        # Heatmap) that have no `name` attribute set.
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=palette["text"]),
            orientation="h", yanchor="bottom", y=1.02, x=0,
            title=dict(text=""),
        ),
        # NOTE: namelength=0 suppresses Plotly's default trace-name suffix that
        # otherwise renders as "undefined" when a trace has no `name` set.
        hoverlabel=dict(
            bgcolor=palette["panel"],
            bordercolor=palette["accent_1"],
            font=dict(color=palette["text"], family="Inter"),
            namelength=0,
        ),
    )




def _pitch_shapes(palette: dict) -> list[dict]:
    """Return Plotly `shapes` describing a standard pitch outline."""
    line = dict(color=palette["pitch_line"], width=2)
    shapes = [
        # outer rectangle
        dict(type="rect", x0=0, y0=0, x1=PITCH_LENGTH, y1=PITCH_WIDTH, line=line),
        # midline
        dict(type="line", x0=CENTER_X, y0=0, x1=CENTER_X, y1=PITCH_WIDTH, line=line),
        # center circle
        dict(type="circle",
             x0=CENTER_X - CENTER_CIRCLE_RADIUS, y0=CENTER_Y - CENTER_CIRCLE_RADIUS,
             x1=CENTER_X + CENTER_CIRCLE_RADIUS, y1=CENTER_Y + CENTER_CIRCLE_RADIUS,
             line=line),
        # left penalty
        dict(type="rect", x0=0, y0=PENALTY_Y_TOP, x1=LEFT_PENALTY_X, y1=PENALTY_Y_BOTTOM, line=line),
        # right penalty
        dict(type="rect", x0=RIGHT_PENALTY_X, y0=PENALTY_Y_TOP, x1=PITCH_LENGTH, y1=PENALTY_Y_BOTTOM, line=line),
        # left 6-yard
        dict(type="rect", x0=0, y0=GOAL_AREA_Y_TOP, x1=LEFT_GOAL_AREA_X, y1=GOAL_AREA_Y_BOTTOM, line=line),
        # right 6-yard
        dict(type="rect", x0=RIGHT_GOAL_AREA_X, y0=GOAL_AREA_Y_TOP, x1=PITCH_LENGTH, y1=GOAL_AREA_Y_BOTTOM, line=line),
        # center spot
        dict(type="circle",
             x0=CENTER_X - 0.4, y0=CENTER_Y - 0.4,
             x1=CENTER_X + 0.4, y1=CENTER_Y + 0.4,
             line=dict(color=palette["pitch_line"], width=1), fillcolor=palette["pitch_line"]),
        # penalty spots
        dict(type="circle",
             x0=LEFT_PENALTY_SPOT_X - 0.4, y0=CENTER_Y - 0.4,
             x1=LEFT_PENALTY_SPOT_X + 0.4, y1=CENTER_Y + 0.4,
             line=dict(color=palette["pitch_line"], width=1), fillcolor=palette["pitch_line"]),
        dict(type="circle",
             x0=RIGHT_PENALTY_SPOT_X - 0.4, y0=CENTER_Y - 0.4,
             x1=RIGHT_PENALTY_SPOT_X + 0.4, y1=CENTER_Y + 0.4,
             line=dict(color=palette["pitch_line"], width=1), fillcolor=palette["pitch_line"]),
    ]
    # Penalty arcs (as SVG paths)
    th = np.arccos((LEFT_PENALTY_X - LEFT_PENALTY_SPOT_X) / PENALTY_ARC_RADIUS)
    ang_l = np.linspace(-th, th, 30)
    ang_r = np.linspace(np.pi - th, np.pi + th, 30)
    xs_l = LEFT_PENALTY_SPOT_X + PENALTY_ARC_RADIUS * np.cos(ang_l)
    ys_l = CENTER_Y + PENALTY_ARC_RADIUS * np.sin(ang_l)
    xs_r = RIGHT_PENALTY_SPOT_X + PENALTY_ARC_RADIUS * np.cos(ang_r)
    ys_r = CENTER_Y + PENALTY_ARC_RADIUS * np.sin(ang_r)
    for xs, ys in [(xs_l, ys_l), (xs_r, ys_r)]:
        path = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
        shapes.append(dict(type="path", path=path,
                           line=dict(color=palette["pitch_line"], width=2)))
    return shapes


def _pitch_axes(palette: dict, height: int = 460) -> dict:
    return dict(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=palette["pitch_bg"],
        font=dict(color=palette["text"], family="Inter, sans-serif"),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h", yanchor="bottom", y=1.02, x=0,
            font=dict(color=palette["text"]),
            title=dict(text=""),
        ),

        xaxis=dict(
            range=[-2, PITCH_LENGTH + 2],
            showgrid=False, zeroline=False, showticklabels=False,
            scaleanchor="y", scaleratio=1, constrain="domain",
        ),
        yaxis=dict(
            range=[-2, PITCH_WIDTH + 2],
            showgrid=False, zeroline=False, showticklabels=False,
        ),
        shapes=_pitch_shapes(palette),
        hoverlabel=dict(
            bgcolor=palette["panel"],
            bordercolor=palette["accent_1"],
            font=dict(color=palette["text"]),
            namelength=0,
        ),
    )



# ─── Donut: possession ───────────────────────────────────────────────────────
def build_possession_donut(palette: dict, t1_pct: float, t2_pct: float,
                           t1_label: str = "Team 1", t2_label: str = "Team 2") -> go.Figure:
    contested = max(0.0, 100.0 - t1_pct - t2_pct)
    labels = [t1_label, "Contested", t2_label]
    values = [t1_pct, contested, t2_pct]
    colors = [palette["team1"], palette["panel_alt"], palette["team2"]]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        name="Possession",
        marker=dict(colors=colors, line=dict(color=palette["panel"], width=2)),
        textinfo="label+percent",
        textfont=dict(size=13, color=palette["text"]),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
        sort=False, direction="clockwise",
    ))
    fig.add_annotation(
        x=0.5, y=0.5, showarrow=False,
        text=f"<b style='font-size:22px;color:{palette['text']}'>{t1_pct:.0f} : {t2_pct:.0f}</b>"
             f"<br><span style='font-size:11px;color:{palette['text_dim']}'>POSSESSION</span>",
        font=dict(family="Inter"), align="center",
    )
    fig.update_layout(_base_layout(palette, height=340))
    fig.update_layout(legend_title_text="")
    return fig




# ─── Possession timeline (rolling) ───────────────────────────────────────────
def build_possession_timeline(palette: dict, game_data: list, window: int = 30) -> go.Figure:
    """Rolling possession share line chart.

    Possession is determined by bbox overlap between the ball and a
    player's bounding box, with carry-forward so the chart is sticky
    (possession only changes when the OTHER team's player bbox overlaps
    the ball bbox).
    """
    from game_analyzer import GameAnalyzer
    owners = GameAnalyzer.compute_ball_owner_per_frame(game_data)
    buf, t1_series, t2_series, frames_axis = [], [], [], []
    for i, entry in enumerate(game_data):
        owner = owners[i]
        team = int(owner) if owner is not None else -1
        buf.append(team)
        if len(buf) > window:
            buf.pop(0)
        valid = [b for b in buf if b in (0, 1)]
        if valid:
            t1pc = round(100.0 * sum(1 for b in valid if b == 0) / len(valid), 1)
            t2pc = round(100.0 * sum(1 for b in valid if b == 1) / len(valid), 1)
        else:
            t1pc = t2pc = 0.0
        t1_series.append(t1pc)
        t2_series.append(t2pc)
        frames_axis.append(int(entry.get("frame_idx", len(frames_axis) + 1)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frames_axis, y=t1_series, mode="lines",
        name="Team 1", line=dict(color=palette["team1"], width=2.4),
        fill="tozeroy", fillcolor=_alpha(palette["team1"], 0.18),
        hovertemplate="Frame %{x}<br><b>Team 1</b>: %{y}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=frames_axis, y=t2_series, mode="lines",
        name="Team 2", line=dict(color=palette["team2"], width=2.4),
        fill="tozeroy", fillcolor=_alpha(palette["team2"], 0.18),
        hovertemplate="Frame %{x}<br><b>Team 2</b>: %{y}%<extra></extra>",
    ))
    layout = _base_layout(palette, height=340)
    layout.update(dict(
        xaxis=dict(title="Frame", gridcolor=palette["grid"], zerolinecolor=palette["grid"]),
        yaxis=dict(title="Possession %", range=[0, 100],
                   gridcolor=palette["grid"], zerolinecolor=palette["grid"]),
    ))
    fig.update_layout(layout)
    fig.update_layout(legend_title_text="")
    return fig



# ─── Team radar ──────────────────────────────────────────────────────────────
def build_team_radar(palette: dict, formation: dict, stats: dict, possession: dict) -> go.Figure:
    """5-axis tactical DNA radar without defensive-depth metrics."""
    def clip(v, lo=0.0, hi=100.0):
        return float(max(lo, min(hi, v)))

    t1_attack = clip(formation["team1_avg_center"][0] / PITCH_LENGTH * 100) if formation.get("team1_avg_center") else 50.0
    t2_attack = clip(formation["team2_avg_center"][0] / PITCH_LENGTH * 100) if formation.get("team2_avg_center") else 50.0
    max_spread = 40.0
    t1_comp = clip(100 - (formation["team1_avg_spread"] / max_spread * 100))
    t2_comp = clip(100 - (formation["team2_avg_spread"] / max_spread * 100))
    t1_width = clip(formation["team1_avg_spread"] / max_spread * 100)
    t2_width = clip(formation["team2_avg_spread"] / max_spread * 100)
    t1_poss = clip(possession.get("team1_possession_pct", 0.0))
    t2_poss = clip(possession.get("team2_possession_pct", 0.0))
    tempo = clip(stats.get("ball_progression_m", 0) / 1000.0 * 100)
    t1_tempo = tempo * (t1_poss / 100 + 0.5)
    t2_tempo = tempo * (t2_poss / 100 + 0.5)

    categories = ["Attacking Intent", "Compactness", "Width", "Possession", "Tempo"]
    t1_vals = [t1_attack, t1_comp, t1_width, t1_poss, clip(t1_tempo)]
    t2_vals = [t2_attack, t2_comp, t2_width, t2_poss, clip(t2_tempo)]

    # Close the polygon
    cats_loop = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=t1_vals + [t1_vals[0]], theta=cats_loop, fill="toself",
        name="Team 1", line=dict(color=palette["team1"], width=2.4),
        fillcolor=_alpha(palette["team1"], 0.28),
        hovertemplate="<b>Team 1</b> · %{theta}<br>%{r:.1f} / 100<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=t2_vals + [t2_vals[0]], theta=cats_loop, fill="toself",
        name="Team 2", line=dict(color=palette["team2"], width=2.4),
        fillcolor=_alpha(palette["team2"], 0.28),
        hovertemplate="<b>Team 2</b> · %{theta}<br>%{r:.1f} / 100<extra></extra>",
    ))
    fig.update_layout(_base_layout(palette, height=460))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                range=[0, 100], showline=False, showticklabels=True,
                gridcolor=palette["grid"], tickfont=dict(size=10, color=palette["text_dim"]),
            ),
            angularaxis=dict(
                gridcolor=palette["grid"], tickfont=dict(size=11, color=palette["text"]),
            ),
        ),
        legend_title_text="",
    )
    return fig



# ─── Territory 3x3 ───────────────────────────────────────────────────────────
def build_territory_grid(palette: dict, territory: dict) -> go.Figure:
    """Plotly heatmap of zone dominance (T1 share %) with hover details."""
    zone_grid = territory["zone_grid"]
    # rows in zone_grid are pitch-WIDTH bands; cols are LENGTH thirds
    # Display rows top→bottom (Left/Cent/Right), cols left→right (Def/Mid/Att).
    z = []
    text = []
    custom = []
    row_labels = ["Top",  "Centre", "Bottom"]   # pitch-width bands
    col_labels = ["Defensive", "Middle", "Attacking"]
    for r in range(3):
        z_row, t_row, c_row = [], [], []
        for c in range(3):
            zone = zone_grid[r][c]
            t1 = zone["team1_pct"]
            t2 = zone["team2_pct"]
            # Signed value: positive = T1 dominance, negative = T2 dominance
            signed = t1 - t2
            z_row.append(signed)
            t_row.append(
                f"<b>{zone['zone_name']}</b><br>"
                f"T1 {t1:.0f}% · T2 {t2:.0f}%"
            )
            c_row.append([t1, t2, zone.get("total_frames", 0)])
        z.append(z_row); text.append(t_row); custom.append(c_row)

    fig = go.Figure(go.Heatmap(
        z=z, text=text,
        x=col_labels, y=row_labels,
        customdata=custom,
        name="",
        showlegend=False,
        hoverlabel=dict(namelength=0),
        colorscale=[
            [0.0, palette["team2"]],
            [0.5, palette["panel_alt"]],
            [1.0, palette["team1"]],
        ],
        zmin=-100, zmax=100,
        hovertemplate=(
            "<b>%{y} · %{x}</b><br>"
            "Team 1: %{customdata[0]:.1f}%<br>"
            "Team 2: %{customdata[1]:.1f}%<br>"
            "Player-frames: %{customdata[2]}<extra></extra>"
        ),
        showscale=True,
        colorbar=dict(
            title=dict(text="Dominance", font=dict(color=palette["text_dim"])),
            tickfont=dict(color=palette["text_dim"]),
            ticktext=["T2", "Neutral", "T1"], tickvals=[-80, 0, 80],
        ),
        texttemplate="%{text}",
        textfont=dict(color=palette["text"], size=13),
    ))

    fig.update_layout(_base_layout(palette, height=420))
    fig.update_layout(legend_title_text="", showlegend=False)
    fig.update_xaxes(side="bottom", showgrid=False, tickfont=dict(color=palette["text"], size=13))
    fig.update_yaxes(showgrid=False, autorange="reversed", tickfont=dict(color=palette["text"], size=13))
    return fig



# ─── Density heatmap on pitch (interactive tooltip) ──────────────────────────
def build_density_heatmap(palette: dict, game_data: list, team_id: int,
                          *, bins=(18, 12), name="Team 1") -> go.Figure:
    """Player density heatmap overlaid on a pitch; hovering a cell shows a tooltip."""
    from game_analyzer import GameAnalyzer
    heat = GameAnalyzer.compute_heatmaps(game_data, bins=bins)
    if team_id == 0:
        matrix = heat["team1_heatmap"]
        color = palette["team1"]
        sample_count = heat["team1_count"]
    else:
        matrix = heat["team2_heatmap"]
        color = palette["team2"]
        sample_count = heat["team2_count"]

    x_edges = heat["x_edges"]
    y_edges = heat["y_edges"]
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

    # Plotly expects z indexed as [y][x]
    z = matrix.T  # shape (n_y, n_x)
    max_v = float(z.max()) if z.size else 0.0
    # Hover customdata: (count, third label, lateral band)
    # NOTE: must use object dtype so the numeric count is preserved alongside
    # the string labels — np.dstack with mixed types would coerce everything
    # to strings, which makes plotly's "%{customdata[0]:.0f}" format render
    # as "undefined" on hover.
    n_y, n_x = z.shape
    custom = np.empty((n_y, n_x, 3), dtype=object)
    for yi, yc in enumerate(y_centers):
        if yc < PITCH_WIDTH / 3:
            band = "Bottom Band"
        elif yc < 2 * PITCH_WIDTH / 3:
            band = "Centre Band"
        else:
            band = "Top Band"
        for xi, xc in enumerate(x_centers):
            if xc < PITCH_LENGTH / 3:
                third = "Defensive Third"
            elif xc < 2 * PITCH_LENGTH / 3:
                third = "Middle Third"
            else:
                third = "Attacking Third"
            custom[yi, xi, 0] = float(z[yi, xi])
            custom[yi, xi, 1] = third
            custom[yi, xi, 2] = band


    # Build a colorscale: transparent → team color
    scale = [[0.0, "rgba(0,0,0,0)"], [0.15, _alpha(color, 0.35)],
             [0.55, _alpha(color, 0.75)], [1.0, color]]

    fig = go.Figure(go.Heatmap(
        z=z.tolist(),
        x=x_centers.tolist(),
        y=y_centers.tolist(),
        zsmooth="best",
        zmin=0, zmax=max_v if max_v > 0 else 1,
        customdata=custom,
        name=name,
        showlegend=False,
        hoverlabel=dict(namelength=0),
        colorscale=scale,
        showscale=True,
        colorbar=dict(
            title=dict(text="Density", font=dict(color=palette["text_dim"])),
            tickfont=dict(color=palette["text_dim"]),
            outlinewidth=0,
        ),
        hovertemplate=(
            f"<b>{name}</b><br>"
            "x: %{x:.1f} m · y: %{y:.1f} m<br>"
            "Detections: %{customdata[0]:.0f}<br>"
            "Zone: %{customdata[1]} · %{customdata[2]}"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(_pitch_axes(palette, height=460))
    fig.update_layout(legend_title_text="", showlegend=False)
    fig.add_annotation(
        x=PITCH_LENGTH / 2, y=PITCH_WIDTH + 0.5, showarrow=False,
        text=f"<b>{name}</b> · {sample_count} samples",
        font=dict(color=palette["text"], size=12), xanchor="center",
    )
    return fig



# ─── Combined formation scatter ──────────────────────────────────────────────
def build_formation_scatter(palette: dict, game_data: list,
                            max_frames: int = 200) -> go.Figure:
    from game_analyzer import GameAnalyzer, TEAM0, TEAM1

    registry = GameAnalyzer.build_registry(game_data)

    step = max(1, len(game_data) // max_frames)
    t1_xs, t1_ys, t1_fr = [], [], []
    t2_xs, t2_ys, t2_fr = [], [], []
    for entry in game_data[::step]:
        tids = entry.get("track_ids")
        positions = entry.get("player_positions")
        fi = int(entry.get("frame_idx", 0))
        if registry.has_track_ids and tids is not None and positions is not None:
            for i, tid in enumerate(np.asarray(tids)):
                rec = registry.tracks.get(int(tid))
                if rec is None:
                    continue
                if rec.canonical_team == TEAM0:
                    x, y = float(positions[i][0]), float(positions[i][1])
                elif rec.canonical_team == TEAM1:
                    x, y = float(positions[i][0]), float(positions[i][1])
                else:
                    continue
                if -2 <= x <= PITCH_LENGTH + 2 and -2 <= y <= PITCH_WIDTH + 2:
                    if rec.canonical_team == TEAM0:
                        t1_xs.append(x); t1_ys.append(y); t1_fr.append(fi)
                    else:
                        t2_xs.append(x); t2_ys.append(y); t2_fr.append(fi)
        else:
            _, _, t1, t2 = GameAnalyzer._split_teams(entry)
            if t1 is not None and len(t1) > 0:
                for x, y in t1:
                    if -2 <= x <= PITCH_LENGTH + 2 and -2 <= y <= PITCH_WIDTH + 2:
                        t1_xs.append(float(x)); t1_ys.append(float(y)); t1_fr.append(fi)
            if t2 is not None and len(t2) > 0:
                for x, y in t2:
                    if -2 <= x <= PITCH_LENGTH + 2 and -2 <= y <= PITCH_WIDTH + 2:
                        t2_xs.append(float(x)); t2_ys.append(float(y)); t2_fr.append(fi)

    ball_xs, ball_ys, ball_fr = [], [], []
    for entry in game_data:
        bp = entry.get("ball_position")
        if bp is None:
            continue
        bp = np.asarray(bp, dtype=float).reshape(-1)
        if bp.shape[0] < 2:
            continue
        x, y = float(bp[0]), float(bp[1])
        if -5 <= x <= PITCH_LENGTH + 5 and -5 <= y <= PITCH_WIDTH + 5:
            ball_xs.append(x); ball_ys.append(y); ball_fr.append(int(entry.get("frame_idx", 0)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t1_xs, y=t1_ys, mode="markers", name="Team 1",
        marker=dict(color=palette["team1"], size=7, opacity=0.55,
                    line=dict(width=0)),
        customdata=t1_fr,
        hovertemplate="<b>Team 1</b><br>x %{x:.1f}m · y %{y:.1f}m<br>Frame %{customdata}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=t2_xs, y=t2_ys, mode="markers", name="Team 2",
        marker=dict(color=palette["team2"], size=7, opacity=0.55,
                    line=dict(width=0)),
        customdata=t2_fr,
        hovertemplate="<b>Team 2</b><br>x %{x:.1f}m · y %{y:.1f}m<br>Frame %{customdata}<extra></extra>",
    ))
    if ball_xs:
        fig.add_trace(go.Scatter(
            x=ball_xs, y=ball_ys, mode="lines+markers", name="Ball trail",
            line=dict(color=palette["warn"], width=1.5),
            marker=dict(color=palette["warn"], size=4, opacity=0.8),
            customdata=ball_fr,
            hovertemplate="<b>Ball</b><br>x %{x:.1f}m · y %{y:.1f}m<br>Frame %{customdata}<extra></extra>",
        ))
    fig.update_layout(_pitch_axes(palette, height=520))
    fig.update_layout(legend_title_text="")
    return fig


# ─── Region pie + bar (pitch segmentation) ───────────────────────────────────
def build_region_pie(palette: dict, region_items: list[dict]) -> go.Figure:
    if not region_items:
        return _empty_fig(palette, "No region detections recorded")
    labels = [r["label"] for r in region_items]
    values = [r["count"] for r in region_items]
    colors = _categorical_palette(palette, len(labels))

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.45,
        name="Regions",
        marker=dict(colors=colors, line=dict(color=palette["panel"], width=2)),
        textinfo="percent",
        textfont=dict(size=12, color="#ffffff"),
        hovertemplate="<b>%{label}</b><br>%{value} detections<br>%{percent}<extra></extra>",
        sort=False,
    ))
    fig.update_layout(_base_layout(palette, height=360))
    fig.update_layout(legend_title_text="")
    return fig


def build_region_bar(palette: dict, region_items: list[dict]) -> go.Figure:
    if not region_items:
        return _empty_fig(palette, "No region detections recorded")
    labels = [r["label"] for r in region_items]
    values = [r["count"] for r in region_items]
    colors = _categorical_palette(palette, len(labels))

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        name="",
        showlegend=False,
        hoverlabel=dict(namelength=0),
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:,}" for v in values], textposition="outside",
        textfont=dict(color=palette["text"]),
        hovertemplate="<b>%{y}</b><br>%{x} detections<extra></extra>",
    ))
    layout = _base_layout(palette, height=360)
    layout.update(dict(
        xaxis=dict(title="Detections", gridcolor=palette["grid"], zerolinecolor=palette["grid"]),
        yaxis=dict(autorange="reversed", showgrid=False),
    ))
    fig.update_layout(layout)
    fig.update_layout(legend_title_text="", showlegend=False)
    return fig


# ─── Zone time bar + zone timeline (deeper segmentation analytics) ───────────
SEG_LABELS = {
    "18Yard":               "Penalty Area (18yd)",
    "18Yard Circle":        "Penalty Arc",
    "5Yard":                "Goal Area (6yd)",
    "Half Central Circle":  "Center Circle",
    "Half Field":           "Half Field",
}


def build_zone_time_bar(palette: dict, summary: dict) -> go.Figure:
    """Horizontal bar of seconds in each pitch region (from compute_zone_summary)."""
    region_time = (summary or {}).get("region_time_s", {}) or {}
    if not region_time:
        return _empty_fig(palette, "No region detections recorded")
    items = sorted(region_time.items(), key=lambda kv: kv[1], reverse=True)
    labels = [SEG_LABELS.get(k, k) for k, _ in items]
    values = [v for _, v in items]
    colors = _categorical_palette(palette, len(labels))

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        name="",
        showlegend=False,
        hoverlabel=dict(namelength=0),
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}s" for v in values], textposition="outside",
        textfont=dict(color=palette["text"]),
        hovertemplate="<b>%{y}</b><br>%{x:.2f} seconds<extra></extra>",
    ))
    layout = _base_layout(palette, height=320)
    layout.update(dict(
        xaxis=dict(title="Time in zone (s)", gridcolor=palette["grid"],
                   zerolinecolor=palette["grid"]),
        yaxis=dict(autorange="reversed", showgrid=False),
    ))
    fig.update_layout(layout)
    fig.update_layout(legend_title_text="", showlegend=False)
    return fig


def build_zone_timeline(palette: dict, timeline: dict) -> go.Figure:
    """Multi-line timeline of rolling-window region detection counts."""
    if not timeline or not timeline.get("x") or not timeline.get("region_names"):
        return _empty_fig(palette, "No region detections recorded")
    cycle = _categorical_palette(palette, len(timeline["region_names"]))
    fig = go.Figure()
    for rname, color in zip(timeline["region_names"], cycle):
        y = timeline["series"].get(rname, [])
        fig.add_trace(go.Scatter(
            x=timeline["x"], y=y, mode="lines",
            name=SEG_LABELS.get(rname, rname),
            line=dict(color=color, width=2.2),
            hovertemplate=f"<b>{SEG_LABELS.get(rname, rname)}</b><br>"
                          "Frame %{x}<br>Detections: %{y}<extra></extra>",
        ))
    layout = _base_layout(palette, height=340)
    layout.update(dict(
        xaxis=dict(title="Frame", gridcolor=palette["grid"],
                   zerolinecolor=palette["grid"]),
        yaxis=dict(title="Detections in window", rangemode="tozero",
                   gridcolor=palette["grid"], zerolinecolor=palette["grid"]),
    ))
    fig.update_layout(layout)
    fig.update_layout(legend_title_text="")
    return fig


# ─── Attacking direction diagram ─────────────────────────────────────────────
def build_attacking_direction_diagram(palette: dict, direction_info: dict,
                                      team1_color: str, team2_color: str) -> go.Figure:
    """Two-arrow pitch diagram showing each team's attacking direction."""
    t1_dir = (direction_info or {}).get("team1_attacks")
    t2_dir = (direction_info or {}).get("team2_attacks")
    if not t1_dir or not t2_dir:
        return _empty_fig(palette, "Attacking direction could not be inferred")

    fig = go.Figure()
    arrow_y_t1 = PITCH_WIDTH * 0.30
    arrow_y_t2 = PITCH_WIDTH * 0.70

    def _arrow(y: float, direction: str, color: str, label: str):
        if direction == "right":
            x0, x1 = PITCH_LENGTH * 0.10, PITCH_LENGTH * 0.90
        else:
            x0, x1 = PITCH_LENGTH * 0.90, PITCH_LENGTH * 0.10
        fig.add_annotation(
            x=x1, y=y, ax=x0, ay=y, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.6, arrowwidth=4,
            arrowcolor=color, text="",
        )
        fig.add_annotation(
            x=(x0 + x1) / 2, y=y + 4, showarrow=False,
            text=f"<b>{label}</b> → {'Right' if direction == 'right' else 'Left'}",
            font=dict(color=color, size=14),
        )

    _arrow(arrow_y_t1, t1_dir, team1_color, "Team 1")
    _arrow(arrow_y_t2, t2_dir, team2_color, "Team 2")

    layout = _pitch_axes(palette, height=380)
    layout.pop("legend", None)
    layout["showlegend"] = False
    layout["xaxis"].update(dict(showticklabels=False))
    layout["yaxis"].update(dict(showticklabels=False))
    fig.update_layout(layout)
    return fig



# ─── Helpers ─────────────────────────────────────────────────────────────────
def _alpha(hex_color: str, a: float) -> str:
    """Convert #rrggbb to rgba string with alpha."""
    if hex_color.startswith("#") and len(hex_color) == 7:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f"rgba({r},{g},{b},{a})"
    return hex_color


def _categorical_palette(palette: dict, n: int) -> list[str]:
    cycle = [
        palette["accent_1"], palette["accent_2"], palette["team1"],
        palette["team2"], palette["good"], palette["warn"], palette["bad"],
    ]
    return [cycle[i % len(cycle)] for i in range(n)]


def _empty_fig(palette: dict, msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                       font=dict(color=palette["text_dim"]))
    fig.update_layout(_base_layout(palette, height=300))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# ─── Passing network ─────────────────────────────────────────────────────────
def build_passing_network(palette: dict, network: dict, team_label: str,
                          team_color: str) -> go.Figure:
    """Draw a directed passing graph on a pitch layout."""
    nodes = (network or {}).get("nodes", [])
    edges = (network or {}).get("edges", [])
    if not nodes:
        return _empty_fig(palette, f"No track data for {team_label}")
    fig = go.Figure()
    # Edges (drawn first so they sit under nodes)
    max_w = max((e["count"] for e in edges), default=1) or 1
    for e in edges:
        a = next((n for n in nodes if n["track_id"] == e["from"]), None)
        b = next((n for n in nodes if n["track_id"] == e["to"]), None)
        if a is None or b is None:
            continue
        w = max(1.0, 6.0 * e["count"] / max_w)
        fig.add_trace(go.Scatter(
            x=[a["x"], b["x"]], y=[a["y"], b["y"]],
            mode="lines",
            line=dict(color=team_color, width=w),
            opacity=0.55,
            hoverinfo="skip",
            showlegend=False,
        ))
    # Nodes
    nx = [n["x"] for n in nodes]
    ny = [n["y"] for n in nodes]
    nlabels = [f"#{n['track_id']}" for n in nodes]
    fig.add_trace(go.Scatter(
        x=nx, y=ny, mode="markers+text",
        marker=dict(color=team_color, size=18, line=dict(color="#ffffff", width=2)),
        text=nlabels, textposition="top center",
        textfont=dict(color=palette["text"], size=10),
        name=team_label,
        hovertemplate=f"<b>{team_label}</b><br>Track #%{{text}}<br>"
                      "x %{x:.1f} · y %{y:.1f}<extra></extra>",
    ))
    layout = _pitch_axes(palette, height=420)
    layout["showlegend"] = False
    layout["xaxis"].update(dict(showticklabels=False))
    layout["yaxis"].update(dict(showticklabels=False))
    fig.update_layout(layout)
    fig.add_annotation(
        x=PITCH_LENGTH / 2, y=PITCH_WIDTH + 1.5, showarrow=False,
        text=f"<b>{team_label}</b> · {len(edges)} passing connections",
        font=dict(color=palette["text"], size=12), xanchor="center",
    )
    return fig


def _region_for_key(region_key: str) -> dict:
    for region in PITCH_THIRD_REGIONS:
        if region["key"] == region_key:
            return region
    return PITCH_THIRD_REGIONS[0]


def _x_in_region(x: float, region: dict) -> bool:
    x = float(x)
    if region["key"] == "attacking":
        return region["x_min"] <= x <= region["x_max"]
    return region["x_min"] <= x < region["x_max"]


def _third_key_for_x(x: float) -> Optional[str]:
    for region in PITCH_THIRD_REGIONS:
        if _x_in_region(x, region):
            return region["key"]
    return None


def _profile_rows(profiles: list[dict] | dict) -> list[dict]:
    if isinstance(profiles, dict):
        return list(profiles.get("profiles", []))
    return list(profiles or [])


def filter_passing_network_by_pitch_third(network: dict, region_key: str) -> dict:
    """Return a passing network containing only nodes inside one pitch third."""
    region = _region_for_key(region_key)
    nodes = []
    for node in (network or {}).get("nodes", []):
        try:
            x = float(node.get("x", float("nan")))
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and _x_in_region(x, region):
            nodes.append(dict(node))
    node_ids = {int(n["track_id"]) for n in nodes if "track_id" in n}
    edges = []
    for edge in (network or {}).get("edges", []):
        try:
            a = int(edge.get("from"))
            b = int(edge.get("to"))
        except (TypeError, ValueError):
            continue
        if a in node_ids and b in node_ids:
            edges.append(dict(edge))
    return {"nodes": nodes, "edges": edges}


def build_player_region_distance_bar(palette: dict, profiles: list[dict] | dict,
                                     team_id: int, team_label: str) -> go.Figure:
    """Distance covered by one team, weighted by each player's time in thirds."""
    rows = [p for p in _profile_rows(profiles) if int(p.get("team", -1)) == int(team_id)]
    if not rows:
        return _empty_fig(palette, f"No player profiles for {team_label}")
    totals = []
    player_counts = []
    for idx, _region in enumerate(PITCH_THIRD_REGIONS):
        total = 0.0
        count = 0
        for profile in rows:
            thirds = profile.get("time_in_thirds_pct") or [0.0, 0.0, 0.0]
            pct = float(thirds[idx]) if idx < len(thirds) else 0.0
            if pct > 0.0:
                count += 1
            total += float(profile.get("distance_m", 0.0)) * pct / 100.0
        totals.append(total)
        player_counts.append(count)

    color = palette["team1"] if int(team_id) == 0 else palette["team2"]
    custom = [[c, v / 1000.0] for c, v in zip(player_counts, totals)]
    fig = go.Figure(go.Bar(
        x=_THIRD_SHORTS,
        y=totals,
        name=team_label,
        marker=dict(color=color, line=dict(width=0)),
        customdata=custom,
        text=[f"{v / 1000.0:.2f} km" for v in totals],
        textposition="outside",
        textfont=dict(color=palette["text"]),
        hovertemplate=(
            f"<b>{team_label}</b><br>Region: %{{x}}<br>"
            "Distance: %{y:.1f} m (%{customdata[1]:.2f} km)<br>"
            "Players present: %{customdata[0]}<extra></extra>"
        ),
    ))
    layout = _base_layout(palette, height=320)
    layout.update(dict(
        showlegend=False,
        xaxis=dict(title="Pitch third", gridcolor=palette["grid"]),
        yaxis=dict(title="Weighted distance (m)", gridcolor=palette["grid"],
                   rangemode="tozero"),
    ))
    fig.update_layout(layout)
    fig.update_layout(legend_title_text="", showlegend=False)
    return fig


def build_player_region_top_speed_bar(palette: dict, profiles: list[dict] | dict,
                                      team_id: int, team_label: str) -> go.Figure:
    """Top speed by dominant pitch third for one team."""
    rows = [p for p in _profile_rows(profiles) if int(p.get("team", -1)) == int(team_id)]
    if not rows:
        return _empty_fig(palette, f"No player profiles for {team_label}")
    speeds = []
    track_ids = []
    for region in PITCH_THIRD_REGIONS:
        candidates = [
            p for p in rows
            if str(p.get("dominant_third", "")).lower().startswith(region["short"].lower())
        ]
        if not candidates:
            speeds.append(0.0)
            track_ids.append("—")
            continue
        top_profile = max(candidates, key=lambda p: float(p.get("top_speed_m_s", 0.0)))
        speeds.append(float(top_profile.get("top_speed_m_s", 0.0)))
        track_ids.append(f"#{int(top_profile.get('track_id', 0))}")

    color = palette["team1"] if int(team_id) == 0 else palette["team2"]
    custom = [[tid] for tid in track_ids]
    fig = go.Figure(go.Bar(
        x=_THIRD_SHORTS,
        y=speeds,
        name=team_label,
        marker=dict(color=color, line=dict(width=0)),
        customdata=custom,
        text=[f"{v:.1f}" if v > 0 else "—" for v in speeds],
        textposition="outside",
        textfont=dict(color=palette["text"]),
        hovertemplate=(
            f"<b>{team_label}</b><br>Dominant region: %{{x}}<br>"
            "Top speed: %{y:.2f} m/s<br>Player: %{customdata[0]}<extra></extra>"
        ),
    ))
    layout = _base_layout(palette, height=320)
    layout.update(dict(
        showlegend=False,
        xaxis=dict(title="Dominant pitch third", gridcolor=palette["grid"]),
        yaxis=dict(title="Top speed (m/s)", gridcolor=palette["grid"],
                   rangemode="tozero"),
    ))
    fig.update_layout(layout)
    fig.update_layout(legend_title_text="", showlegend=False)
    return fig


def build_team_pressing_by_region_timeline(palette: dict, pressing: dict,
                                           game_data: list, team_id: int,
                                           team_label: str) -> go.Figure:
    """One team's pressing timeline split by ball-location pitch third."""
    if not pressing or not pressing.get("x"):
        return _empty_fig(palette, f"No pressing data for {team_label}")

    team_key = "team1" if int(team_id) == 0 else "team2"
    x_axis = list(pressing.get("x", []))
    values = list(pressing.get(team_key, []))
    n = min(len(x_axis), len(values), len(game_data or []))
    if n == 0:
        return _empty_fig(palette, f"No pressing data for {team_label}")

    series = {r["key"]: [None] * n for r in PITCH_THIRD_REGIONS}
    has_values = False
    for i in range(n):
        value = values[i]
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        ball = (game_data[i] or {}).get("ball_position")
        if ball is None:
            continue
        arr = np.asarray(ball, dtype=float).reshape(-1)
        if arr.shape[0] < 2 or not np.isfinite(arr[0]):
            continue
        region_key = _third_key_for_x(float(arr[0]))
        if region_key is None:
            continue
        series[region_key][i] = value
        has_values = True

    if not has_values:
        return _empty_fig(palette, f"No region-tagged pressing data for {team_label}")

    team_color = palette["team1"] if int(team_id) == 0 else palette["team2"]
    colors = [palette["accent_1"], palette["warn"], team_color]
    fig = go.Figure()
    for region, color in zip(PITCH_THIRD_REGIONS, colors):
        fig.add_trace(go.Scatter(
            x=x_axis[:n],
            y=series[region["key"]],
            mode="lines",
            name=region["short"],
            line=dict(color=color, width=2.0),
            connectgaps=False,
            hovertemplate=(
                f"<b>{team_label}</b><br>Region: {region['label']}<br>"
                "Frame %{x}<br>Nearest opponent: %{y:.1f} m<extra></extra>"
            ),
        ))
    layout = _base_layout(palette, height=320)
    layout.update(dict(
        xaxis=dict(title="Frame", gridcolor=palette["grid"],
                   zerolinecolor=palette["grid"]),
        yaxis=dict(title="Nearest opponent to ball (m)", rangemode="tozero",
                   gridcolor=palette["grid"], zerolinecolor=palette["grid"]),
    ))
    fig.update_layout(layout)
    fig.update_layout(legend_title_text="")
    return fig


# ─── Pressing timeline ───────────────────────────────────────────────────────
def build_pressing_timeline(palette: dict, pressing: dict) -> go.Figure:
    """Two-line chart of nearest-opponent distance to the ball (lower = more press)."""
    if not pressing or not pressing.get("x"):
        return _empty_fig(palette, "No pressing data")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pressing["x"], y=pressing["team1"], mode="lines",
        name="Team 1", line=dict(color=palette["team1"], width=2),
        hovertemplate="Frame %{x}<br>Team 1 press: %{y:.1f} m<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=pressing["x"], y=pressing["team2"], mode="lines",
        name="Team 2", line=dict(color=palette["team2"], width=2),
        hovertemplate="Frame %{x}<br>Team 2 press: %{y:.1f} m<extra></extra>",
    ))
    layout = _base_layout(palette, height=320)
    layout.update(dict(
        xaxis=dict(title="Frame", gridcolor=palette["grid"],
                   zerolinecolor=palette["grid"]),
        yaxis=dict(title="Nearest opponent to ball (m)", rangemode="tozero",
                   gridcolor=palette["grid"], zerolinecolor=palette["grid"]),
    ))
    fig.update_layout(layout)
    return fig


# ─── Defensive line height timeline ──────────────────────────────────────────
def build_defensive_line_timeline(palette: dict, dline: dict) -> go.Figure:
    """Deepest outfield defender's X for each team per frame (excludes GK)."""
    if not dline or not dline.get("x"):
        return _empty_fig(palette, "No defensive-line data")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dline["x"], y=dline["team1"], mode="lines",
        name="Team 1", line=dict(color=palette["team1"], width=2.2),
        connectgaps=False,
        hovertemplate="Frame %{x}<br>Team 1 line: %{y:.1f} m<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dline["x"], y=dline["team2"], mode="lines",
        name="Team 2", line=dict(color=palette["team2"], width=2.2),
        connectgaps=False,
        hovertemplate="Frame %{x}<br>Team 2 line: %{y:.1f} m<extra></extra>",
    ))
    # Reference: midline
    fig.add_hline(y=CENTER_X, line=dict(color=palette["grid"], width=1, dash="dash"))
    layout = _base_layout(palette, height=320)
    layout.update(dict(
        xaxis=dict(title="Frame", gridcolor=palette["grid"],
                   zerolinecolor=palette["grid"]),
        yaxis=dict(title="Defensive line X (m)", range=[-2, PITCH_LENGTH + 2],
                   gridcolor=palette["grid"], zerolinecolor=palette["grid"]),
    ))
    fig.update_layout(layout)
    return fig


# ─── xT pitch-value heatmap ──────────────────────────────────────────────────
def build_xt_heatmap(palette: dict, xt: dict, team_id: int) -> go.Figure:
    """Pitch-value heatmap (danger-weighted ball-possession grid)."""
    if team_id == 0:
        matrix = xt.get("team1_matrix", np.zeros((1, 1)))
        total = xt.get("team1_total_value", 0.0)
        color = palette["team1"]
        name = "Team 1"
    else:
        matrix = xt.get("team2_matrix", np.zeros((1, 1)))
        total = xt.get("team2_total_value", 0.0)
        color = palette["team2"]
        name = "Team 2"
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0 or matrix.max() <= 0:
        return _empty_fig(palette, "No xT data")
    x_edges = xt.get("x_edges")
    y_edges = xt.get("y_edges")
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    z = matrix.T
    max_v = float(z.max())
    scale = [
        [0.0, "rgba(0,0,0,0)"],
        [0.2, _alpha(color, 0.35)],
        [0.6, _alpha(color, 0.75)],
        [1.0, color],
    ]
    fig = go.Figure(go.Heatmap(
        z=z.tolist(), x=x_centers.tolist(), y=y_centers.tolist(),
        zsmooth="best", zmin=0, zmax=max_v,
        colorscale=scale, showscale=True,
        hovertemplate=f"<b>{name} xT</b><br>x %{{x:.1f}} m · y %{{y:.1f}} m<br>"
                      "Value: %{z:.2f}<extra></extra>",
        colorbar=dict(title=dict(text="xT", font=dict(color=palette["text_dim"])),
                      tickfont=dict(color=palette["text_dim"])),
    ))
    layout = _pitch_axes(palette, height=440)
    layout["showlegend"] = False
    fig.update_layout(layout)
    fig.add_annotation(
        x=PITCH_LENGTH / 2, y=PITCH_WIDTH + 1.5, showarrow=False,
        text=f"<b>{name}</b> · total xT {total:.1f}",
        font=dict(color=palette["text"], size=12), xanchor="center",
    )
    return fig


# ─── Voronoi pitch control ───────────────────────────────────────────────────
def build_voronoi_control(palette: dict, voronoi: dict) -> go.Figure:
    """Per-cell signed control heatmap (-1 = Team 2, +1 = Team 1)."""
    matrix = np.asarray(voronoi.get("matrix", np.zeros((1, 1))), dtype=np.float32)
    if matrix.size == 0:
        return _empty_fig(palette, "No Voronoi data")
    x_edges = voronoi["x_edges"]; y_edges = voronoi["y_edges"]
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    z = matrix.T
    scale = [
        [0.0, palette["team2"]],
        [0.5, palette["panel_alt"]],
        [1.0, palette["team1"]],
    ]
    fig = go.Figure(go.Heatmap(
        z=z.tolist(), x=x_centers.tolist(), y=y_centers.tolist(),
        zmin=-1, zmax=1, zsmooth="best",
        colorscale=scale, showscale=True,
        hovertemplate="x %{x:.1f} m · y %{y:.1f} m<br>Control: %{z:.2f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Control", font=dict(color=palette["text_dim"])),
            tickfont=dict(color=palette["text_dim"]),
            tickvals=[-0.8, 0, 0.8], ticktext=["T2", "—", "T1"],
        ),
    ))
    layout = _pitch_axes(palette, height=440)
    layout["showlegend"] = False
    fig.update_layout(layout)
    fig.add_annotation(
        x=PITCH_LENGTH / 2, y=PITCH_WIDTH + 1.5, showarrow=False,
        text=(f"<b>Pitch control</b> · "
              f"T1 {voronoi.get('team1_pct', 0):.0f}% · "
              f"T2 {voronoi.get('team2_pct', 0):.0f}% · "
              f"Contested {voronoi.get('contested_pct', 0):.0f}%"),
        font=dict(color=palette["text"], size=12), xanchor="center",
    )
    return fig


# ─── Possession chain length histogram ───────────────────────────────────────
def build_chain_length_histogram(palette: dict, chains: dict) -> go.Figure:
    """Histogram of possession chain lengths per team (in frames, ~30-frame buckets)."""
    t1_hist = (chains.get("team1") or {}).get("histogram", {})
    t2_hist = (chains.get("team2") or {}).get("histogram", {})
    if not t1_hist and not t2_hist:
        return _empty_fig(palette, "No possession chains")
    # Union of x buckets
    xs = sorted(set(t1_hist) | set(t2_hist))
    if not xs:
        return _empty_fig(palette, "No possession chains")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"{x}+" for x in xs], y=[t1_hist.get(x, 0) for x in xs],
        name="Team 1", marker=dict(color=palette["team1"]),
        hovertemplate="<b>Team 1</b><br>%{x} frames<br>Chains: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=[f"{x}+" for x in xs], y=[t2_hist.get(x, 0) for x in xs],
        name="Team 2", marker=dict(color=palette["team2"]),
        hovertemplate="<b>Team 2</b><br>%{x} frames<br>Chains: %{y}<extra></extra>",
    ))
    layout = _base_layout(palette, height=320)
    layout.update(dict(
        barmode="group",
        xaxis=dict(title="Chain length (frames)", gridcolor=palette["grid"]),
        yaxis=dict(title="Number of chains", gridcolor=palette["grid"],
                   rangemode="tozero"),
    ))
    fig.update_layout(layout)
    return fig
