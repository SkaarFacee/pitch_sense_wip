"""
charts.py — Plotly figure builders for the PitchSense Streamlit dashboard.

All builders accept the theme palette (dict) so dark/light modes look consistent.

Charts produced:
    • build_possession_donut       — donut % chart for nearest-player possession
    • build_possession_timeline    — area chart of rolling possession
    • build_team_radar             — 6-axis tactical DNA radar
    • build_territory_grid         — 3×3 dominance heatmap (with hover details)
    • build_density_heatmap        — Player density heatmap on a pitch (interactive tooltips)
    • build_formation_scatter      — Combined positioning scatter on a pitch
    • build_region_pie             — Pie of pitch region detection frequency
    • build_region_bar             — Horizontal bar of region detection counts
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

    Track-aware: uses the canonical per-track team from the registry, so a
    single-frame misclassification of one player cannot flip the
    possession assignment for that frame.
    """
    from game_analyzer import GameAnalyzer
    registry = GameAnalyzer.build_registry(game_data)
    buf, t1_series, t2_series, frames_axis = [], [], [], []
    for entry in game_data:
        ball = entry.get("ball_position")
        team = -1
        if ball is not None:
            ball_arr = np.asarray(ball, dtype=np.float32).reshape(1, 2)
            team = GameAnalyzer._nearest_team_to_ball(entry, ball_arr, registry)
            if team is not None:
                team = int(team)
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
    """6-axis tactical DNA radar."""
    def clip(v, lo=0.0, hi=100.0):
        return float(max(lo, min(hi, v)))

    t1_attack = clip(formation["team1_avg_center"][0] / PITCH_LENGTH * 100) if formation.get("team1_avg_center") else 50.0
    t2_attack = clip(formation["team2_avg_center"][0] / PITCH_LENGTH * 100) if formation.get("team2_avg_center") else 50.0
    t1_def = clip(100 - (formation["team1_defensive_depth"] / PITCH_LENGTH * 100))
    t2_def = clip(100 - (formation["team2_defensive_depth"] / PITCH_LENGTH * 100))
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

    categories = ["Attacking Intent", "Defensive Depth", "Compactness", "Width", "Possession", "Tempo"]
    t1_vals = [t1_attack, t1_def, t1_comp, t1_width, t1_poss, clip(t1_tempo)]
    t2_vals = [t2_attack, t2_def, t2_comp, t2_width, t2_poss, clip(t2_tempo)]

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
