"""
ui_theme.py — Dark / light theme, custom CSS, and HTML widgets
(ring progress, hero possession bar, KPI cards) for the Streamlit app.
"""
from __future__ import annotations
import math


# ─── Theme palettes ──────────────────────────────────────────────────────────
DARK = {
    "bg":          "#0a0e17",
    "bg_alt":      "#0f1420",
    "panel":       "#141b2c",
    "panel_alt":   "#1a2238",
    "border":      "#2a3552",
    "text":        "#e9edf7",
    "text_dim":    "#8b94ad",
    "accent_1":    "#00e5ff",     # cyan
    "accent_2":    "#7c4dff",     # violet
    "team1":       "#3aa1ff",     # blue
    "team2":       "#ff4d6d",     # red
    "good":        "#00e6a8",
    "warn":        "#ffaa33",
    "bad":         "#ff5577",
    "grid":        "rgba(255,255,255,0.06)",
    "pitch_bg":    "#0f2a18",
    "pitch_line":  "rgba(255,255,255,0.55)",
}

LIGHT = {
    "bg":          "#f4f6fb",
    "bg_alt":      "#ffffff",
    "panel":       "#ffffff",
    "panel_alt":   "#f7f9fd",
    "border":      "#dfe5ee",
    "text":        "#0e1424",
    "text_dim":    "#5a657a",
    "accent_1":    "#0b7cff",
    "accent_2":    "#7a3df0",
    "team1":       "#1a73e8",
    "team2":       "#d63354",
    "good":        "#0fa66a",
    "warn":        "#cc7a00",
    "bad":         "#c8333d",
    "grid":        "rgba(20,30,50,0.07)",
    "pitch_bg":    "#e8f5e9",
    "pitch_line":  "rgba(20,40,30,0.55)",
}


def palette(theme: str) -> dict:
    return DARK if theme == "dark" else LIGHT


# ─── Global CSS ──────────────────────────────────────────────────────────────
def inject_css(theme: str) -> str:
    p = palette(theme)
    return f"""
    <style>
      :root {{
        --ps-bg:         {p['bg']};
        --ps-bg-alt:     {p['bg_alt']};
        --ps-panel:      {p['panel']};
        --ps-panel-alt:  {p['panel_alt']};
        --ps-border:     {p['border']};
        --ps-text:       {p['text']};
        --ps-text-dim:   {p['text_dim']};
        --ps-accent-1:   {p['accent_1']};
        --ps-accent-2:   {p['accent_2']};
        --ps-team1:      {p['team1']};
        --ps-team2:      {p['team2']};
        --ps-good:       {p['good']};
        --ps-warn:       {p['warn']};
        --ps-bad:        {p['bad']};
      }}

      /* Base Streamlit overrides */
      html, body, .stApp, [data-testid="stAppViewContainer"] {{
        background: var(--ps-bg) !important;
        color: var(--ps-text) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      }}
      [data-testid="stHeader"], [data-testid="stToolbar"] {{ background: transparent !important; }}
      .block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px; }}

      /* Sidebar */
      [data-testid="stSidebar"] {{
        background: var(--ps-bg-alt) !important;
        border-right: 1px solid var(--ps-border);
      }}
      [data-testid="stSidebar"] * {{ color: var(--ps-text) !important; }}

      /* Headings & text */
      h1, h2, h3, h4 {{ color: var(--ps-text) !important; font-weight: 700; letter-spacing: -0.01em; }}
      p, label, span, div {{ color: var(--ps-text); }}
      small {{ color: var(--ps-text-dim) !important; }}

      /* Inputs */
      .stSelectbox > div > div,
      .stNumberInput input,
      .stTextInput input {{
        background: var(--ps-panel) !important;
        border: 1px solid var(--ps-border) !important;
        color: var(--ps-text) !important;
        border-radius: 10px !important;
      }}
      .stSelectbox div[data-baseweb="select"] > div {{
        background: var(--ps-panel) !important;
        color: var(--ps-text) !important;
      }}

      /* Buttons */
      .stButton > button {{
        background: linear-gradient(135deg, var(--ps-accent-1), var(--ps-accent-2));
        color: #fff !important;
        font-weight: 700;
        border: 0;
        border-radius: 12px;
        padding: 0.65rem 1.4rem;
        box-shadow: 0 10px 30px -10px var(--ps-accent-2);
        transition: transform 120ms ease, box-shadow 200ms ease, filter 200ms ease;
      }}
      .stButton > button:hover {{
        transform: translateY(-1px);
        filter: brightness(1.07);
        box-shadow: 0 14px 36px -8px var(--ps-accent-1);
      }}
      .stButton > button:disabled {{
        opacity: 0.5;
        cursor: not-allowed;
        background: var(--ps-panel-alt) !important;
        color: var(--ps-text-dim) !important;
        box-shadow: none;
      }}

      /* Tabs */
      .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: var(--ps-panel-alt);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid var(--ps-border);
      }}
      .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        color: var(--ps-text-dim) !important;
        border-radius: 10px !important;
        padding: 8px 18px !important;
        font-weight: 600 !important;
        border: none !important;
      }}
      .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, var(--ps-accent-1), var(--ps-accent-2)) !important;
        color: #fff !important;
      }}

      /* Cards */
      .ps-card {{
        background: linear-gradient(160deg, var(--ps-panel), var(--ps-panel-alt));
        border: 1px solid var(--ps-border);
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px -20px rgba(0,0,0,0.5);
      }}
      .ps-card__title {{
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: -0.01em;
        margin: 0 0 4px 0;
      }}
      .ps-card__sub {{
        font-size: 0.83rem;
        color: var(--ps-text-dim);
        margin: 0;
      }}

      /* Badge */
      .ps-badge {{
        display: inline-flex; align-items: center; gap: 6px;
        background: var(--ps-panel-alt);
        border: 1px solid var(--ps-border);
        padding: 4px 10px; border-radius: 999px;
        font-size: 0.78rem; color: var(--ps-text-dim);
        font-weight: 600;
      }}
      .ps-badge .dot {{
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--ps-good);
        box-shadow: 0 0 8px var(--ps-good);
      }}
      .ps-badge.bad .dot {{ background: var(--ps-bad); box-shadow: 0 0 8px var(--ps-bad); }}

      /* Model status row */
      .ps-status-row {{
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 10px; border-radius: 10px;
        background: var(--ps-panel-alt);
        margin-bottom: 6px;
      }}
      .ps-status-row__name {{
        font-weight: 600; text-transform: capitalize;
      }}
      .ps-status-row__state {{
        font-size: 0.82rem; font-weight: 700; letter-spacing: 0.04em;
      }}
      .ok  {{ color: var(--ps-good); }}
      .err {{ color: var(--ps-bad); }}

      /* KPI cards */
      .ps-kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 14px;
        margin: 4px 0 22px 0;
      }}
      .ps-kpi {{
        background: linear-gradient(150deg, var(--ps-panel), var(--ps-panel-alt));
        border: 1px solid var(--ps-border);
        border-radius: 14px;
        padding: 14px 16px;
        position: relative;
        overflow: hidden;
      }}
      .ps-kpi::after {{
        content: "";
        position: absolute; inset: 0;
        background: radial-gradient(circle at 100% 0%, var(--ps-accent-1) 0%, transparent 40%);
        opacity: 0.12; pointer-events: none;
      }}
      .ps-kpi__label {{
        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em;
        color: var(--ps-text-dim); font-weight: 700;
      }}
      .ps-kpi__value {{
        font-size: 1.7rem; font-weight: 800; margin-top: 4px;
        font-feature-settings: "tnum";
      }}
      .ps-kpi__hint {{
        font-size: 0.75rem; color: var(--ps-text-dim); margin-top: 2px;
      }}

      /* Hero possession bar */
      .ps-hero {{
        display: grid;
        grid-template-columns: 1fr 2.2fr 1fr;
        gap: 18px;
        align-items: center;
        background: linear-gradient(160deg, var(--ps-panel), var(--ps-panel-alt));
        border: 1px solid var(--ps-border);
        border-radius: 18px;
        padding: 22px 26px;
        margin: 4px 0 22px 0;
      }}
      .ps-hero__team {{ text-align: center; }}
      .ps-hero__crest {{
        width: 64px; height: 64px; margin: 0 auto 8px;
        border-radius: 16px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.6rem; font-weight: 800;
        background: linear-gradient(135deg, var(--ps-team1), color-mix(in srgb, var(--ps-team1) 60%, black));
        color: white;
        box-shadow: 0 12px 30px -10px var(--ps-team1);
      }}
      .ps-hero__team--away .ps-hero__crest {{
        background: linear-gradient(135deg, var(--ps-team2), color-mix(in srgb, var(--ps-team2) 60%, black));
        box-shadow: 0 12px 30px -10px var(--ps-team2);
      }}
      .ps-hero__name {{ font-weight: 700; font-size: 1.05rem; }}
      .ps-hero__sub  {{ font-size: 0.78rem; color: var(--ps-text-dim); }}
      .ps-hero__poss {{ display: flex; flex-direction: column; gap: 10px; }}
      .ps-poss-bar {{
        height: 16px; border-radius: 999px; overflow: hidden;
        display: flex; background: var(--ps-bg-alt);
        border: 1px solid var(--ps-border);
      }}
      .ps-poss-bar__fill {{ height: 100%; transition: width 400ms ease; }}
      .ps-poss-bar__home {{ background: linear-gradient(90deg, var(--ps-team1), color-mix(in srgb, var(--ps-team1) 60%, white)); }}
      .ps-poss-bar__neutral {{ background: var(--ps-panel-alt); }}
      .ps-poss-bar__away {{ background: linear-gradient(90deg, color-mix(in srgb, var(--ps-team2) 60%, white), var(--ps-team2)); }}
      .ps-poss-labels {{
        display: flex; justify-content: space-between; align-items: center;
        font-weight: 700; font-size: 0.95rem;
      }}
      .ps-poss-labels__mid {{ font-size: 0.74rem; color: var(--ps-text-dim); letter-spacing: 0.18em; text-transform: uppercase; font-weight: 700; }}

      /* Ring progress */
      .ps-ring-wrap {{
        display: grid;
        grid-template-columns: 230px 1fr;
        gap: 22px;
        align-items: center;
        background: linear-gradient(160deg, var(--ps-panel), var(--ps-panel-alt));
        border: 1px solid var(--ps-border);
        border-radius: 18px;
        padding: 18px 22px;
        margin: 8px 0 18px 0;
      }}
      .ps-ring {{
        position: relative;
        width: 200px; height: 200px;
      }}
      .ps-ring svg {{ width: 100%; height: 100%; transform: rotate(-90deg); }}
      .ps-ring__bg {{
        fill: none; stroke: var(--ps-border); stroke-width: 14;
      }}
      .ps-ring__fg {{
        fill: none; stroke: url(#psRingGrad); stroke-width: 14;
        stroke-linecap: round;
        transition: stroke-dashoffset 320ms cubic-bezier(.22,.61,.36,1);
        filter: drop-shadow(0 0 8px rgba(0, 229, 255, 0.35));
      }}
      .ps-ring__pct {{
        position: absolute; inset: 0;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
      }}
      .ps-ring__pct-big {{
        font-size: 2.4rem; font-weight: 800; line-height: 1;
        background: linear-gradient(135deg, var(--ps-accent-1), var(--ps-accent-2));
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
      }}
      .ps-ring__pct-sub {{
        font-size: 0.78rem; color: var(--ps-text-dim);
        margin-top: 6px; letter-spacing: 0.05em;
      }}
      .ps-ring-stats {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }}
      .ps-ring-stat {{
        background: var(--ps-bg-alt);
        border: 1px solid var(--ps-border);
        border-radius: 12px;
        padding: 12px 14px;
      }}
      .ps-ring-stat__label {{
        text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.12em;
        color: var(--ps-text-dim); font-weight: 700;
      }}
      .ps-ring-stat__value {{
        font-size: 1.25rem; font-weight: 800; margin-top: 2px;
        font-feature-settings: "tnum";
      }}

      /* Zone summary chip */
      .ps-chip {{
        display: inline-flex; gap: 6px;
        background: var(--ps-panel-alt);
        border: 1px solid var(--ps-border);
        padding: 4px 10px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600;
        color: var(--ps-text-dim);
      }}

      /* Footer */
      .ps-footer {{
        text-align: center; padding: 14px 0 4px;
        color: var(--ps-text-dim); font-size: 0.78rem;
        border-top: 1px solid var(--ps-border); margin-top: 18px;
      }}

      /* Streamlit progress bar override (we hide it; ring replaces it) */
      [data-testid="stProgress"] > div > div {{
        background: linear-gradient(90deg, var(--ps-accent-1), var(--ps-accent-2)) !important;
      }}

      /* Hide Streamlit default footer/menu for a cleaner pro look */
      footer {{ visibility: hidden; }}
    </style>
    """


# ─── Widgets ─────────────────────────────────────────────────────────────────
def ring_html(pct: float, label: str = "Processing", sublabel: str = "",
              stat_pairs: list[tuple[str, str]] | None = None) -> str:
    """Render an SVG progress ring with center % + optional side stats."""
    pct = max(0.0, min(100.0, float(pct)))
    radius = 96
    circumference = 2 * math.pi * radius
    offset = circumference * (1 - pct / 100.0)
    stat_pairs = stat_pairs or []
    stats_html = "".join(
        f'<div class="ps-ring-stat"><div class="ps-ring-stat__label">{k}</div>'
        f'<div class="ps-ring-stat__value">{v}</div></div>'
        for k, v in stat_pairs
    )
    return f"""
    <div class="ps-ring-wrap">
      <div class="ps-ring">
        <svg viewBox="0 0 220 220">
          <defs>
            <linearGradient id="psRingGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%"  stop-color="var(--ps-accent-1)"/>
              <stop offset="100%" stop-color="var(--ps-accent-2)"/>
            </linearGradient>
          </defs>
          <circle class="ps-ring__bg" cx="110" cy="110" r="{radius}"/>
          <circle class="ps-ring__fg" cx="110" cy="110" r="{radius}"
                  stroke-dasharray="{circumference:.2f}"
                  stroke-dashoffset="{offset:.2f}" />
        </svg>
        <div class="ps-ring__pct">
          <div class="ps-ring__pct-big">{pct:.0f}%</div>
          <div class="ps-ring__pct-sub">{label}</div>
        </div>
      </div>
      <div>
        <div class="ps-card__sub" style="margin-bottom:10px;">{sublabel}</div>
        <div class="ps-ring-stats">{stats_html}</div>
      </div>
    </div>
    """


def kpi_card(label: str, value: str, hint: str = "") -> str:
    return (
        f'<div class="ps-kpi">'
        f'<div class="ps-kpi__label">{label}</div>'
        f'<div class="ps-kpi__value">{value}</div>'
        f'<div class="ps-kpi__hint">{hint}</div>'
        f'</div>'
    )


def kpi_grid(items: list[tuple[str, str, str]]) -> str:
    return '<div class="ps-kpi-grid">' + "".join(kpi_card(l, v, h) for l, v, h in items) + "</div>"


def hero_possession(t1_label: str, t2_label: str,
                    t1_pct: float, t2_pct: float,
                    t1_frames: int, t2_frames: int) -> str:
    contested = max(0.0, 100.0 - t1_pct - t2_pct)
    return f"""
    <div class="ps-hero">
      <div class="ps-hero__team team--home">
        <div class="ps-hero__crest">{t1_label[:1].upper()}</div>
        <div class="ps-hero__name">{t1_label}</div>
        <div class="ps-hero__sub">{t1_frames} frames</div>
      </div>
      <div class="ps-hero__poss">
        <div class="ps-poss-bar">
          <div class="ps-poss-bar__fill ps-poss-bar__home"    style="width:{t1_pct}%"></div>
          <div class="ps-poss-bar__fill ps-poss-bar__neutral" style="width:{contested}%"></div>
          <div class="ps-poss-bar__fill ps-poss-bar__away"    style="width:{t2_pct}%"></div>
        </div>
        <div class="ps-poss-labels">
          <span>{t1_pct:.0f}%</span>
          <span class="ps-poss-labels__mid">Possession</span>
          <span>{t2_pct:.0f}%</span>
        </div>
      </div>
      <div class="ps-hero__team ps-hero__team--away">
        <div class="ps-hero__crest">{t2_label[:1].upper()}</div>
        <div class="ps-hero__name">{t2_label}</div>
        <div class="ps-hero__sub">{t2_frames} frames</div>
      </div>
    </div>
    """


def card_open(title: str, sub: str = "", chip: str = "") -> str:
    chip_html = f'<span class="ps-chip">{chip}</span>' if chip else ""
    return (
        f'<div class="ps-card">'
        f'<div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:10px;">'
        f'<div><div class="ps-card__title">{title}</div>'
        f'<div class="ps-card__sub">{sub}</div></div>'
        f'{chip_html}'
        f'</div>'
    )


def card_close() -> str:
    return "</div>"
