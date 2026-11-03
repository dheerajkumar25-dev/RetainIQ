"""Design tokens, global CSS and the shared Plotly look."""
import streamlit as st

# ---- colour system: navy surfaces, one teal accent, coral for churn, amber for "medium"
BACKGROUND = "#0B1426"
SURFACE = "#111C33"
SURFACE_RAISED = "#16233F"
BORDER = "rgba(148, 163, 184, 0.16)"
TEXT = "#E4ECF7"
MUTED = "#8FA1BC"
TEAL = "#34D1BF"
CORAL = "#F4727C"
AMBER = "#F2B45A"
BLUE = "#6F8FD9"
GRID = "rgba(148, 163, 184, 0.12)"

STATUS_COLORS = {"Retained": TEAL, "Churned": CORAL}
RISK_COLORS = {"Low": TEAL, "Medium": AMBER, "High": CORAL}
MODEL_COLORS = {"XGBoost": TEAL, "Logistic Regression": BLUE, "Random Forest": AMBER}
FONT_STACK = "Manrope, 'Segoe UI', system-ui, -apple-system, sans-serif"

_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

.stApp {{ background: {BACKGROUND}; font-family: {FONT_STACK}; color: {TEXT}; }}
.stApp button, .stApp input, .stApp textarea, .stApp select {{ font-family: inherit; }}
[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding: 2.4rem 2.6rem 4rem; max-width: 1320px; }}
h1, h2, h3, h4 {{ font-family: {FONT_STACK}; letter-spacing: -0.015em; color: {TEXT}; }}
p, li {{ line-height: 1.6; }}
hr {{ border-color: {BORDER}; }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{ background: #0D172B; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {{
    border-radius: 10px; padding: 0.5rem 0.75rem; margin-bottom: 2px; font-weight: 600; color: {MUTED};
}}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover {{ background: rgba(148,163,184,0.08); color: {TEXT}; }}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {{
    background: rgba(52,209,191,0.12); color: {TEAL};
}}
.riq-sidebar-note {{ font-size: 0.78rem; color: {MUTED}; line-height: 1.55; padding: 0.4rem 0.4rem 0; }}
.riq-sidebar-note b {{ color: {TEXT}; font-weight: 600; }}

/* ---------- page header ---------- */
.riq-header {{ margin-bottom: 1.6rem; }}
.riq-header h1 {{ font-size: 2rem; font-weight: 800; margin: 0 0 0.35rem 0; padding: 0; }}
.riq-header p {{ color: {MUTED}; font-size: 1rem; margin: 0 0 0.8rem 0; max-width: 68ch; }}
.riq-pill {{
    display: inline-block; padding: 0.18rem 0.65rem; margin-right: 0.4rem; border-radius: 999px;
    font-size: 0.76rem; font-weight: 600; color: {MUTED}; border: 1px solid {BORDER}; background: rgba(148,163,184,0.06);
}}
.riq-pill.teal {{ color: {TEAL}; border-color: rgba(52,209,191,0.35); background: rgba(52,209,191,0.08); }}
.riq-pill.amber {{ color: {AMBER}; border-color: rgba(242,180,90,0.4); background: rgba(242,180,90,0.08); }}
.riq-pill.coral {{ color: {CORAL}; border-color: rgba(244,114,124,0.4); background: rgba(244,114,124,0.08); }}

/* ---------- section titles ---------- */
.riq-section {{ margin: 1.9rem 0 0.8rem; }}
.riq-section h3 {{ font-size: 1.12rem; font-weight: 700; margin: 0; padding: 0; }}
.riq-section p {{ color: {MUTED}; font-size: 0.9rem; margin: 0.15rem 0 0; }}

/* ---------- KPI cards ---------- */
.riq-kpi {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 1.05rem 1.2rem 1rem;
    box-shadow: 0 8px 24px rgba(2,6,18,0.28); height: 100%;
}}
.riq-kpi-label {{ color: {MUTED}; font-size: 0.84rem; font-weight: 600; }}
.riq-kpi-value {{ font-size: 1.95rem; font-weight: 800; letter-spacing: -0.02em; margin-top: 0.25rem; font-variant-numeric: tabular-nums; }}
.riq-kpi-caption {{ color: {MUTED}; font-size: 0.78rem; margin-top: 0.15rem; }}
.riq-tone-accent .riq-kpi-value {{ color: {TEAL}; }}
.riq-tone-danger .riq-kpi-value {{ color: {CORAL}; }}

/* ---------- bordered containers act as cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"], [class*="st-key-riq-card-"] {{
    background: {SURFACE}; border-color: {BORDER} !important; border-radius: 14px;
    box-shadow: 0 8px 24px rgba(2,6,18,0.22);
}}

/* ---------- callouts ---------- */
.riq-callout {{
    border: 1px solid {BORDER}; border-left: 3px solid {TEAL}; background: rgba(52,209,191,0.05);
    border-radius: 10px; padding: 0.75rem 1rem; font-size: 0.88rem; color: {MUTED}; margin: 0.6rem 0;
}}
.riq-callout.amber {{ border-left-color: {AMBER}; background: rgba(242,180,90,0.06); }}
.riq-callout b {{ color: {TEXT}; }}

/* ---------- prediction verdict ---------- */
.riq-verdict {{
    border: 1px solid {BORDER}; border-radius: 14px; padding: 1.2rem 1.4rem; background: {SURFACE_RAISED};
    border-top: 3px solid {TEAL};
}}
.riq-verdict.risk-medium {{ border-top-color: {AMBER}; }}
.riq-verdict.risk-high {{ border-top-color: {CORAL}; }}
.riq-verdict-label {{ font-size: 1.7rem; font-weight: 800; letter-spacing: -0.02em; }}
.riq-verdict-sub {{ color: {MUTED}; font-size: 0.92rem; margin-top: 0.2rem; }}
.riq-placeholder {{
    border: 1px dashed rgba(148,163,184,0.3); border-radius: 14px; padding: 2.6rem 1.6rem; text-align: center; color: {MUTED};
}}
.riq-list {{ margin: 0; padding-left: 1.1rem; }}
.riq-list li {{ margin-bottom: 0.45rem; color: {TEXT}; font-size: 0.93rem; }}

/* ---------- widgets ---------- */
button[kind="primary"], [data-testid="stBaseButton-primary"] {{
    background: {TEAL}; color: #06201C; border: none; font-weight: 700; border-radius: 10px;
}}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{ background: #5FE0D1; color: #06201C; }}
button[kind="secondary"], [data-testid="stDownloadButton"] button {{ border-radius: 10px; border-color: {BORDER}; }}
button[data-baseweb="tab"] {{ font-weight: 600; color: {MUTED}; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {TEAL}; }}
[data-baseweb="tab-highlight"] {{ background-color: {TEAL}; }}
[data-testid="stFileUploader"] section {{ background: {SURFACE}; border: 1px dashed rgba(148,163,184,0.35); border-radius: 12px; }}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; overflow: hidden; }}
[data-testid="stExpander"] {{ border-color: {BORDER}; border-radius: 12px; }}

@media (max-width: 768px) {{
    .block-container {{ padding: 1.4rem 1rem 3rem; }}
    .riq-header h1 {{ font-size: 1.6rem; }}
    .riq-kpi-value {{ font-size: 1.6rem; }}
}}
"""


def apply_theme() -> None:
    """Inject the global stylesheet. Called once from app.py on every run."""
    st.session_state["_riq_card_counter"] = 0  # card keys restart at 1 on each script run
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def style_figure(fig, height: int = 340, title: str | None = None, show_legend: bool = True):
    """Apply the shared look to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, color=TEXT, size=12),
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=TEXT)) if title else None,
        margin=dict(l=8, r=12, t=58 if title else 16, b=8),
        height=height,
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0, font=dict(size=11), title_text=""),
        hoverlabel=dict(bgcolor=SURFACE_RAISED, bordercolor="rgba(148,163,184,0.3)", font=dict(family=FONT_STACK, color=TEXT)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED))
    return fig
