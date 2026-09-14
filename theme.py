# Palette
BG        = "#0E1117"
SURFACE   = "#161B22"
BORDER    = "#30363D"
TEXT      = "#E6EDF3"
MUTED     = "#8B949E"
ACCENT    = "#4F8BF9"
POSITIVE  = "#3FB950"
NEGATIVE  = "#F85149"
WARNING   = "#D29922"

# Chart-specific
GRID      = "rgba(139, 148, 158, 0.15)"
TRANSPARENT = "rgba(0,0,0,0)"

# Plotly config passed to st.plotly_chart
PLOTLY_CONFIG = {"displayModeBar": False, "displaylogo": False}

# Factor colors (used by rolling betas chart)
FACTOR_COLORS = {
    "Mkt-RF": ACCENT,
    "SMB":    POSITIVE,
    "HML":    WARNING,
    "RMW":    "#A371F7",
    "CMA":    "#F778BA",
    "Alpha":  MUTED,
}