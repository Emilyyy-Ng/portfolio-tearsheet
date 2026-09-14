import plotly.graph_objects as go
import pandas as pd
import numpy as np
import theme
from analytics import AnalysisResult, compute_wealth_index

# ========== Shared styling ==========
def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the app's dark theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=theme.TRANSPARENT,   # inherit Streamlit background
        plot_bgcolor=theme.TRANSPARENT,
        font=dict(
            family="Inter, system-ui, -apple-system, sans-serif",
            size=12,
            color=theme.TEXT,
        ),
        margin=dict(l=40, r=20, t=20, b=40),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=theme.SURFACE,
            bordercolor=theme.BORDER,
            font=dict(size=12, color=theme.TEXT),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            bgcolor=theme.TRANSPARENT,
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=False,
        tickcolor=theme.MUTED,
        tickfont=dict(color=theme.MUTED),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=theme.GRID,
        gridwidth=1,
        zeroline=False,
        showline=False,
        tickcolor=theme.MUTED,
        tickfont=dict(color=theme.MUTED),
    )
    return fig


# ========== Charts ==========
def build_equity_curve(result: AnalysisResult) -> go.Figure:
    """Growth of $1 — portfolio vs benchmark."""
    wealth = result.portfolio_stats.wealth_index
    # Prepend anchor at $1
    wealth = pd.concat([pd.Series([1.0], index=[wealth.index[0] - pd.Timedelta(days=1)]), wealth], axis=0)
    bench_returns = result.returns["benchmark_returns"]
    bench_wealth = compute_wealth_index(bench_returns)

    fig = go.Figure()
    # Benchmark first so portfolio renders on top
    fig.add_trace(go.Scatter(
        x=bench_wealth.index, y=bench_wealth.values,
        name="Benchmark",
        line=dict(color=theme.MUTED, width=1.5),
        hovertemplate="Benchmark: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=wealth.index, y=wealth.values,
        name="Portfolio",
        line=dict(color=theme.ACCENT, width=2),
        hovertemplate="Portfolio: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(yaxis_title="Growth of $1", height=520)
    return _apply_theme(fig)

def build_monthly_heatmap(result: AnalysisResult) -> go.Figure:
    """Monthly returns heatmap: rows = year, columns = month."""
    monthly = (1 + result.returns["portfolio_returns"]).resample("ME").prod() - 1
    df = monthly.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month

    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot = pivot.reindex(columns=range(1, 13))   # ensure Jan..Dec order
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    # Color scale: red for losses, green for gains, centered at 0
    zmax = float(np.nanmax(np.abs(pivot.values))) if pivot.notna().any().any() else 0.05

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=month_labels,
        y=pivot.index.astype(str),
        colorscale=[
            [0.0, "#AD0900"],
            [0.5, "#fcffc3"],
            [1.0, "#009E15"],
        ],
        zmid=0, zmin=-zmax, zmax=zmax,
        hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>",
        colorbar=dict(
            title="Return",
            tickformat=".0%",
            outlinewidth=0,
            thickness=12,
        ),
    ))

    # Overlay text with per-cell colors
    xs, ys, texts, colors = [], [], [], []
    for yi, year in enumerate(pivot.index):
        for xi, month in enumerate(pivot.columns):
            v = pivot.iloc[yi, xi]
            if pd.isna(v):
                continue
            xs.append(month_labels[xi])
            ys.append(str(year))
            texts.append(f"{v:.2%}")
            colors.append(theme.TEXT if abs(v) >= 0.5 * zmax else theme.BORDER)

    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="text",
        text=texts,
        textfont=dict(size=12, color=colors),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_layout(height=120 + 40 * len(pivot), yaxis_autorange="reversed")
    return _apply_theme(fig)

def build_drawdown(result: AnalysisResult) -> go.Figure:
    """Underwater drawdown plot."""
    dd = result.portfolio_stats.drawdown_series

    fig = go.Figure(go.Scatter(
        x=dd.index, y=dd.values,
        name="Drawdown",
        fill="tozeroy",
        line=dict(color=theme.NEGATIVE, width=1.5),
        fillcolor="rgba(248, 81, 73, 0.20)",
        hovertemplate="%{y:.1%}<extra></extra>",
    ))
    fig.update_layout(
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        height=400,
    )

    # Annotate max drawdown
    if len(dd) and dd.min() < 0:
        max_dd_date = dd.idxmin()
        max_dd_val = dd.min()
        fig.add_annotation(
            x=max_dd_date, y=max_dd_val,
            text=f"Max DD {max_dd_val:.2%}",
            showarrow=True, arrowhead=0, arrowcolor=theme.MUTED,
            font=dict(color=theme.NEGATIVE, size=12),
            bgcolor=theme.SURFACE, 
            bordercolor=theme.BORDER, 
            borderwidth=1, borderpad=3,
            ax=-80, ay=10,
        )
    return _apply_theme(fig)

def build_returns_histogram(result: AnalysisResult) -> go.Figure:
    """Plot the portfolio return distribution with the 95% VaR marked."""
    returns = result.returns["portfolio_returns"]
    var = result.portfolio_stats.var

    fig = go.Figure(go.Histogram(
        x=returns, 
        nbinsx=min(80, max(20, len(returns)//10)),
        marker=dict(color=theme.MUTED, line=dict(color=theme.BORDER, width=1))
    ))
    if np.isfinite(var):
        fig.add_vline(
            var, 
            line_dash="dash", 
            line_color=theme.NEGATIVE, 
            line_width=1
        )
    fig.add_annotation(
        x=var, yref="paper", y=1.0,
        xanchor="right",
        yanchor="top",
        text=f"VaR 95%: {var:.2%}",
        showarrow=False,
        font=dict(color=theme.NEGATIVE, size=12),
        bgcolor=theme.SURFACE, 
        bordercolor=theme.BORDER, 
        borderwidth=1,
        borderpad=3,
        xshift=-8
    )
    fig.update_layout(
        xaxis_title="Daily Return",
        yaxis_title="Frequency",
        height=400
    )
    return _apply_theme(fig)

def build_rolling_sharpe(result: AnalysisResult) -> go.Figure:
    """Rolling Sharpe ratio."""
    rolling = result.rolling_sharpe.dropna()
    overall = result.portfolio_stats.sharpe

    fig = go.Figure(go.Scatter(
        x=rolling.index, y=rolling.values,
        name="Rolling Sharpe",
        fill="tozeroy",
        line=dict(color=theme.ACCENT, width=1.5),
        fillcolor="rgba(79, 139, 249, 0.2)",
        hovertemplate="%{y:.2f}<extra></extra>",
    ))
    fig.add_hline(0, line_dash="dot", line_color=theme.MUTED, line_width=1)
    fig.add_hline(
        overall,
        line_dash="solid",
        line_color=theme.WARNING,
        line_width=1.5,
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=1.0, y=1.0,
        xanchor="right", yanchor="middle",
        text=f"Overall Sharpe: {overall:.2f}",
        showarrow=False,
        font=dict(color=theme.WARNING, size=12),
        bgcolor=theme.SURFACE,
        bordercolor=theme.BORDER,
        borderwidth=1,
        borderpad=3,
    )
    fig.update_layout(
        yaxis_title="Sharpe ratio",
        yaxis_tickformat=".0f",
        height=400,
    )
    return _apply_theme(fig)


def build_rolling_betas(result: AnalysisResult) -> go.Figure:
    """Rolling factor exposures."""
    betas = result.rolling_betas

    fig = go.Figure()
    for col in betas.columns:
        fig.add_trace(go.Scatter(
            x=betas.index, y=betas[col],
            name=col,
            line=dict(width=1.5, color=theme.FACTOR_COLORS.get(col, theme.MUTED)),
            hovertemplate=f"{col}: %{{y:.2f}}<extra></extra>",
        ))
    fig.add_hline(0, line_dash="dot", line_color=theme.MUTED, line_width=1)
    fig.update_layout(yaxis_title="Beta", height=520)
    return _apply_theme(fig)

def build_rolling_volatility(result: AnalysisResult) -> go.Figure:
    """Rolling volatility over time."""
    vol = result.rolling_volatility
    overall = result.portfolio_stats.ann_vol

    fig = go.Figure(go.Scatter(
        x=vol.index, y=vol.values,
        line=dict(width=1.5, color=theme.WARNING),
        hovertemplate="%{y:.2%}<extra></extra>"
    ))
    fig.update_layout(
        yaxis_title="Volatility", 
        yaxis_tickformat=".0%",
        height=400)
    fig.add_hline(overall, line_dash="solid", line_color=theme.MUTED, line_width=1.5)
    fig.add_annotation(
        xref="paper", yref="paper",
        x=1.0, y=1.0,
        xanchor="right", yanchor="middle",
        text=f"Overall Volatility: {overall:.2%}",
        showarrow=False,
        font=dict(color=theme.MUTED, size=12),
        bgcolor=theme.SURFACE,
        bordercolor=theme.BORDER,
        borderwidth=1,
        borderpad=3,
    )
    return _apply_theme(fig)

def _loadings_fig(loadings: list) -> go.Figure:
    names_map = {
        "const":    "Alpha",
        "Mkt-RF": "Mkt"
    }
    names = [names_map.get(l.name, l.name) for l in loadings]
    coefs = [l.coef for l in loadings]
    colors = [theme.ACCENT if l.significant else theme.MUTED for l in loadings]

    fig = go.Figure(go.Bar(
        x=names, y=coefs,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{c:+.4f}" for c in coefs],
        textposition="outside",
        textfont=dict(color=theme.TEXT, size=12),
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    ))

    ymax = max(coefs + [0])
    ymin = min(coefs + [0])
    span = (ymax - ymin) or 1.0
    fig.update_yaxes(range=[ymin - 0.20 * span, ymax + 0.25 * span])
    fig.update_layout(
        yaxis_title="Loading",
        bargap=0.4,
        height=420
    )
    return _apply_theme(fig)


def build_ff3_factor_loadings(result: AnalysisResult) -> go.Figure:
    return _loadings_fig(result.ff3.loadings)

def build_ff5_factor_loadings(result: AnalysisResult) -> go.Figure:
    return _loadings_fig(result.ff5.loadings)