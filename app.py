import streamlit as st
import pandas as pd
import traceback
import tempfile, os
from pathlib import Path

from analytics import run_full_analysis, BENCHMARK_OPTIONS, AnalysisResult, BenchmarkFetchError, FactorLoading
from charts import build_equity_curve, build_drawdown, build_returns_histogram, \
                    build_rolling_sharpe, build_rolling_betas, build_ff3_factor_loadings, \
                    build_ff5_factor_loadings, build_monthly_heatmap, build_rolling_volatility
from theme import PLOTLY_CONFIG

# ========== Page Config ==========
st.set_page_config(
    page_title="Portfolio Tearsheet",
    layout="wide",
)

# ========== Custom CSS ==========
_CSS_PATH = Path(__file__).parent / "static" / "style.css"
if _CSS_PATH.is_file():
    st.markdown(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ========== Session State ==========
if "page" not in st.session_state:
    st.session_state.page = "landing"

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "run_params" not in st.session_state:
    st.session_state.run_params = None

# ========== Cache ==========
@st.cache_data(show_spinner="Running analysis... this may take a moment.")
def get_analysis(file_bytes: bytes, file_name: str, benchmark_symbol: str, window: int) -> AnalysisResult:
    # Write to a temp file so load_returns can keep its current signature.
    suffix = os.path.splitext(file_name)[1] or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return run_full_analysis(tmp_path, benchmark_symbol, window)
    finally:
        os.unlink(tmp_path)

# ========== Landing Page ==========
def render_landing() -> None:
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.title("Portfolio Tearsheet")
        st.caption("Upload your portfolio daily returns to get Fama-French factor loadings, rolling attribution, and a complete performance breakdown vs your selected benchmark.")

    with mid:
        with st.container(border=True):
            uploaded_file = st.file_uploader("Returns file (CSV)", type=["csv"])
            st.caption("Analysis runs in-memory. Your data is never stored.")

            c1, c2 = st.columns(2)
            with c1:
                benchmark_choice = st.selectbox(
                    "Benchmark (Defaults to S&P 500)",
                    options=list(BENCHMARK_OPTIONS.keys()),
                    index=0,
                    help="Used for benchmarking performance."
                )
            with c2:
                window_choice = st.selectbox(
                    "Rolling Window",
                    options=[21, 63, 126, 252],
                    index=1,
                    format_func=lambda d: {
                        21: "21d",
                        63: "63d",
                        126: "126d", 
                        252: "252d"
                    }[d]
                )

        benchmark_symbol = BENCHMARK_OPTIONS[benchmark_choice]
        run_clicked = st.button("Generate tearsheet", type="primary", width="stretch")

        with mid:
            st.write("")
            with st.expander("Expected CSV format", expanded=False):
                st.markdown(
                    "A `date` column plus daily simple returns "
                    "(decimals, e.g. `0.001` = 0.1%):"
                )
                st.code(
                    "date,portfolio_return\n"
                    "2023-01-03,-0.0008\n"
                    "2023-01-04,-0.0068\n"
                    "2023-01-05, 0.0120\n"
                    "2023-01-06, 0.0109",
                    language="csv",
                )
                st.caption(
                    "Only the first column after `date` is analysed as the portfolio. "
                    "The benchmark is fetched automatically from your selection above."
                )

    if run_clicked:
        if uploaded_file is None:
            st.error("Upload a returns file to continue.")
            return
        try:
            file_bytes = uploaded_file.getvalue()   # bytes, hashable by st.cache_data
            result = get_analysis(file_bytes, uploaded_file.name, benchmark_symbol, window_choice)
        except BenchmarkFetchError as e:
            st.error(f"Benchmark data unavailable. {e}")
            st.caption("Check your internet connection, or try a different benchmark.")
            return
        except Exception:
            st.error("Couldn't process that file. CSV must have a date index and at least one numeric column.")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
            return

        st.session_state.analysis_result = result
        st.session_state.run_params = {
            "file_name": uploaded_file.name,
            "benchmark_symbol": benchmark_symbol,
            "window": window_choice,
        }
        st.session_state.page = "analysis"
        st.rerun()

# ========== Analysis Page ==========
def render_metrics(result: AnalysisResult) -> None:
    p = result.portfolio_stats
    b = result.benchmark_stats
    capm = result.capm
    beta = next((l.coef for l in capm.loadings if l.name == "Mkt-RF"), float("nan"))
    alpha = capm.alpha_annualized
    bench_sym = (st.session_state.run_params or {}).get("benchmark_symbol", "Benchmark")

    # --- Row 1: Headline ---
    r1 = st.columns(5)
    r1[0].metric("CAGR", f"{p.cagr:.2%}")
    r1[1].metric("Sharpe", f"{p.sharpe:.2f}")
    r1[2].metric("Sortino", f"{p.sortino:.2f}")
    r1[3].metric("Max Drawdown", f"{p.max_drawdown:.2%}")
    r1[4].metric("Volatility", f"{p.ann_vol:.2%}")

    # --- Row 2: vs Benchmark ---
    st.write(f"##### vs {bench_sym}")
    r3 = st.columns(5)
    r3[0].metric("Beta", f"{beta:.2f}")
    r3[1].metric("Alpha (ann.)", f"{alpha:.2%}")
    r3[2].metric("Correlation", f"{p.correlation:.2f}")
    r3[3].metric("Up Capture", f"{p.up_capture:.0%}")
    r3[4].metric("Down Capture", f"{p.down_capture:.0%}")

    # --- Detail table ---
    with st.expander("Detailed statistics"):
        detail = pd.DataFrame({
            "Metric": [
                "Total Return", "CAGR", "Volatility", "Max Drawdown",
                "Avg DD Duration (days)", "Max DD Duration (days)",
                "Sharpe", "Sortino", "Calmar", "Information Ratio",
                "Skewness", "Excess Kurtosis",
                "VaR (95%, 1d)", "CVaR (95%, 1d)",
                "Win Rate", "Profit Factor", "Overall Capture"
            ],
            "Portfolio": [
                f"{p.total_return:.2%}", f"{p.cagr:.2%}", f"{p.ann_vol:.2%}",
                f"{p.max_drawdown:.2%}", f"{p.avg_drawdown_duration:.0f}",
                f"{p.max_drawdown_duration:.0f}",
                f"{p.sharpe:.2f}", f"{p.sortino:.2f}", f"{p.calmar:.2f}",
                f"{p.information_ratio:.2f}",
                f"{p.skewness:.2f}", f"{p.excess_kurtosis:.2f}",
                f"{p.var:.2%}", f"{p.cvar:.2%}",
                f"{p.win_rate:.2%}", f"{p.profit_factor:.2f}",
                f"{p.overall_capture:.2f}"
            ],
            "Benchmark": [
                f"{b.total_return:.2%}", f"{b.cagr:.2%}", f"{b.ann_vol:.2%}",
                f"{b.max_drawdown:.2%}", f"{b.avg_drawdown_duration:.0f}",
                f"{b.max_drawdown_duration:.0f}",
                f"{b.sharpe:.2f}", f"{b.sortino:.2f}", f"{b.calmar:.2f}",
                "—",  # IR is undefined for the benchmark itself
                f"{b.skewness:.2f}", f"{b.excess_kurtosis:.2f}",
                f"{b.var:.2%}", f"{b.cvar:.2%}",
                f"{b.win_rate:.2%}", f"{b.profit_factor:.2f}",
                "—"   # Overall Capture is undefined for the benchmark
            ],
        })
        st.dataframe(detail, hide_index=True, width="stretch")

def render_performance(result: AnalysisResult) -> None:
    equity_curve = build_equity_curve(result)
    heatmap = build_monthly_heatmap(result)

    st.write("##### Equity Curve")
    st.plotly_chart(equity_curve, width="stretch", config=PLOTLY_CONFIG)

    st.write("##### Monthly Returns Heatmap")
    st.plotly_chart(heatmap, width="stretch", config=PLOTLY_CONFIG)

def render_risk(result: AnalysisResult) -> None:
    params = st.session_state.run_params or {}
    window = params.get("window", "—")

    c1, c2 = st.columns(2)
    with c1:
        st.write("##### Rolling Sharpe Ratio")
        if result.rolling_sharpe.dropna().empty:
            st.info("Not enough data for this rolling window. Try a shorter window.")
        else:
            st.plotly_chart(build_rolling_sharpe(result), width="stretch", config=PLOTLY_CONFIG)

        st.write("##### Underwater Plot — Drawdown from Peak")
        st.plotly_chart(build_drawdown(result), width="stretch", config=PLOTLY_CONFIG)

    with c2:
        st.write("##### Rolling Volatility")
        if result.rolling_volatility.dropna().empty:
            st.info("Not enough data for this rolling window. Try a shorter window.")
        else:
            st.plotly_chart(build_rolling_volatility(result), width="stretch", config=PLOTLY_CONFIG)

        st.write("##### Return Distribution")
        st.plotly_chart(build_returns_histogram(result), width="stretch", config=PLOTLY_CONFIG)

    st.write("##### Rolling Fama-French Factor Betas")
    if result.rolling_betas.dropna(how="all").empty:
        st.info("Not enough data for this rolling window. Try a shorter window.")
    else:
        st.plotly_chart(build_rolling_betas(result), width="stretch", config=PLOTLY_CONFIG)

# ========== Helper ==========
def _loadings_to_df(loadings: list[FactorLoading]) -> pd.DataFrame:
    names_map = {
        "const":    "Alpha",
        "Mkt-RF": "Mkt"
    }
    return pd.DataFrame({
        "Factor": [names_map.get(l.name, l.name) for l in loadings],
        "Coef": [f"{l.coef:+.4f}" for l in loadings],
        "t-stat": [f"{l.tstat:.2f}" for l in loadings],
        "p-value": [f"{l.pvalue:.4f}" for l in loadings],
        "Significant": ["✓" if l.significant else "⛌" for l in loadings],
    })

def render_factors(result: AnalysisResult) -> None:
    ff3_factor_loadings = build_ff3_factor_loadings(result)
    ff5_factor_loadings = build_ff5_factor_loadings(result)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.write("##### Fama-French 3-Factor Loadings")
        st.plotly_chart(ff3_factor_loadings, width="stretch", config=PLOTLY_CONFIG)
    with r1c2:
        st.write("##### Fama-French 5-Factor Loadings")
        st.plotly_chart(ff5_factor_loadings, width="stretch", config=PLOTLY_CONFIG)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.dataframe(_loadings_to_df(result.ff3.loadings), hide_index=True, width="stretch")
    with r2c2:
        st.dataframe(_loadings_to_df(result.ff5.loadings), hide_index=True, width="stretch")

def render_analysis() -> None:
    result = st.session_state.analysis_result
    params = st.session_state.run_params or {}

    if result is None:
        st.warning("Session expired — starting a new analysis.")
        st.session_state.page = "landing"
        st.rerun()
        return
    
    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    with c1:
        st.write("## Portfolio Tearsheet")
        st.caption(
            f"`{params.get('file_name', '—')}` · "
            f"Benchmark: `{params.get('benchmark_symbol', '—')}` · "
            f"Window: `{params.get('window', '—')}d` · "
            f"Period: `{result.start_date} → {result.end_date}` · "
            f"Observations: `{len(result.returns)} days`" 
        )
    with c2:
        if st.button("← New analysis", key="home_btn", width="stretch"):
            st.session_state.page = "landing"
            st.session_state.analysis_result = None
            st.session_state.run_params = None
            st.rerun()

    st.space("small")
    with st.container(border=True):
        render_metrics(result)

    st.space("small")
    tab_perf, tab_risk, tab_factors = st.tabs(["Performance", "Risk", "Factors"])
    with tab_perf:
        render_performance(result)
    with tab_risk:
        render_risk(result)
    with tab_factors:
        render_factors(result)

# ========== Router ==========
def main() -> None:
    if st.session_state.page == "landing":
        render_landing()
    else:
        render_analysis()

if __name__ == "__main__":
    main()

