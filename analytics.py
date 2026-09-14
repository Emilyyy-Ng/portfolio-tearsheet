from dataclasses import dataclass
import pandas as pd
import numpy as np
import statsmodels.api as sm
from fama_french import get_ff5
import pandas_datareader as pdr
from typing import List, cast
import yfinance as yf
from datetime import datetime

# ========== Constants ==========
TRADING_DAYS = 252
BENCHMARK_OPTIONS = {
    "S&P 500": "SPY",
    "NASDAQ-100": "QQQ",
    "Dow Jones": "DIA",
    "Russell 2000": "IWM",
    "Total Stock Market": "VTI"
}

class BenchmarkFetchError(RuntimeError):
    """Raised when benchmark data cannot be retrieved."""

# ========== Data Loading ==========
def load_returns(source) -> pd.DataFrame:
    """Load a returns CSV into a single-column DataFrame indexed by date.

    Expects the first column to be a date index and the first data column
    to be the returns series. Values with |r| > 1 are treated as percentages
    and divided by 100. The first column is renamed to ``portfolio_returns``.

    Args:
        source: Path to a CSV file. May be a tempfile written by the caller.

    Returns:
        DataFrame with one column, ``portfolio_returns``, on a DatetimeIndex.

    Raises:
        FileNotFoundError: If ``source`` does not exist.
        ValueError: If the file is empty, has no data columns, or the
            returns column is entirely NaN.
    """
    try:
        with open(source, "r") as f:
            ret_df = pd.read_csv(f, index_col=0)
        ret_df.index = pd.to_datetime(ret_df.index, errors="raise")
    except FileNotFoundError:
        raise FileNotFoundError(f"The csv file {source} does not exist.")

    if ret_df.empty or ret_df.shape[1] == 0:
        raise ValueError("CSV must contain a returns column")

    ret_df = ret_df.iloc[:, [0]]
    ret_df.columns = ["portfolio_returns"]

    if ret_df["portfolio_returns"].isna().all():
        raise ValueError("CSV returns column is entirely empty")

    ret_df["portfolio_returns"] = pd.to_numeric(ret_df["portfolio_returns"], errors="coerce")

    if ret_df["portfolio_returns"].abs().max() > 1:
        ret_df["portfolio_returns"] /= 100

    return ret_df 

def fetch_benchmark(start_date, end_date, benchmark_symbol: str = "SPY") -> pd.Series:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if start_ts >= end_ts:
        raise ValueError("Invalid input — start date must be before end date.")

    benchmark_symbol = benchmark_symbol.upper()
    try:
        df = yf.download(
            benchmark_symbol,
            start=start_ts.strftime("%Y-%m-%d"),
            end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), # yfinance treats end as exclusive
            interval="1d",
            progress=False,
        )
    except Exception as e:
        raise BenchmarkFetchError(f"Could not fetch {benchmark_symbol} from Yahoo Finance: {e}") from e

    if df.empty:
        raise BenchmarkFetchError(
            f"No price data found for {benchmark_symbol} from {start_ts.date()} to {end_ts.date()}. "
            f"The ticker may be delisted or the date range may be outside Yahoo's coverage."
        )

    close_series: pd.Series = df.iloc[:, 0]
    return close_series.pct_change()

def fetch_risk_free_rate(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch the daily 3-month T-bill rate (FRED ``DTB3``) and de-annualize it.

    Args:
        start_date: ISO date string (``YYYY-MM-DD``), inclusive.
        end_date: ISO date string (``YYYY-MM-DD``), inclusive.

    Returns:
        DataFrame with a single ``rf`` column: daily risk-free rate,
        compounded from the annualized percentage reported by FRED.
    """
    rf = pdr.get_data_fred("DTB3", start=start_date, end=end_date)
    rf.columns = ["rf"]
    rf = (1 + rf / 100) ** (1 / TRADING_DAYS) - 1
    return rf

def fetch_fama_french(frequency: str, start_date: str, end_date: str) -> pd.DataFrame:
    ff_df = get_ff5(frequency, start_date, end_date, return_type="datetime")
    return ff_df

# ========== Core Calculations ==========
def compute_wealth_index(returns: pd.Series) -> pd.Series:
    wealth_index = (1 + returns).cumprod()
    return wealth_index

def compute_total_return(returns: pd.Series) -> float:
    wealth_index = compute_wealth_index(returns)
    total_return = wealth_index.iloc[-1] - 1
    return total_return

def compute_drawdowns_series(returns: pd.Series) -> pd.Series:
    wealth_index = compute_wealth_index(returns)
    peak = wealth_index.cummax()
    drawdowns = (wealth_index - peak) / peak
    return drawdowns

def compute_drawdown_stats(returns: pd.Series) -> tuple[float, float, float]:
    dd = compute_drawdowns_series(returns)
    max_dd = dd.min()
    underwater = dd < 0
    grp = (underwater != underwater.shift(1)).cumsum()
    duration = dd.groupby(grp[underwater]).size()
    avg_dd_duration = duration.mean()
    max_dd_duration = duration.max()
    return max_dd, avg_dd_duration, max_dd_duration

def compute_cagr(returns: pd.Series) -> float:
    n_obs = len(returns)
    cagr = np.prod(1 + returns) ** (TRADING_DAYS / n_obs) - 1
    return cagr

def compute_annualized_vol(returns: pd.Series) -> float:
    vol = returns.std()
    ann_vol = vol * np.sqrt(TRADING_DAYS)
    return ann_vol

def compute_sharpe_ratio(returns: pd.Series, rf: pd.Series) -> float:
    excess_ret = returns - rf
    daily_sharpe = excess_ret.mean() / excess_ret.std() if excess_ret.std() else np.nan
    ann_sharpe = daily_sharpe * np.sqrt(TRADING_DAYS)
    return ann_sharpe

def compute_sortino_ratio(returns: pd.Series, rf: pd.Series) -> float:
    excess_ret = returns - rf
    ann_excess_ret = excess_ret.mean() * TRADING_DAYS 
    downside_vol = np.sqrt((np.minimum(0, excess_ret) ** 2).sum() / (len(excess_ret) - 1)) * np.sqrt(TRADING_DAYS)
    sortino = ann_excess_ret / downside_vol if downside_vol else np.nan
    return sortino

def compute_calmar_ratio(returns: pd.Series) -> float:
    cagr = compute_cagr(returns)
    dd = compute_drawdowns_series(returns)
    max_drawdown = dd.min()
    calmar = cagr / abs(max_drawdown) if max_drawdown else np.nan
    return calmar

def compute_win_rate(returns: pd.Series) -> float:
    win_rate = (returns > 0).sum() / len(returns)
    return win_rate

def compute_profit_factor(returns: pd.Series) -> float:
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss else np.nan
    return profit_factor

def compute_information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    active_return = returns - benchmark_returns
    ann_active_return = active_return.mean() * TRADING_DAYS 
    tracking_error = active_return.std() * np.sqrt(TRADING_DAYS)
    information_ratio = ann_active_return / tracking_error if tracking_error else np.nan
    return information_ratio

def compute_var(returns: pd.Series) -> float:
    var = returns.quantile(0.05)
    return var

def compute_cvar(returns: pd.Series) -> float:
    var = compute_var(returns)
    cvar = returns[returns <= var].mean()
    return cvar

def compute_correlation(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    return returns.corr(benchmark_returns)

def compute_capture_ratios(returns: pd.Series, benchmark_returns: pd.Series) -> tuple[float, float, float]:
    up   = benchmark_returns > 0
    down = benchmark_returns < 0
    up_capture   = returns[up].mean() / benchmark_returns[up].mean() if benchmark_returns[up].mean() else np.nan
    down_capture = returns[down].mean() / benchmark_returns[down].mean() if benchmark_returns[down].mean() else np.nan
    overall_capture = up_capture / down_capture if down_capture else np.nan
    return up_capture, down_capture, overall_capture

# ========== Regression Models ==========

# Helper function
def _extract_loading(model: sm.regression.linear_model.RegressionResultsWrapper, factor: str) -> FactorLoading:
    p = model.params
    t = model.tvalues
    pval = model.pvalues

    coef = p.loc[factor]
    tstat = t.loc[factor]
    pvalue = pval.loc[factor]
    significant = pvalue < 0.05

    return FactorLoading(
        factor, 
        coef,
        tstat,
        pvalue,
        significant
    )

def _build_regression_result(model, label, loadings) -> RegressionResult:
    loadings_list = []
    for l in loadings:
        loadings_list.append(_extract_loading(model, l))

    # Extract the results
    alpha = _extract_loading(model, "const").coef
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    num_obs = model.nobs
    ann_alpha = alpha * TRADING_DAYS
    fstat = model.fvalue
    f_pvalue = model.f_pvalue

    return RegressionResult(
        label,
        loadings_list,
        r_squared,
        adj_r_squared,
        num_obs,
        ann_alpha,
        fstat,
        f_pvalue
    )

def capm_regression(portfolio_returns: pd.Series, factor_data: pd.DataFrame) -> RegressionResult:
    df = pd.concat([portfolio_returns, factor_data[["Mkt-RF", "RF"]]], axis=1, join="inner").dropna()
    y = df["portfolio_returns"] - df["RF"]
    X = df["Mkt-RF"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    loadings = ["const", "Mkt-RF"]
    res = _build_regression_result(model, "capm", loadings)
    return res

def ff3_regression(portfolio_returns: pd.Series, factor_data: pd.DataFrame) -> RegressionResult:
    # Align index
    df = pd.merge(factor_data, portfolio_returns, how="inner", left_index=True, right_index=True)
    df.dropna(inplace=True)
    y = df["portfolio_returns"] - df["RF"]
    X = df[["Mkt-RF", "SMB", "HML"]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    loadings = ["const", "Mkt-RF", "SMB", "HML"]
    res = _build_regression_result(model, "ff3", loadings)
    return res

def ff5_regression(portfolio_returns: pd.Series, factor_data: pd.DataFrame) -> RegressionResult:
    df = pd.merge(factor_data, portfolio_returns, how="inner", left_index=True, right_index=True)
    df.dropna(inplace=True)
    y = df["portfolio_returns"] - df["RF"]
    X = df[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    loadings = ["const", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    res = _build_regression_result(model, "ff5", loadings)
    return res

def compute_rolling_betas(portfolio_returns: pd.Series, factor_data: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    df = pd.merge(factor_data, portfolio_returns, how="inner", left_index=True, right_index=True)
    df.dropna(inplace=True)
    n = len(df)

    betas_mkt = []
    betas_smb = []
    betas_hml = []
    betas_rmw = []
    betas_cma = []

    for i in range(window - 1, n):
        y = (df["portfolio_returns"] - df["RF"]).iloc[i - window + 1 : i + 1]
        X = df[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].iloc[i - window + 1 : i + 1]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        betas_mkt.append(model.params.loc["Mkt-RF"])
        betas_smb.append(model.params.loc["SMB"])
        betas_hml.append(model.params.loc["HML"])
        betas_rmw.append(model.params.loc["RMW"])
        betas_cma.append(model.params.loc["CMA"])

    rolling_betas = pd.DataFrame({
        "Mkt-RF": betas_mkt,
        "SMB": betas_smb,
        "HML": betas_hml,
        "RMW": betas_rmw,
        "CMA": betas_cma
    }, index = df.index[window-1:]
    )
    return rolling_betas

def compute_rolling_sharpe(portfolio_returns: pd.Series, rf: pd.Series, window: int = 63) -> pd.Series:
    excess_ret = portfolio_returns - rf
    rolling_mean = excess_ret.rolling(window).mean()
    rolling_std = excess_ret.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(TRADING_DAYS)
    return rolling_sharpe

def compute_rolling_volatility(portfolio_returns: pd.Series, window: int=63) -> pd.Series:
    rolling_vol = portfolio_returns.rolling(window=window).std() * float(np.sqrt(TRADING_DAYS))
    return rolling_vol

def _compute_performance_stats(portfolio_ret: pd.Series, rf: pd.Series, bench_ret: pd.Series, is_benchmark: bool) -> PerformanceStats:
    if is_benchmark:
        ret = bench_ret
    else:
        ret = portfolio_ret

    wealth_index = compute_wealth_index(ret)
    drawdown_series = compute_drawdowns_series(ret)
    total_return = compute_total_return(ret)
    cagr = compute_cagr(ret)
    ann_vol = compute_annualized_vol(ret)
    max_dd, avg_dd_duration, max_dd_duration = compute_drawdown_stats(ret)
    sharpe = compute_sharpe_ratio(ret, rf)
    sortino = compute_sortino_ratio(ret, rf)
    calmar = compute_calmar_ratio(ret)
    information_ratio = np.nan if is_benchmark else compute_information_ratio(ret, bench_ret)
    skewness = cast(float, ret.skew())
    excess_kurtosis = cast(float, ret.kurtosis())
    var = compute_var(ret)
    cvar = compute_cvar(ret)
    win_rate = compute_win_rate(ret)
    profit_factor = compute_profit_factor(ret)
    correlation = compute_correlation(ret, bench_ret)
    if is_benchmark:
        up_capture = down_capture = overall_capture = np.nan
    else:
        up_capture, down_capture, overall_capture = compute_capture_ratios(ret, bench_ret)

    performance_stats = PerformanceStats(
        wealth_index,
        drawdown_series,
        total_return,
        cagr,
        ann_vol,
        max_dd,
        avg_dd_duration,
        max_dd_duration,
        sharpe,
        sortino,
        calmar,
        information_ratio,
        skewness,
        excess_kurtosis,
        var,
        cvar,
        win_rate,
        profit_factor,
        correlation,
        up_capture,
        down_capture,
        overall_capture
    )

    return performance_stats

# ========== Dataclasses ==========
@dataclass
class FactorLoading:
    name: str
    coef: float
    tstat: float
    pvalue: float
    significant: bool # p < 0.05

@dataclass
class RegressionResult:
    label: str
    loadings: List[FactorLoading]
    r_squared: float
    adj_r_squared: float    # Penalizes extra variables
    num_obs: int            # Number of observation used
    alpha_annualized: float
    fstat: float
    f_pvalue: float

@dataclass
class PerformanceStats:
    """Comprehensive risk/return metrics"""
    # Series
    wealth_index: pd.Series
    drawdown_series: pd.Series

    # Return metrics
    total_return: float
    cagr: float

    # Risk metrics
    ann_vol: float
    max_drawdown: float
    avg_drawdown_duration: float
    max_drawdown_duration: float

    # Risk-adjusted returns
    sharpe: float
    sortino: float
    calmar: float
    information_ratio: float

    # Distribution metrics
    skewness: float
    excess_kurtosis: float

    # Risk measures
    var: float
    cvar: float

    # Trading metrics
    win_rate: float
    profit_factor: float

    # Benchmark relative metrics
    correlation: float
    up_capture: float
    down_capture: float
    overall_capture: float

@dataclass
class AnalysisResult:
    """Main container passed to charts.py"""
    start_date: str
    end_date: str
    returns: pd.DataFrame
    portfolio_stats: PerformanceStats
    benchmark_stats: PerformanceStats
    capm: RegressionResult
    ff3: RegressionResult
    ff5: RegressionResult
    rolling_betas: pd.DataFrame
    rolling_sharpe: pd.Series
    rolling_volatility: pd.Series
    window: int

# ========== Orchestration ==========
def run_full_analysis(source: str, benchmark_symbol: str, window: int = 63) -> AnalysisResult:
    """Run the full tearsheet pipeline on a returns file.

    Loads portfolio returns, aligns them with the benchmark, risk-free rate,
    and Fama-French factors, then computes performance stats, CAPM/FF3/FF5
    regressions, rolling betas, and rolling Sharpe.

    Args:
        source: Path to the portfolio returns CSV (see :func:`load_returns`).
        benchmark_symbol: Yahoo Finance ticker for the benchmark (e.g. ``"SPY"``).
        window: Rolling window length in trading days. Defaults to 63.

    Returns:
        AnalysisResult bundling the aligned returns frame, stats, regression
        results, and rolling series.

    Raises:
        ValueError: If fewer than 2 portfolio return observations are present.
        BenchmarkFetchError: If benchmark price data cannot be retrieved.
    """
    portfolio_returns = load_returns(source)
    if len(portfolio_returns) < 2: 
        raise ValueError("Need at least 2 return observations")
    start_date = portfolio_returns.index[0].strftime("%Y-%m-%d")
    end_date = portfolio_returns.index[-1].strftime("%Y-%m-%d")
    risk_free = fetch_risk_free_rate(start_date, end_date)
    risk_free = risk_free.reindex(portfolio_returns.index).ffill()
    benchmark_returns = fetch_benchmark(start_date, end_date, benchmark_symbol)

    returns = pd.concat([portfolio_returns, benchmark_returns, risk_free], join="inner", axis=1).dropna()
    returns.columns = ["portfolio_returns", "benchmark_returns", "rf"]

    port_ret = returns["portfolio_returns"]
    rf = returns["rf"]
    bench_ret = returns["benchmark_returns"]

    portfolio_stats = _compute_performance_stats(port_ret, rf, bench_ret, is_benchmark=False)
    benchmark_stats = _compute_performance_stats(port_ret, rf, bench_ret, is_benchmark=True)

    factor_data = fetch_fama_french("D", start_date, end_date)
    capm = capm_regression(port_ret, factor_data)
    ff3 = ff3_regression(port_ret, factor_data)
    ff5 = ff5_regression(port_ret, factor_data)
    rolling_betas = compute_rolling_betas(port_ret, factor_data, window)
    rolling_sharpe = compute_rolling_sharpe(port_ret, rf, window)
    rolling_volatility = compute_rolling_volatility(port_ret, window)

    return AnalysisResult(
        start_date, 
        end_date,
        returns,
        portfolio_stats,
        benchmark_stats,
        capm,
        ff3,
        ff5,
        rolling_betas,
        rolling_sharpe,
        rolling_volatility,
        window
    )