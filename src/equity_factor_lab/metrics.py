import pandas as pd
import numpy as np


def compute_ic_series(
    scores: pd.DataFrame,
    future_returns: pd.DataFrame,
    min_assets: int = 50,
) -> pd.Series:
    """Computes date-level cross-sectional Spearman IC series."""
    scores, future_returns = scores.align(future_returns, join="inner")

    valid = scores.notna() & future_returns.notna()
    n_valid = valid.sum(axis=1)

    ic = scores.corrwith(future_returns, axis=1, method="spearman")
    ic = ic.where(n_valid >= min_assets).dropna()

    return ic


def choose_nw_lag(T: int) -> int:
    """Chooses a Newey-West lag length from sample size."""
    if T <= 0:
        return 0

    L = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    L = max(1, min(L, T - 1))
    return L


def compute_ic_stats(
    ic: pd.Series,
    max_lag: int = 5,
    nw_lag: int | None = None,
) -> dict:
    """Computes IC summary statistics, ACF, and Newey-West t-statistics."""
    ic = ic.dropna()
    T = len(ic)
    if T == 0:
        return {
            "mean_ic": np.nan,
            "std_ic": np.nan,
            "t_naive": np.nan,
            "acf": {},
            "t_newey_west": np.nan,
            "nw_lag": nw_lag,
            "n_obs": 0,
            "ci95_naive": (np.nan, np.nan),
            "ci95_newey_west": (np.nan, np.nan),
        }

    mean_ic = ic.mean()
    std_ic = ic.std(ddof=1)
    se_naive = std_ic / np.sqrt(T)
    t_naive = np.nan

    if std_ic > 0 and T > 1:
        t_naive = mean_ic / se_naive

    acf = {}
    for lag in range(1, max_lag + 1):
        acf[lag] = ic.autocorr(lag=lag)

    t_nw = np.nan
    se_nw = np.nan

    nw_lag = choose_nw_lag(T) if nw_lag is None else nw_lag
    if nw_lag is not None and nw_lag > 0 and T > 1:
        x = ic.values
        x_centered = x - mean_ic

        gamma0 = np.dot(x_centered, x_centered) / T
        var_hat = gamma0

        for k in range(1, min(nw_lag, T - 1) + 1):
            cov = np.dot(x_centered[k:], x_centered[:-k]) / T
            weight = 1.0 - k / (nw_lag + 1)
            var_hat += 2.0 * weight * cov

        se_nw = np.sqrt(var_hat / T)
        if se_nw > 0:
            t_nw = mean_ic / se_nw

    return {
        "mean_ic": float(mean_ic),
        "std_ic": float(std_ic),
        "t_naive": float(t_naive),
        "acf": {int(k): float(v) for k, v in acf.items()},
        "t_newey_west": float(t_nw) if not np.isnan(t_nw) else np.nan,
        "nw_lag": nw_lag,
        "n_obs": int(T),
        "ci95_naive": (mean_ic - 1.96 * se_naive, mean_ic + 1.96 * se_naive),
        "ci95_newey_west": (mean_ic - 1.96 * se_nw, mean_ic + 1.96 * se_nw),
    }


def compute_performance_stats(equity_curve: pd.Series, trading_days_per_year: int = 252) -> dict:
    """
    Computes standard performance statistics from an equity curve.

    Annual return is computed as CAGR when both start/end equity are positive.
    Sharpe is computed from annualized daily return mean and volatility.
    """
    if equity_curve.empty:
        return {
            "total_return": np.nan,
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }

    daily_ret = equity_curve.pct_change().dropna()

    start_value = float(equity_curve.iloc[0])
    end_value = float(equity_curve.iloc[-1])
    total_return = np.nan if start_value == 0.0 else (end_value / start_value) - 1.0

    n_periods = int(len(daily_ret))
    annual_return = np.nan
    if n_periods > 0 and start_value > 0.0 and end_value > 0.0:
        annual_return = (end_value / start_value) ** (trading_days_per_year / n_periods) - 1.0

    avg_daily = float(daily_ret.mean()) if n_periods > 0 else np.nan
    std_daily = float(daily_ret.std(ddof=1)) if n_periods > 1 else np.nan

    annual_vol = std_daily * np.sqrt(trading_days_per_year)

    sharpe = np.nan
    if np.isfinite(avg_daily) and np.isfinite(std_daily) and std_daily > 0.0:
        sharpe = (avg_daily / std_daily) * np.sqrt(trading_days_per_year)

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_drawdown = drawdown.min()

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
    }
