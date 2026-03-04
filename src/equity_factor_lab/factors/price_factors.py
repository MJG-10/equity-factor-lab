import pandas as pd
import numpy as np


def winsorize_cross_section(
    scores: pd.DataFrame,
    *,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """Clips each cross-section at fixed lower and upper quantiles."""
    lower = scores.quantile(lower_q, axis=1, interpolation="linear")
    upper = scores.quantile(upper_q, axis=1, interpolation="linear")
    return scores.clip(lower=lower, upper=upper, axis=0)


def _rolling_compound_returns(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Computes rolling compounded returns via log1p / expm1 for stability."""
    safe_returns = returns.where(returns > -1.0)
    log_returns = np.log1p(safe_returns)
    rolling_log_sum = log_returns.rolling(window=window, min_periods=window).sum()
    return np.expm1(rolling_log_sum)


def _leave_one_out_equal_weight_market_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Builds a leave-one-out equal-weight market proxy for each asset return series."""
    total_returns = returns.sum(axis=1, min_count=1)
    return_counts = returns.notna().sum(axis=1).astype(float)
    other_returns = returns.rsub(total_returns, axis=0)
    other_counts = returns.notna().astype(float).rsub(return_counts, axis=0)
    market_proxy = other_returns.div(other_counts.where(other_counts > 0.0))
    return market_proxy.where(returns.notna())


def get_rebalance_dates(scores: pd.DataFrame, freq: str = "ME") -> pd.DatetimeIndex:
    """Returns last observed trading dates per rebalance period."""
    if scores.empty:
        return pd.DatetimeIndex([])
    if freq is None or freq == "D":
        return pd.DatetimeIndex(scores.index.unique()).sort_values()

    # Anchored frequencies (e.g., "W-FRI") rebalance on the last available trading day in each bucket.
    period_last = pd.Series(scores.index, index=scores.index).resample(freq).last()
    return pd.DatetimeIndex(period_last.dropna().to_numpy())


def compute_period_forward_returns(
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Computes forward returns between rebalance dates.

    Terminal endpoint-missing prices are treated as -100%; non-terminal misses stay NaN.
    """
    fwd_list = []
    idx_list = []
    has_observation = prices.notna()
    obs_after = has_observation.iloc[::-1].cumsum().iloc[::-1].shift(-1).fillna(0.0)

    for i in range(len(rebalance_dates) - 1):
        start = rebalance_dates[i]
        end = rebalance_dates[i + 1]

        endpoints = prices.reindex(index=[start, end])
        if endpoints.empty:
            continue

        start_prices = endpoints.loc[start]
        end_prices = endpoints.loc[end]
        period_return = end_prices.div(start_prices).sub(1.0)
        obs_after_end = obs_after.reindex(index=[end]).iloc[0]
        terminal_missing_end = end_prices.isna() & (obs_after_end == 0.0)
        period_return = period_return.where(~terminal_missing_end, -1.0)
        fwd_list.append(period_return)
        idx_list.append(start)

    if not fwd_list:
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=prices.columns)

    fwd = pd.DataFrame(fwd_list, index=pd.DatetimeIndex(idx_list))
    return fwd


def standardize_cross_section(scores: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally z-scores each row in a score panel."""
    cs_mean = scores.mean(axis=1)
    cs_std = scores.std(axis=1, ddof=0)

    z = scores.sub(cs_mean, axis=0)
    z = z.div(cs_std.replace(0.0, np.nan), axis=0)
    z = z.where(scores.notna())

    return z


def compute_momentum_scores(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip_recent: int = 21,
) -> pd.DataFrame:
    """Computes raw momentum scores with configurable lookback and skip window."""
    returns = prices.pct_change(fill_method=None)

    effective_window = lookback - skip_recent
    if effective_window <= 0:
        raise ValueError("lookback must be greater than skip_recent.")

    shifted_returns = returns.shift(skip_recent)
    scores = _rolling_compound_returns(shifted_returns, window=effective_window)
    return scores


def compute_short_term_reversal_scores(
    prices: pd.DataFrame,
    lookback: int = 21,
) -> pd.DataFrame:
    """Computes raw short-term reversal scores from recent cumulative returns."""
    returns = prices.pct_change(fill_method=None)

    cum_ret = _rolling_compound_returns(returns, window=lookback)

    scores = -cum_ret
    return scores



def compute_low_volatility_scores(
    prices: pd.DataFrame,
    lookback: int = 252,
    beta_min_periods: int = 252,
) -> pd.DataFrame:
    """Computes raw low-volatility scores from same-window idiosyncratic return volatility."""
    returns = prices.pct_change(fill_method=None)
    if lookback <= 1:
        raise ValueError("lookback must be greater than 1.")
    min_periods = int(beta_min_periods)
    if min_periods <= 1 or min_periods > lookback:
        raise ValueError("beta_min_periods must be in [2, lookback].")

    market_returns = _leave_one_out_equal_weight_market_returns(returns)
    asset_var = returns.rolling(window=lookback, min_periods=min_periods).var(ddof=0)
    market_var = market_returns.rolling(window=lookback, min_periods=min_periods).var(ddof=0)
    rolling_cov = returns.rolling(window=lookback, min_periods=min_periods).cov(market_returns, ddof=0)
    explained_var = rolling_cov.pow(2).div(market_var.where(market_var > 0.0))
    residual_var = asset_var.sub(explained_var).clip(lower=0.0)
    scores = -np.sqrt(residual_var)
    return scores

