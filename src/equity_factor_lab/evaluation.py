import pandas as pd

from .factors.price_factors import compute_period_forward_returns, get_rebalance_dates


def build_ic_inputs(
    scores: pd.DataFrame,
    future_returns: pd.DataFrame,
    rebalance_freq: str,
    prices: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds score/return panels aligned to the IC evaluation horizon."""
    if rebalance_freq is None or rebalance_freq == "D":
        ic_scores, ic_future_returns = scores.align(future_returns, join="inner")
        return ic_scores, ic_future_returns

    if prices is None:
        raise ValueError("prices are required for non-daily IC input construction.")

    rebalance_dates = get_rebalance_dates(scores, freq=rebalance_freq)
    if len(rebalance_dates) < 2:
        raise ValueError(
            f"Need at least 2 rebalance dates for freq='{rebalance_freq}', "
            f"but found {len(rebalance_dates)}."
        )

    period_forward_returns = compute_period_forward_returns(
        prices=prices,
        rebalance_dates=rebalance_dates,
    )
    if period_forward_returns.empty:
        raise ValueError(
            f"No period forward returns computed for rebalance_freq='{rebalance_freq}'."
        )

    rebalance_scores = scores.reindex(period_forward_returns.index)
    ic_scores, ic_future_returns = rebalance_scores.align(period_forward_returns, join="inner")
    if ic_scores.empty or ic_future_returns.empty:
        raise ValueError("No data available for IC evaluation after rebalance alignment.")

    return ic_scores, ic_future_returns
