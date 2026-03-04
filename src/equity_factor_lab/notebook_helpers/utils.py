"""Notebook helpers for slicing and validation metrics."""

import numpy as np
import pandas as pd

from ..backtest import long_short_decile_backtest
from ..evaluation import build_ic_inputs
from ..metrics import compute_ic_series, compute_ic_stats, compute_performance_stats

def slice_panel(
    panel: pd.DataFrame,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Slices a panel by optional inclusive start/end dates."""
    out = panel
    if start_date is not None:
        out = out.loc[out.index >= start_date]
    if end_date is not None:
        out = out.loc[out.index <= end_date]
    return out


def window_metrics(
    scores: pd.DataFrame,
    future_returns: pd.DataFrame,
    prices: pd.DataFrame,
    rebalance_freq: str,
    turnover_cost_rate: float,
    borrow_cost_rate_annual: float,
    ic_min_assets: int = 50,
    compute_ic: bool = True,
) -> dict[str, float]:
    """Computes IC/backtest metrics for one panel window."""
    scores, future_returns = scores.align(future_returns, join="inner")
    if scores.empty or future_returns.empty:
        raise ValueError("No overlapping score and return data in the requested window.")

    if compute_ic:
        ic_scores, ic_future_returns = build_ic_inputs(
            scores=scores,
            future_returns=future_returns,
            rebalance_freq=rebalance_freq,
            prices=prices,
        )
        ic_series = compute_ic_series(ic_scores, ic_future_returns, min_assets=ic_min_assets)
        ic_stats = compute_ic_stats(ic_series)
    else:
        ic_stats = {"mean_ic": np.nan, "t_newey_west": np.nan, "n_obs": 0.0}

    equity_curve = long_short_decile_backtest(
        scores=scores,
        future_returns=future_returns,
        rebalance_freq=rebalance_freq,
        turnover_cost_rate=turnover_cost_rate,
        borrow_cost_rate_annual=borrow_cost_rate_annual,
    )
    perf_stats = compute_performance_stats(equity_curve)
    turnover_mean = float(equity_curve.attrs.get("turnover_mean", np.nan))
    turnover_cost_drag_ann = float(equity_curve.attrs.get("turnover_cost_drag_ann", np.nan))
    borrow_cost_drag_ann = float(equity_curve.attrs.get("borrow_cost_drag_ann", np.nan))

    return {
        "mean_ic": float(ic_stats["mean_ic"]) if pd.notna(ic_stats["mean_ic"]) else np.nan,
        "t_ic_newey_west": (
            float(ic_stats["t_newey_west"]) if pd.notna(ic_stats["t_newey_west"]) else np.nan
        ),
        "sharpe": float(perf_stats["sharpe"]) if pd.notna(perf_stats["sharpe"]) else np.nan,
        "max_drawdown": (
            float(perf_stats["max_drawdown"]) if pd.notna(perf_stats["max_drawdown"]) else np.nan
        ),
        "ic_n_obs": float(ic_stats["n_obs"]),
        "turnover_mean": float(turnover_mean) if pd.notna(turnover_mean) else np.nan,
        "turnover_cost_drag_ann": (
            float(turnover_cost_drag_ann) if pd.notna(turnover_cost_drag_ann) else np.nan
        ),
        "borrow_cost_drag_ann": (
            float(borrow_cost_drag_ann) if pd.notna(borrow_cost_drag_ann) else np.nan
        ),
    }


def build_equity_curve(
    scores: pd.DataFrame,
    future_returns: pd.DataFrame,
    rebalance_freq: str,
    turnover_cost_rate: float,
    borrow_cost_rate_annual: float,
) -> pd.Series:
    """Builds a long-short equity curve for a score panel."""
    return long_short_decile_backtest(
        scores=scores,
        future_returns=future_returns,
        rebalance_freq=rebalance_freq,
        turnover_cost_rate=turnover_cost_rate,
        borrow_cost_rate_annual=borrow_cost_rate_annual,
    )
