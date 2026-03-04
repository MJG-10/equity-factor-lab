import pandas as pd
import numpy as np
from .factors.price_factors import get_rebalance_dates
from .missing_returns import build_terminal_missing_mask


def _build_rebalance_long_short_weights(
    rebalance_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds disjoint equal-weight long/short decile targets from ranked scores."""
    assets = rebalance_scores.columns
    long_weights = pd.DataFrame(0.0, index=rebalance_scores.index, columns=assets, dtype=float)
    short_weights = pd.DataFrame(0.0, index=rebalance_scores.index, columns=assets, dtype=float)

    for dt, row in rebalance_scores.iterrows():
        valid_scores = row.dropna()
        n_assets = len(valid_scores)
        if n_assets < 2:
            continue

        decile_count = max(1, int(np.floor(0.1 * n_assets)))
        n_short = min(decile_count, n_assets)
        n_long = min(decile_count, n_assets - n_short)

        if n_long <= 0 or n_short <= 0:
            continue

        ordered = valid_scores.sort_values(ascending=True, kind="mergesort")
        short_assets = ordered.index[:n_short]
        long_assets = ordered.index[-n_long:]

        short_weights.loc[dt, short_assets] = 1.0 / float(n_short)
        long_weights.loc[dt, long_assets] = 1.0 / float(n_long)

    return long_weights, short_weights


def long_short_decile_backtest(
    scores: pd.DataFrame,
    future_returns: pd.DataFrame,
    rebalance_freq: str = "ME",
    turnover_cost_rate: float = 0.001,
    borrow_cost_rate_annual: float = 0.0,
) -> pd.Series:
    """Runs a long-short decile backtest; returns equity (summary in attrs)."""
    if turnover_cost_rate < 0.0:
        raise ValueError("turnover_cost_rate must be non-negative.")
    if borrow_cost_rate_annual < 0.0:
        raise ValueError("borrow_cost_rate_annual must be non-negative.")

    scores, future_returns = scores.align(future_returns, join="inner")

    if rebalance_freq is None or rebalance_freq == "D":
        rebalance_dates = scores.index
    else:
        rebalance_dates = get_rebalance_dates(scores, freq=rebalance_freq)

    if len(rebalance_dates) == 0:
        raise ValueError("No rebalance dates found. Check rebalance_freq and index frequency.")

    reb_scores = scores.loc[rebalance_dates]
    terminal_missing_mask = build_terminal_missing_mask(future_returns)
    long_w_reb, short_w_reb = _build_rebalance_long_short_weights(reb_scores)

    dates = scores.index
    assets = scores.columns
    rebalance_set = set(rebalance_dates)

    prev_long = pd.Series(0.0, index=assets)
    prev_short = pd.Series(0.0, index=assets)

    daily_ret_list: list[float] = []
    turnover_cost_sum = 0.0
    borrow_cost_sum = 0.0
    rebalance_turnovers: list[float] = []

    for dt in dates:
        turnover_cost_t = 0.0
        is_rebalance = dt in rebalance_set

        if is_rebalance:
            target_long = long_w_reb.loc[dt].reindex(assets).fillna(0.0)
            target_short = short_w_reb.loc[dt].reindex(assets).fillna(0.0)

            turnover_long = (target_long - prev_long).abs().sum()
            turnover_short = (target_short - prev_short).abs().sum()
            total_turnover = float(turnover_long + turnover_short)
            rebalance_turnovers.append(total_turnover)

            if turnover_cost_rate != 0.0:
                turnover_cost_t = turnover_cost_rate * total_turnover

            curr_long = target_long
            curr_short = target_short
        else:
            curr_long = prev_long
            curr_short = prev_short

        r_t = future_returns.loc[dt].reindex(assets)
        terminal_missing_t = terminal_missing_mask.loc[dt].reindex(assets).fillna(False)
        held_mask = (curr_long.abs() + curr_short.abs()) > 0.0

        delist_like_missing = r_t.isna() & terminal_missing_t & held_mask
        r_t = r_t.copy()
        r_t.loc[delist_like_missing] = -1.0
        r_t = r_t.fillna(0.0)

        gross_long_ret_t = (curr_long * r_t).sum()
        gross_short_ret_t = (curr_short * r_t).sum()
        gross_daily_ret_t = gross_long_ret_t - gross_short_ret_t

        short_gross_exposure_t = curr_short.abs().sum()
        borrow_cost_t = 0.0
        if borrow_cost_rate_annual != 0.0:
            borrow_cost_t = (borrow_cost_rate_annual / 252.0) * short_gross_exposure_t

        daily_ret_t = gross_daily_ret_t - turnover_cost_t - borrow_cost_t
        daily_ret_list.append(float(daily_ret_t))
        turnover_cost_sum += float(turnover_cost_t)
        borrow_cost_sum += float(borrow_cost_t)

        if curr_long.sum() != 0.0:
            long_notional = curr_long * (1.0 + r_t)
            long_total = long_notional.sum()
            if long_total != 0.0 and not np.isnan(long_total):
                prev_long = long_notional / long_total
            else:
                prev_long = curr_long * 0.0
        else:
            prev_long = curr_long * 0.0

        if curr_short.sum() != 0.0:
            short_notional = curr_short * (1.0 + r_t)
            short_total = short_notional.sum()
            if short_total != 0.0 and not np.isnan(short_total):
                prev_short = short_notional / short_total
            else:
                prev_short = curr_short * 0.0
        else:
            prev_short = curr_short * 0.0

    daily_ret = pd.Series(daily_ret_list, index=dates, dtype=float)
    equity_curve = (1.0 + daily_ret).cumprod()

    if len(rebalance_turnovers) > 0:
        reb_turnover = pd.Series(rebalance_turnovers, dtype=float)
        turnover_mean = float(reb_turnover.mean())
    else:
        turnover_mean = np.nan

    n_days = len(daily_ret)
    if n_days > 0:
        turnover_cost_drag_ann = float((turnover_cost_sum / float(n_days)) * 252.0)
        borrow_cost_drag_ann = float((borrow_cost_sum / float(n_days)) * 252.0)
    else:
        turnover_cost_drag_ann = np.nan
        borrow_cost_drag_ann = np.nan

    equity_curve.attrs["turnover_mean"] = turnover_mean
    equity_curve.attrs["turnover_cost_drag_ann"] = turnover_cost_drag_ann
    equity_curve.attrs["borrow_cost_drag_ann"] = borrow_cost_drag_ann
    return equity_curve
