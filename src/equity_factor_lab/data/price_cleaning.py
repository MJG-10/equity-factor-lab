from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PriceCleaningResult:
    """Stores the cleaned price panel, anomaly masks, and summary statistics."""

    prices_clean: pd.DataFrame
    invalid_price_mask: pd.DataFrame
    hard_event_mask: pd.DataFrame
    suspect_event_mask: pd.DataFrame
    spike_reversal_mask: pd.DataFrame
    persistent_shift_mask: pd.DataFrame
    stale_jump_mask: pd.DataFrame
    summary_stats: dict[str, float]


def _false_mask_like(frame: pd.DataFrame) -> pd.DataFrame:
    """Builds an all-False boolean mask aligned to `frame`."""
    return pd.DataFrame(False, index=frame.index, columns=frame.columns, dtype=bool)


def _compute_spike_reversal_mask(
    returns: pd.DataFrame,
    *,
    min_next_abs_return: float,
    max_two_day_net_abs: float,
) -> pd.DataFrame:
    """Flags large positive jumps followed by a sharp opposite-day reversal."""
    next_returns = returns.shift(-1)
    two_day_net = returns.add(1.0).mul(next_returns.add(1.0)).sub(1.0)
    sign_flip = (returns > 0.0) & (next_returns < 0.0)
    return sign_flip & (next_returns.abs() >= float(min_next_abs_return)) & (
        two_day_net.abs() <= float(max_two_day_net_abs)
    )


def _compute_persistent_shift_mask(
    prices: pd.DataFrame,
    *,
    horizon_days: int,
    stability_abs_threshold: float,
    candidate_cols: pd.Index | None = None,
) -> pd.DataFrame:
    """Flags jumps after which prices stay near the new level over a short horizon."""
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    if prices.empty:
        return _false_mask_like(prices)

    if candidate_cols is None:
        selected_cols = prices.columns
    else:
        selected_cols = pd.Index(candidate_cols)
        if len(selected_cols) == 0:
            return _false_mask_like(prices)

    prices_view = prices.reindex(columns=selected_cols)

    horizon_abs_ratios: list[pd.DataFrame] = []
    for k in range(1, horizon_days + 1):
        ratio_abs = prices_view.shift(-k).div(prices_view).sub(1.0).abs()
        horizon_abs_ratios.append(ratio_abs)
    median_abs_post_move = (
        pd.concat(horizon_abs_ratios, keys=range(1, horizon_days + 1))
        .groupby(level=1)
        .median()
    )
    persistent_subset = (median_abs_post_move <= float(stability_abs_threshold)).fillna(False).astype(bool)

    if candidate_cols is None:
        return persistent_subset

    persistent_mask = _false_mask_like(prices)
    persistent_mask.loc[:, selected_cols] = persistent_subset.reindex(
        index=prices.index,
        columns=selected_cols,
        fill_value=False,
    )
    return persistent_mask


def _compute_stale_jump_mask(
    returns: pd.DataFrame,
    *,
    stale_lookback_days: int,
    stale_abs_return_epsilon: float,
) -> pd.DataFrame:
    """Flags dates preceded by near-zero returns for `stale_lookback_days`."""
    if stale_lookback_days <= 0:
        raise ValueError("stale_lookback_days must be positive.")
    return (
        returns.abs()
        .shift(1)
        .le(float(stale_abs_return_epsilon))
        .rolling(window=stale_lookback_days, min_periods=stale_lookback_days)
        .sum()
        >= stale_lookback_days
    )


def clean_price_panel(
    prices: pd.DataFrame,
    *,
    hard_max_return: float = 10.0,
    classify_min_return: float = 3.0,
    reversal_next_abs_return: float = 0.6,
    reversal_two_day_net_abs: float = 0.2,
    stale_lookback_days: int = 5,
    stale_abs_return_epsilon: float = 1e-12,
    shift_horizon_days: int = 5,
    shift_stability_abs_threshold: float = 0.2,
) -> PriceCleaningResult:
    """
    Applies deterministic price cleaning for non-physical and artifact-like jumps.

    Suspect-event labels are diagnostic-only. The trading-path cleaning drops
    invalid prices and hard impossible returns, while keeping suspect events in
    the cleaned panel.
    """
    if prices.empty or prices.shape[1] == 0:
        empty_mask = _false_mask_like(prices)
        return PriceCleaningResult(
            prices_clean=prices.copy(),
            invalid_price_mask=empty_mask,
            hard_event_mask=empty_mask,
            suspect_event_mask=empty_mask,
            spike_reversal_mask=empty_mask,
            persistent_shift_mask=empty_mask,
            stale_jump_mask=empty_mask,
            summary_stats={
                "n_classify_candidates": 0.0,
                "n_spike_reversal": 0.0,
                "n_persistent_shift": 0.0,
                "n_stale_jump": 0.0,
                "n_suspect_events": 0.0,
                "n_assets_with_any_suspect_event": 0.0,
            },
        )

    if stale_lookback_days <= 0:
        raise ValueError("stale_lookback_days must be positive.")
    if shift_horizon_days <= 0:
        raise ValueError("shift_horizon_days must be positive.")

    prices_clean = prices.astype(float, copy=True)
    invalid_price_mask = ~np.isfinite(prices_clean) | (prices_clean <= 0.0)
    prices_clean = prices_clean.where(~invalid_price_mask)

    returns = prices_clean.pct_change(fill_method=None)
    hard_ret_gt = returns >= float(hard_max_return)
    hard_ret_lt = returns < -1.0
    hard_event_mask = hard_ret_gt | hard_ret_lt

    classify_candidate = (returns > float(classify_min_return)) & (~hard_event_mask)
    candidate_col_mask = classify_candidate.any(axis=0)
    candidate_cols = candidate_col_mask[candidate_col_mask].index
    if len(candidate_cols) == 0:
        spike_reversal_mask = _false_mask_like(returns)
        persistent_shift_mask = _false_mask_like(returns)
        stale_jump_mask = _false_mask_like(returns)
    else:
        candidate_returns = returns.reindex(columns=candidate_cols)
        spike_reversal_subset = _compute_spike_reversal_mask(
            candidate_returns,
            min_next_abs_return=reversal_next_abs_return,
            max_two_day_net_abs=reversal_two_day_net_abs,
        )
        stale_jump_subset = _compute_stale_jump_mask(
            candidate_returns,
            stale_lookback_days=stale_lookback_days,
            stale_abs_return_epsilon=stale_abs_return_epsilon,
        )
        spike_reversal_mask = _false_mask_like(returns)
        spike_reversal_mask.loc[:, candidate_cols] = spike_reversal_subset.fillna(False).astype(bool)
        stale_jump_mask = _false_mask_like(returns)
        stale_jump_mask.loc[:, candidate_cols] = stale_jump_subset.fillna(False).astype(bool)
        persistent_shift_mask = _compute_persistent_shift_mask(
            prices_clean,
            horizon_days=shift_horizon_days,
            stability_abs_threshold=shift_stability_abs_threshold,
            candidate_cols=candidate_cols,
        )

    suspect_event_mask = classify_candidate & (
        spike_reversal_mask | persistent_shift_mask | stale_jump_mask
    )
    event_drop_mask = hard_event_mask

    # Event at t corresponds to move from t-1 to t, so null price[t].
    prices_clean = prices_clean.where(~event_drop_mask)

    summary_stats = {
        "n_classify_candidates": float(classify_candidate.sum().sum()),
        "n_spike_reversal": float((classify_candidate & spike_reversal_mask).sum().sum()),
        "n_persistent_shift": float((classify_candidate & persistent_shift_mask).sum().sum()),
        "n_stale_jump": float((classify_candidate & stale_jump_mask).sum().sum()),
        "n_suspect_events": float(suspect_event_mask.sum().sum()),
        "n_assets_with_any_suspect_event": float((suspect_event_mask.sum(axis=0) > 0.0).sum()),
    }

    return PriceCleaningResult(
        prices_clean=prices_clean,
        invalid_price_mask=invalid_price_mask.fillna(False),
        hard_event_mask=hard_event_mask.fillna(False),
        suspect_event_mask=suspect_event_mask.fillna(False),
        spike_reversal_mask=spike_reversal_mask.fillna(False),
        persistent_shift_mask=persistent_shift_mask.fillna(False),
        stale_jump_mask=stale_jump_mask.fillna(False),
        summary_stats=summary_stats,
    )
