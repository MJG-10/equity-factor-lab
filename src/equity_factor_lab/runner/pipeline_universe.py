import pandas as pd


def build_dynamic_universe_mask(
    *,
    prices: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    max_tickers: int | None,
    adv_lookback: int,
    required_data_mask: pd.DataFrame | None = None,
    quality_mask: pd.DataFrame | None = None,
    min_prev_close: float | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Builds a PIT-safe date-by-date tradable-universe mask."""
    if adv_lookback <= 0:
        raise ValueError("dynamic_universe_adv_lookback must be positive.")
    if max_tickers is not None and max_tickers <= 0:
        raise ValueError("universe_max_tickers must be positive when provided.")
    if min_prev_close is not None and min_prev_close <= 0.0:
        raise ValueError("min_prev_close must be positive when provided.")

    prices, dollar_volume = prices.align(dollar_volume, join="inner")
    if prices.empty or prices.shape[1] == 0:
        raise ValueError("No overlapping price and dollar-volume data for dynamic universe.")

    has_price = prices.notna()
    if min_prev_close is None:
        prev_close_ok = pd.DataFrame(True, index=prices.index, columns=prices.columns, dtype=bool)
    else:
        prev_close_ok = prices.shift(1).ge(float(min_prev_close)).fillna(False)

    # Uses only information available through t-1.
    adv = dollar_volume.shift(1).rolling(
        window=adv_lookback,
        min_periods=adv_lookback,
    ).mean()
    liquidity_ok = adv.notna()

    def _select_with_cap(eligible_mask: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if max_tickers is None:
            return eligible_mask, pd.Series(0.0, index=prices.index, dtype=float)
        adv_rank = adv.where(eligible_mask).rank(axis=1, ascending=False, method="first")
        cap_ok = adv_rank <= float(max_tickers)
        selected = eligible_mask & cap_ok
        cap_excluded = (eligible_mask & ~cap_ok).sum(axis=1).astype(float)
        return selected, cap_excluded

    eligible_base = has_price & liquidity_ok & prev_close_ok

    if quality_mask is None:
        aligned_quality_mask = pd.DataFrame(True, index=prices.index, columns=prices.columns, dtype=bool)
    else:
        aligned_quality_mask = quality_mask.reindex(
            index=prices.index,
            columns=prices.columns,
            fill_value=False,
        ).astype(bool)
    eligible_quality = eligible_base & aligned_quality_mask
    quality_excluded_count = (eligible_base & ~aligned_quality_mask).sum(axis=1).astype(float)

    raw_selected_mask, _ = _select_with_cap(eligible_quality)
    if required_data_mask is None:
        universe_mask, cap_excluded_count = _select_with_cap(eligible_quality)
        data_completeness_excluded_count = pd.Series(0.0, index=prices.index, dtype=float)
    else:
        required = required_data_mask.reindex(
            index=prices.index,
            columns=prices.columns,
            fill_value=False,
        ).astype(bool)
        eligible_final = eligible_quality & required
        universe_mask, cap_excluded_count = _select_with_cap(eligible_final)
        data_completeness_excluded_count = (eligible_quality & ~required).sum(axis=1).astype(float)
    diagnostics = {
        "raw_selected_count": raw_selected_mask.sum(axis=1).astype(float),
        "liquidity_excluded_count": (has_price & ~liquidity_ok).sum(axis=1).astype(float),
        "price_floor_excluded_count": (has_price & liquidity_ok & ~prev_close_ok).sum(axis=1).astype(float),
        "quality_excluded_count": quality_excluded_count,
        "cap_excluded_count": cap_excluded_count,
        "data_completeness_excluded_count": data_completeness_excluded_count,
    }
    return universe_mask, diagnostics
