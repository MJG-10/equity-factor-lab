from dataclasses import dataclass
from typing import TYPE_CHECKING
import pandas as pd
from simfin.names import SIMFIN_ID
from ..backtest import long_short_decile_backtest
from ..data.factor_qc import summarize_factor_panels
from ..data.price_cleaning import clean_price_panel
from ..data.simfin_prices import (
    build_adj_close_panel,
    build_close_dollar_volume_panel,
    load_universe_market_data,
)
from ..data.simfin_universe import load_company_universe_simfin_ids
from ..evaluation import build_ic_inputs
from ..metrics import compute_ic_series, compute_ic_stats, compute_performance_stats
from ..models.ridge import build_walk_forward_ridge_scores_fixed_alpha
from .pipeline_steps import (
    _apply_fixed_winsorization,
    _align_factor_panels,
    _build_fundamental_factor_scores,
    _build_price_factor_scores,
    _combine_equal_weight,
    _compute_base_future_returns,
    _neutralize_factor_panels,
    _standardize_factor_panels,
)
from .pipeline_universe import build_dynamic_universe_mask

if TYPE_CHECKING:
    from .pipeline import PipelineSettings


@dataclass
class LoadedPipelineData:
    """Stores data artifacts from the load/universe-selection stage."""

    market_data: pd.DataFrame
    prices_raw: pd.DataFrame
    prices: pd.DataFrame
    price_clean_summary_stats: dict[str, float]
    price_quality_mask: pd.DataFrame
    dollar_volume: pd.DataFrame


@dataclass
class BuiltScorePanels:
    """Stores aligned score panels and one-step-ahead returns before tradable masking."""

    all_scores: dict[str, pd.DataFrame]
    selected_scores: pd.DataFrame
    aligned_future_returns_raw: pd.DataFrame


@dataclass
class TradablePanels:
    """Stores tradable-masked scores and aligned returns."""

    all_scores: dict[str, pd.DataFrame]
    selected_scores: pd.DataFrame
    tradable_mask: pd.DataFrame
    aligned_future_returns: pd.DataFrame


@dataclass
class EvaluationArtifacts:
    """Stores diagnostics and evaluation outputs for one pipeline run."""

    ic_diagnostics: pd.DataFrame
    factor_qc_stats: pd.DataFrame
    ic_stats: dict[str, object]
    performance_stats: dict[str, float]


def _safe_quantile(series: pd.Series, q: float) -> float:
    """Returns a quantile with graceful empty-series handling."""
    clean = series.dropna().astype(float)
    if clean.empty:
        return float("nan")
    return float(clean.quantile(q))


def build_ic_coverage_diagnostic_rows(
    *,
    ic_valid_count: pd.Series,
    ic_coverage_frac: pd.Series,
    ic_min_assets: int,
) -> dict[str, int | float]:
    """Build slim diagnostics for IC cross-sectional coverage."""
    return {
        "n_dates_below_ic_min_assets": int((ic_valid_count < float(ic_min_assets)).sum()),
        "ic_min_assets": int(ic_min_assets),
        "ic_coverage_frac_p10": _safe_quantile(ic_coverage_frac, 0.10),
    }


def _build_price_quality_mask(
    *,
    prices: pd.DataFrame,
    hard_event_mask: pd.DataFrame,
    lookback_days: int,
    max_hard_events: int,
    min_coverage_frac: float,
) -> pd.DataFrame:
    """Builds a causal walk-forward quality mask from rolling coverage and hard-event counts."""
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")
    aligned_hard = hard_event_mask.reindex(
        index=prices.index,
        columns=prices.columns,
        fill_value=False,
    ).astype(bool)
    rolling_coverage = (
        prices.notna()
        .shift(1)
        .rolling(window=lookback_days, min_periods=lookback_days)
        .mean()
    )
    coverage_ok = rolling_coverage >= float(min_coverage_frac)
    rolling_hard = (
        aligned_hard.astype(float)
        .shift(1)
        .rolling(window=lookback_days, min_periods=lookback_days)
        .sum()
    )
    hard_ok = rolling_hard <= float(max_hard_events)
    quality_ok = (coverage_ok & hard_ok).fillna(True)
    return quality_ok.astype(bool)


def load_pipeline_data(settings: "PipelineSettings") -> LoadedPipelineData:
    """Loads market data and builds the base price and liquidity panels."""
    simfin_ids = load_company_universe_simfin_ids(
        api_key=settings.simfin_api_key,
        data_dir=settings.simfin_data_dir,
        market="us",
        refresh_days=settings.simfin_refresh_days,
    )
    market_data = load_universe_market_data(
        simfin_ids=simfin_ids,
        refresh_days=settings.simfin_refresh_days,
        start=settings.start_date,
        end=settings.end_date,
        api_key=settings.simfin_api_key,
        data_dir=settings.simfin_data_dir,
    )
    prices_vendor_raw_unfiltered = build_adj_close_panel(market_data)
    clean_result = clean_price_panel(
        prices_vendor_raw_unfiltered,
        hard_max_return=settings.price_clean_hard_max_return,
        classify_min_return=settings.price_clean_classify_min_return,
        reversal_next_abs_return=settings.price_clean_reversal_next_abs_return,
        reversal_two_day_net_abs=settings.price_clean_reversal_two_day_net_abs,
        stale_lookback_days=settings.price_clean_stale_lookback_days,
        stale_abs_return_epsilon=settings.price_clean_stale_abs_return_epsilon,
        shift_horizon_days=settings.price_clean_shift_horizon_days,
        shift_stability_abs_threshold=settings.price_clean_shift_stability_abs_threshold,
    )
    prices_causal_raw = clean_result.prices_clean
    prices = prices_causal_raw.dropna(axis=1, how="all")
    if prices.shape[1] == 0:
        raise ValueError("No price columns remain after initial price filtering.")

    dollar_volume_raw = build_close_dollar_volume_panel(market_data)
    dollar_volume = dollar_volume_raw.reindex(index=prices.index, columns=prices.columns)

    date_coverage = prices.notna().sum(axis=1).astype(float)
    if settings.min_assets_per_date is None:
        keep_dates = prices.index
    else:
        keep_dates = date_coverage.loc[
            date_coverage >= float(settings.min_assets_per_date)
        ].index
        if len(keep_dates) == 0:
            raise ValueError(
                "min_assets_per_date filter removed all dates. "
                "Lower min_assets_per_date or widen the universe."
            )

    prices_vendor_raw = prices_vendor_raw_unfiltered.reindex(index=keep_dates)
    prices_causal_raw = prices_causal_raw.reindex(index=keep_dates)
    prices = prices.reindex(index=keep_dates)
    dollar_volume = dollar_volume.reindex(index=keep_dates)

    # Re-apply cleaning after sparse-date filtering. Dropping dates can otherwise
    # create large one-step jumps across gaps in the retained panel.
    clean_result = clean_price_panel(
        prices_causal_raw,
        hard_max_return=settings.price_clean_hard_max_return,
        classify_min_return=settings.price_clean_classify_min_return,
        reversal_next_abs_return=settings.price_clean_reversal_next_abs_return,
        reversal_two_day_net_abs=settings.price_clean_reversal_two_day_net_abs,
        stale_lookback_days=settings.price_clean_stale_lookback_days,
        stale_abs_return_epsilon=settings.price_clean_stale_abs_return_epsilon,
        shift_horizon_days=settings.price_clean_shift_horizon_days,
        shift_stability_abs_threshold=settings.price_clean_shift_stability_abs_threshold,
    )
    prices_causal_clean = clean_result.prices_clean
    prices = prices_causal_clean.dropna(axis=1, how="all")
    if prices.shape[1] == 0:
        raise ValueError("No price columns remain after post-filter cleaning.")
    prices_vendor_raw = prices_vendor_raw.reindex(index=prices.index, columns=prices.columns)
    dollar_volume = dollar_volume.reindex(index=prices.index, columns=prices.columns)
    price_quality_mask_raw = _build_price_quality_mask(
        prices=prices_causal_clean,
        hard_event_mask=clean_result.hard_event_mask,
        lookback_days=settings.price_quality_lookback_days,
        max_hard_events=settings.price_quality_max_hard_events,
        min_coverage_frac=settings.price_quality_min_coverage_frac,
    )
    price_quality_mask = price_quality_mask_raw.reindex(
        index=prices.index,
        columns=prices.columns,
        fill_value=True,
    ).astype(bool)
    active_asset_set = set(int(asset) for asset in prices.columns.tolist())
    market_data = market_data.loc[
        market_data.index.get_level_values(SIMFIN_ID).astype(int).isin(active_asset_set)
    ]

    return LoadedPipelineData(
        market_data=market_data,
        prices_raw=prices_vendor_raw,
        prices=prices,
        price_clean_summary_stats={
            key: float(value) for key, value in clean_result.summary_stats.items()
        },
        price_quality_mask=price_quality_mask,
        dollar_volume=dollar_volume,
    )


def build_score_panels(
    *,
    settings: "PipelineSettings",
    prices: pd.DataFrame,
    market_data: pd.DataFrame,
) -> BuiltScorePanels:
    """Builds processed factor panels, composite or ridge outputs, and aligned returns."""
    future_returns = _compute_base_future_returns(prices)

    factor_scores: dict[str, pd.DataFrame] = {}
    factor_scores.update(_build_price_factor_scores(prices, settings.price_factors))
    if not settings.fundamental_factors:
        raise ValueError("fundamental_factors cannot be empty. Fundamentals are required.")

    fundamental_scores = _build_fundamental_factor_scores(
        prices=prices,
        factor_names=settings.fundamental_factors,
        settings=settings,
        market_data=market_data,
    )
    factor_scores.update(fundamental_scores)

    aligned_scores, aligned_future_returns = _align_factor_panels(
        factor_scores=factor_scores,
        future_returns=future_returns,
    )
    aligned_future_returns_raw = aligned_future_returns.copy()

    winsorized_scores = _apply_fixed_winsorization(aligned_scores)
    neutralized_scores = _neutralize_factor_panels(
        factor_scores=winsorized_scores,
        prices=prices,
        settings=settings,
    )
    standardized_scores = _standardize_factor_panels(neutralized_scores)

    composite_scores = _combine_equal_weight(
        standardized_scores,
        price_factor_names=settings.price_factors,
        fundamental_factor_names=settings.fundamental_factors,
    )
    all_scores = dict(standardized_scores)
    all_scores["composite"] = composite_scores

    if settings.output_factor == "ridge":
        all_scores["ridge"] = build_walk_forward_ridge_scores_fixed_alpha(
            factor_scores=all_scores,
            future_returns=aligned_future_returns,
            prices=prices,
            rebalance_freq=settings.rebalance_freq,
            alpha=settings.ridge_alpha,
            window_type=settings.ridge_window_type,
            window_size=settings.ridge_window_size,
            min_train_periods=settings.ridge_min_train_periods,
            refit_every_n_rebalances=settings.ridge_refit_every_n_rebalances,
            min_train_rows=settings.ridge_min_train_rows,
        )

    if settings.output_factor not in all_scores:
        raise ValueError(
            f"Output factor '{settings.output_factor}' not available. "
            f"Available: {sorted(all_scores.keys())}"
        )

    selected_scores = all_scores[settings.output_factor]
    return BuiltScorePanels(
        all_scores=all_scores,
        selected_scores=selected_scores,
        aligned_future_returns_raw=aligned_future_returns_raw,
    )


def apply_tradable_mask(
    *,
    all_scores: dict[str, pd.DataFrame],
    selected_scores: pd.DataFrame,
    prices: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    aligned_future_returns_raw: pd.DataFrame,
    max_tickers: int | None,
    adv_lookback: int,
    quality_mask: pd.DataFrame | None = None,
    min_prev_close: float | None = None,
) -> TradablePanels:
    """Builds the selected-score tradable mask and applies it to the score panels."""
    dynamic_universe_mask, _ = build_dynamic_universe_mask(
        prices=prices,
        dollar_volume=dollar_volume,
        max_tickers=max_tickers,
        adv_lookback=adv_lookback,
        quality_mask=quality_mask,
        min_prev_close=min_prev_close,
        required_data_mask=selected_scores.notna(),
    )
    aligned_universe_mask = dynamic_universe_mask.reindex(
        index=selected_scores.index,
        columns=selected_scores.columns,
        fill_value=False,
    )
    tradable_mask = aligned_universe_mask & selected_scores.notna()

    masked_scores = {name: panel.where(tradable_mask) for name, panel in all_scores.items()}
    selected_masked_scores = selected_scores.where(tradable_mask)

    return TradablePanels(
        all_scores=masked_scores,
        selected_scores=selected_masked_scores,
        tradable_mask=tradable_mask,
        aligned_future_returns=aligned_future_returns_raw,
    )


def evaluate_pipeline_artifacts(
    *,
    settings: "PipelineSettings",
    prices: pd.DataFrame,
    tradable_panels: TradablePanels,
) -> EvaluationArtifacts:
    """Builds selected-panel factor, IC, and backtest diagnostics."""
    factor_qc_stats = summarize_factor_panels(
        tradable_panels.all_scores,
        coverage_base_mask=tradable_panels.tradable_mask,
    )

    ic_scores, ic_future_returns = build_ic_inputs(
        scores=tradable_panels.selected_scores,
        future_returns=tradable_panels.aligned_future_returns,
        rebalance_freq=settings.rebalance_freq,
        prices=prices,
    )
    ic_valid_mask = ic_scores.notna() & ic_future_returns.notna()
    ic_valid_count = ic_valid_mask.sum(axis=1).astype(float)
    ic_base_count = ic_scores.notna().sum(axis=1).astype(float).replace(0.0, float("nan"))
    ic_coverage_frac = ic_valid_count.div(ic_base_count)

    ic_coverage_rows = build_ic_coverage_diagnostic_rows(
        ic_valid_count=ic_valid_count,
        ic_coverage_frac=ic_coverage_frac,
        ic_min_assets=settings.ic_min_assets,
    )
    ic_diagnostics = pd.DataFrame(
        [
            {
                "rebalance_freq": settings.rebalance_freq,
                **ic_coverage_rows,
            }
        ]
    )

    ic_series = compute_ic_series(
        ic_scores,
        ic_future_returns,
        min_assets=settings.ic_min_assets,
    )
    ic_stats = compute_ic_stats(ic_series)

    equity_curve = long_short_decile_backtest(
        scores=tradable_panels.selected_scores,
        future_returns=tradable_panels.aligned_future_returns,
        rebalance_freq=settings.rebalance_freq,
        turnover_cost_rate=settings.turnover_cost_rate,
        borrow_cost_rate_annual=settings.borrow_cost_rate_annual,
    )
    performance_stats = compute_performance_stats(equity_curve)

    return EvaluationArtifacts(
        ic_diagnostics=ic_diagnostics,
        factor_qc_stats=factor_qc_stats,
        ic_stats=ic_stats,
        performance_stats=performance_stats,
    )
