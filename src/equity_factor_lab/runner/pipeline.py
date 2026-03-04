from dataclasses import dataclass
import pandas as pd
from ..data.factor_qc import summarize_factor_panels
from ..data.price_qc import (
    summarize_price_panel,
    summarize_removed_prices,
)
from ..factors.registry import (
    DEFAULT_FUNDAMENTAL_FACTORS,
    DEFAULT_PRICE_FACTORS,
)
from .pipeline_stages import (
    TradablePanels,
    apply_tradable_mask,
    build_score_panels,
    evaluate_pipeline_artifacts,
    load_pipeline_data,
)


def _coerce_qc_stat_value(stat_name: str, stat_value: float) -> int | float:
    """Keep count-like QC stats as integers for presentation."""
    if stat_name.startswith("n_"):
        return int(round(float(stat_value)))
    return float(stat_value)


@dataclass(frozen=True)
class PipelineSettings:
    """Runtime settings for one end-to-end factor pipeline run."""

    start_date: str = "2008-01-01"
    end_date: str | None = None
    rebalance_freq: str = "ME"
    ic_min_assets: int = 50
    neutralization_mode: str = "none"
    # Market-proxy asset SimFinId used by market neutralization mode.
    market_simfin_id: int | None = None
    price_factors: tuple[str, ...] = DEFAULT_PRICE_FACTORS
    fundamental_factors: tuple[str, ...] = DEFAULT_FUNDAMENTAL_FACTORS
    output_factor: str = "composite"
    ridge_alpha: float = 10.0
    ridge_window_type: str = "rolling"
    ridge_window_size: int | None = None
    ridge_min_train_periods: int = 24
    ridge_refit_every_n_rebalances: int = 1
    ridge_min_train_rows: int = 500
    turnover_cost_rate: float = 0.001
    borrow_cost_rate_annual: float = 0.0
    simfin_api_key: str | None = None
    simfin_data_dir: str | None = None
    simfin_refresh_days: int = 30
    simfin_publish_shift_business_days: int = 1
    universe_max_tickers: int | None = None
    dynamic_universe_adv_lookback: int = 63
    min_assets_per_date: int | None = 100
    min_prev_close: float | None = 1.0
    price_clean_hard_max_return: float = 10.0
    price_clean_classify_min_return: float = 3.0
    price_clean_reversal_next_abs_return: float = 0.6
    price_clean_reversal_two_day_net_abs: float = 0.2
    price_clean_stale_lookback_days: int = 5
    price_clean_stale_abs_return_epsilon: float = 1e-12
    price_clean_shift_horizon_days: int = 5
    price_clean_shift_stability_abs_threshold: float = 0.2
    price_quality_lookback_days: int = 252
    price_quality_max_hard_events: int = 3
    price_quality_min_coverage_frac: float = 0.95


@dataclass
class PipelineCoreResult:
    """Core pipeline panels and QC tables."""

    prices: pd.DataFrame
    qc_raw_daily: pd.DataFrame
    qc_tradable_daily: pd.DataFrame
    qc_cleaning_diff: pd.DataFrame
    factor_qc_stats: pd.DataFrame
    factor_scores: dict[str, pd.DataFrame]
    future_returns: pd.DataFrame


@dataclass
class PipelineResult(PipelineCoreResult):
    """Stores the full pipeline output including the compact evaluation summary."""

    evaluation_summary: dict[str, float]


def _validate_pipeline_settings(settings: PipelineSettings) -> None:
    """Validates pipeline settings."""
    if settings.ic_min_assets <= 0:
        raise ValueError("ic_min_assets must be positive.")
    if settings.min_assets_per_date is not None and settings.min_assets_per_date <= 0:
        raise ValueError("min_assets_per_date must be positive when provided.")
    if settings.min_prev_close is not None and settings.min_prev_close <= 0.0:
        raise ValueError("min_prev_close must be positive when provided.")
    if settings.price_clean_hard_max_return <= settings.price_clean_classify_min_return:
        raise ValueError(
            "price_clean_hard_max_return must be greater than price_clean_classify_min_return."
        )
    if settings.price_quality_lookback_days <= 0:
        raise ValueError("price_quality_lookback_days must be positive.")
    if settings.price_quality_max_hard_events < 0:
        raise ValueError("price_quality_max_hard_events must be non-negative.")
    if not (0.0 <= settings.price_quality_min_coverage_frac <= 1.0):
        raise ValueError("price_quality_min_coverage_frac must be in [0, 1].")


@dataclass
class _PreparedPipelineRun:
    """Internal bundle of prepared core panels plus evaluation inputs."""

    core_result: PipelineCoreResult
    tradable_panels: TradablePanels


def _prepare_pipeline_run(settings: PipelineSettings) -> _PreparedPipelineRun:
    """Runs the core pipeline stages and prepares inputs for evaluation."""
    _validate_pipeline_settings(settings)

    loaded = load_pipeline_data(settings)
    built_scores = build_score_panels(
        settings=settings,
        prices=loaded.prices,
        market_data=loaded.market_data,
    )
    tradable_panels = apply_tradable_mask(
        all_scores=built_scores.all_scores,
        selected_scores=built_scores.selected_scores,
        prices=loaded.prices,
        dollar_volume=loaded.dollar_volume,
        aligned_future_returns_raw=built_scores.aligned_future_returns_raw,
        max_tickers=settings.universe_max_tickers,
        adv_lookback=settings.dynamic_universe_adv_lookback,
        quality_mask=loaded.price_quality_mask,
        min_prev_close=settings.min_prev_close,
    )
    tradable_price_mask = tradable_panels.tradable_mask.reindex(
        index=loaded.prices.index,
        columns=loaded.prices.columns,
        fill_value=False,
    ).astype(bool)
    tradable_prices = loaded.prices.where(tradable_price_mask)
    qc_raw_daily = summarize_price_panel(loaded.prices_raw, scope="raw_daily")
    qc_tradable_daily = summarize_price_panel(
        tradable_prices,
        scope="tradable_daily",
    )
    for stat_name, stat_value in loaded.price_clean_summary_stats.items():
        qc_value = _coerce_qc_stat_value(stat_name, stat_value)
        qc_raw_daily[f"clean_{stat_name}"] = qc_value
        qc_tradable_daily[f"clean_{stat_name}"] = qc_value
    qc_cleaning_diff = summarize_removed_prices(
        loaded.prices_raw,
        loaded.prices,
    )
    factor_qc_stats = summarize_factor_panels(
        tradable_panels.all_scores,
        coverage_base_mask=tradable_panels.tradable_mask,
    )
    core_result = PipelineCoreResult(
        prices=loaded.prices,
        qc_raw_daily=qc_raw_daily,
        qc_tradable_daily=qc_tradable_daily,
        qc_cleaning_diff=qc_cleaning_diff,
        factor_qc_stats=factor_qc_stats,
        factor_scores=tradable_panels.all_scores,
        future_returns=tradable_panels.aligned_future_returns,
    )
    return _PreparedPipelineRun(
        core_result=core_result,
        tradable_panels=tradable_panels,
    )


def run_pipeline_core(settings: PipelineSettings) -> PipelineCoreResult:
    """Runs the core pipeline stages and returns panels plus QC tables."""
    return _prepare_pipeline_run(settings).core_result


def run_pipeline(settings: PipelineSettings) -> PipelineResult:
    """Runs the full pipeline and returns core panels plus a compact evaluation summary."""
    prepared = _prepare_pipeline_run(settings)
    evaluation = evaluate_pipeline_artifacts(
        settings=settings,
        prices=prepared.core_result.prices,
        tradable_panels=prepared.tradable_panels,
    )
    return PipelineResult(
        prices=prepared.core_result.prices,
        qc_raw_daily=prepared.core_result.qc_raw_daily,
        qc_tradable_daily=prepared.core_result.qc_tradable_daily,
        qc_cleaning_diff=prepared.core_result.qc_cleaning_diff,
        factor_qc_stats=evaluation.factor_qc_stats,
        factor_scores=prepared.core_result.factor_scores,
        future_returns=prepared.core_result.future_returns,
        evaluation_summary={
            "mean_ic": float(evaluation.ic_stats["mean_ic"]),
            "t_newey_west": float(evaluation.ic_stats["t_newey_west"]),
            "ic_n_obs": float(evaluation.ic_stats["n_obs"]),
            "sharpe": float(evaluation.performance_stats["sharpe"]),
            "max_drawdown": float(evaluation.performance_stats["max_drawdown"]),
        },
    )
