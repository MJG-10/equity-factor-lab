from typing import TYPE_CHECKING
import pandas as pd
from ..data.simfin_prices import build_adj_close_panel, load_universe_market_data
from ..data.simfin_signals import load_universe_signals
from ..data.simfin_universe import load_sector_map
from ..factors.price_factors import standardize_cross_section, winsorize_cross_section
from ..factors.registry import (
    get_fundamental_factor_builders,
    get_price_factor_builders,
    validate_selected_factors,
)
from ..models.neutralization import (
    _compute_rolling_market_betas,
    neutralize_scores,
    uses_market_neutralization,
    uses_sector_neutralization,
    validate_neutralization_mode,
)

if TYPE_CHECKING:
    from .pipeline import PipelineSettings

FIXED_WINSOR_LOWER_Q = 0.01
FIXED_WINSOR_UPPER_Q = 0.99


def _compute_base_future_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Computes one-step-ahead returns from the filtered price panel."""
    returns = prices.pct_change(fill_method=None)
    future_returns = returns.shift(-1).dropna(how="all")
    return future_returns


def _align_factor_panels(
    factor_scores: dict[str, pd.DataFrame],
    future_returns: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Aligns factor score panels and forward returns to common dates while
    preserving the full future-return asset set.
    """
    if not factor_scores:
        raise ValueError("No factor scores were produced.")

    common_index = future_returns.index

    for scores in factor_scores.values():
        common_index = common_index.intersection(scores.index)

    common_columns = future_returns.columns
    if len(common_index) == 0 or len(common_columns) == 0:
        raise ValueError("No common index/columns found when aligning factors and forward returns.")

    aligned_future_returns = future_returns.loc[common_index, common_columns]
    aligned_factor_scores = {
        name: scores.reindex(index=common_index, columns=common_columns)
        for name, scores in factor_scores.items()
    }
    return aligned_factor_scores, aligned_future_returns


def _apply_fixed_winsorization(
    factor_scores: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Applies fixed cross-sectional winsorization to each factor panel."""
    return {
        name: winsorize_cross_section(
            scores,
            lower_q=FIXED_WINSOR_LOWER_Q,
            upper_q=FIXED_WINSOR_UPPER_Q,
        )
        for name, scores in factor_scores.items()
    }


def _standardize_factor_panels(
    factor_scores: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Applies a single final cross-sectional z-score to each factor panel."""
    return {
        name: standardize_cross_section(scores)
        for name, scores in factor_scores.items()
    }


def _combine_equal_weight(
    factor_scores: dict[str, pd.DataFrame],
    *,
    price_factor_names: tuple[str, ...],
    fundamental_factor_names: tuple[str, ...],
) -> pd.DataFrame:
    """
    Builds a raw equal-weight composite from available factor values.
    Requires minimum coverage across total/price/fundamental buckets:
    at least min(5, total_factors), min(2, price_factors), and
    min(2, fundamental_factors) non-null components per date/asset.
    """
    if not factor_scores:
        raise ValueError("factor_scores cannot be empty.")

    first = next(iter(factor_scores.values()))
    total = first * 0.0
    total_count = first.notna().astype(float) * 0.0
    price_count = first.notna().astype(float) * 0.0
    fundamental_count = first.notna().astype(float) * 0.0

    price_set = set(price_factor_names)
    fundamental_set = set(fundamental_factor_names)

    for name, scores in factor_scores.items():
        aligned = scores.reindex_like(first)
        present = aligned.notna().astype(float)
        total = total + aligned.fillna(0.0)
        total_count = total_count + present
        if name in price_set:
            price_count = price_count + present
        if name in fundamental_set:
            fundamental_count = fundamental_count + present

    required_total = min(5, len(factor_scores))
    required_price = min(2, len(price_set))
    required_fundamental = min(2, len(fundamental_set))

    composite = total.div(total_count.where(total_count > 0.0))
    composite = composite.where(total_count >= float(required_total))
    composite = composite.where(price_count >= float(required_price))
    composite = composite.where(fundamental_count >= float(required_fundamental))
    return composite


def _load_market_returns(
    settings: "PipelineSettings",
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """Loads daily returns for the market-proxy asset used in neutralization."""
    if settings.market_simfin_id is None:
        raise ValueError(
            "market_simfin_id is required when using market neutralization modes."
        )
    market_simfin_id = int(settings.market_simfin_id)

    market_data = load_universe_market_data(
        simfin_ids=[market_simfin_id],
        refresh_days=settings.simfin_refresh_days,
        start=settings.start_date,
        end=settings.end_date,
        api_key=settings.simfin_api_key,
        data_dir=settings.simfin_data_dir,
    )
    market_prices = build_adj_close_panel(market_data)

    if market_simfin_id not in market_prices.columns:
        raise ValueError(
            f"Could not load market SimFinId '{market_simfin_id}' from local data."
        )

    market_returns = market_prices[market_simfin_id].pct_change(fill_method=None).reindex(target_index)
    if market_returns.notna().sum() < 30:
        raise ValueError(
            f"Insufficient market return history for SimFinId '{market_simfin_id}' "
            "to run market neutralization."
        )

    return market_returns


def _neutralize_factor_panels(
    factor_scores: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    settings: "PipelineSettings",
) -> dict[str, pd.DataFrame]:
    """Applies the selected neutralization mode to all factor score panels."""
    mode = validate_neutralization_mode(settings.neutralization_mode)
    if mode == "none":
        return factor_scores

    simfin_ids = prices.columns.astype(int).tolist()
    sector_map: pd.Series | None = None
    market_betas: pd.DataFrame | None = None

    if uses_sector_neutralization(mode):
        sector_map = load_sector_map(
            simfin_ids=simfin_ids,
            api_key=settings.simfin_api_key,
            data_dir=settings.simfin_data_dir,
            refresh_days=settings.simfin_refresh_days,
        )
    if uses_market_neutralization(mode):
        market_returns = _load_market_returns(
            settings=settings,
            target_index=prices.index,
        )
        asset_returns = prices.pct_change(fill_method=None)
        market_betas = _compute_rolling_market_betas(
            asset_returns=asset_returns,
            market_returns=market_returns.reindex(asset_returns.index),
        )

    return {
        name: neutralize_scores(
            scores=scores,
            mode=mode,
            sector_map=sector_map,
            market_betas=market_betas,
        )
        for name, scores in factor_scores.items()
    }


def _build_price_factor_scores(
    prices: pd.DataFrame,
    factor_names: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Builds the requested price-factor score panels."""
    builders = get_price_factor_builders()
    validate_selected_factors(factor_names, builders, label="price")
    return {name: builders[name](prices) for name in factor_names}


def _build_fundamental_factor_scores(
    prices: pd.DataFrame,
    factor_names: tuple[str, ...],
    settings: "PipelineSettings",
    market_data: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Builds requested fundamental-factor score panels from SimFin signals."""
    builders = get_fundamental_factor_builders()
    validate_selected_factors(factor_names, builders, label="fundamental")

    signals = load_universe_signals(
        market_data=market_data,
        start=settings.start_date,
        end=settings.end_date,
        api_key=settings.simfin_api_key,
        data_dir=settings.simfin_data_dir,
        refresh_days=settings.simfin_refresh_days,
        publish_shift_business_days=settings.simfin_publish_shift_business_days,
    )
    return {name: builders[name](prices, signals) for name in factor_names}
