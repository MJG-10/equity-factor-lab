import difflib
import numpy as np
import pandas as pd
from simfin.names import (
    ASSETS_GROWTH,
    ASSET_TURNOVER,
    BOOK_MARKET,
    DEBT_RATIO,
    DIVIDEND_YIELD,
    EARNINGS_GROWTH,
    EARNINGS_YIELD,
    FCF_YIELD,
    GROSS_PROFIT_MARGIN,
    ROA,
    OPERATING_MARGIN,
    SALES_GROWTH,
    SIMFIN_ID,
)
from .price_factors import standardize_cross_section, winsorize_cross_section
from ..data.simfin_signals import OPERATING_INCOME_GROWTH, get_signal_columns

DEFAULT_SIGNAL_FFILL_LIMIT = 252


VALUE_SIGNAL_COLUMNS = (BOOK_MARKET, EARNINGS_YIELD, FCF_YIELD, DIVIDEND_YIELD)
QUALITY_CORE_SIGNAL_COLUMNS = (ROA, ASSET_TURNOVER, DEBT_RATIO)
QUALITY_MARGIN_SIGNAL_COLUMNS = (GROSS_PROFIT_MARGIN, OPERATING_MARGIN)
INVEST_SIGNAL_COLUMNS = (ASSETS_GROWTH,)
GROWTH_SIGNAL_COLUMNS = (SALES_GROWTH, OPERATING_INCOME_GROWTH, EARNINGS_GROWTH)

VALUE_MIN_COMPONENTS = 2
QUALITY_MIN_COMPONENTS = 3
GROWTH_MIN_COMPONENTS = 2


def _require_signal_columns(
    signals: pd.DataFrame,
    required_columns: tuple[str, ...],
    factor_name: str,
) -> None:
    """Raises a helpful error when required SimFin columns are missing."""
    available_columns = get_signal_columns(signals)
    missing_columns = [column for column in required_columns if column not in signals.columns]
    if not missing_columns:
        return

    details: list[str] = []
    for missing in missing_columns:
        suggestions = difflib.get_close_matches(
            str(missing),
            available_columns,
            n=4,
            cutoff=0.5,
        )
        if suggestions:
            details.append(f"- {missing!r} (close matches: {suggestions})")
        else:
            details.append(f"- {missing!r}")

    available_preview = ", ".join(available_columns[:25])
    raise KeyError(
        f"Missing SimFin column(s) for '{factor_name}' factor:\n"
        + "\n".join(details)
        + "\nUpdate the factor column mapping in fundamental_factors.py.\n"
        + f"Available columns (first 25): {available_preview}"
    )


def _combine_component_panels(
    components: list[pd.DataFrame],
    *,
    min_components: int,
) -> pd.DataFrame:
    """Averages available components and requires a minimum non-null count."""
    if not components:
        raise ValueError("components cannot be empty.")
    if min_components <= 0:
        raise ValueError("min_components must be positive.")

    combined_index = components[0].index
    combined_columns = components[0].columns
    for panel in components[1:]:
        combined_index = combined_index.union(panel.index)
        combined_columns = combined_columns.union(panel.columns)

    aligned_components = [
        panel.reindex(index=combined_index, columns=combined_columns)
        for panel in components
    ]
    total = aligned_components[0].fillna(0.0).copy()
    counts = aligned_components[0].notna().astype(float)

    for aligned in aligned_components[1:]:
        total = total + aligned.fillna(0.0)
        counts = counts + aligned.notna().astype(float)

    combined = total.div(counts.where(counts > 0.0))
    combined = combined.where(counts >= float(min_components))
    return combined


def _normalize_component_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Winsorizes and z-scores one component panel before factor aggregation."""
    return standardize_cross_section(winsorize_cross_section(panel))


def _get_first_available_signal_column(
    signals: pd.DataFrame,
    candidate_columns: tuple[str, ...],
    *,
    factor_name: str,
) -> str:
    """Returns the first non-empty signal column from an ordered preference list."""
    for column in candidate_columns:
        if column in signals.columns and bool(signals[column].notna().any()):
            return column

    available_columns = get_signal_columns(signals)
    raise KeyError(
        f"Missing SimFin column(s) for '{factor_name}' factor: {list(candidate_columns)}. "
        f"Available columns (first 25): {', '.join(available_columns[:25])}"
    )


def build_daily_panel_from_signal(
    signals: pd.DataFrame,
    column: str,
    prices_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Takes a SimFin signal column indexed by [SimFinId, Date] and aligns it to a
    daily wide panel on `prices_index` with a fixed forward-fill freshness cap.
    """
    if column not in signals.columns:
        raise KeyError(
            f"Column {column!r} not found in SimFin signals. "
            "Run `print(signals.columns)` and adjust."
        )

    wide = signals[column].unstack(SIMFIN_ID).sort_index()

    wide = wide.reindex(prices_index).sort_index()
    wide = wide.ffill(limit=DEFAULT_SIGNAL_FFILL_LIMIT)
    return wide


def compute_value_scores(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """Builds the value composite from B/M, E/P, FCF/P, and dividend yield."""
    _require_signal_columns(signals, VALUE_SIGNAL_COLUMNS, factor_name="value")

    bm = build_daily_panel_from_signal(signals, BOOK_MARKET, prices.index)
    ep = build_daily_panel_from_signal(signals, EARNINGS_YIELD, prices.index)
    cfp = build_daily_panel_from_signal(signals, FCF_YIELD, prices.index)
    dy = build_daily_panel_from_signal(signals, DIVIDEND_YIELD, prices.index)

    components = [_normalize_component_panel(panel) for panel in [bm, ep, cfp, dy]]
    return _combine_component_panels(
        components,
        min_components=VALUE_MIN_COMPONENTS,
    )


def compute_quality_style_from_signals(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Builds QUALITY_STYLE as a composite of profitability, margins, efficiency, and safety.
    """
    _require_signal_columns(signals, QUALITY_CORE_SIGNAL_COLUMNS, factor_name="quality")
    margin_column = _get_first_available_signal_column(
        signals,
        QUALITY_MARGIN_SIGNAL_COLUMNS,
        factor_name="quality",
    )

    roa = build_daily_panel_from_signal(
        signals, ROA, prices.index,
    )
    margin = build_daily_panel_from_signal(
        signals, margin_column, prices.index,
    )
    asset_turnover = build_daily_panel_from_signal(
        signals,
        ASSET_TURNOVER,
        prices.index,
    )
    debt_ratio = build_daily_panel_from_signal(
        signals,
        DEBT_RATIO,
        prices.index,
    )

    components = [
        _normalize_component_panel(panel)
        for panel in [roa, margin, asset_turnover, -debt_ratio]
    ]
    quality_scores = _combine_component_panels(
        components,
        min_components=QUALITY_MIN_COMPONENTS,
    )
    return quality_scores


def compute_invest_style_from_signals(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Builds INVEST_STYLE as a conservative-minus-aggressive investment factor from asset growth.
    """
    _require_signal_columns(signals, INVEST_SIGNAL_COLUMNS, factor_name="invest")

    asset_growth = build_daily_panel_from_signal(
        signals,
        ASSETS_GROWTH,
        prices.index,
    )

    return -asset_growth


def compute_growth_style_from_signals(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Builds GROWTH_STYLE as a composite of sales, operating-income, and earnings growth.
    """
    _require_signal_columns(signals, GROWTH_SIGNAL_COLUMNS, factor_name="growth")

    sales_g = build_daily_panel_from_signal(
        signals,
        SALES_GROWTH,
        prices.index,
    )
    op_income_g = build_daily_panel_from_signal(
        signals,
        OPERATING_INCOME_GROWTH,
        prices.index,
    )
    eps_g = build_daily_panel_from_signal(
        signals,
        EARNINGS_GROWTH,
        prices.index,
    )

    components = [_normalize_component_panel(panel) for panel in [sales_g, op_income_g, eps_g]]
    growth_scores = _combine_component_panels(
        components,
        min_components=GROWTH_MIN_COMPONENTS,
    )
    return growth_scores
