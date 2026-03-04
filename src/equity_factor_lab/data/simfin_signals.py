from pathlib import Path
import numpy as np
import pandas as pd
import simfin as sf
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
    NET_INCOME,
    OPERATING_INCOME,
    OPERATING_MARGIN,
    PBOOK,
    REVENUE,
    ROA,
    SALES_GROWTH,
    SIMFIN_ID,
    TOTAL_ASSETS,
)
from .simfin_fundamentals import FUNDAMENTAL_DATE_INDEX, load_universe_financial_statements

OPERATING_INCOME_GROWTH = "Operating Income Growth"

REQUIRED_SIGNAL_COLUMNS: tuple[str, ...] = (
    BOOK_MARKET,
    EARNINGS_YIELD,
    FCF_YIELD,
    DIVIDEND_YIELD,
    ROA,
    ASSET_TURNOVER,
    DEBT_RATIO,
    GROSS_PROFIT_MARGIN,
    OPERATING_MARGIN,
    ASSETS_GROWTH,
    SALES_GROWTH,
    OPERATING_INCOME_GROWTH,
    EARNINGS_GROWTH,
)


def get_signal_columns(signals: pd.DataFrame) -> list[str]:
    """Returns sorted SimFin signal column names."""
    return sorted(str(column) for column in signals.columns)


def _raise_on_duplicate_signal_columns(signals: pd.DataFrame) -> None:
    """Raises when duplicate signal column names are present."""
    duplicate_mask = signals.columns.duplicated(keep=False)
    if not duplicate_mask.any():
        return
    duplicate_columns = sorted({str(column) for column in signals.columns[duplicate_mask]})
    raise ValueError(
        "Duplicate signal columns detected across SimFin signal builders: "
        f"{duplicate_columns}"
    )


def _replace_infinite_with_nan(frame: pd.DataFrame) -> pd.DataFrame:
    """Replaces +/-inf numeric values with NaN to keep factor inputs finite."""
    cleaned = frame.copy(deep=False)
    for column in frame.columns:
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series.dtype):
            continue
        inf_mask = series.eq(np.inf) | series.eq(-np.inf)
        if bool(inf_mask.any()):
            cleaned[column] = series.mask(inf_mask, np.nan)
    return cleaned


def _build_operating_margin_fallback(income_ttm: pd.DataFrame) -> pd.DataFrame:
    """Builds fallback operating margin while avoiding divide-by-zero infinities."""
    revenue = income_ttm[REVENUE].where(income_ttm[REVENUE] != 0.0)
    return (income_ttm[OPERATING_INCOME] / revenue).to_frame(OPERATING_MARGIN)


def _get_universe_from_market_data(market_data: pd.DataFrame) -> set[int]:
    """Extracts the SimFinId universe from market-data index values."""
    if market_data.empty:
        raise ValueError("market_data is empty.")
    if not isinstance(market_data.index, pd.MultiIndex):
        raise TypeError("market_data must use a MultiIndex [SimFinId, Date].")
    if SIMFIN_ID not in list(market_data.index.names):
        raise ValueError(
            "market_data index must include "
            f"'{SIMFIN_ID}'. Found: {list(market_data.index.names)}"
        )

    simfin_ids = pd.Index(market_data.index.get_level_values(SIMFIN_ID), copy=False)
    if simfin_ids.hasnans:
        raise ValueError("market_data index contains missing SimFinId values.")
    try:
        universe_ids = set(simfin_ids.astype(int).tolist())
    except (TypeError, ValueError) as exc:
        raise ValueError("market_data index contains non-integer SimFinId values.") from exc
    if not universe_ids:
        raise ValueError("No valid SimFinIds found in market_data index.")
    return universe_ids


def _build_growth_signals(
    *,
    income_ttm: pd.DataFrame,
    balance_quarterly: pd.DataFrame,
    market_data: pd.DataFrame | None,
) -> pd.DataFrame:
    """Builds only the growth columns used by the current factor builders."""
    income_growth_frame = income_ttm.loc[
        :,
        [column for column in (REVENUE, OPERATING_INCOME, NET_INCOME) if column in income_ttm.columns],
    ]
    income_name_map = {
        REVENUE: SALES_GROWTH,
        OPERATING_INCOME: OPERATING_INCOME_GROWTH,
        NET_INCOME: EARNINGS_GROWTH,
    }
    income_name_map = {
        source_column: target_column
        for source_column, target_column in income_name_map.items()
        if source_column in income_growth_frame.columns
    }
    income_growth_signals = sf.rel_change(
        df=income_growth_frame,
        freq="q",
        quarters=4,
        future=False,
        annualized=False,
        new_names=income_name_map,
        group_index=SIMFIN_ID,
    )
    asset_growth_signals = sf.rel_change(
        df=balance_quarterly[[TOTAL_ASSETS]],
        freq="q",
        quarters=4,
        future=False,
        annualized=False,
        new_names={TOTAL_ASSETS: ASSETS_GROWTH},
        group_index=SIMFIN_ID,
    )
    growth_signals = pd.concat([income_growth_signals, asset_growth_signals], axis=1)

    if market_data is not None:
        growth_signals = sf.reindex(
            df_src=growth_signals,
            df_target=market_data,
            method="ffill",
            group_index=SIMFIN_ID,
        )

    growth_signals = growth_signals.loc[
        :,
        [
            column
            for column in (
                ASSETS_GROWTH,
                SALES_GROWTH,
                OPERATING_INCOME_GROWTH,
                EARNINGS_GROWTH,
            )
            if column in growth_signals.columns
        ],
    ]
    growth_signals.sort_index(axis="columns", inplace=True)
    return growth_signals


def _select_required_signal_columns(signals: pd.DataFrame) -> pd.DataFrame:
    """Keeps only the signal columns consumed by the current factor builders."""
    available_required = [column for column in REQUIRED_SIGNAL_COLUMNS if column in signals.columns]
    if not available_required:
        return signals.iloc[:, 0:0]
    return signals.loc[:, available_required]


def load_universe_signals(
    *,
    market_data: pd.DataFrame,
    start: str,
    end: str | None = None,
    api_key: str | None = None,
    data_dir: str | Path | None = None,
    refresh_days: int = 30,
    publish_shift_business_days: int = 1,
) -> pd.DataFrame:
    """
    Loads SimFin valuation, quality, and growth signals for a market-data universe.
    Expects market data indexed by [SimFinId, Date].
    """
    universe_simfin_ids = _get_universe_from_market_data(market_data)

    statements = load_universe_financial_statements(
        simfin_ids=universe_simfin_ids,
        start=start,
        end=end,
        api_key=api_key,
        data_dir=data_dir,
        refresh_days=refresh_days,
        publish_shift_business_days=publish_shift_business_days,
        duplicate_policy="keep_last",
    )
    income_ttm = statements["income_ttm"]
    balance_ttm = statements["balance_ttm"]
    balance_quarterly = statements["balance_quarterly"]
    cashflow_ttm = statements["cashflow_ttm"]

    valuation_signals = sf.val_signals(
        df_prices=market_data,
        df_income_ttm=income_ttm,
        df_balance_ttm=balance_ttm,
        df_cashflow_ttm=cashflow_ttm,
        date_index=FUNDAMENTAL_DATE_INDEX,
        group_index=SIMFIN_ID,
    )
    financial_signals = sf.fin_signals(
        df_income_ttm=income_ttm,
        df_balance_ttm=balance_ttm,
        df_cashflow_ttm=cashflow_ttm,
        df_prices=market_data,
        date_index=FUNDAMENTAL_DATE_INDEX,
        group_index=SIMFIN_ID,
    )
    growth_signals = _build_growth_signals(
        income_ttm=income_ttm,
        balance_quarterly=balance_quarterly,
        market_data=market_data,
    )

    signals = pd.concat([valuation_signals, financial_signals, growth_signals], axis=1)

    if BOOK_MARKET not in signals.columns and PBOOK in signals.columns:
        pbook = signals[PBOOK].where(signals[PBOOK] != 0.0)
        signals[BOOK_MARKET] = 1.0 / pbook

    if OPERATING_MARGIN not in signals.columns:
        op_margin = _build_operating_margin_fallback(income_ttm)
        op_margin = sf.reindex(
            df_src=op_margin,
            df_target=market_data,
            method="ffill",
            group_index=SIMFIN_ID,
        )
        signals[OPERATING_MARGIN] = op_margin[OPERATING_MARGIN]
    signals = _select_required_signal_columns(signals)
    _raise_on_duplicate_signal_columns(signals)
    signals = _replace_infinite_with_nan(signals)
    signals.sort_index(axis="columns", inplace=True)

    return signals
