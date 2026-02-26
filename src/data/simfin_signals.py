from pathlib import Path
import numpy as np
import pandas as pd
import simfin as sf
from simfin.names import (
    ASSETS_GROWTH,
    ASSETS_GROWTH_QOQ,
    ASSETS_GROWTH_YOY,
    BOOK_MARKET,
    EARNINGS_GROWTH,
    EARNINGS_GROWTH_QOQ,
    EARNINGS_GROWTH_YOY,
    FCF,
    FCF_GROWTH,
    FCF_GROWTH_QOQ,
    FCF_GROWTH_YOY,
    NET_INCOME,
    OPERATING_INCOME,
    OPERATING_MARGIN,
    PBOOK,
    REVENUE,
    SALES_GROWTH,
    SALES_GROWTH_QOQ,
    SALES_GROWTH_YOY,
    SIMFIN_ID,
    TOTAL_ASSETS,
)
from .simfin_fundamentals import FUNDAMENTAL_DATE_INDEX, load_universe_financial_statements


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
    return frame.replace([np.inf, -np.inf], np.nan)


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
    income_quarterly: pd.DataFrame,
    balance_ttm: pd.DataFrame,
    balance_quarterly: pd.DataFrame,
    cashflow_ttm: pd.DataFrame,
    cashflow_quarterly: pd.DataFrame,
    market_data: pd.DataFrame | None,
    fill_method: str = "ffill",
    offset: pd.DateOffset | None = None,
    func=None,
    date_index: str = FUNDAMENTAL_DATE_INDEX,
    group_index: str = SIMFIN_ID,
) -> pd.DataFrame:
    """Builds growth signals while explicitly grouping by the requested index key."""
    ttm_frame = pd.concat(
        [
            income_ttm[[REVENUE, NET_INCOME]],
            sf.free_cash_flow(cashflow_ttm),
            balance_ttm[[TOTAL_ASSETS]],
        ],
        axis=1,
    )
    ttm_name_map = {
        REVENUE: SALES_GROWTH,
        NET_INCOME: EARNINGS_GROWTH,
        FCF: FCF_GROWTH,
        TOTAL_ASSETS: ASSETS_GROWTH,
    }
    growth_ttm = sf.rel_change(
        df=ttm_frame,
        freq="q",
        quarters=4,
        future=False,
        annualized=False,
        new_names=ttm_name_map,
        group_index=group_index,
    )

    quarterly_frame = pd.concat(
        [
            income_quarterly[[REVENUE, NET_INCOME]],
            sf.free_cash_flow(cashflow_quarterly),
            balance_quarterly[[TOTAL_ASSETS]],
        ],
        axis=1,
    )
    yoy_name_map = {
        REVENUE: SALES_GROWTH_YOY,
        NET_INCOME: EARNINGS_GROWTH_YOY,
        FCF: FCF_GROWTH_YOY,
        TOTAL_ASSETS: ASSETS_GROWTH_YOY,
    }
    growth_yoy = sf.rel_change(
        df=quarterly_frame,
        freq="q",
        quarters=4,
        future=False,
        annualized=False,
        new_names=yoy_name_map,
        group_index=group_index,
    )

    qoq_name_map = {
        REVENUE: SALES_GROWTH_QOQ,
        NET_INCOME: EARNINGS_GROWTH_QOQ,
        FCF: FCF_GROWTH_QOQ,
        TOTAL_ASSETS: ASSETS_GROWTH_QOQ,
    }
    growth_qoq = sf.rel_change(
        df=quarterly_frame,
        freq="q",
        quarters=1,
        future=False,
        annualized=False,
        new_names=qoq_name_map,
        group_index=group_index,
    )

    growth_signals = pd.concat([growth_ttm, growth_yoy, growth_qoq], axis=1)

    if offset is not None:
        growth_signals = sf.add_date_offset(
            df=growth_signals,
            offset=offset,
            date_index=date_index,
        )

    if func is not None:
        growth_signals = sf.apply(
            df=growth_signals,
            func=func,
            group_index=group_index,
        )

    if market_data is not None:
        growth_signals = sf.reindex(
            df_src=growth_signals,
            df_target=market_data,
            method=fill_method,
            group_index=group_index,
        )

    growth_signals.sort_index(axis="columns", inplace=True)
    return growth_signals


def load_universe_signals(
    *,
    market_data: pd.DataFrame,
    start: str,
    end: str | None = None,
    api_key: str | None = None,
    data_dir: str | Path | None = None,
    verbose: bool = False,
    refresh_days: int = 30,
    align_to_publish_date: bool = True,
    publish_shift_business_days: int = 1,
    fundamentals_duplicate_policy: str = "keep_last",
) -> pd.DataFrame:
    """
    Loads SimFin valuation, quality, and growth signals for a market-data universe.
    Expects market data indexed by [SimFinId, Date].
    """
    if verbose:
        print("simfin version:", sf.__version__)

    universe_simfin_ids = _get_universe_from_market_data(market_data)

    if verbose:
        print("Using SimFin market data with shape:", market_data.shape)
        print(market_data.head())

    statements = load_universe_financial_statements(
        simfin_ids=universe_simfin_ids,
        start=start,
        end=end,
        api_key=api_key,
        data_dir=data_dir,
        refresh_days=refresh_days,
        align_to_publish_date=align_to_publish_date,
        publish_shift_business_days=publish_shift_business_days,
        duplicate_policy=fundamentals_duplicate_policy,
    )
    income_ttm = statements["income_ttm"]
    income_quarterly = statements["income_quarterly"]
    balance_ttm = statements["balance_ttm"]
    balance_quarterly = statements["balance_quarterly"]
    cashflow_ttm = statements["cashflow_ttm"]
    cashflow_quarterly = statements["cashflow_quarterly"]

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
        income_quarterly=income_quarterly,
        balance_ttm=balance_ttm,
        balance_quarterly=balance_quarterly,
        cashflow_ttm=cashflow_ttm,
        cashflow_quarterly=cashflow_quarterly,
        market_data=market_data,
        date_index=FUNDAMENTAL_DATE_INDEX,
        group_index=SIMFIN_ID,
    )

    signals = pd.concat([valuation_signals, financial_signals, growth_signals], axis=1)
    _raise_on_duplicate_signal_columns(signals)

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
    signals = _replace_infinite_with_nan(signals)
    signals.sort_index(axis="columns", inplace=True)

    if verbose:
        print("Loaded SimFin signals with shape:", signals.shape)
        print(signals.head())
        print("SimFin signal columns:", get_signal_columns(signals))

    return signals
