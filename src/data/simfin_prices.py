from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import simfin as sf
from simfin.names import DATE, SIMFIN_ID, TICKER
from .simfin_setup import configure_simfin


def _require_multiindex_levels(
    data_frame: pd.DataFrame,
    *,
    required_levels: tuple[str, ...],
    context: str,
) -> list[str]:
    """Ensures a DataFrame uses a MultiIndex with the required level names."""
    if not isinstance(data_frame.index, pd.MultiIndex):
        raise TypeError(f"{context} must use a MultiIndex.")
    index_names = list(data_frame.index.names)
    missing_levels = [level for level in required_levels if level not in index_names]
    if missing_levels:
        raise ValueError(
            f"{context} index must include {required_levels}. Found: {index_names}"
        )
    return index_names


def normalize_id_and_filter_universe(
    data_frame: pd.DataFrame,
    simfin_ids: set[int],
    *,
    dedupe: bool,
    context: str,
) -> pd.DataFrame:
    """Normalizes SimFinId values and filters rows to the requested SimFinId universe."""
    if data_frame.empty:
        return data_frame
    index_names = _require_multiindex_levels(
        data_frame,
        required_levels=(SIMFIN_ID,),
        context=context,
    )

    numeric_ids = pd.to_numeric(
        pd.Series(data_frame.index.get_level_values(SIMFIN_ID), copy=False),
        errors="coerce",
    )
    invalid_ids = numeric_ids.isna()
    if invalid_ids.any():
        warnings.warn(
            f"Dropping {int(invalid_ids.sum())} row(s) from {context} due to invalid SimFinId values."
        )

    keep_rows = (~invalid_ids) & numeric_ids.isin(simfin_ids)
    filtered = data_frame.loc[keep_rows.to_numpy()]
    if filtered.empty:
        return filtered

    filtered_data = filtered.reset_index()
    filtered_data[SIMFIN_ID] = numeric_ids.loc[keep_rows].astype(int).to_numpy()

    if dedupe:
        duplicate_rows = int(filtered_data.duplicated(subset=index_names, keep="last").sum())
        if duplicate_rows > 0:
            warnings.warn(
                f"Dropping {duplicate_rows} duplicate row(s) in {context} after SimFinId canonicalization."
            )
        filtered_data = filtered_data.drop_duplicates(subset=index_names, keep="last")

    return filtered_data.set_index(index_names).sort_index()


def load_universe_market_data(
    *,
    simfin_ids: list[int],
    refresh_days: int,
    start: str = "2008-01-01",
    end: str | None = None,
    api_key: str | None = None,
    data_dir: str | Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Loads SimFin daily shareprices filtered and normalized to a SimFinId universe."""
    configure_simfin(api_key=api_key, data_dir=data_dir)

    universe_ids = (
        pd.to_numeric(pd.Series(simfin_ids, copy=False), errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if not universe_ids:
        raise ValueError("No valid SimFinIds provided for SimFin market-data loading.")

    market_data = sf.load_shareprices(
        variant="daily",
        market="us",
        start_date=start,
        end_date=end,
        refresh_days=refresh_days,
        index=[SIMFIN_ID, DATE],
    )
    market_data = normalize_id_and_filter_universe(
        market_data,
        simfin_ids=set(universe_ids),
        dedupe=True,
        context="market_data",
    )
    if market_data.empty:
        raise ValueError("No SimFin market data found for the requested SimFinId universe.")

    if verbose:
        print(f"Loaded SimFin market data with shape {market_data.shape}.")
    return market_data


def build_adj_close_panel(
    market_data: pd.DataFrame,
    *,
    asset_level: str = SIMFIN_ID,
) -> pd.DataFrame:
    """Builds a date-by-asset adjusted-close panel from shareprice rows.

    Non-finite and non-positive adjusted closes are treated as missing.
    """
    price_column = "Adj. Close"
    if price_column not in market_data.columns:
        raise KeyError(
            f"SimFin market data missing '{price_column}' column. "
            f"Available: {list(market_data.columns)}"
        )

    adj_close_panel = (
        market_data[price_column]
        .unstack(asset_level)
        .sort_index()
        .dropna(how="all")
    )
    adj_close_panel = adj_close_panel.astype(float)
    finite_positive = np.isfinite(adj_close_panel) & (adj_close_panel > 0.0)
    adj_close_panel = adj_close_panel.where(finite_positive).dropna(how="all")
    if adj_close_panel.empty:
        raise ValueError("Adjusted-close panel is empty after wide conversion.")
    return adj_close_panel


def build_close_dollar_volume_panel(
    market_data: pd.DataFrame,
    *,
    asset_level: str = SIMFIN_ID,
) -> pd.DataFrame:
    """Builds a date-by-asset dollar-volume panel (Close * Volume) from shareprice rows."""
    price_column = "Close"
    volume_column = "Volume"
    missing_columns = [
        column_name
        for column_name in (price_column, volume_column)
        if column_name not in market_data.columns
    ]
    if missing_columns:
        raise KeyError(
            "SimFin market data missing required columns for dollar volume: "
            f"{missing_columns}. Available: {list(market_data.columns)}"
        )

    dollar_volume_panel = (
        market_data[price_column] * market_data[volume_column]
    ).unstack(asset_level).sort_index().dropna(how="all")
    if dollar_volume_panel.empty:
        raise ValueError("Dollar-volume panel is empty after wide conversion.")
    return dollar_volume_panel


def build_simfin_id_to_latest_ticker_map(market_data: pd.DataFrame) -> pd.Series:
    """
    Builds a SimFinId to ticker mapping from shareprice rows for display/reporting.

    Uses the last non-null ticker observed for each SimFinId.
    """
    if market_data.empty:
        return pd.Series(dtype="string", name=TICKER)

    if not isinstance(market_data.index, pd.MultiIndex):
        raise TypeError("market_data must be a MultiIndex DataFrame.")
    if SIMFIN_ID not in list(market_data.index.names) or DATE not in list(market_data.index.names):
        raise ValueError(
            "market_data index must include both "
            f"'{SIMFIN_ID}' and '{DATE}'. Found: {list(market_data.index.names)}"
        )
    if TICKER not in market_data.columns:
        raise KeyError(
            f"market_data is missing '{TICKER}' column required to build a ticker mapping."
        )

    tickers = (
        market_data[TICKER]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
    )
    mapping = tickers.groupby(level=SIMFIN_ID, sort=True).last()
    mapping.index.name = SIMFIN_ID
    mapping.name = TICKER
    return mapping
