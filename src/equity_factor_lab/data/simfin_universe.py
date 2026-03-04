import warnings
from pathlib import Path
import pandas as pd
import simfin as sf
from simfin.names import INDUSTRY_ID, SECTOR, SIMFIN_ID
from .simfin_setup import configure_simfin


def load_company_universe_simfin_ids(
    *,
    api_key: str | None = None,
    data_dir: str | Path | None = None,
    market: str = "us",
    refresh_days: int = 30,
) -> list[int]:
    """Loads a broad SimFinId universe from SimFin company metadata."""
    configure_simfin(api_key=api_key, data_dir=data_dir)

    companies = sf.load_companies(market=market, index=SIMFIN_ID, refresh_days=refresh_days)
    if companies.empty:
        raise ValueError("SimFin company universe is empty.")

    companies = companies.loc[~companies.index.duplicated(keep="first")]
    simfin_ids = pd.to_numeric(pd.Series(companies.index, copy=False), errors="coerce")
    simfin_ids = simfin_ids.dropna().astype(int).unique().tolist()
    if not simfin_ids:
        raise ValueError("No SimFinIds found in SimFin company universe.")
    return simfin_ids


def load_sector_map(
    simfin_ids: list[int],
    api_key: str | None = None,
    data_dir: str | Path | None = None,
    market: str = "us",
    refresh_days: int = 30,
) -> pd.Series:
    """Loads a SimFinId-to-sector mapping from SimFin company metadata."""
    configure_simfin(api_key=api_key, data_dir=data_dir)

    companies = sf.load_companies(market=market, index=SIMFIN_ID, refresh_days=refresh_days)
    if companies.empty:
        raise ValueError("SimFin companies metadata is empty; cannot build sector map.")
    companies.index = pd.to_numeric(companies.index, errors="coerce")
    companies = companies.loc[companies.index.notna()].copy()
    if companies.empty:
        raise ValueError("No valid SimFinIds found in SimFin companies metadata.")
    companies.index = companies.index.astype(int)

    if SECTOR in companies.columns:
        sector_raw = companies[SECTOR].replace("", pd.NA)
    elif INDUSTRY_ID in companies.columns:
        industries = sf.load_industries(index=INDUSTRY_ID, refresh_days=refresh_days)
        if industries.empty:
            raise ValueError("SimFin industries metadata is empty; cannot derive sectors.")
        if SECTOR not in industries.columns:
            raise KeyError(
                f"SimFin industries dataset is missing '{SECTOR}' column. "
                f"Available columns: {list(industries.columns)}"
            )
        industries = industries[~industries.index.duplicated(keep="first")]
        sector_lookup = industries[SECTOR].replace("", pd.NA)
        sector_lookup.index = pd.to_numeric(sector_lookup.index, errors="coerce")
        company_industry_id = pd.to_numeric(companies[INDUSTRY_ID], errors="coerce")
        sector_raw = company_industry_id.map(sector_lookup)
    else:
        raise KeyError(
            "SimFin companies dataset is missing both "
            f"'{SECTOR}' and '{INDUSTRY_ID}' columns. "
            f"Available columns: {list(companies.columns)}"
        )

    # Duplicate SimFinId rows may exist; keep the first non-null sector per id.
    sector_by_simfin_id = sector_raw.groupby(companies.index, sort=True).first()
    sector_by_simfin_id.name = SECTOR

    simfin_id_series = pd.Series(simfin_ids, copy=False)
    if simfin_id_series.isna().any():
        raise ValueError("simfin_ids contains missing SimFinId values.")
    try:
        simfin_id_numeric = pd.to_numeric(simfin_id_series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("simfin_ids contains non-numeric SimFinId values.") from exc
    simfin_id_as_int = simfin_id_numeric.astype(int)
    if not simfin_id_numeric.eq(simfin_id_as_int).all():
        raise ValueError("simfin_ids contains non-integer SimFinId values.")
    simfin_id_index = pd.Index(simfin_id_as_int.tolist(), name=SIMFIN_ID)
    if simfin_id_index.empty:
        raise ValueError("No valid SimFinIds provided to build sector map.")
    sectors = sector_by_simfin_id.reindex(simfin_id_index).replace("", pd.NA)
    if sectors.notna().sum() == 0:
        raise ValueError(
            "No sector classifications available for requested SimFinIds. "
            "Sector neutralization cannot run with all sectors missing."
        )

    missing = sectors[sectors.isna()].index.tolist()
    if missing:
        warnings.warn(
            f"Missing SimFin sector classification for {len(missing)} SimFinId(s). "
            f"Assigning 'Unknown' sector. Examples: {missing[:20]}"
        )
        sectors.loc[missing] = "Unknown"

    sectors = sectors.astype(str)
    sectors.index = simfin_id_index

    return sectors
