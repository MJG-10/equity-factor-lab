import pandas as pd
import pytest
from simfin.names import SECTOR, SIMFIN_ID

from equity_factor_lab.data import simfin_universe


def test_load_sector_map_raises_when_companies_metadata_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(simfin_universe, "configure_simfin", lambda **_: None)
    monkeypatch.setattr(simfin_universe.sf, "load_companies", lambda **_: pd.DataFrame())

    with pytest.raises(ValueError, match="companies metadata is empty"):
        simfin_universe.load_sector_map(simfin_ids=[101, 202])


def test_load_sector_map_raises_when_all_requested_sectors_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    companies = pd.DataFrame(
        {
            SECTOR: [pd.NA, ""],
        },
        index=pd.Index([101, 202], name=SIMFIN_ID),
    )
    monkeypatch.setattr(simfin_universe, "configure_simfin", lambda **_: None)
    monkeypatch.setattr(simfin_universe.sf, "load_companies", lambda **_: companies)

    with pytest.raises(ValueError, match="No sector classifications available"):
        simfin_universe.load_sector_map(simfin_ids=[101, 202])


def test_load_sector_map_warns_and_fills_unknown_for_partial_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    companies = pd.DataFrame(
        {
            SECTOR: ["Technology", pd.NA],
        },
        index=pd.Index([101, 202], name=SIMFIN_ID),
    )
    monkeypatch.setattr(simfin_universe, "configure_simfin", lambda **_: None)
    monkeypatch.setattr(simfin_universe.sf, "load_companies", lambda **_: companies)

    with pytest.warns(UserWarning, match="Missing SimFin sector classification"):
        sectors = simfin_universe.load_sector_map(simfin_ids=[101, 202])

    assert sectors.loc[101] == "Technology"
    assert sectors.loc[202] == "Unknown"


def test_load_sector_map_raises_on_non_integer_simfin_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    companies = pd.DataFrame(
        {
            SECTOR: ["Technology"],
        },
        index=pd.Index([101], name=SIMFIN_ID),
    )
    monkeypatch.setattr(simfin_universe, "configure_simfin", lambda **_: None)
    monkeypatch.setattr(simfin_universe.sf, "load_companies", lambda **_: companies)

    with pytest.raises(ValueError, match="non-integer SimFinId values"):
        simfin_universe.load_sector_map(simfin_ids=[101.5])
