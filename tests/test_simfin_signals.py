import numpy as np
import pandas as pd
import pytest
from simfin.names import (
    ASSETS_GROWTH,
    EARNINGS_GROWTH,
    NET_INCOME,
    OPERATING_INCOME,
    OPERATING_MARGIN,
    REVENUE,
    SALES_GROWTH,
    SIMFIN_ID,
    TOTAL_ASSETS,
)

from equity_factor_lab.data.simfin_signals import (
    OPERATING_INCOME_GROWTH,
    _build_growth_signals,
    _build_operating_margin_fallback,
    _get_universe_from_market_data,
    _raise_on_duplicate_signal_columns,
    _replace_infinite_with_nan,
)


def test_build_operating_margin_fallback_avoids_infinite_values() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            (1001, pd.Timestamp("2020-03-31")),
            (1001, pd.Timestamp("2020-06-30")),
        ],
        names=[SIMFIN_ID, "Date"],
    )
    income_ttm = pd.DataFrame(
        {
            OPERATING_INCOME: [10.0, -5.0],
            REVENUE: [0.0, 20.0],
        },
        index=index,
    )

    op_margin = _build_operating_margin_fallback(income_ttm)

    assert pd.isna(op_margin.iloc[0, 0])
    assert op_margin.iloc[1, 0] == -0.25
    assert np.isfinite(op_margin.dropna().to_numpy()).all()


def test_replace_infinite_with_nan_replaces_both_signs() -> None:
    frame = pd.DataFrame({"x": [1.0, np.inf, -np.inf]})

    clean = _replace_infinite_with_nan(frame)

    assert clean.loc[0, "x"] == 1.0
    assert pd.isna(clean.loc[1, "x"])
    assert pd.isna(clean.loc[2, "x"])


def test_get_universe_from_market_data_raises_on_missing_ids() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            (1001, pd.Timestamp("2020-01-01")),
            (np.nan, pd.Timestamp("2020-01-02")),
        ],
        names=[SIMFIN_ID, "Date"],
    )
    market_data = pd.DataFrame({"Close": [1.0, 2.0]}, index=index)

    with pytest.raises(ValueError, match="missing SimFinId values"):
        _get_universe_from_market_data(market_data)


def test_get_universe_from_market_data_raises_on_non_integer_ids() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            ("A12", pd.Timestamp("2020-01-01")),
        ],
        names=[SIMFIN_ID, "Date"],
    )
    market_data = pd.DataFrame({"Close": [1.0]}, index=index)

    with pytest.raises(ValueError, match="non-integer SimFinId values"):
        _get_universe_from_market_data(market_data)


def test_raise_on_duplicate_signal_columns() -> None:
    signals = pd.DataFrame(
        [[1.0, 2.0]],
        columns=["dup", "dup"],
    )

    with pytest.raises(ValueError, match="Duplicate signal columns"):
        _raise_on_duplicate_signal_columns(signals)

def test_build_growth_signals_uses_simfin_id_for_rel_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.MultiIndex.from_tuples(
        [
            (1001, pd.Timestamp("2020-03-31")),
            (1001, pd.Timestamp("2020-06-30")),
            (1001, pd.Timestamp("2020-09-30")),
            (1001, pd.Timestamp("2020-12-31")),
            (1001, pd.Timestamp("2021-03-31")),
        ],
        names=[SIMFIN_ID, "Date"],
    )

    income_ttm = pd.DataFrame(
        {
            REVENUE: [10.0, 11.0, 12.0, 13.0, 14.0],
            OPERATING_INCOME: [2.0, 2.1, 2.2, 2.3, 2.4],
            NET_INCOME: [1.0, 1.1, 1.2, 1.3, 1.4],
        },
        index=index,
    )
    balance_quarterly = pd.DataFrame({TOTAL_ASSETS: [100.0, 101.0, 102.0, 103.0, 104.0]}, index=index)

    group_index_calls: list[str] = []

    def fake_rel_change(*, df: pd.DataFrame, new_names: dict[str, str], group_index: str, **kwargs) -> pd.DataFrame:
        group_index_calls.append(group_index)
        return df.rename(columns=new_names).astype(float)

    monkeypatch.setattr("equity_factor_lab.data.simfin_signals.sf.rel_change", fake_rel_change)

    signals = _build_growth_signals(
        income_ttm=income_ttm,
        balance_quarterly=balance_quarterly,
        market_data=None,
    )

    assert group_index_calls == [SIMFIN_ID, SIMFIN_ID]
    expected_columns = {
        SALES_GROWTH,
        OPERATING_INCOME_GROWTH,
        EARNINGS_GROWTH,
        ASSETS_GROWTH,
    }
    assert set(signals.columns) == expected_columns

