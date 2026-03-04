import numpy as np
import pandas as pd
from simfin.names import SIMFIN_ID

from equity_factor_lab.data.simfin_prices import (
    build_adj_close_panel,
    build_close_dollar_volume_panel,
)


def _build_mock_market_data_rows() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [
            (101, pd.Timestamp("2020-01-01")),
            (101, pd.Timestamp("2020-01-02")),
            (202, pd.Timestamp("2020-01-01")),
            (202, pd.Timestamp("2020-01-02")),
            (303, pd.Timestamp("2020-01-02")),
        ],
        names=[SIMFIN_ID, "Date"],
    )
    return pd.DataFrame(
        {
            "Adj. Close": [10.0, 11.0, 20.0, 21.0, 99.0],
            "Close": [10.0, 11.0, 20.0, 21.0, 99.0],
            "Volume": [100.0, 110.0, 200.0, 210.0, 1.0],
        },
        index=index,
    )


def test_build_adj_close_panel_from_market_data_rows() -> None:
    market_data_rows = _build_mock_market_data_rows()

    prices = build_adj_close_panel(market_data_rows)

    assert list(prices.columns) == [101, 202, 303]
    assert prices.loc[pd.Timestamp("2020-01-02"), 101] == 11.0
    assert prices.loc[pd.Timestamp("2020-01-02"), 202] == 21.0


def test_build_close_dollar_volume_panel_from_market_data_rows() -> None:
    market_data_rows = _build_mock_market_data_rows()

    dollar_volume = build_close_dollar_volume_panel(market_data_rows)

    assert list(dollar_volume.columns) == [101, 202, 303]
    assert dollar_volume.loc[pd.Timestamp("2020-01-01"), 101] == 1000.0
    assert dollar_volume.loc[pd.Timestamp("2020-01-02"), 202] == 4410.0


def test_build_adj_close_panel_sanitizes_nonpositive_and_nonfinite_values() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            (101, pd.Timestamp("2020-01-01")),
            (101, pd.Timestamp("2020-01-02")),
            (202, pd.Timestamp("2020-01-01")),
            (202, pd.Timestamp("2020-01-02")),
            (303, pd.Timestamp("2020-01-01")),
            (303, pd.Timestamp("2020-01-02")),
        ],
        names=[SIMFIN_ID, "Date"],
    )
    market_data_rows = pd.DataFrame(
        {
            "Adj. Close": [10.0, 11.0, 0.0, 22.0, np.inf, 33.0],
            "Close": [10.0, 11.0, 0.0, 22.0, np.inf, 33.0],
            "Volume": [100.0, 110.0, 200.0, 210.0, 1.0, 1.0],
        },
        index=index,
    )

    prices = build_adj_close_panel(market_data_rows)

    assert np.isnan(prices.loc[pd.Timestamp("2020-01-01"), 202])
    assert np.isnan(prices.loc[pd.Timestamp("2020-01-01"), 303])
    assert prices.loc[pd.Timestamp("2020-01-02"), 202] == 22.0
    assert prices.loc[pd.Timestamp("2020-01-02"), 303] == 33.0
