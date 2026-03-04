import numpy as np
import pandas as pd

from equity_factor_lab.factors.price_factors import compute_period_forward_returns, get_rebalance_dates


def test_get_rebalance_dates_uses_last_trading_day_per_month() -> None:
    idx = pd.bdate_range("2024-01-01", "2024-04-30")
    scores = pd.DataFrame({"asset": np.arange(len(idx), dtype=float)}, index=idx)

    rebalance_dates = get_rebalance_dates(scores, freq="ME")

    expected = pd.DatetimeIndex(
        ["2024-01-31", "2024-02-29", "2024-03-29", "2024-04-30"]
    )
    pd.testing.assert_index_equal(rebalance_dates, expected)


def test_compute_period_forward_returns_uses_endpoint_prices() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame({"asset": [100.0, 101.0, np.nan, 103.0, 104.0, 105.0]}, index=idx)
    rebalance_dates = pd.DatetimeIndex([idx[0], idx[-1]])

    period = compute_period_forward_returns(
        prices=prices,
        rebalance_dates=rebalance_dates,
    )

    assert np.isclose(period.loc[idx[0], "asset"], 0.05)


def test_compute_period_forward_returns_terminal_missing_endpoint_is_delist_loss() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame({"asset": [100.0, 101.0, 102.0, 103.0, 104.0, np.nan]}, index=idx)
    rebalance_dates = pd.DatetimeIndex([idx[0], idx[-1]])

    period = compute_period_forward_returns(
        prices=prices,
        rebalance_dates=rebalance_dates,
    )

    assert float(period.loc[idx[0], "asset"]) == -1.0


def test_compute_period_forward_returns_non_terminal_missing_endpoint_is_nan() -> None:
    idx = pd.bdate_range("2024-01-01", periods=7)
    prices = pd.DataFrame(
        {"asset": [100.0, 101.0, 102.0, np.nan, 104.0, 105.0, 106.0]},
        index=idx,
    )
    rebalance_dates = pd.DatetimeIndex([idx[0], idx[3], idx[-1]])

    period = compute_period_forward_returns(
        prices=prices,
        rebalance_dates=rebalance_dates,
    )

    assert pd.isna(period.loc[idx[0], "asset"])
