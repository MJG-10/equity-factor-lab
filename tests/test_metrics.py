import numpy as np
import pandas as pd
import pytest

from equity_factor_lab.metrics import compute_performance_stats


def test_compute_performance_stats_uses_cagr_and_daily_sharpe() -> None:
    daily_returns = np.array([0.01, -0.005, 0.002, 0.004], dtype=float)
    equity_values = [1.0]
    for r in daily_returns:
        equity_values.append(equity_values[-1] * (1.0 + float(r)))
    equity = pd.Series(
        equity_values,
        index=pd.date_range("2020-01-01", periods=len(equity_values), freq="D"),
        dtype=float,
    )

    stats = compute_performance_stats(equity, trading_days_per_year=252)

    start_value = equity_values[0]
    end_value = equity_values[-1]
    n_periods = len(daily_returns)

    expected_total_return = (end_value / start_value) - 1.0
    expected_annual_return = (end_value / start_value) ** (252 / n_periods) - 1.0
    expected_annual_vol = float(np.std(daily_returns, ddof=1) * np.sqrt(252))
    expected_sharpe = float(
        (np.mean(daily_returns) / np.std(daily_returns, ddof=1)) * np.sqrt(252)
    )
    expected_max_drawdown = float((equity / equity.cummax() - 1.0).min())

    assert stats["total_return"] == pytest.approx(expected_total_return)
    assert stats["annual_return"] == pytest.approx(expected_annual_return)
    assert stats["annual_vol"] == pytest.approx(expected_annual_vol)
    assert stats["sharpe"] == pytest.approx(expected_sharpe)
    assert stats["max_drawdown"] == pytest.approx(expected_max_drawdown)


def test_compute_performance_stats_sets_annual_return_nan_when_end_non_positive() -> None:
    equity = pd.Series(
        [1.0, 0.5, -0.1],
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
        dtype=float,
    )

    stats = compute_performance_stats(equity, trading_days_per_year=252)

    assert np.isnan(stats["annual_return"])
    assert stats["total_return"] == pytest.approx(-1.1)
