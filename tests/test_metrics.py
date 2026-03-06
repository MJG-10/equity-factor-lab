import numpy as np
import pandas as pd
import pytest

from equity_factor_lab.metrics import compute_ic_series, compute_performance_stats


def test_compute_performance_stats_uses_daily_sharpe_and_drawdown() -> None:
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

    expected_annual_vol = float(np.std(daily_returns, ddof=1) * np.sqrt(252))
    expected_sharpe = float(
        (np.mean(daily_returns) / np.std(daily_returns, ddof=1)) * np.sqrt(252)
    )
    expected_max_drawdown = float((equity / equity.cummax() - 1.0).min())

    assert stats["annual_vol"] == pytest.approx(expected_annual_vol)
    assert stats["sharpe"] == pytest.approx(expected_sharpe)
    assert stats["max_drawdown"] == pytest.approx(expected_max_drawdown)


def test_compute_performance_stats_handles_non_positive_end_equity() -> None:
    equity = pd.Series(
        [1.0, 0.5, -0.1],
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
        dtype=float,
    )

    stats = compute_performance_stats(equity, trading_days_per_year=252)

    assert np.isfinite(stats["annual_vol"])
    assert np.isfinite(stats["max_drawdown"])


def test_compute_ic_series_uses_spearman_rank_correlation() -> None:
    scores = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0]],
        index=[pd.Timestamp("2020-01-01")],
        columns=["A", "B", "C", "D"],
    )
    future_returns = pd.DataFrame(
        [[1.0, 8.0, 27.0, 64.0]],
        index=scores.index,
        columns=scores.columns,
    )

    ic = compute_ic_series(scores, future_returns, min_assets=2)

    assert ic.iloc[0] == pytest.approx(1.0)
