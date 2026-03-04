import pandas as pd
import pytest

from equity_factor_lab.backtest import long_short_decile_backtest


def test_backtest_uses_disjoint_ranked_buckets_when_scores_are_tied() -> None:
    dates = pd.DatetimeIndex([pd.Timestamp("2020-01-02")])
    columns = ["A", "B", "C", "D"]

    scores = pd.DataFrame([[1.0, 1.0, 1.0, 1.0]], index=dates, columns=columns)
    future_returns = pd.DataFrame([[-0.5, 0.0, 0.0, 0.5]], index=dates, columns=columns)

    equity = long_short_decile_backtest(
        scores=scores,
        future_returns=future_returns,
        rebalance_freq="D",
        turnover_cost_rate=0.0,
        borrow_cost_rate_annual=0.0,
    )

    # Stable ties with disjoint rank buckets pick A short and D long.
    # Return = +0.5 - (-0.5) = +1.0, so equity should be 2.0.
    assert equity.iloc[-1] == pytest.approx(2.0)


def test_backtest_attaches_turnover_and_cost_summaries() -> None:
    dates = pd.DatetimeIndex([pd.Timestamp("2020-01-02")])
    columns = ["A", "B", "C", "D"]

    scores = pd.DataFrame([[1.0, 1.0, 1.0, 1.0]], index=dates, columns=columns)
    future_returns = pd.DataFrame([[-0.5, 0.0, 0.0, 0.5]], index=dates, columns=columns)

    equity = long_short_decile_backtest(
        scores=scores,
        future_returns=future_returns,
        rebalance_freq="D",
        turnover_cost_rate=0.001,
        borrow_cost_rate_annual=0.03,
    )

    assert equity.attrs["turnover_mean"] == pytest.approx(2.0)
    # 0.001 cost rate * 2.0 turnover on the only day, annualized by mean(daily)*252
    assert equity.attrs["turnover_cost_drag_ann"] == pytest.approx(0.504)
    # borrow cost is annual rate * 1.0 short exposure
    assert equity.attrs["borrow_cost_drag_ann"] == pytest.approx(0.03)
