import numpy as np
import pandas as pd
import pytest

from equity_factor_lab.backtest import long_short_decile_backtest


def test_backtest_rejects_negative_turnover_cost_rate() -> None:
    dates = pd.DatetimeIndex([pd.Timestamp("2020-01-02")])
    scores = pd.DataFrame([[1.0, 2.0]], index=dates, columns=["A", "B"])
    future_returns = pd.DataFrame([[0.0, 0.0]], index=dates, columns=["A", "B"])

    with pytest.raises(ValueError, match="turnover_cost_rate must be non-negative"):
        long_short_decile_backtest(
            scores=scores,
            future_returns=future_returns,
            turnover_cost_rate=-0.001,
        )


def test_backtest_rejects_negative_borrow_cost_rate() -> None:
    dates = pd.DatetimeIndex([pd.Timestamp("2020-01-02")])
    scores = pd.DataFrame([[1.0, 2.0]], index=dates, columns=["A", "B"])
    future_returns = pd.DataFrame([[0.0, 0.0]], index=dates, columns=["A", "B"])

    with pytest.raises(ValueError, match="borrow_cost_rate_annual must be non-negative"):
        long_short_decile_backtest(
            scores=scores,
            future_returns=future_returns,
            borrow_cost_rate_annual=-0.01,
        )


def test_terminal_missing_return_is_treated_as_delist_loss_for_held_long() -> None:
    dates = pd.DatetimeIndex([pd.Timestamp("2020-01-02")])
    scores = pd.DataFrame([[2.0, 1.0]], index=dates, columns=["A", "B"])
    future_returns = pd.DataFrame([[np.nan, 0.0]], index=dates, columns=["A", "B"])

    equity = long_short_decile_backtest(
        scores=scores,
        future_returns=future_returns,
        rebalance_freq="D",
        turnover_cost_rate=0.0,
        borrow_cost_rate_annual=0.0,
    )

    assert equity.iloc[-1] == pytest.approx(0.0)


def test_non_terminal_missing_return_is_not_forced_to_delist_loss() -> None:
    dates = pd.DatetimeIndex([pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")])
    scores = pd.DataFrame(
        [[2.0, 1.0], [2.0, 1.0]],
        index=dates,
        columns=["A", "B"],
    )
    future_returns = pd.DataFrame(
        [[np.nan, 0.0], [0.0, 0.0]],
        index=dates,
        columns=["A", "B"],
    )

    equity = long_short_decile_backtest(
        scores=scores,
        future_returns=future_returns,
        rebalance_freq="D",
        turnover_cost_rate=0.0,
        borrow_cost_rate_annual=0.0,
    )

    assert equity.iloc[-1] == pytest.approx(1.0)
