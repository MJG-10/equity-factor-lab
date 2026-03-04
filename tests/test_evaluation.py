import pandas as pd
import pytest

from equity_factor_lab.evaluation import build_ic_inputs


def test_build_ic_inputs_daily_alignment_with_explicit_d_frequency() -> None:
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    )
    scores = pd.DataFrame(
        [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]],
        index=idx,
        columns=["A", "B"],
    )
    future_returns = pd.DataFrame(
        [[0.01, 0.02], [0.011, 0.021], [0.012, 0.022]],
        index=idx,
        columns=["A", "B"],
    )

    ic_scores, ic_future_returns = build_ic_inputs(
        scores=scores,
        future_returns=future_returns,
        rebalance_freq="D",
        prices=None,
    )

    assert ic_scores.equals(scores)
    assert ic_future_returns.equals(future_returns)


def test_build_ic_inputs_requires_prices_for_non_daily() -> None:
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")]
    )
    scores = pd.DataFrame(
        [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]],
        index=idx,
        columns=["A", "B"],
    )
    future_returns = pd.DataFrame(
        [[0.01, 0.02], [0.011, 0.021], [0.012, 0.022]],
        index=idx,
        columns=["A", "B"],
    )

    with pytest.raises(ValueError, match="prices are required for non-daily IC input construction"):
        build_ic_inputs(
            scores=scores,
            future_returns=future_returns,
            rebalance_freq="ME",
            prices=None,
        )
