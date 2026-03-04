import pandas as pd

from equity_factor_lab.runner.pipeline_steps import _align_factor_panels


def test_align_factor_panels_preserves_future_return_asset_set() -> None:
    future_index = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-02"),
            pd.Timestamp("2020-01-03"),
        ]
    )
    future_returns = pd.DataFrame(
        [
            [0.01, 0.02, 0.03],
            [0.04, 0.05, 0.06],
            [0.07, 0.08, 0.09],
        ],
        index=future_index,
        columns=["A", "B", "C"],
    )

    factor_scores = {
        "f1": pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            index=pd.DatetimeIndex([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]),
            columns=["A", "B"],
        ),
        "f2": pd.DataFrame(
            [[5.0, 6.0], [7.0, 8.0]],
            index=pd.DatetimeIndex([pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]),
            columns=["B", "C"],
        ),
    }

    aligned_scores, aligned_future_returns = _align_factor_panels(
        factor_scores=factor_scores,
        future_returns=future_returns,
    )

    assert list(aligned_future_returns.index) == [pd.Timestamp("2020-01-02")]
    assert list(aligned_future_returns.columns) == ["A", "B", "C"]
    assert aligned_scores["f1"]["C"].isna().all()
    assert aligned_scores["f2"]["A"].isna().all()
