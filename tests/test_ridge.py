import numpy as np
import pandas as pd
import pytest

from equity_factor_lab.models.ridge import (
    build_walk_forward_ridge_scores_fixed_alpha,
    fit_walk_forward_ridge_scores,
)


def _minimal_ridge_inputs() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    idx = pd.DatetimeIndex(["2020-01-01", "2020-01-02"])
    cols = ["A", "B"]
    factor = pd.DataFrame([[1.0, 2.0], [1.1, 2.1]], index=idx, columns=cols)
    future_returns = pd.DataFrame([[0.01, 0.02], [0.011, 0.019]], index=idx, columns=cols)
    return {"f1": factor, "f2": factor + 0.5}, future_returns


@pytest.mark.parametrize(
    ("alpha_grid", "message"),
    [
        (tuple(), "alpha_grid cannot be empty"),
        ((0.1, np.nan), "alpha_grid must contain only finite values"),
        ((0.1, -1.0), "alpha_grid must be non-negative"),
    ],
)
def test_fit_walk_forward_ridge_scores_validates_alpha_grid(
    alpha_grid: tuple[float, ...],
    message: str,
) -> None:
    factor_scores, future_returns = _minimal_ridge_inputs()
    with pytest.raises(ValueError, match=message):
        fit_walk_forward_ridge_scores(
            factor_scores=factor_scores,
            future_returns=future_returns,
            rebalance_freq="D",
            tuning_cutoff_date="2020-02-01",
            alpha_grid=alpha_grid,
        )


def test_fit_walk_forward_ridge_scores_keeps_missing_predictions_as_nan() -> None:
    idx = pd.DatetimeIndex(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]
    )
    cols = ["A", "B"]
    factor_one = pd.DataFrame(
        [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]],
        index=idx,
        columns=cols,
    )
    factor_two = pd.DataFrame(
        [[0.5, 1.5], [0.6, 1.6], [0.7, 1.7], [0.8, np.nan], [0.9, 1.9]],
        index=idx,
        columns=cols,
    )
    future_returns = pd.DataFrame(
        [[0.01, 0.02], [0.011, 0.019], [0.012, 0.018], [0.013, 0.017], [0.014, 0.016]],
        index=idx,
        columns=cols,
    )

    ridge_fit = fit_walk_forward_ridge_scores(
        factor_scores={"f1": factor_one, "f2": factor_two},
        future_returns=future_returns,
        rebalance_freq="D",
        tuning_cutoff_date="2020-02-01",
        alpha_grid=(1.0,),
        window_type="expanding",
        min_train_periods=1,
        refit_every_n_rebalances=1,
        min_train_rows=2,
        min_assets=1,
    )

    assert pd.isna(ridge_fit.scores.loc[pd.Timestamp("2020-01-04"), "B"])
    assert pd.notna(ridge_fit.scores.loc[pd.Timestamp("2020-01-05"), "B"])


def test_build_walk_forward_ridge_scores_fixed_alpha_keeps_missing_predictions_as_nan() -> None:
    idx = pd.DatetimeIndex(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]
    )
    cols = ["A", "B"]
    factor_one = pd.DataFrame(
        [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]],
        index=idx,
        columns=cols,
    )
    factor_two = pd.DataFrame(
        [[0.5, 1.5], [0.6, 1.6], [0.7, 1.7], [0.8, np.nan], [0.9, 1.9]],
        index=idx,
        columns=cols,
    )
    future_returns = pd.DataFrame(
        [[0.01, 0.02], [0.011, 0.019], [0.012, 0.018], [0.013, 0.017], [0.014, 0.016]],
        index=idx,
        columns=cols,
    )

    ridge_scores = build_walk_forward_ridge_scores_fixed_alpha(
        factor_scores={"f1": factor_one, "f2": factor_two},
        future_returns=future_returns,
        rebalance_freq="D",
        alpha=1.0,
        window_type="expanding",
        min_train_periods=1,
        refit_every_n_rebalances=1,
        min_train_rows=2,
    )

    assert pd.isna(ridge_scores.loc[pd.Timestamp("2020-01-04"), "B"])
    assert pd.notna(ridge_scores.loc[pd.Timestamp("2020-01-05"), "B"])


def test_build_walk_forward_ridge_scores_fixed_alpha_allows_partial_missing_when_gate_passes() -> None:
    idx = pd.date_range("2020-01-01", periods=7, freq="D")
    cols = ["A", "B"]
    base = pd.DataFrame(
        [[1.0 + 0.1 * i, 2.0 + 0.1 * i] for i in range(len(idx))],
        index=idx,
        columns=cols,
    )
    factor_scores = {
        "momentum": base.copy(),
        "reversal": (base + 0.1).copy(),
        "low_vol": (base + 0.2).copy(),
        "value": (base + 0.3).copy(),
        "quality": (base + 0.4).copy(),
        "invest": (base + 0.5).copy(),
        "growth": (base + 0.6).copy(),
    }

    # B is missing one factor on 2020-01-04 but still satisfies coverage gate:
    # min_total=5, min_price=2, min_fundamental=2.
    pred_date = pd.Timestamp("2020-01-06")
    factor_scores["growth"].loc[pred_date, "B"] = np.nan

    # A is missing too many factors and should fail coverage gate.
    factor_scores["reversal"].loc[pred_date, "A"] = np.nan
    factor_scores["low_vol"].loc[pred_date, "A"] = np.nan
    factor_scores["quality"].loc[pred_date, "A"] = np.nan
    factor_scores["growth"].loc[pred_date, "A"] = np.nan

    future_returns = pd.DataFrame(
        [[0.01 + 0.001 * i, 0.02 - 0.001 * i] for i in range(len(idx))],
        index=idx,
        columns=cols,
    )

    ridge_scores = build_walk_forward_ridge_scores_fixed_alpha(
        factor_scores=factor_scores,
        future_returns=future_returns,
        rebalance_freq="D",
        alpha=1.0,
        window_type="expanding",
        min_train_periods=1,
        refit_every_n_rebalances=1,
        min_train_rows=2,
    )

    assert pd.notna(ridge_scores.loc[pred_date, "B"])
    assert pd.isna(ridge_scores.loc[pred_date, "A"])
