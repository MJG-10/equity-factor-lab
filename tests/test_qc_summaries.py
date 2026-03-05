import numpy as np
import pandas as pd

from equity_factor_lab.data.factor_qc import summarize_factor_panels
from equity_factor_lab.data.price_qc import summarize_price_panel


def test_summarize_factor_panels_uses_coverage_base_mask() -> None:
    dates = pd.DatetimeIndex(["2021-01-04", "2021-01-05"])
    cols = ["A", "B", "C"]
    panel = pd.DataFrame(
        [[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]],
        index=dates,
        columns=cols,
    )
    coverage_base = pd.DataFrame(
        [[True, True, False], [True, False, True]],
        index=dates,
        columns=cols,
    )
    qc = summarize_factor_panels(
        {"factor_a": panel},
        coverage_base_mask=coverage_base,
    )

    row = qc.iloc[0]
    assert row["factor"] == "factor_a"
    assert row["non_null_frac"] == 0.5
    assert row["date_coverage_p50"] == 0.5
    assert row["stock_coverage_p50"] == 0.0
    assert set(qc.columns) == {
        "factor",
        "non_null_frac",
        "date_coverage_p50",
        "stock_coverage_p50",
        "value_p01",
        "value_p99",
    }


def test_summarize_price_panel_returns_compact_canonical_view() -> None:
    dates = pd.DatetimeIndex(["2021-01-04", "2021-01-05", "2021-01-06"])
    prices = pd.DataFrame(
        {
            "A": [10.0, 10.5, 10.0],
            "B": [5.0, np.nan, 5.0],
        },
        index=dates,
    )

    compact = summarize_price_panel(prices)

    assert compact.shape[0] == 1
    assert set(compact.columns) == {
        "scope",
        "n_observed_prices",
        "daily_ret_alldays_min",
        "daily_ret_alldays_p01",
        "daily_ret_alldays_p99",
        "daily_ret_alldays_p999",
        "daily_ret_alldays_max",
    }
    assert float(compact.loc[0, "n_observed_prices"]) == 5.0


def test_summarize_price_panel_ignores_nonfinite_returns_in_tail_stats() -> None:
    dates = pd.DatetimeIndex(["2021-01-04", "2021-01-05", "2021-01-06"])
    prices = pd.DataFrame(
        {
            "A": [1.0, 0.0, 1.0],  # 0->1 would be +inf if untreated.
            "B": [2.0, 2.0, 2.0],
        },
        index=dates,
    )

    compact = summarize_price_panel(prices)

    assert pd.notna(compact.loc[0, "daily_ret_alldays_max"])
