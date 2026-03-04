import numpy as np
import pandas as pd
import pytest

from simfin.names import (
    ASSETS_GROWTH,
    ASSET_TURNOVER,
    DEBT_RATIO,
    EARNINGS_GROWTH,
    GROSS_PROFIT_MARGIN,
    OPERATING_MARGIN,
    ROA,
    SALES_GROWTH,
)

from equity_factor_lab.data.simfin_signals import OPERATING_INCOME_GROWTH
from equity_factor_lab.factors.fundamental_factors import (
    build_daily_panel_from_signal,
    compute_growth_style_from_signals,
    compute_invest_style_from_signals,
    compute_quality_style_from_signals,
)
from equity_factor_lab.factors.price_factors import (
    compute_low_volatility_scores,
    compute_momentum_scores,
    compute_short_term_reversal_scores,
    standardize_cross_section,
)
from equity_factor_lab.models.neutralization import validate_neutralization_mode


def test_compute_momentum_scores_matches_log_compounded_return() -> None:
    dates = pd.bdate_range("2020-01-01", periods=6)
    prices = pd.DataFrame(
        {
            "A": [100.0, 102.0, 101.0, 104.0, 106.0, 107.0],
        },
        index=dates,
    )

    scores = compute_momentum_scores(
        prices,
        lookback=4,
        skip_recent=1,
    )

    expected = (104.0 / 100.0) - 1.0
    np.testing.assert_allclose(scores.loc[dates[4], "A"], expected, rtol=1e-12, atol=1e-12)


def test_compute_short_term_reversal_scores_matches_negated_cumulative_return() -> None:
    dates = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame(
        {
            "A": [100.0, 110.0, 121.0, 118.58],
        },
        index=dates,
    )

    scores = compute_short_term_reversal_scores(
        prices,
        lookback=3,
    )

    expected = -((118.58 / 100.0) - 1.0)
    np.testing.assert_allclose(scores.loc[dates[3], "A"], expected, rtol=1e-12, atol=1e-12)


def test_build_daily_panel_from_signal_applies_freshness_limit() -> None:
    signal_index = pd.MultiIndex.from_tuples(
        [
            (101, pd.Timestamp("2020-01-01")),
        ],
        names=["SimFinId", "Date"],
    )
    signals = pd.DataFrame({"Metric": [1.0]}, index=signal_index)
    prices_index = pd.bdate_range("2020-01-01", periods=255)

    panel = build_daily_panel_from_signal(signals, "Metric", prices_index)

    assert panel.loc[prices_index[0], 101] == 1.0
    assert panel.loc[prices_index[252], 101] == 1.0
    assert pd.isna(panel.loc[prices_index[253], 101])


def test_standardize_cross_section_uses_population_std() -> None:
    scores = pd.DataFrame([[1.0, 2.0, 3.0]], index=[pd.Timestamp("2020-01-01")], columns=["A", "B", "C"])

    standardized = standardize_cross_section(scores)

    expected = np.array([-(np.sqrt(3 / 2)), 0.0, np.sqrt(3 / 2)])
    np.testing.assert_allclose(standardized.iloc[0].to_numpy(dtype=float), expected, rtol=1e-12, atol=1e-12)


def test_compute_low_volatility_scores_prefers_lower_idiosyncratic_volatility() -> None:
    dates = pd.bdate_range("2020-01-01", periods=8)
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02],
            "B": [0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02],
            "C": [0.08, -0.04, 0.07, -0.03, 0.09, -0.02, 0.06, -0.05],
        },
        index=dates,
    )
    prices = 100.0 * (1.0 + returns).cumprod()

    scores = compute_low_volatility_scores(
        prices,
        lookback=5,
        beta_min_periods=5,
    )

    last_row = scores.loc[dates[-1]]
    assert last_row["A"] > last_row["C"]
    assert last_row["B"] > last_row["C"]


def test_compute_quality_style_from_signals_combines_profitability_efficiency_and_safety() -> None:
    prices_index = pd.bdate_range("2020-01-01", periods=2)
    prices = pd.DataFrame({101: [10.0, 11.0], 202: [20.0, 21.0], 303: [30.0, 31.0]}, index=prices_index)
    signal_index = pd.MultiIndex.from_tuples(
        [
            (101, prices_index[0]),
            (202, prices_index[0]),
            (303, prices_index[0]),
        ],
        names=["SimFinId", "Date"],
    )
    signals = pd.DataFrame(
        {
            ROA: [0.18, 0.12, 0.06],
            GROSS_PROFIT_MARGIN: [0.60, 0.40, 0.20],
            OPERATING_MARGIN: [0.05, 0.20, 0.35],
            ASSET_TURNOVER: [1.6, 1.1, 0.6],
            DEBT_RATIO: [0.20, 0.45, 0.70],
        },
        index=signal_index,
    )

    quality_scores = compute_quality_style_from_signals(prices, signals)
    last_row = quality_scores.loc[prices_index[0]]

    assert last_row[101] > last_row[202] > last_row[303]


def test_compute_quality_style_from_signals_falls_back_to_operating_margin() -> None:
    prices_index = pd.bdate_range("2020-01-01", periods=1)
    prices = pd.DataFrame({101: [10.0], 202: [20.0], 303: [30.0]}, index=prices_index)
    signal_index = pd.MultiIndex.from_tuples(
        [
            (101, prices_index[0]),
            (202, prices_index[0]),
            (303, prices_index[0]),
        ],
        names=["SimFinId", "Date"],
    )
    signals = pd.DataFrame(
        {
            ROA: [0.10, 0.10, 0.10],
            GROSS_PROFIT_MARGIN: [np.nan, np.nan, np.nan],
            OPERATING_MARGIN: [0.30, 0.20, 0.10],
            ASSET_TURNOVER: [1.0, 1.0, 1.0],
            DEBT_RATIO: [0.10, 0.20, 0.30],
        },
        index=signal_index,
    )

    quality_scores = compute_quality_style_from_signals(prices, signals)
    row = quality_scores.loc[prices_index[0]]

    assert row.notna().all()
    assert row[101] > row[202] > row[303]


def test_compute_quality_style_from_signals_requires_three_components() -> None:
    prices_index = pd.bdate_range("2020-01-01", periods=1)
    prices = pd.DataFrame({101: [10.0], 202: [20.0], 303: [30.0]}, index=prices_index)
    signal_index = pd.MultiIndex.from_tuples(
        [
            (101, prices_index[0]),
            (202, prices_index[0]),
            (303, prices_index[0]),
        ],
        names=["SimFinId", "Date"],
    )
    signals = pd.DataFrame(
        {
            ROA: [0.18, 0.12, 0.06],
            GROSS_PROFIT_MARGIN: [0.60, 0.40, 0.20],
            ASSET_TURNOVER: [1.6, np.nan, 0.6],
            DEBT_RATIO: [0.20, np.nan, 0.70],
        },
        index=signal_index,
    )

    quality_scores = compute_quality_style_from_signals(prices, signals)
    row = quality_scores.loc[prices_index[0]]

    assert pd.notna(row[101])
    assert pd.isna(row[202])
    assert pd.notna(row[303])


def test_compute_invest_style_from_signals_negates_asset_growth() -> None:
    prices_index = pd.bdate_range("2020-01-01", periods=2)
    prices = pd.DataFrame({101: [10.0, 11.0], 202: [20.0, 21.0]}, index=prices_index)
    signal_index = pd.MultiIndex.from_tuples(
        [
            (101, prices_index[0]),
            (202, prices_index[0]),
        ],
        names=["SimFinId", "Date"],
    )
    signals = pd.DataFrame(
        {
            ASSETS_GROWTH: [0.10, 0.25],
        },
        index=signal_index,
    )

    invest_scores = compute_invest_style_from_signals(prices, signals)

    np.testing.assert_allclose(
        invest_scores.loc[prices_index[0], [101, 202]].to_numpy(dtype=float),
        np.array([-0.10, -0.25]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_compute_growth_style_from_signals_combines_three_growth_components() -> None:
    prices_index = pd.bdate_range("2020-01-01", periods=1)
    prices = pd.DataFrame({101: [10.0], 202: [20.0], 303: [30.0]}, index=prices_index)
    signal_index = pd.MultiIndex.from_tuples(
        [
            (101, prices_index[0]),
            (202, prices_index[0]),
            (303, prices_index[0]),
        ],
        names=["SimFinId", "Date"],
    )
    signals = pd.DataFrame(
        {
            SALES_GROWTH: [0.30, 0.15, 0.00],
            OPERATING_INCOME_GROWTH: [0.35, 0.10, -0.05],
            EARNINGS_GROWTH: [0.25, 0.05, -0.10],
        },
        index=signal_index,
    )

    growth_scores = compute_growth_style_from_signals(prices, signals)
    row = growth_scores.loc[prices_index[0]]

    assert row[101] > row[202] > row[303]


def test_compute_growth_style_from_signals_requires_two_components() -> None:
    prices_index = pd.bdate_range("2020-01-01", periods=1)
    prices = pd.DataFrame({101: [10.0], 202: [20.0], 303: [30.0]}, index=prices_index)
    signal_index = pd.MultiIndex.from_tuples(
        [
            (101, prices_index[0]),
            (202, prices_index[0]),
            (303, prices_index[0]),
        ],
        names=["SimFinId", "Date"],
    )
    signals = pd.DataFrame(
        {
            SALES_GROWTH: [0.30, 0.15, np.nan],
            OPERATING_INCOME_GROWTH: [0.35, np.nan, np.nan],
            EARNINGS_GROWTH: [0.25, 0.05, -0.10],
        },
        index=signal_index,
    )

    growth_scores = compute_growth_style_from_signals(prices, signals)
    row = growth_scores.loc[prices_index[0]]

    assert pd.notna(row[101])
    assert pd.notna(row[202])
    assert pd.isna(row[303])

def test_validate_neutralization_mode_rejects_removed_sector_market_mode() -> None:
    with pytest.raises(ValueError, match="Unknown neutralization_mode"):
        validate_neutralization_mode("sector_market")
