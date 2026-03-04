import pandas as pd
import pytest

from equity_factor_lab.data.price_cleaning import clean_price_panel


def test_clean_price_panel_drops_nonpositive_prices_and_hard_extremes() -> None:
    dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2021-01-04"),
            pd.Timestamp("2021-01-05"),
            pd.Timestamp("2021-01-06"),
        ]
    )
    prices = pd.DataFrame(
        {
            101: [1.0, 1.0, 20.0],  # +1900% hard-drop event on last day.
            202: [1.0, 0.0, 1.0],  # nonpositive middle price gets nulled.
            303: [1.0, 1.0, 11.0],  # +1000% should also hard-drop (inclusive cap).
        },
        index=dates,
    )
    result = clean_price_panel(prices)

    assert pd.isna(result.prices_clean.loc[dates[1], 202])
    assert bool(result.hard_event_mask.loc[dates[2], 101]) is True
    assert pd.isna(result.prices_clean.loc[dates[2], 101])
    assert bool(result.hard_event_mask.loc[dates[2], 303]) is True
    assert pd.isna(result.prices_clean.loc[dates[2], 303])


def test_clean_price_panel_detects_spike_reversal_suspect_pattern() -> None:
    dates = pd.date_range("2021-01-04", periods=4, freq="D")
    prices = pd.DataFrame(
        {
            101: [1.0, 5.0, 1.2, 1.1],  # +400%, then strong opposite move; 2-day net small.
        },
        index=dates,
    )
    result = clean_price_panel(
        prices,
        hard_max_return=10.0,
        classify_min_return=3.0,
        reversal_next_abs_return=0.6,
        reversal_two_day_net_abs=0.25,
        shift_horizon_days=2,
    )

    assert float(result.summary_stats["n_spike_reversal"]) == 1.0
    assert float(result.summary_stats["n_suspect_events"]) == 1.0
    assert pd.notna(result.prices_clean.loc[dates[1], 101])


def test_clean_price_panel_keeps_suspects_in_causal_execution_path() -> None:
    dates = pd.date_range("2021-01-04", periods=4, freq="D")
    prices = pd.DataFrame(
        {
            101: [1.0, 5.0, 1.2, 1.1],  # Spike-reversal suspect.
        },
        index=dates,
    )
    result = clean_price_panel(
        prices,
        hard_max_return=10.0,
        classify_min_return=3.0,
        reversal_next_abs_return=0.6,
        reversal_two_day_net_abs=0.25,
    )

    assert float(result.summary_stats["n_suspect_events"]) == 1.0
    assert pd.notna(result.prices_clean.loc[dates[1], 101])


def test_clean_price_panel_detects_persistent_shift_suspect_pattern() -> None:
    dates = pd.date_range("2021-01-04", periods=7, freq="D")
    prices = pd.DataFrame(
        {
            101: [1.0, 5.0, 5.1, 4.9, 5.0, 5.05, 4.95],  # jump then stable near new level.
        },
        index=dates,
    )
    result = clean_price_panel(
        prices,
        hard_max_return=10.0,
        classify_min_return=3.0,
        shift_horizon_days=5,
        shift_stability_abs_threshold=0.2,
    )

    assert float(result.summary_stats["n_persistent_shift"]) == 1.0
    assert float(result.summary_stats["n_suspect_events"]) == 1.0
    assert pd.notna(result.prices_clean.loc[dates[1], 101])


def test_clean_price_panel_detects_stale_jump_suspect_pattern() -> None:
    dates = pd.date_range("2021-01-04", periods=6, freq="D")
    prices = pd.DataFrame(
        {
            101: [1.0, 1.0, 1.0, 1.0, 1.0, 5.0],  # stale then +400% jump.
        },
        index=dates,
    )
    result = clean_price_panel(
        prices,
        hard_max_return=10.0,
        classify_min_return=3.0,
        stale_lookback_days=4,
        stale_abs_return_epsilon=1e-12,
        shift_horizon_days=2,
    )

    assert float(result.summary_stats["n_stale_jump"]) == 1.0
    assert float(result.summary_stats["n_suspect_events"]) == 1.0
    assert pd.notna(result.prices_clean.loc[dates[5], 101])


def test_clean_price_panel_raises_on_nonpositive_lookback_parameters() -> None:
    dates = pd.date_range("2021-01-04", periods=3, freq="D")
    prices = pd.DataFrame({101: [1.0, 1.0, 1.1]}, index=dates)

    with pytest.raises(ValueError, match="stale_lookback_days must be positive"):
        clean_price_panel(prices, stale_lookback_days=0)

    with pytest.raises(ValueError, match="shift_horizon_days must be positive"):
        clean_price_panel(prices, shift_horizon_days=0)
