import pandas as pd

from equity_factor_lab.runner.pipeline_universe import build_dynamic_universe_mask


def test_build_dynamic_universe_mask_backfills_under_cap_with_required_data() -> None:
    dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2021-01-04"),
            pd.Timestamp("2021-01-05"),
            pd.Timestamp("2021-01-06"),
        ]
    )
    assets = [101, 202, 303]
    prices = pd.DataFrame(1.0, index=dates, columns=assets)
    dollar_volume = pd.DataFrame(
        [
            [100.0, 90.0, 80.0],
            [300.0, 200.0, 100.0],
            [300.0, 200.0, 100.0],
        ],
        index=dates,
        columns=assets,
    )
    required_data = pd.DataFrame(True, index=dates, columns=assets)
    required_data.loc[dates[-1], 101] = False

    universe_mask, diagnostics = build_dynamic_universe_mask(
        prices=prices,
        dollar_volume=dollar_volume,
        max_tickers=2,
        adv_lookback=1,
        required_data_mask=required_data,
    )

    assert bool(universe_mask.loc[dates[-1], 101]) is False
    assert bool(universe_mask.loc[dates[-1], 202]) is True
    assert bool(universe_mask.loc[dates[-1], 303]) is True
    assert float(diagnostics["raw_selected_count"].loc[dates[-1]]) == 2.0
    assert float(universe_mask.loc[dates[-1]].sum()) == 2.0
    assert float(diagnostics["data_completeness_excluded_count"].loc[dates[-1]]) == 1.0


def test_build_dynamic_universe_mask_without_required_data_matches_raw_selection() -> None:
    dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2021-01-04"),
            pd.Timestamp("2021-01-05"),
        ]
    )
    assets = [101, 202]
    prices = pd.DataFrame(1.0, index=dates, columns=assets)
    dollar_volume = pd.DataFrame(100.0, index=dates, columns=assets)

    universe_mask, diagnostics = build_dynamic_universe_mask(
        prices=prices,
        dollar_volume=dollar_volume,
        max_tickers=None,
        adv_lookback=1,
    )

    selected_count = universe_mask.sum(axis=1).astype(float)
    assert selected_count.equals(diagnostics["raw_selected_count"])
    assert float(diagnostics["data_completeness_excluded_count"].sum()) == 0.0


def test_build_dynamic_universe_mask_applies_min_prev_close_floor() -> None:
    dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2021-01-04"),
            pd.Timestamp("2021-01-05"),
            pd.Timestamp("2021-01-06"),
        ]
    )
    assets = [101, 202]
    prices = pd.DataFrame(
        {
            101: [0.8, 0.8, 0.8],
            202: [2.0, 2.0, 2.0],
        },
        index=dates,
    )
    dollar_volume = pd.DataFrame(100.0, index=dates, columns=assets)

    universe_mask, diagnostics = build_dynamic_universe_mask(
        prices=prices,
        dollar_volume=dollar_volume,
        max_tickers=None,
        adv_lookback=1,
        min_prev_close=1.0,
    )

    assert bool(universe_mask.loc[dates[-1], 101]) is False
    assert bool(universe_mask.loc[dates[-1], 202]) is True
    assert float(diagnostics["price_floor_excluded_count"].loc[dates[-1]]) == 1.0


def test_build_dynamic_universe_mask_applies_quality_mask_before_selection() -> None:
    dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2021-01-04"),
            pd.Timestamp("2021-01-05"),
            pd.Timestamp("2021-01-06"),
        ]
    )
    assets = [101, 202, 303]
    prices = pd.DataFrame(2.0, index=dates, columns=assets)
    dollar_volume = pd.DataFrame(100.0, index=dates, columns=assets)
    quality_mask = pd.DataFrame(True, index=dates, columns=assets)
    quality_mask.loc[dates[-1], 202] = False

    universe_mask, diagnostics = build_dynamic_universe_mask(
        prices=prices,
        dollar_volume=dollar_volume,
        max_tickers=None,
        adv_lookback=1,
        quality_mask=quality_mask,
    )

    assert bool(universe_mask.loc[dates[-1], 202]) is False
    assert float(diagnostics["quality_excluded_count"].loc[dates[-1]]) == 1.0


def test_build_dynamic_universe_mask_never_selects_missing_prices() -> None:
    dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2021-01-04"),
            pd.Timestamp("2021-01-05"),
            pd.Timestamp("2021-01-06"),
        ]
    )
    assets = [101, 202]
    prices = pd.DataFrame(
        {
            101: [2.0, 2.1, 2.2],
            202: [3.0, 3.1, float("nan")],
        },
        index=dates,
    )
    dollar_volume = pd.DataFrame(100.0, index=dates, columns=assets)

    universe_mask, _ = build_dynamic_universe_mask(
        prices=prices,
        dollar_volume=dollar_volume,
        max_tickers=None,
        adv_lookback=1,
    )

    assert bool(universe_mask.loc[dates[-1], 202]) is False
