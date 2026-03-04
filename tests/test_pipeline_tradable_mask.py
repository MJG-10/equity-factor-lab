import pandas as pd

from equity_factor_lab.runner.pipeline_stages import apply_tradable_mask


def test_apply_tradable_mask_is_subset_of_non_missing_prices() -> None:
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
            101: [10.0, 10.1, 10.2],
            202: [20.0, 20.1, float("nan")],
        },
        index=dates,
    )
    selected_scores = pd.DataFrame(1.0, index=dates, columns=assets)
    all_scores = {"composite": selected_scores.copy()}
    dollar_volume = pd.DataFrame(1000.0, index=dates, columns=assets)
    aligned_future_returns = pd.DataFrame(0.0, index=dates, columns=assets)

    panels = apply_tradable_mask(
        all_scores=all_scores,
        selected_scores=selected_scores,
        prices=prices,
        dollar_volume=dollar_volume,
        aligned_future_returns_raw=aligned_future_returns,
        max_tickers=None,
        adv_lookback=1,
    )

    assert bool(panels.tradable_mask.loc[dates[-1], 202]) is False
    assert float((panels.tradable_mask & prices.isna()).sum().sum()) == 0.0
