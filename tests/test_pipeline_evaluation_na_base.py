import pandas as pd

from equity_factor_lab.runner.pipeline import PipelineSettings
from equity_factor_lab.runner.pipeline_stages import TradablePanels, evaluate_pipeline_artifacts


def test_evaluate_pipeline_artifacts_handles_zero_selected_base_without_natype_cast_error() -> None:
    dates = pd.DatetimeIndex([pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05")])
    assets = [101, 202]

    selected_scores = pd.DataFrame(float("nan"), index=dates, columns=assets)
    tradable_mask = pd.DataFrame(False, index=dates, columns=assets)
    aligned_future_returns = pd.DataFrame(float("nan"), index=dates, columns=assets)

    tradable_panels = TradablePanels(
        all_scores={"composite": selected_scores},
        selected_scores=selected_scores,
        tradable_mask=tradable_mask,
        aligned_future_returns=aligned_future_returns,
    )

    prices = pd.DataFrame({101: [100.0, 100.0], 202: [50.0, 50.0]}, index=dates)
    settings = PipelineSettings(rebalance_freq="D", ic_min_assets=1)

    artifacts = evaluate_pipeline_artifacts(
        settings=settings,
        prices=prices,
        tradable_panels=tradable_panels,
    )

    assert not artifacts.ic_diagnostics.empty
    assert artifacts.ic_diagnostics.loc[0, "n_dates_below_ic_min_assets"] == 2
    assert artifacts.ic_diagnostics.loc[0, "ic_min_assets"] == 1
    assert pd.isna(artifacts.ic_diagnostics.loc[0, "ic_coverage_frac_p10"])
