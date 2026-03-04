"""Runs one configured pipeline execution and prints key diagnostics."""

import os
from ..config import SIMFIN_DATA_DIR
from .pipeline import PipelineSettings, run_pipeline


def main() -> None:
    """Runs a single pipeline configuration and prints key diagnostics."""
    settings = PipelineSettings(
        start_date="2011-01-01",
        end_date=None,
        rebalance_freq="ME",
        neutralization_mode="sector",
        output_factor="composite",
        turnover_cost_rate=0.001,
        borrow_cost_rate_annual=0.03,
        # SimFin API key is required by data loaders; keep it outside source control.
        simfin_api_key=os.getenv("SIMFIN_API_KEY"),
        simfin_data_dir=str(SIMFIN_DATA_DIR),
        simfin_refresh_days=0,
    )
    result = run_pipeline(settings=settings)

    print("Selected factor:", settings.output_factor)
    if settings.output_factor == "ridge":
        print("Ridge alpha:", settings.ridge_alpha)
    print("Neutralization mode:", settings.neutralization_mode)
    print("Turnover cost rate:", settings.turnover_cost_rate)
    print("Borrow cost annual:", settings.borrow_cost_rate_annual)
    print(
        "Fundamental timing:",
        f"publish_shift_bdays={settings.simfin_publish_shift_business_days}"
    )
    print("Available factors:", sorted(result.factor_scores.keys()))
    print("Prices shape:", result.prices.shape)
    print("Factor QC rows:", len(result.factor_qc_stats))
    print("Future returns shape:", result.future_returns.shape)
    print("Mean IC:", result.evaluation_summary["mean_ic"])
    print("IC Newey-West t-stat:", result.evaluation_summary["t_newey_west"])
    print("Backtest sharpe:", result.evaluation_summary["sharpe"])
    print("Backtest max drawdown:", result.evaluation_summary["max_drawdown"])
