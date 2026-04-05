# Equity Factor Lab

In this project we implement a walk-forward research workflow for a simple cross-sectional equity factor strategy on SimFin US data. We compare a small set of weekly rebalanced strategy variants in pre-holdout validation and then carry one selected strategy variant into holdout with point-in-time investability filters and trading costs.

The objective is to test whether a simple factor mix still performs out of sample once those implementation constraints are applied.

## What we test specifically

- **Data**: SimFin US daily prices, market fields, and lagged fundamental / signal inputs.
- **Prediction target**: cross-sectional forward stock returns after the score date.
- **Evaluation**: Spearman IC for rank quality and long/short decile backtests after costs.
- **Timing convention**: scores at date `t` are evaluated on forward returns after `t`; for weekly `W-FRI` evaluation, returns are computed using prices at rebalance dates.
- **Experiment design**: load data from `2007-01-01` so the first design date in `2008` starts with fully formed long-lookback factors; compare strategy variants on two expanding pre-holdout folds, using `2014-2016` and `2017-2019` as validation windows; then carry one selected strategy variant into holdout from `2020-01-01`, with a scheduled re-estimation at `2023-01-01`.
- **Factor set**: `momentum`, `reversal`, `low_vol`, `value`, `quality`, `invest`, and `growth`.
- **Processing**: cross-sectional winsorization (1% / 99%), `sector` neutralization in the notebook run, and cross-sectional standardization.
- **Universe / eligibility**: price floor, point-in-time liquidity and data-quality filters, score availability, and a top-`1500` point-in-time universe cap by lagged average daily dollar volume (ADV).
- **Candidate strategy variants**: `EW__ALL`, `EW__DROP_STABLY_NEGATIVE`, and `RIDGE__ALL`.
- **Equal-weight screening rule**: `EW__DROP_STABLY_NEGATIVE` removes only factors that are negative on full-train IC and on both train halves.
- **Ridge implementation**: `RIDGE__ALL` uses a rolling window of `156` weekly rebalances (about 3 years), refits every `12` rebalances, and selects alpha from `(0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)` by train-window mean IC.
- **Ridge setting choice**: the ridge window type, size, and refit frequency were chosen based on preliminary pre-holdout trial runs and then kept fixed in the main comparison.
- **Diagnostics**: price QC tables, train-window factor diagnostics, fold-level validation comparisons, pooled validation recap, and holdout block summaries.
- **Costs**: `10 bps` turnover cost and `3%` annualized borrow cost on the short book.

## Results

The primary output of this project is `notebooks/01_results.ipynb`, which contains the tables, figures, and narrative for the validation and holdout comparison.

The installable package `equity_factor_lab` provides the reusable pipeline, factor construction, QC, signal combination, and evaluation code behind that notebook.


**Illustrative findings from the current saved notebook run:**

- The pooled pre-holdout comparison between the two equal-weight strategy variants is close: `EW__DROP_STABLY_NEGATIVE` is slightly better on IC, but `EW__ALL` is slightly better on net Sharpe and drawdown, so `EW__ALL` is carried into holdout.
- `RIDGE__ALL` is the clear weak link: it trails both equal-weight strategy variants in both folds and is worst on pooled IC, net Sharpe, and drawdown.
- In holdout, `EW__ALL` stays positive in both `2020-2022` and `2023+`, but the overall result is modest and still marked by large drawdowns.

## Quickstart

```bash
git clone https://github.com/MJG-10/equity-factor-lab.git
cd equity-factor-lab
python -m venv .venv
```

**Activate the virtual environment**

```powershell
. .\.venv\Scripts\Activate.ps1
```

```bash
source .venv/bin/activate
```

**Install**

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook]"
```

## Configuration

- Set your SimFin API key before running the pipeline or rerunning the notebook.

```powershell
$env:SIMFIN_API_KEY = "your_simfin_key"
```

```bash
export SIMFIN_API_KEY="your_simfin_key"
```

- End-to-end pipeline runs and notebook reruns require a SimFin API key and SimFin Basic access.
- Repo-local data cache defaults to `simfin_data/` via `src/equity_factor_lab/config.py`.
- Runtime settings live in `src/equity_factor_lab/runner/pipeline.py` on the `PipelineSettings` dataclass.
- The convenience script in `src/equity_factor_lab/runner/run_pipeline.py` is an example configuration; the results notebook has its own explicit settings.

## How to run

1. **Pipeline API**

   Use `run_pipeline_core(settings)` when you want prices, QC tables, factor panels, and future returns without evaluation summary packaging. Use `run_pipeline(settings)` when you also want the compact summary metrics. The notebook builds on these pipeline functions plus `src/equity_factor_lab/notebook_helpers/` for the fold and holdout reporting.

   Path: `src/equity_factor_lab/runner/pipeline.py`

2. **Provided pipeline script**

   Path: `src/equity_factor_lab/runner/run_pipeline.py`

   This file contains one example pipeline configuration. You can also run it directly from the terminal:

```bash
python -m equity_factor_lab.runner.run_pipeline
```

   This runs one configured pipeline instance and prints a compact diagnostic summary to the console.

3. **Run tests**

```bash
pytest -q
```

## Pipeline return objects

- `run_pipeline_core(settings)` returns a `PipelineCoreResult` with:
  - cleaned `prices`,
  - QC tables for `raw_daily`, `tradable_daily`, and hard-cleaning impact,
  - `factor_qc_stats`,
  - tradable `factor_scores`,
  - aligned `future_returns`.
- `run_pipeline(settings)` returns a `PipelineResult`, which adds `evaluation_summary` with `mean_ic`, `t_newey_west`, `ic_n_obs`, `sharpe`, and `max_drawdown`.
- These results are returned in memory and then consumed by the notebook or by the console script.

## Repository layout

- `src/equity_factor_lab/data/`: SimFin loaders, price cleaning, price QC, and factor QC.
- `src/equity_factor_lab/factors/`: price and fundamental factor builders plus registry/default selections.
- `src/equity_factor_lab/models/`: neutralization and ridge combination modules.
- `src/equity_factor_lab/notebook_helpers/`: notebook-specific QC, triage, matrix, and utility helpers.
- `src/equity_factor_lab/runner/`: `PipelineSettings`, pipeline stages, tradable-universe construction, and the convenience runner.
- `tests/`: targeted regression and behavior tests.
- `notebooks/01_results.ipynb`: primary research notebook.

## Identifier policy

- SimFinId is the canonical asset identifier for joins and panel alignment.
- `PipelineSettings.market_simfin_id` is the benchmark SimFinId used for `market` neutralization.
