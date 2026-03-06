"""Factor-triage helpers for the notebook."""

import pandas as pd

from ..evaluation import build_ic_inputs
from ..metrics import compute_ic_series, compute_ic_stats
from .utils import slice_panel


def select_drop_stably_negative_factors(report: pd.DataFrame) -> list[str]:
    ordered = report.sort_values(
        ["train_ic_t_nw_spearman", "train_ic_mean_spearman"],
        ascending=False,
    ).reset_index(drop=True)
    stable_negative_mask = (
        (ordered["train_ic_mean_spearman"] < 0.0)
        & (ordered["train_ic_mean_first_half"] < 0.0)
        & (ordered["train_ic_mean_second_half"] < 0.0)
    )
    return ordered.loc[~stable_negative_mask, "factor"].tolist()

def build_train_factor_report(
    *,
    factor_panels: dict[str, pd.DataFrame],
    future_returns: pd.DataFrame,
    prices: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    rebalance_freq: str,
    ic_min_assets: int = 50,
) -> pd.DataFrame:
    """Builds a factor diagnostics table for one explicit train window."""
    if not factor_panels:
        raise ValueError("factor_panels cannot be empty.")
    train_start = pd.Timestamp(train_start)
    train_end = pd.Timestamp(train_end)
    train_years = list(range(int(train_start.year), int(train_end.year) + 1))
    if not train_years:
        raise ValueError("train window must include at least one calendar year.")
    split_idx = (len(train_years) + 1) // 2
    first_half_end = (
        train_end
        if len(train_years) == 1
        else pd.Timestamp(f"{train_years[split_idx - 1]}-12-31")
    )
    second_half_start = (
        train_end + pd.Timedelta(days=1)
        if len(train_years) == 1
        else pd.Timestamp(f"{train_years[split_idx]}-01-01")
    )

    train_prices = slice_panel(prices, start_date=train_start, end_date=train_end)
    train_returns = slice_panel(future_returns, start_date=train_start, end_date=train_end)
    rows: list[dict[str, object]] = []

    for name, panel in factor_panels.items():
        scores_train = slice_panel(panel, start_date=train_start, end_date=train_end)
        if scores_train.empty:
            raise ValueError(f"Factor '{name}' has no data in the requested train window.")
        ic_scores, ic_fwd = build_ic_inputs(
            scores=scores_train,
            future_returns=train_returns,
            rebalance_freq=rebalance_freq,
            prices=train_prices,
        )

        ic_s = compute_ic_series(ic_scores, ic_fwd, min_assets=ic_min_assets)
        ic_stats = compute_ic_stats(ic_s)
        mean_ic = float(ic_stats["mean_ic"]) if pd.notna(ic_stats["mean_ic"]) else float("nan")

        row: dict[str, object] = {
            "factor": name,
            "train_ic_mean_spearman": mean_ic,
            "train_ic_t_nw_spearman": (
                float(ic_stats["t_newey_west"])
                if pd.notna(ic_stats["t_newey_west"])
                else float("nan")
            ),
            "train_ic_n_obs": int(ic_stats["n_obs"]) if pd.notna(ic_stats["n_obs"]) else 0,
        }
        first_half = ic_s.loc[(ic_s.index >= train_start) & (ic_s.index <= first_half_end)]
        second_half = ic_s.loc[(ic_s.index >= second_half_start) & (ic_s.index <= train_end)]
        row["train_ic_mean_first_half"] = float(first_half.mean()) if len(first_half) else float("nan")
        row["train_ic_mean_second_half"] = (
            float(second_half.mean()) if len(second_half) else float("nan")
        )
        rows.append(row)

    report = (
        pd.DataFrame(rows)
        .sort_values(["train_ic_t_nw_spearman", "train_ic_mean_spearman"], ascending=False)
        .reset_index(drop=True)
    )
    if report.empty:
        raise ValueError("No factor train diagnostics were produced.")
    return report
