"""Composite-panel and validation helpers for the notebook."""

import numpy as np
import pandas as pd

from ..models.ridge import fit_walk_forward_ridge_scores
from .triage import build_train_factor_report, select_drop_stably_negative_factors
from .utils import slice_panel, window_metrics

COMPOSITE_MIN_COVERAGE_FRAC = 0.60
SELECTION_RULES: tuple[str, ...] = ("all", "drop_stably_negative")
COMBINE_METHODS: tuple[str, ...] = ("ew", "ridge")


def _window_metrics_pair(
    *,
    scores: pd.DataFrame,
    future_returns: pd.DataFrame,
    prices: pd.DataFrame,
    rebalance_freq: str,
    turnover_cost_rate: float,
    borrow_cost_rate_annual: float,
    ic_min_assets: int,
) -> tuple[dict[str, float], dict[str, float]]:
    scores, future_returns = scores.align(future_returns, join="inner")
    gross = window_metrics(
        scores=scores,
        future_returns=future_returns,
        prices=prices,
        rebalance_freq=rebalance_freq,
        turnover_cost_rate=0.0,
        borrow_cost_rate_annual=0.0,
        ic_min_assets=ic_min_assets,
        compute_ic=False,
    )
    net = window_metrics(
        scores=scores,
        future_returns=future_returns,
        prices=prices,
        rebalance_freq=rebalance_freq,
        turnover_cost_rate=turnover_cost_rate,
        borrow_cost_rate_annual=borrow_cost_rate_annual,
        ic_min_assets=ic_min_assets,
    )
    return gross, net


def build_candidate_panels(
    *,
    factor_panels: dict[str, pd.DataFrame],
    candidate_factors: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    """Build raw equal-weight composite panels from explicit notebook-defined factor lists."""
    out: dict[str, pd.DataFrame] = {}
    for candidate_name, selected_factors in candidate_factors.items():
        present_factors = [factor for factor in selected_factors if factor in factor_panels]
        first = factor_panels[present_factors[0]]
        total = first * 0.0
        available = first.notna().astype(float) * 0.0
        required_count = max(
            1,
            int(np.ceil(COMPOSITE_MIN_COVERAGE_FRAC * len(present_factors))),
        )
        for factor in present_factors:
            panel = factor_panels[factor].reindex_like(first)
            present = panel.notna().astype(float)
            total = total + panel.fillna(0.0)
            available = available + present
        composite = total.div(available.where(available > 0.0))
        composite = composite.where(available >= float(required_count))
        out[str(candidate_name)] = composite
    return out


def _select_factor_list(
    *,
    factor_report: pd.DataFrame,
    selection_rule: str,
) -> list[str]:
    """Select factor names from the train report using an explicit notebook rule."""
    if selection_rule == "all":
        return factor_report["factor"].tolist()
    if selection_rule == "drop_stably_negative":
        return select_drop_stably_negative_factors(factor_report)
    raise ValueError(
        f"Unknown selection_rule '{selection_rule}'. "
        f"Available: {list(SELECTION_RULES)}"
    )


def evaluate_validation_matrix(
    *,
    constructs: dict[str, pd.DataFrame],
    future_returns: pd.DataFrame,
    prices: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    rebalance_freq: str,
    turnover_cost_rate: float,
    borrow_cost_rate_annual: float,
    ic_min_assets: int = 50,
) -> pd.DataFrame:
    window_prices = slice_panel(prices, start_date=start, end_date=end)
    window_returns = slice_panel(future_returns, start_date=start, end_date=end)
    rows: list[dict[str, object]] = []
    for construct, panel in constructs.items():
        scores = slice_panel(panel, start_date=start, end_date=end)
        gross, net = _window_metrics_pair(
            scores=scores,
            future_returns=window_returns,
            prices=window_prices,
            rebalance_freq=rebalance_freq,
            turnover_cost_rate=float(turnover_cost_rate),
            borrow_cost_rate_annual=float(borrow_cost_rate_annual),
            ic_min_assets=ic_min_assets,
        )
        rows.append(
            {
                "construct": construct,
                "mean_ic": net["mean_ic"],
                "t_ic_newey_west": net["t_ic_newey_west"],
                "sharpe_gross": gross["sharpe"],
                "sharpe_net": net["sharpe"],
                "max_drawdown_net": net["max_drawdown"],
                "turnover_mean": net["turnover_mean"],
                "turnover_cost_drag_ann": net["turnover_cost_drag_ann"],
                "borrow_cost_drag_ann": net["borrow_cost_drag_ann"],
            }
        )
    matrix = (
        pd.DataFrame(rows)
        .sort_values(["sharpe_net", "t_ic_newey_west"], ascending=False)
        .reset_index(drop=True)
    )
    return matrix


def build_selected_protocol_scores(
    *,
    combine_method: str,
    selection_rule: str,
    factor_panels: dict[str, pd.DataFrame],
    future_returns: pd.DataFrame,
    prices: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    rebalance_freq: str,
    score_start: pd.Timestamp,
    score_end: pd.Timestamp | None = None,
    alpha_grid: tuple[float, ...] | None = None,
    refit_every_n_rebalances: int | None = None,
    ridge_window_type: str = "rolling",
    ridge_window_size: int | None = None,
) -> pd.DataFrame:
    factor_report = build_train_factor_report(
        factor_panels=factor_panels,
        future_returns=future_returns,
        prices=prices,
        train_start=train_start,
        train_end=train_end,
        rebalance_freq=rebalance_freq,
    )
    factor_list = _select_factor_list(
        factor_report=factor_report,
        selection_rule=selection_rule,
    )

    if combine_method == "ew":
        scores = build_candidate_panels(
            factor_panels=factor_panels,
            candidate_factors={"selected": factor_list},
        )["selected"]
    elif combine_method == "ridge":
        if alpha_grid is None:
            raise ValueError("alpha_grid is required when combine_method='ridge'.")
        if refit_every_n_rebalances is None:
            raise ValueError(
                "refit_every_n_rebalances is required when combine_method='ridge'."
            )
        if ridge_window_type == "rolling":
            if ridge_window_size is None or ridge_window_size <= 0:
                raise ValueError(
                    "ridge_window_size must be positive when combine_method='ridge' "
                    "and ridge_window_type='rolling'."
                )
        scores = fit_walk_forward_ridge_scores(
            factor_scores={name: factor_panels[name] for name in factor_list},
            future_returns=future_returns,
            prices=prices,
            rebalance_freq=rebalance_freq,
            tuning_cutoff_date=score_start.strftime("%Y-%m-%d"),
            alpha_grid=alpha_grid,
            window_type=ridge_window_type,
            window_size=None if ridge_window_size is None else int(ridge_window_size),
            refit_every_n_rebalances=int(refit_every_n_rebalances),
        ).scores
    else:
        raise ValueError(
            f"Unknown combine_method '{combine_method}'. "
            f"Available: {list(COMBINE_METHODS)}"
        )

    return slice_panel(scores, start_date=score_start, end_date=score_end)
