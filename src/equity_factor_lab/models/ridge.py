from dataclasses import dataclass
import numpy as np
import pandas as pd
from ..evaluation import build_ic_inputs
from ..factors.registry import get_fundamental_factor_builders, get_price_factor_builders
from ..metrics import compute_ic_series


@dataclass(frozen=True)
class RidgeFitResult:
    """Stores walk-forward ridge scores, selected alpha, and alpha validation metrics."""

    scores: pd.DataFrame
    best_alpha: float
    validation_table: pd.DataFrame


@dataclass(frozen=True)
class RidgeCoverageSpec:
    """Coverage gate used for ridge train/predict row eligibility."""

    price_feature_names: tuple[str, ...]
    fundamental_feature_names: tuple[str, ...]
    min_total: int
    min_price: int
    min_fundamental: int


def build_default_coverage_spec(feature_names: tuple[str, ...]) -> RidgeCoverageSpec:
    """Builds the default ridge coverage gate from price/fundamental factor membership."""
    price_available = set(get_price_factor_builders().keys())
    fundamental_available = set(get_fundamental_factor_builders().keys())
    price_feature_names = tuple(name for name in feature_names if name in price_available)
    fundamental_feature_names = tuple(
        name for name in feature_names if name in fundamental_available
    )
    return RidgeCoverageSpec(
        price_feature_names=price_feature_names,
        fundamental_feature_names=fundamental_feature_names,
        min_total=min(5, len(feature_names)),
        min_price=min(2, len(price_feature_names)),
        min_fundamental=min(2, len(fundamental_feature_names)),
    )


def _coverage_mask(features: pd.DataFrame, coverage_spec: RidgeCoverageSpec) -> pd.Series:
    """Returns True where feature presence satisfies the ridge coverage gate."""
    if features.empty:
        return pd.Series(dtype=bool, index=features.index)

    present = features.notna()
    valid = present.sum(axis=1) >= int(coverage_spec.min_total)

    if coverage_spec.min_price > 0 and coverage_spec.price_feature_names:
        price_present = present.loc[:, list(coverage_spec.price_feature_names)].sum(axis=1)
        valid = valid & (price_present >= int(coverage_spec.min_price))
    if coverage_spec.min_fundamental > 0 and coverage_spec.fundamental_feature_names:
        fundamental_present = present.loc[:, list(coverage_spec.fundamental_feature_names)].sum(
            axis=1
        )
        valid = valid & (fundamental_present >= int(coverage_spec.min_fundamental))

    return valid


def align_factors_and_returns(
    factor_scores: dict[str, pd.DataFrame],
    future_returns: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Aligns factor panels and forward returns to a common panel shape."""
    if not factor_scores:
        raise ValueError("No factor panels provided for ridge fitting.")

    common_index = future_returns.index
    common_columns = future_returns.columns
    for panel in factor_scores.values():
        common_index = common_index.intersection(panel.index)
        common_columns = common_columns.intersection(panel.columns)

    if len(common_index) == 0 or len(common_columns) == 0:
        raise ValueError("No common index/columns across factor panels and future returns.")

    aligned_returns = future_returns.loc[common_index, common_columns]
    aligned_factors = {
        name: panel.loc[common_index, common_columns]
        for name, panel in factor_scores.items()
    }
    return aligned_factors, aligned_returns


def panel_to_long(
    factor_panels: dict[str, pd.DataFrame],
    target_panel: pd.DataFrame,
    coverage_spec: RidgeCoverageSpec,
) -> pd.DataFrame:
    """Stacks factor panels and target into a long regression table with coverage gating."""
    features = pd.concat(
        {name: panel.stack(future_stack=True) for name, panel in factor_panels.items()},
        axis=1,
    )
    target = target_panel.stack(future_stack=True).rename("target")
    valid = _coverage_mask(features, coverage_spec)
    data = features.fillna(0.0).join(target)
    data = data[valid]
    data = data.dropna(subset=["target"])
    return data


def fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    """Fits closed-form ridge coefficients and intercept."""
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0, ddof=0)
    x_std = np.where(x_std == 0.0, 1.0, x_std)

    y_mean = float(y.mean())
    y_centered = y - y_mean
    x_scaled = (X - x_mean) / x_std

    xtx = x_scaled.T @ x_scaled
    reg = float(alpha) * np.eye(xtx.shape[0], dtype=float)
    xty = x_scaled.T @ y_centered

    try:
        coef_scaled = np.linalg.solve(xtx + reg, xty)
    except np.linalg.LinAlgError:
        coef_scaled = np.linalg.lstsq(xtx + reg, xty, rcond=None)[0]

    coef_raw = coef_scaled / x_std
    intercept = y_mean - float(np.dot(x_mean, coef_raw))
    return coef_raw, intercept


def training_dates_for_step(
    all_rebalance_dates: pd.DatetimeIndex,
    step_idx: int,
    window_type: str,
    window_size: int | None,
) -> pd.DatetimeIndex:
    """Returns training dates for a walk-forward step."""
    if step_idx <= 0:
        return pd.DatetimeIndex([])

    hist_dates = all_rebalance_dates[:step_idx]
    if window_type == "expanding":
        return hist_dates
    if window_type != "rolling":
        raise ValueError(f"Unknown ridge window_type '{window_type}'. Use 'rolling' or 'expanding'.")
    if window_size is None or window_size <= 0:
        raise ValueError("window_size must be a positive integer when window_type='rolling'.")
    return hist_dates[-window_size:]


def predict_cross_section_for_date(
    factor_panels: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    coef: np.ndarray,
    intercept: float,
    coverage_spec: RidgeCoverageSpec,
) -> pd.Series:
    """Predicts cross-sectional scores for one rebalance date."""
    features = pd.DataFrame(
        {name: panel.loc[date] for name, panel in factor_panels.items()}
    )
    valid_assets = _coverage_mask(features, coverage_spec)

    pred = pd.Series(np.nan, index=features.index, dtype=float)
    if valid_assets.sum() == 0:
        return pred

    X = features.fillna(0.0).loc[valid_assets].to_numpy(dtype=float)
    pred.loc[valid_assets] = X @ coef + intercept
    return pred


def _prepare_rebalance_ridge_inputs(
    factor_scores: dict[str, pd.DataFrame],
    future_returns: pd.DataFrame,
    prices: pd.DataFrame | None,
    *,
    rebalance_freq: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, RidgeCoverageSpec]:
    """Builds aligned rebalance-date ridge inputs from factor panels and returns."""
    base_factors = {
        name: panel
        for name, panel in factor_scores.items()
        if name != "composite"
    }
    if len(base_factors) < 2:
        raise ValueError("Need at least two non-composite factor panels for ridge fitting.")

    aligned_factors, aligned_returns = align_factors_and_returns(base_factors, future_returns)
    aligned_prices = None
    if prices is not None:
        aligned_prices = prices.reindex(index=aligned_returns.index, columns=aligned_returns.columns)

    reference_panel = next(iter(aligned_factors.values()))
    rebal_reference_scores, rebal_target_returns = build_ic_inputs(
        scores=reference_panel,
        future_returns=aligned_returns,
        rebalance_freq=rebalance_freq,
        prices=aligned_prices,
    )
    rebal_factor_panels = {
        name: panel.reindex(index=rebal_reference_scores.index, columns=rebal_reference_scores.columns)
        for name, panel in aligned_factors.items()
    }
    coverage_spec = build_default_coverage_spec(tuple(rebal_factor_panels.keys()))
    return rebal_factor_panels, rebal_target_returns, aligned_returns, coverage_spec


def _split_long_data_by_date(
    long_data: pd.DataFrame,
) -> tuple[dict[pd.Timestamp, pd.DataFrame], dict[pd.Timestamp, int]]:
    """Splits the long ridge table into per-date blocks and row counts."""
    date_blocks: dict[pd.Timestamp, pd.DataFrame] = {}
    date_row_counts: dict[pd.Timestamp, int] = {}
    for date, block in long_data.groupby(level=0, sort=False):
        ts = pd.Timestamp(date)
        date_blocks[ts] = block
        date_row_counts[ts] = int(len(block))
    return date_blocks, date_row_counts


def walk_forward_predict_rebalance(
    factor_panels: dict[str, pd.DataFrame],
    target_panel: pd.DataFrame,
    *,
    alpha: float,
    coverage_spec: RidgeCoverageSpec,
    window_type: str,
    window_size: int | None,
    min_train_periods: int,
    refit_every_n_rebalances: int,
    min_train_rows: int,
) -> pd.DataFrame:
    """Runs walk-forward ridge fitting and prediction on rebalance dates."""
    rebalance_dates = target_panel.index
    assets = target_panel.columns
    predictions = pd.DataFrame(
        np.nan,
        index=rebalance_dates,
        columns=assets,
        dtype=float,
    )

    feature_names = list(factor_panels.keys())
    n_features = len(feature_names)
    min_train_rows_effective = max(int(min_train_rows), n_features + 1)

    long_data = panel_to_long(factor_panels, target_panel, coverage_spec=coverage_spec)
    if long_data.empty:
        return predictions
    date_blocks, date_row_counts = _split_long_data_by_date(long_data)

    coef: np.ndarray | None = None
    intercept: float | None = None
    last_refit_idx = -10**9

    for step_idx, date in enumerate(rebalance_dates):
        train_dates = training_dates_for_step(
            all_rebalance_dates=rebalance_dates,
            step_idx=step_idx,
            window_type=window_type,
            window_size=window_size,
        )
        if len(train_dates) < min_train_periods:
            continue

        should_refit = (
            coef is None
            or intercept is None
            or (step_idx - last_refit_idx) >= refit_every_n_rebalances
        )
        if should_refit:
            train_row_count = sum(date_row_counts.get(pd.Timestamp(dt), 0) for dt in train_dates)
            if train_row_count < min_train_rows_effective:
                coef = None
                intercept = None
                continue

            train_blocks = [
                date_blocks[pd.Timestamp(dt)]
                for dt in train_dates
                if pd.Timestamp(dt) in date_blocks
            ]
            if not train_blocks:
                coef = None
                intercept = None
                continue

            train_data = train_blocks[0] if len(train_blocks) == 1 else pd.concat(train_blocks, axis=0)
            X_train = train_data[feature_names].to_numpy(dtype=float)
            y_train = train_data["target"].to_numpy(dtype=float)
            coef, intercept = fit_ridge(X_train, y_train, alpha=float(alpha))
            last_refit_idx = step_idx

        if coef is None or intercept is None:
            continue

        predictions.loc[date] = predict_cross_section_for_date(
            factor_panels=factor_panels,
            date=date,
            coef=coef,
            intercept=intercept,
            coverage_spec=coverage_spec,
        )

    return predictions


def score_alpha_on_pre_holdout(
    prediction_panel: pd.DataFrame,
    target_panel: pd.DataFrame,
    pre_dates: pd.DatetimeIndex,
    min_assets: int,
) -> tuple[float, int]:
    """Scores one alpha by pre-holdout mean IC and observation count."""
    pred_pre = prediction_panel[prediction_panel.index.isin(pre_dates)]
    target_pre = target_panel.reindex(index=pred_pre.index, columns=pred_pre.columns)
    ic = compute_ic_series(
        pred_pre,
        target_pre,
        min_assets=min_assets,
    )
    if len(ic) == 0:
        return np.nan, 0
    return float(ic.mean()), int(len(ic))


def build_walk_forward_ridge_scores_fixed_alpha(
    factor_scores: dict[str, pd.DataFrame],
    future_returns: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    *,
    rebalance_freq: str,
    alpha: float,
    window_type: str = "rolling",
    window_size: int | None = None,
    min_train_periods: int = 24,
    refit_every_n_rebalances: int = 1,
    min_train_rows: int = 500,
) -> pd.DataFrame:
    """Builds walk-forward ridge scores with a fixed alpha and no tuning."""
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite.")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative for ridge.")
    if rebalance_freq != "D" and prices is None:
        raise ValueError("prices are required for ridge when rebalance_freq is not 'D'.")

    rebal_factor_panels, rebal_target_returns, aligned_returns, coverage_spec = (
        _prepare_rebalance_ridge_inputs(
            factor_scores=factor_scores,
            future_returns=future_returns,
            prices=prices,
            rebalance_freq=rebalance_freq,
        )
    )

    prediction_rebal = walk_forward_predict_rebalance(
        factor_panels=rebal_factor_panels,
        target_panel=rebal_target_returns,
        alpha=float(alpha),
        coverage_spec=coverage_spec,
        window_type=window_type,
        window_size=window_size,
        min_train_periods=min_train_periods,
        refit_every_n_rebalances=refit_every_n_rebalances,
        min_train_rows=min_train_rows,
    )

    ridge_scores = prediction_rebal.reindex(aligned_returns.index)
    ridge_scores = ridge_scores.reindex(columns=aligned_returns.columns)
    return ridge_scores


def fit_walk_forward_ridge_scores(
    factor_scores: dict[str, pd.DataFrame],
    future_returns: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    *,
    rebalance_freq: str,
    tuning_cutoff_date: str,
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 5.0, 10.0, 25.0, 50.0),
    window_type: str = "rolling",
    window_size: int | None = None,
    min_train_periods: int = 24,
    refit_every_n_rebalances: int = 1,
    min_train_rows: int = 500,
    min_assets: int = 50,
) -> RidgeFitResult:
    """Fits walk-forward ridge scores and tunes alpha before the cutoff date.

    Runtime scales with `len(alpha_grid)` and refit frequency.
    """
    if len(alpha_grid) == 0:
        raise ValueError("alpha_grid cannot be empty.")
    try:
        alpha_values = np.asarray(alpha_grid, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha_grid must contain numeric values.") from exc
    if not np.isfinite(alpha_values).all():
        raise ValueError("alpha_grid must contain only finite values.")
    if (alpha_values < 0.0).any():
        raise ValueError("alpha_grid must be non-negative for ridge.")
    if rebalance_freq != "D" and prices is None:
        raise ValueError("prices are required for ridge when rebalance_freq is not 'D'.")
    alpha_candidates = tuple(float(value) for value in alpha_values.tolist())

    rebal_factor_panels, rebal_target_returns, aligned_returns, coverage_spec = (
        _prepare_rebalance_ridge_inputs(
            factor_scores=factor_scores,
            future_returns=future_returns,
            prices=prices,
            rebalance_freq=rebalance_freq,
        )
    )

    tuning_cutoff = pd.Timestamp(tuning_cutoff_date)
    pre_dates = rebal_target_returns.index[rebal_target_returns.index < tuning_cutoff]
    if len(pre_dates) == 0:
        raise ValueError("No pre-cutoff dates found for ridge tuning.")

    validation_rows: list[dict[str, float]] = []
    best_prediction_rebal: pd.DataFrame | None = None
    best_mean_ic = -np.inf
    best_alpha_candidate: float | None = None
    for alpha in alpha_candidates:
        prediction_rebal = walk_forward_predict_rebalance(
            factor_panels=rebal_factor_panels,
            target_panel=rebal_target_returns,
            alpha=float(alpha),
            coverage_spec=coverage_spec,
            window_type=window_type,
            window_size=window_size,
            min_train_periods=min_train_periods,
            refit_every_n_rebalances=refit_every_n_rebalances,
            min_train_rows=min_train_rows,
        )
        pre_mean_ic, pre_n_obs = score_alpha_on_pre_holdout(
            prediction_panel=prediction_rebal,
            target_panel=rebal_target_returns,
            pre_dates=pre_dates,
            min_assets=min_assets,
        )
        validation_rows.append(
            {
                "alpha": float(alpha),
                "pre_mean_ic": pre_mean_ic,
                "pre_n_obs": float(pre_n_obs),
            }
        )
        if np.isfinite(pre_mean_ic):
            is_better = (
                best_prediction_rebal is None
                or pre_mean_ic > best_mean_ic
                or (pre_mean_ic == best_mean_ic and best_alpha_candidate is not None and float(alpha) < best_alpha_candidate)
            )
            if is_better:
                best_prediction_rebal = prediction_rebal
                best_mean_ic = float(pre_mean_ic)
                best_alpha_candidate = float(alpha)
    validation_table = pd.DataFrame(validation_rows).sort_values(
        ["pre_mean_ic", "alpha"],
        ascending=[False, True],
    ).reset_index(drop=True)

    if not validation_table["pre_mean_ic"].notna().any():
        raise ValueError(
            "Ridge alpha tuning failed: all candidates produced NaN pre-holdout IC. "
            "Try more history, larger universe, or looser training settings."
        )
    best_alpha = float(validation_table.iloc[0]["alpha"])
    if best_prediction_rebal is None:
        raise ValueError("Ridge alpha tuning failed to retain the best prediction panel.")

    ridge_scores = best_prediction_rebal.reindex(aligned_returns.index)
    ridge_scores = ridge_scores.reindex(columns=aligned_returns.columns)
    return RidgeFitResult(
        scores=ridge_scores,
        best_alpha=best_alpha,
        validation_table=validation_table,
    )
