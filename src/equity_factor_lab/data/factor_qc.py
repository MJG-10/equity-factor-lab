import numpy as np
import pandas as pd


def _align_coverage_base_mask(
    panel: pd.DataFrame,
    coverage_base_mask: pd.DataFrame | None,
) -> pd.DataFrame:
    """Aligns an optional coverage base to one factor panel."""
    if coverage_base_mask is None:
        return pd.DataFrame(True, index=panel.index, columns=panel.columns, dtype=bool)
    return (
        coverage_base_mask.reindex(index=panel.index, columns=panel.columns, fill_value=False)
        .fillna(False)
        .astype(bool)
    )


def _safe_quantile(series: pd.Series, q: float) -> float:
    """Returns a quantile with graceful empty-series handling."""
    clean = series.dropna().astype(float)
    if clean.empty:
        return float("nan")
    return float(clean.quantile(q))


def _summarize_factor_panel(
    name: str,
    panel: pd.DataFrame,
    *,
    coverage_base_mask: pd.DataFrame | None,
) -> dict[str, object]:
    """Computes compact QC statistics for one factor panel."""
    base_mask = _align_coverage_base_mask(panel, coverage_base_mask)
    values = panel.where(base_mask).to_numpy(dtype=float, copy=False)
    valid_region = base_mask.to_numpy(dtype=bool, copy=False)
    n_cells = int(valid_region.sum())

    if n_cells == 0:
        return {
            "factor": name,
            "non_null_frac": np.nan,
            "date_coverage_p50": np.nan,
            "stock_coverage_p50": np.nan,
            "value_p01": np.nan,
            "value_p99": np.nan,
        }

    finite_mask = np.isfinite(values) & valid_region
    finite_values = values[finite_mask]
    if finite_values.size > 0:
        value_p01 = float(np.quantile(finite_values, 0.01))
        value_p99 = float(np.quantile(finite_values, 0.99))
    else:
        value_p01 = np.nan
        value_p99 = np.nan

    covered = panel.notna() & base_mask
    non_null_count = float(covered.sum(axis=1).sum())
    date_base = base_mask.sum(axis=1).astype(float)
    stock_base = base_mask.sum(axis=0).astype(float)
    date_valid = covered.sum(axis=1).astype(float)
    stock_valid = covered.sum(axis=0).astype(float)
    date_coverage = date_valid.div(date_base.replace(0.0, np.nan))
    stock_coverage = stock_valid.div(stock_base.replace(0.0, np.nan))

    return {
        "factor": name,
        "non_null_frac": float(non_null_count / n_cells),
        "date_coverage_p50": _safe_quantile(date_coverage, 0.50),
        "stock_coverage_p50": _safe_quantile(stock_coverage, 0.50),
        "value_p01": value_p01,
        "value_p99": value_p99,
    }


def summarize_factor_panels(
    factor_panels: dict[str, pd.DataFrame],
    *,
    coverage_base_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Builds a one-row-per-factor QC table with dynamic-universe-aware coverage."""
    if not factor_panels:
        return pd.DataFrame()

    rows = [
        _summarize_factor_panel(
            name,
            panel,
            coverage_base_mask=coverage_base_mask,
        )
        for name, panel in factor_panels.items()
    ]
    return pd.DataFrame(rows).sort_values("factor").reset_index(drop=True)
