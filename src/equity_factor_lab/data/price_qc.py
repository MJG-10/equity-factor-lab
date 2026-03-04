import numpy as np
import pandas as pd


def _safe_quantile(values: np.ndarray, q: float) -> float:
    """Returns `np.quantile(values, q)` or NaN when `values` is empty."""
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def summarize_price_panel(
    prices: pd.DataFrame,
    *,
    scope: str | None = None,
) -> pd.DataFrame:
    """Returns a compact price QC summary row."""
    prices_float = prices.astype(float, copy=False)
    n_observed = int(prices_float.notna().sum().sum())

    if prices.empty or prices.shape[1] == 0:
        return pd.DataFrame(
            [
                {
                    "scope": scope,
                    "n_observed_prices": 0,
                    "daily_ret_alldays_min": float("nan"),
                    "daily_ret_alldays_p01": float("nan"),
                    "daily_ret_alldays_p99": float("nan"),
                    "daily_ret_alldays_p999": float("nan"),
                    "daily_ret_alldays_max": float("nan"),
                }
            ]
        )

    returns = prices_float.pct_change(fill_method=None).to_numpy(dtype=float).ravel()
    finite_values = returns[np.isfinite(returns)]

    row = {
        "scope": scope,
        "n_observed_prices": n_observed,
        "daily_ret_alldays_min": float(finite_values.min()) if finite_values.size > 0 else float("nan"),
        "daily_ret_alldays_p01": _safe_quantile(finite_values, 0.01),
        "daily_ret_alldays_p99": _safe_quantile(finite_values, 0.99),
        "daily_ret_alldays_p999": _safe_quantile(finite_values, 0.999),
        "daily_ret_alldays_max": float(finite_values.max()) if finite_values.size > 0 else float("nan"),
    }
    return pd.DataFrame([row])


def summarize_removed_prices(
    prices_before: pd.DataFrame,
    prices_after: pd.DataFrame,
) -> pd.DataFrame:
    """Summarizes the cleaning-impact fields used in QC displays."""
    before, after = prices_before.align(prices_after, join="inner")
    removed_mask = before.notna() & after.isna()
    observed_before = float(before.notna().sum().sum())
    removed_asset_days = int(removed_mask.sum().sum())
    removed_frac = (
        float(removed_asset_days / observed_before)
        if observed_before > 0.0
        else float("nan")
    )

    removed_by_asset = removed_mask.sum(axis=0).astype(float)
    affected_assets = int((removed_by_asset > 0.0).sum())

    summary = pd.DataFrame(
        [
            {
                "removed_asset_days": removed_asset_days,
                "removed_assets_affected": affected_assets,
                "removed_frac_of_observed_asset_days": removed_frac,
            }
        ]
    )
    return summary
