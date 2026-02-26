import numpy as np
import pandas as pd

from ..factors.price_factors import get_rebalance_dates


def _safe_quantile(values: np.ndarray, q: float) -> float:
    """Returns `np.quantile(values, q)` or NaN when `values` is empty."""
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def summarize_price_panel(
    prices: pd.DataFrame,
    *,
    scope: str | None = None,
    schedule: str | None = None,
) -> pd.DataFrame:
    """Returns a one-row QC summary with explicit return-metric semantics."""
    n_dates = int(prices.shape[0])
    n_tickers = int(prices.shape[1])
    n_asset_days = int(n_dates * n_tickers)

    if prices.empty or prices.shape[1] == 0:
        row: dict[str, object] = {
            "scope": scope,
            "schedule": schedule,
            "n_dates": float(n_dates),
            "n_tickers": float(n_tickers),
            "n_asset_days": float(n_asset_days),
            "n_observed_prices": 0.0,
            "price_nan_frac": float("nan"),
            "price_nonpositive_or_nonfinite_count": 0.0,
            "median_obs_per_asset": float("nan"),
            "median_nan_frac_per_asset": float("nan"),
            "max_nan_frac_per_asset": float("nan"),
            "n_const_price_assets": 0.0,
            "n_assets_with_abs_ret_gt_1": 0.0,
            "ret_nonfinite_count": 0.0,
            "daily_ret_alldays_min": float("nan"),
            "daily_ret_alldays_p01": float("nan"),
            "daily_ret_alldays_p99": float("nan"),
            "daily_ret_alldays_p999": float("nan"),
            "daily_ret_alldays_max": float("nan"),
            "daily_ret_assetmin_min": float("nan"),
            "daily_ret_assetmin_p01": float("nan"),
            "daily_ret_assetmax_p99": float("nan"),
            "daily_ret_assetmax_max": float("nan"),
        }
        return pd.DataFrame([row])

    prices_float = prices.astype(float)
    observed_mask = prices_float.notna()
    n_observed = int(observed_mask.sum().sum())
    per_ticker_obs = observed_mask.sum(axis=0).astype(float)
    per_ticker_nan_frac = prices_float.isna().mean(axis=0).astype(float)
    per_ticker_const = prices_float.nunique(axis=0, dropna=True).le(1).astype(float)
    nonpositive_or_nonfinite = (~np.isfinite(prices_float)) | (prices_float <= 0.0)

    returns = prices_float.pct_change(fill_method=None)
    finite_returns = returns.where(np.isfinite(returns))
    per_ticker_ret_min = finite_returns.min(axis=0, skipna=True).astype(float)
    per_ticker_ret_max = finite_returns.max(axis=0, skipna=True).astype(float)
    per_ticker_big_moves = (finite_returns.abs() > 1.0).sum(axis=0).astype(float)

    returns_values = returns.to_numpy(dtype=float).ravel()
    finite_values = returns_values[np.isfinite(returns_values)]
    ret_nonfinite_count = int(np.isinf(returns_values).sum())
    asset_min_values = per_ticker_ret_min.dropna().to_numpy(dtype=float)
    asset_max_values = per_ticker_ret_max.dropna().to_numpy(dtype=float)

    row = {
        "scope": scope,
        "schedule": schedule,
        "n_dates": float(n_dates),
        "n_tickers": float(n_tickers),
        "n_asset_days": float(n_asset_days),
        "n_observed_prices": float(n_observed),
        "price_nan_frac": float(1.0 - (n_observed / n_asset_days)) if n_asset_days > 0 else float("nan"),
        "price_nonpositive_or_nonfinite_count": float(nonpositive_or_nonfinite.sum().sum()),
        "median_obs_per_asset": float(per_ticker_obs.median()),
        "median_nan_frac_per_asset": float(per_ticker_nan_frac.median()),
        "max_nan_frac_per_asset": float(per_ticker_nan_frac.max()),
        "n_const_price_assets": float(per_ticker_const.sum()),
        "n_assets_with_abs_ret_gt_1": float((per_ticker_big_moves > 0.0).sum()),
        "ret_nonfinite_count": float(ret_nonfinite_count),
        "daily_ret_alldays_min": float(finite_values.min()) if finite_values.size > 0 else float("nan"),
        "daily_ret_alldays_p01": _safe_quantile(finite_values, 0.01),
        "daily_ret_alldays_p99": _safe_quantile(finite_values, 0.99),
        "daily_ret_alldays_p999": _safe_quantile(finite_values, 0.999),
        "daily_ret_alldays_max": float(finite_values.max()) if finite_values.size > 0 else float("nan"),
        "daily_ret_assetmin_min": float(asset_min_values.min()) if asset_min_values.size > 0 else float("nan"),
        "daily_ret_assetmin_p01": _safe_quantile(asset_min_values, 0.01),
        "daily_ret_assetmax_p99": _safe_quantile(asset_max_values, 0.99),
        "daily_ret_assetmax_max": float(asset_max_values.max()) if asset_max_values.size > 0 else float("nan"),
    }
    return pd.DataFrame([row])


def summarize_decision_step_price_qc(
    prices: pd.DataFrame,
    *,
    schedules: tuple[str, ...] = ("W-FRI", "ME"),
) -> pd.DataFrame:
    """Returns one QC row per decision schedule using rebalance-date subsets."""
    rows: list[pd.DataFrame] = []
    for schedule in schedules:
        rebalance_dates = get_rebalance_dates(prices, freq=schedule)
        if len(rebalance_dates) == 0:
            decision_prices = prices.iloc[0:0]
        else:
            decision_prices = prices.reindex(rebalance_dates)
        rows.append(
            summarize_price_panel(
                decision_prices,
                scope="tradable_decision",
                schedule=schedule,
            )
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_removed_prices(
    prices_before: pd.DataFrame,
    prices_after: pd.DataFrame,
    *,
    top_n_assets: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarizes points removed by cleaning and returns top affected assets."""
    before, after = prices_before.align(prices_after, join="inner")
    removed_mask = before.notna() & after.isna()
    observed_before = float(before.notna().sum().sum())
    removed_asset_days = float(removed_mask.sum().sum())
    removed_frac = (
        float(removed_asset_days / observed_before)
        if observed_before > 0.0
        else float("nan")
    )

    raw_returns = before.pct_change(fill_method=None)
    removed_returns = raw_returns.where(removed_mask).to_numpy(dtype=float).ravel()
    finite_removed_returns = removed_returns[np.isfinite(removed_returns)]

    removed_by_asset = removed_mask.sum(axis=0).astype(float)
    affected_assets = float((removed_by_asset > 0.0).sum())
    top_assets = (
        removed_by_asset[removed_by_asset > 0.0]
        .sort_values(ascending=False)
        .head(int(top_n_assets))
        .rename("removed_count")
        .reset_index()
    )
    first_col = str(top_assets.columns[0]) if len(top_assets.columns) > 0 else "asset"
    top_assets = top_assets.rename(columns={first_col: "asset_id"})

    summary = pd.DataFrame(
        [
            {
                "removed_asset_days": removed_asset_days,
                "removed_assets_affected": affected_assets,
                "removed_frac_of_observed_asset_days": removed_frac,
                "removed_max_daily_return": (
                    float(finite_removed_returns.max())
                    if finite_removed_returns.size > 0
                    else float("nan")
                ),
                "removed_min_daily_return": (
                    float(finite_removed_returns.min())
                    if finite_removed_returns.size > 0
                    else float("nan")
                ),
            }
        ]
    )
    return summary, top_assets
