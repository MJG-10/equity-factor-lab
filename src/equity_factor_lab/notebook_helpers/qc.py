import pandas as pd


def _cast_integer_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        return
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.isna().any():
        frame[column] = numeric.round().astype("Int64")
    else:
        frame[column] = numeric.round().astype(int)


def build_qc_display_tables(
    *,
    qc_raw_daily: pd.DataFrame,
    qc_tradable_daily: pd.DataFrame,
    qc_cleaning_diff: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return the four QC tables used in the notebook diagnostics section."""
    daily_qc = pd.concat([qc_raw_daily, qc_tradable_daily], ignore_index=True)

    overview = daily_qc.loc[:, ["scope", "n_observed_prices"]].rename(
        columns={
            "scope": "panel",
            "n_observed_prices": "observed_prices",
        }
    )
    _cast_integer_column(overview, "observed_prices")

    tails = daily_qc.loc[
        :,
        [
            "scope",
            "daily_ret_alldays_min",
            "daily_ret_alldays_p01",
            "daily_ret_alldays_p99",
            "daily_ret_alldays_p999",
            "daily_ret_alldays_max",
        ],
    ].rename(
        columns={
            "scope": "panel",
            "daily_ret_alldays_min": "daily_ret_min",
            "daily_ret_alldays_p01": "daily_ret_p01",
            "daily_ret_alldays_p99": "daily_ret_p99",
            "daily_ret_alldays_p999": "daily_ret_p999",
            "daily_ret_alldays_max": "daily_ret_max",
        }
    )

    cleaning_impact = qc_cleaning_diff.loc[
        :,
        [
            "removed_asset_days",
            "removed_frac_of_observed_asset_days",
            "removed_assets_affected",
        ],
    ].rename(
        columns={
            "removed_asset_days": "removed_price_observations",
            "removed_frac_of_observed_asset_days": "removed_observation_frac",
            "removed_assets_affected": "assets_with_removed_observations",
        }
    )
    _cast_integer_column(cleaning_impact, "removed_price_observations")
    _cast_integer_column(cleaning_impact, "assets_with_removed_observations")

    suspect_event_qa = qc_raw_daily.loc[
        :,
        [
            "clean_n_classify_candidates",
            "clean_n_suspect_events",
            "clean_n_spike_reversal",
            "clean_n_persistent_shift",
            "clean_n_stale_jump",
            "clean_n_assets_with_any_suspect_event",
        ],
    ].rename(
        columns={
            "clean_n_classify_candidates": "n_large_positive_candidates",
            "clean_n_suspect_events": "n_suspect_events",
            "clean_n_spike_reversal": "n_spike_reversal",
            "clean_n_persistent_shift": "n_persistent_shift",
            "clean_n_stale_jump": "n_stale_jump",
            "clean_n_assets_with_any_suspect_event": "n_assets_with_any_suspect_event",
        }
    )
    if suspect_event_qa.empty:
        suspect_event_qa = pd.DataFrame(columns=["metric", "value"])
    else:
        suspect_event_qa = suspect_event_qa.iloc[0].rename_axis("metric").reset_index(name="value")
        suspect_numeric = pd.to_numeric(suspect_event_qa["value"], errors="coerce")
        if suspect_numeric.isna().any():
            suspect_event_qa["value"] = suspect_numeric.round().astype("Int64")
        else:
            suspect_event_qa["value"] = suspect_numeric.round().astype(int)

    return {
        "overview": overview,
        "tails": tails,
        "cleaning_impact": cleaning_impact,
        "suspect_event_qa": suspect_event_qa,
    }
