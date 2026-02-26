from pathlib import Path
import warnings
import pandas as pd
import simfin as sf
from simfin.names import PUBLISH_DATE, REPORT_DATE, RESTATED_DATE, SIMFIN_ID, TICKER
from .simfin_prices import normalize_id_and_filter_universe
from .simfin_setup import configure_simfin

FUNDAMENTAL_DATE_INDEX = "Date"


def _rename_fundamental_date_level(financials: pd.DataFrame) -> pd.DataFrame:
    """Renames the fundamental date index level to the canonical `Date` name."""
    if not isinstance(financials.index, pd.MultiIndex):
        return financials
    if FUNDAMENTAL_DATE_INDEX in financials.index.names:
        return financials
    if REPORT_DATE not in financials.index.names:
        return financials
    return financials.rename_axis(index={REPORT_DATE: FUNDAMENTAL_DATE_INDEX})


def _align_to_publish_date(
    financials: pd.DataFrame,
    *,
    shift_business_days: int,
    dataset_name: str,
    duplicate_policy: str = "raise",
) -> pd.DataFrame:
    """
    Aligns statement rows to effective dates using Publish Date plus an optional business-day shift.

    Duplicate aligned keys are handled by `duplicate_policy` (`raise` or `keep_last`).
    """
    if financials.empty:
        return financials

    data = financials.reset_index()
    if PUBLISH_DATE not in data.columns:
        raise KeyError(
            f"'{dataset_name}' is missing '{PUBLISH_DATE}' column required for point-in-time alignment."
        )

    publish_dates = pd.to_datetime(data[PUBLISH_DATE], errors="coerce")
    missing_publish = publish_dates.isna()
    if missing_publish.any():
        dropped = int(missing_publish.sum())
        warnings.warn(
            f"Dropping {dropped} row(s) from '{dataset_name}' due to missing publish dates."
        )
        data = data.loc[~missing_publish].copy()
        publish_dates = publish_dates.loc[~missing_publish]
    if data.empty:
        data[FUNDAMENTAL_DATE_INDEX] = publish_dates
        data = data.drop(columns=[REPORT_DATE, PUBLISH_DATE, RESTATED_DATE, TICKER], errors="ignore")
        return data.set_index([SIMFIN_ID, FUNDAMENTAL_DATE_INDEX])

    effective_dates = publish_dates
    if shift_business_days > 0:
        effective_dates = effective_dates + pd.offsets.BDay(shift_business_days)
    data[FUNDAMENTAL_DATE_INDEX] = effective_dates

    sort_columns = [SIMFIN_ID, FUNDAMENTAL_DATE_INDEX]
    duplicate_mask = data.duplicated(subset=sort_columns, keep=False)
    if duplicate_mask.any():
        if duplicate_policy == "raise":
            duplicate_rows = int(duplicate_mask.sum())
            duplicate_examples = (
                data.loc[duplicate_mask, sort_columns]
                .drop_duplicates()
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                f"'{dataset_name}' has {duplicate_rows} row(s) with duplicate aligned keys "
                f"{sort_columns}. Strict point-in-time mode requires unique aligned keys. "
                f"Example duplicates: {duplicate_examples}"
            )
        if duplicate_policy == "keep_last":
            sort_keys = [SIMFIN_ID, FUNDAMENTAL_DATE_INDEX]
            if REPORT_DATE in data.columns:
                sort_keys.append(REPORT_DATE)
            if "Fiscal Year" in data.columns:
                sort_keys.append("Fiscal Year")
            if "Fiscal Period" in data.columns:
                sort_keys.append("Fiscal Period")
            data["_row_id"] = range(len(data))
            data = data.sort_values(sort_keys + ["_row_id"])
            data = data.drop_duplicates(subset=sort_columns, keep="last")
            data = data.drop(columns=["_row_id"])
        else:
            raise ValueError(
                f"Unknown duplicate_policy={duplicate_policy!r}. Use 'raise' or 'keep_last'."
            )

    data = data.drop(columns=[REPORT_DATE, PUBLISH_DATE, RESTATED_DATE, TICKER], errors="ignore")

    aligned = data.set_index(sort_columns).sort_index()
    return aligned


def load_universe_financial_statements(
    *,
    simfin_ids: set[int],
    start: str,
    end: str | None,
    api_key: str | None = None,
    data_dir: str | Path | None = None,
    refresh_days: int,
    align_to_publish_date: bool,
    publish_shift_business_days: int,
    duplicate_policy: str = "keep_last",
) -> dict[str, pd.DataFrame]:
    """Loads SimFin statement datasets for a SimFinId universe and optionally aligns to publish date."""
    configure_simfin(api_key=api_key, data_dir=data_dir)

    # Use as-reported statement variants to avoid timeline collisions caused by
    # restated-date alignment in standardized datasets.
    dataset_specs = [
        ("income_ttm", sf.load_income, "ttm-asreported"),
        ("income_quarterly", sf.load_income, "quarterly-asreported"),
        ("balance_ttm", sf.load_balance, "ttm-asreported"),
        ("balance_quarterly", sf.load_balance, "quarterly-asreported"),
        ("cashflow_ttm", sf.load_cashflow, "ttm-asreported"),
        ("cashflow_quarterly", sf.load_cashflow, "quarterly-asreported"),
    ]

    statements_raw: dict[str, pd.DataFrame] = {}
    for dataset_name, loader_func, variant in dataset_specs:
        raw = loader_func(
            variant=variant,
            market="us",
            start_date=start,
            end_date=end,
            refresh_days=refresh_days,
            index=[SIMFIN_ID, REPORT_DATE],
        )
        statements_raw[dataset_name] = normalize_id_and_filter_universe(
            raw,
            simfin_ids,
            dedupe=False,
            context=f"{dataset_name}_raw",
        )

    if align_to_publish_date:
        if publish_shift_business_days < 0:
            raise ValueError("publish_shift_business_days cannot be negative.")
        statements = {
            dataset_name: _align_to_publish_date(
                frame,
                shift_business_days=publish_shift_business_days,
                dataset_name=dataset_name,
                duplicate_policy=duplicate_policy,
            )
            for dataset_name, frame in statements_raw.items()
        }
    else:
        statements = statements_raw

    statements = {
        dataset_name: _rename_fundamental_date_level(frame)
        for dataset_name, frame in statements.items()
    }
    return statements
