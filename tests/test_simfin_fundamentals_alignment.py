import pandas as pd
import pytest
from simfin.names import PUBLISH_DATE, REPORT_DATE, SIMFIN_ID

from equity_factor_lab.data.simfin_fundamentals import FUNDAMENTAL_DATE_INDEX, _align_to_publish_date


def _build_financials(rows: list[tuple[int, str, str | None, float]]) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(simfin_id, pd.Timestamp(report_date)) for simfin_id, report_date, _, _ in rows],
        names=[SIMFIN_ID, REPORT_DATE],
    )
    return pd.DataFrame(
        {
            PUBLISH_DATE: [
                pd.Timestamp(publish_date) if publish_date is not None else pd.NaT
                for _, _, publish_date, _ in rows
            ],
            "Metric": [metric for _, _, _, metric in rows],
        },
        index=index,
    )


def test_align_to_publish_date_uses_business_day_shift() -> None:
    financials = _build_financials(
        [
            (1001, "2019-12-31", "2020-02-14", 1.0),  # Friday
        ]
    )

    aligned = _align_to_publish_date(
        financials,
        shift_business_days=1,
        dataset_name="income_ttm",
        duplicate_policy="raise",
    )

    assert aligned.index.names == [SIMFIN_ID, FUNDAMENTAL_DATE_INDEX]
    assert aligned.index.get_level_values(FUNDAMENTAL_DATE_INDEX)[0] == pd.Timestamp("2020-02-17")


def test_align_to_publish_date_raises_on_duplicate_aligned_keys() -> None:
    financials = _build_financials(
        [
            (1001, "2019-12-31", "2020-02-15", 1.0),
            (1001, "2020-01-31", "2020-02-15", 2.0),
        ]
    )

    with pytest.raises(ValueError, match="duplicate aligned keys"):
        _align_to_publish_date(
            financials,
            shift_business_days=0,
            dataset_name="income_ttm",
            duplicate_policy="raise",
        )


def test_align_to_publish_date_keep_last_resolves_duplicates() -> None:
    financials = _build_financials(
        [
            (1001, "2019-12-31", "2020-02-15", 1.0),
            (1001, "2020-01-31", "2020-02-15", 3.0),
        ]
    )
    financials["Fiscal Year"] = [2019, 2020]

    aligned = _align_to_publish_date(
        financials,
        shift_business_days=0,
        dataset_name="income_ttm",
        duplicate_policy="keep_last",
    )

    assert len(aligned) == 1
    assert float(aligned["Metric"].iloc[0]) == 3.0


def test_align_to_publish_date_drops_missing_publish_dates() -> None:
    financials = _build_financials(
        [
            (1001, "2019-12-31", None, 1.0),
            (1001, "2020-03-31", "2020-05-15", 2.0),
        ]
    )

    with pytest.warns(UserWarning, match="missing publish dates"):
        aligned = _align_to_publish_date(
            financials,
            shift_business_days=0,
            dataset_name="income_ttm",
            duplicate_policy="raise",
        )

    assert len(aligned) == 1
    assert float(aligned["Metric"].iloc[0]) == 2.0
