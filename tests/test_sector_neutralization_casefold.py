import pandas as pd

from equity_factor_lab.models.neutralization import _sector_neutralize_scores


def test_sector_neutralization_demeans_same_sector_simfin_ids() -> None:
    scores = pd.DataFrame(
        [[1.0, 3.0], [2.0, 4.0]],
        index=pd.DatetimeIndex([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]),
        columns=[101, 202],
    )
    sector_map = pd.Series(
        ["Health Care", "Health Care"],
        index=pd.Index([101, 202], name="SimFinId"),
        name="Sector",
    )

    neutralized = _sector_neutralize_scores(scores, sector_map)

    # Same-sector two-name demean: rows become [-1, +1].
    assert neutralized.loc[pd.Timestamp("2020-01-01"), 101] == -1.0
    assert neutralized.loc[pd.Timestamp("2020-01-01"), 202] == 1.0
    assert neutralized.loc[pd.Timestamp("2020-01-02"), 101] == -1.0
    assert neutralized.loc[pd.Timestamp("2020-01-02"), 202] == 1.0
