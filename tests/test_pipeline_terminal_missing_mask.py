import numpy as np
import pandas as pd

from equity_factor_lab.missing_returns import build_terminal_missing_mask


def test_terminal_missing_mask_flags_only_last_missing_gaps() -> None:
    future_returns = pd.DataFrame(
        [
            [0.01, np.nan],
            [np.nan, 0.02],
            [np.nan, np.nan],
        ],
        index=pd.DatetimeIndex(
            [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
        ),
        columns=["A", "B"],
    )

    mask = build_terminal_missing_mask(future_returns)

    assert bool(mask.loc[pd.Timestamp("2020-01-02"), "A"]) is True
    assert bool(mask.loc[pd.Timestamp("2020-01-01"), "B"]) is False
    assert bool(mask.loc[pd.Timestamp("2020-01-03"), "A"]) is True
    assert bool(mask.loc[pd.Timestamp("2020-01-03"), "B"]) is True
