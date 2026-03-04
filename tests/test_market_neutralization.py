import pandas as pd
import pytest

from equity_factor_lab.models.neutralization import _market_neutralize_scores


def test_market_neutralization_zero_variance_beta_returns_centered_scores() -> None:
    dt = pd.DatetimeIndex([pd.Timestamp("2020-01-01")])
    cols = [101, 202, 303]

    scores = pd.DataFrame([[1.0, 2.0, 4.0]], index=dt, columns=cols)
    betas = pd.DataFrame([[0.5, 0.5, 0.5]], index=dt, columns=cols)

    neutralized = _market_neutralize_scores(scores, betas)

    assert neutralized.loc[dt[0], 101] == pytest.approx(-4.0 / 3.0)
    assert neutralized.loc[dt[0], 202] == pytest.approx(-1.0 / 3.0)
    assert neutralized.loc[dt[0], 303] == pytest.approx(5.0 / 3.0)
