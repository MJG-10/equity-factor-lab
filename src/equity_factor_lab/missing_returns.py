import pandas as pd


def build_terminal_missing_mask(future_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Flags missing return cells with no later observed return for that asset.

    This is used as a delist-like terminal gap detector.
    """
    has_observation = future_returns.notna()
    # Reverse cumulative sum includes current row, so shift to get "strictly after".
    obs_after = has_observation.iloc[::-1].cumsum().iloc[::-1].shift(-1).fillna(0.0)
    return future_returns.isna() & (obs_after == 0.0)
