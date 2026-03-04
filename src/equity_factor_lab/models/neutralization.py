import numpy as np
import pandas as pd


NEUTRALIZATION_MODES: tuple[str, ...] = ("none", "sector", "market")


def validate_neutralization_mode(mode: str) -> str:
    """Validates and returns a supported neutralization mode."""
    if mode not in NEUTRALIZATION_MODES:
        raise ValueError(
            f"Unknown neutralization_mode '{mode}'. "
            f"Available modes: {list(NEUTRALIZATION_MODES)}"
        )
    return mode


def uses_sector_neutralization(mode: str) -> bool:
    """Returns True when the mode includes sector neutralization."""
    mode = validate_neutralization_mode(mode)
    return mode == "sector"


def uses_market_neutralization(mode: str) -> bool:
    """Returns True when the mode includes market(beta) neutralization."""
    mode = validate_neutralization_mode(mode)
    return mode == "market"


def _sector_neutralize_scores(scores: pd.DataFrame, sector_map: pd.Series) -> pd.DataFrame:
    """Demeans scores within sector for each date."""
    sector_labels = sector_map.reindex(scores.columns)
    group_means = scores.T.groupby(sector_labels).transform("mean").T
    return scores - group_means


def _compute_rolling_market_betas(
    asset_returns: pd.DataFrame,
    market_returns: pd.Series,
    lookback: int = 252,
    min_periods: int = 63,
) -> pd.DataFrame:
    """Estimates rolling asset betas against the market return series."""
    aligned_asset_returns, aligned_market_returns = asset_returns.align(
        market_returns,
        join="inner",
        axis=0,
    )
    cov = aligned_asset_returns.rolling(
        window=lookback,
        min_periods=min_periods,
    ).cov(aligned_market_returns)
    var = aligned_market_returns.rolling(
        window=lookback,
        min_periods=min_periods,
    ).var()
    return cov.div(var, axis=0)


def _market_neutralize_scores(scores: pd.DataFrame, betas: pd.DataFrame) -> pd.DataFrame:
    """Regresses scores on beta cross-sectionally and keeps residuals."""
    aligned_scores = scores
    aligned_betas = betas.reindex(index=scores.index, columns=scores.columns)
    neutralized = pd.DataFrame(
        np.nan,
        index=aligned_scores.index,
        columns=aligned_scores.columns,
        dtype=float,
    )

    for dt in aligned_scores.index:
        y = aligned_scores.loc[dt]
        x = aligned_betas.loc[dt]
        valid = y.notna() & x.notna()
        if valid.sum() < 2:
            continue

        y_valid = y.loc[valid].astype(float)
        x_valid = x.loc[valid].astype(float)

        x_centered = x_valid - x_valid.mean()
        y_centered = y_valid - y_valid.mean()

        denom = float((x_centered * x_centered).sum())
        if denom <= 0.0 or np.isnan(denom):
            neutralized.loc[dt, y_valid.index] = y_centered
            continue

        slope = float((x_centered * y_centered).sum() / denom)
        residual = y_centered - slope * x_centered
        neutralized.loc[dt, residual.index] = residual

    return neutralized


def neutralize_scores(
    scores: pd.DataFrame,
    mode: str,
    *,
    sector_map: pd.Series | None = None,
    market_betas: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Applies sector or market neutralization and returns raw residual scores."""
    mode = validate_neutralization_mode(mode)
    if mode == "none":
        return scores

    neutralized = scores.copy()

    if uses_sector_neutralization(mode):
        if sector_map is None:
            raise ValueError("sector_map is required for sector neutralization.")
        neutralized = _sector_neutralize_scores(neutralized, sector_map)

    if uses_market_neutralization(mode):
        if market_betas is None:
            raise ValueError("market_betas are required for market neutralization.")
        neutralized = _market_neutralize_scores(neutralized, market_betas)

    return neutralized
