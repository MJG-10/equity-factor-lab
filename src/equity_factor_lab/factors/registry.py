from typing import Callable
import pandas as pd
from .fundamental_factors import (
    compute_growth_style_from_signals,
    compute_invest_style_from_signals,
    compute_quality_style_from_signals,
    compute_value_scores,
)
from .price_factors import (
    compute_low_volatility_scores,
    compute_momentum_scores,
    compute_short_term_reversal_scores,
)


PriceFactorBuilder = Callable[[pd.DataFrame], pd.DataFrame]
FundamentalFactorBuilder = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


DEFAULT_PRICE_FACTORS: tuple[str, ...] = ("momentum", "reversal", "low_vol")
DEFAULT_FUNDAMENTAL_FACTORS: tuple[str, ...] = ("value", "quality", "invest", "growth")


def get_price_factor_builders() -> dict[str, PriceFactorBuilder]:
    """Returns price-factor builder callables by factor name."""
    return {
        "momentum": compute_momentum_scores,
        "reversal": compute_short_term_reversal_scores,
        "low_vol": compute_low_volatility_scores,
    }


def get_fundamental_factor_builders() -> dict[str, FundamentalFactorBuilder]:
    """Returns fundamental-factor builder callables by factor name."""
    return {
        "value": compute_value_scores,
        "quality": compute_quality_style_from_signals,
        "invest": compute_invest_style_from_signals,
        "growth": compute_growth_style_from_signals,
    }


def validate_selected_factors(
    selected: tuple[str, ...],
    available: dict[str, Callable],
    label: str,
) -> None:
    """Validates selected factor names against available builders."""
    invalid = sorted(name for name in selected if name not in available)
    if invalid:
        raise ValueError(
            f"Unknown {label} factor(s): {invalid}. "
            f"Available {label} factors: {sorted(available.keys())}"
        )
