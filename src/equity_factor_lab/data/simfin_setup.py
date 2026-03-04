from pathlib import Path
import simfin as sf
from ..config import SIMFIN_DATA_DIR


def _resolve_api_key(api_key: str | None) -> str:
    """Validates and returns a non-empty SimFin API key."""
    s = "" if api_key is None else str(api_key).strip()
    if not s:
        raise ValueError("simfin_api_key is required. Set a real API key from your SimFin account.")
    return s


def configure_simfin(
    api_key: str | None,
    data_dir: str | Path | None,
) -> None:
    """Applies SimFin API key and local data directory configuration."""
    sf.set_api_key(_resolve_api_key(api_key))
    resolved_data_dir = data_dir if data_dir else SIMFIN_DATA_DIR
    sf.set_data_dir(str(resolved_data_dir))
