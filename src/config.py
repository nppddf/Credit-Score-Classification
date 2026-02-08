from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


CONFIG_FILENAME = "config.yaml"


def _find_config_path(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for parent in [current] + list(current.parents):
        candidate = parent / CONFIG_FILENAME
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Unable to find {CONFIG_FILENAME} starting from {current}"
    )


@lru_cache(maxsize=1)
def load_config() -> dict:
    config_path = _find_config_path(Path(__file__).resolve())
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}
