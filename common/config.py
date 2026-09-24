"""Centralised configuration loading.

Every experiment reads its settings from ``configs/config.yaml`` through this
module so that values are never scattered across the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class Config:
    """Thin wrapper around the YAML configuration dictionary."""

    def __init__(self, config_path: Path | str = CONFIG_PATH) -> None:
        self._path = Path(config_path)
        if not self._path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self._path}. "
                "Ensure configs/config.yaml exists before running any experiment."
            )
        with self._path.open("r", encoding="utf-8") as fh:
            self._data: dict[str, Any] = yaml.safe_load(fh)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def section(self, name: str) -> dict[str, Any]:
        """Return one top-level section (e.g. 'federated', 'privacy')."""
        if name not in self._data:
            raise KeyError(f"Configuration section '{name}' not found in {self._path}.")
        return self._data[name]

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    @property
    def path(self) -> Path:
        return self._path

    def resolve(self, rel_path: str) -> Path:
        """Resolve a configuration path relative to the project root."""
        return PROJECT_ROOT / rel_path

    def dump_to(self, path: str | Path) -> Path:
        """Write the current (possibly overridden) configuration to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self._data, fh, default_flow_style=False, sort_keys=False)
        return p


def load_config(config_path: str | Path | None = None) -> Config:
    """Load the project configuration."""
    return Config() if config_path is None else Config(config_path)