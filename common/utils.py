"""Shared helpers: seeding, paths, JSON persistence, model save/load."""

from __future__ import annotations

import json
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from common.config import Config

SEEDED = False


def set_seed(seed: int) -> None:
    """Make numpy / random / sklearn reproducible."""
    global SEEDED
    random.seed(seed)
    np.random.seed(seed)
    try:
        import sklearn.utils

        sklearn.utils.check_random_state(seed)
    except Exception:  # pragma: no cover - sklearn always present
        pass
    SEEDED = True


def ensure_dirs(config: Config) -> dict[str, Path]:
    """Create all output directories declared in the configuration."""
    paths = {
        name: config.resolve(p)
        for name, p in config.section("paths").items()
    }
    paths["processed"] = config.resolve(config.section("data")["processed_path"])
    paths["clients"] = config.resolve(config.section("data")["clients_dir"])
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def model_paths(config: Config) -> dict[str, Path]:
    """Canonical locations for saved models/checkpoints."""
    models_dir = config.resolve(config.section("paths")["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    return {
        "ann": models_dir / "baseline_ann.pkl",
        "rf": models_dir / "baseline_rf.pkl",
        "svm": models_dir / "baseline_svm.pkl",
        "global_model": models_dir / "global_model.pkl",
        "global_initial": models_dir / "initial_model.pkl",
        "checkpoints_dir": models_dir / "checkpoints",
        "final": models_dir / "final_global_model.pkl",
    }


def write_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return p


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_model(model: Any, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        pickle.dump(model, fh)
    return p


def load_model(path: str | Path) -> Any:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def to_serializable(obj: Any) -> Any:
    """Best-effort conversion of numpy/flower values to JSON-safe values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj