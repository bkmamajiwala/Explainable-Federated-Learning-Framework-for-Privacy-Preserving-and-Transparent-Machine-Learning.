"""Differential Privacy layer.

The privacy-preserving mechanism applied by each client is **DP-SGD**:
per-example gradients are clipped to an L2 norm bound (``clipping_norm``) and
Gaussian noise with standard deviation ``clipping_norm * noise_multiplier /
batch_size`` is added to the averaged batch gradient *before* the optimizer
update (implemented in :mod:`models.ann`).

Scientific accuracy
-------------------
This is a genuine, configurable DP mechanism. However, this prototype does NOT
report a concrete ``(epsilon, delta)`` value: a rigorous privacy accounting
requires a Renyi-DP composition analysis over all rounds and clients (e.g. via
Opacus/Google's dp-accounting). We therefore state that the mechanism provides
a configurable privacy-preserving layer, and we do not claim a specific
formal privacy budget.
"""

from __future__ import annotations

import numpy as np

from common.config import Config
from models.ann import DPConfig


def build_dp_config(config: Config) -> DPConfig:
    pcfg = config.section("privacy")
    return DPConfig(
        enabled=bool(pcfg.get("dp_enabled", False)),
        noise_multiplier=float(pcfg.get("noise_multiplier", 1.0)),
        clipping_norm=float(pcfg.get("clipping_norm", 1.0)),
    )


def clip_update(update: np.ndarray, clipping_norm: float) -> np.ndarray:
    """Clip a flat update vector to a maximum L2 norm (update-level helper)."""
    norm = float(np.linalg.norm(update))
    if norm > clipping_norm and norm > 0:
        return update * (clipping_norm / norm)
    return update


def add_gaussian_noise(
    update: np.ndarray, clipping_norm: float, noise_multiplier: float
) -> np.ndarray:
    """Add zero-mean Gaussian noise with std ``clip * multiplier``."""
    std = clipping_norm * noise_multiplier
    return update + np.random.normal(0.0, std, size=update.shape)


def apply_dp_update(update: np.ndarray, dp: DPConfig) -> np.ndarray:
    """Clip + Gaussian-noise an update (used for update-level DP)."""
    if not dp.enabled:
        return update
    clipped = clip_update(update, dp.clipping_norm)
    return add_gaussian_noise(clipped, dp.clipping_norm, dp.noise_multiplier)


def privacy_status(config: Config) -> dict:
    """Human-readable privacy status used by the dashboard and reports."""
    pcfg = config.section("privacy")
    return {
        "differential_privacy": bool(pcfg.get("dp_enabled", False)),
        "dp_mechanism": "DP-SGD (per-example gradient clipping + Gaussian noise)",
        "noise_multiplier": float(pcfg.get("noise_multiplier", 1.0)),
        "clipping_norm": float(pcfg.get("clipping_norm", 1.0)),
        "formal_epsilon": "not claimed (see module docstring)",
        "secure_aggregation": bool(pcfg.get("secure_aggregation", False)),
        "note": (
            "Implements a configurable privacy-preserving layer. Federated "
            "learning keeps raw data decentralized; DP-SGD adds quantifiable "
            "noise to model updates."
        ),
    }