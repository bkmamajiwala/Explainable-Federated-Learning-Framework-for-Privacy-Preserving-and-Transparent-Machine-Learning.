"""Secure aggregation (masked vectors).

This module implements a *pairwise-masked* secure aggregation: every client
adds to its model update a private mask shared (pairwise) with the other
clients, so that when the server sums all received updates the masks cancel
out and the server only learns the *sum*, never any individual client's update.

Current prototype limitations (documented honestly, not hidden):
* All ``num_clients`` clients must participate every round, otherwise the
  masks do not cancel (no dropout handling yet).
* The pairwise masks are derived from a shared simulation secret. In a real
  deployment the masks would be established via client-to-client key agreement
  (e.g. Diffie-Hellman / a key distribution centre) - never via the server.
* With masked vectors the aggregation must use a uniform average (equal
  weights) so that masks cancel exactly.

This is therefore a functional, demonstrable secure-aggregation mechanism and
not a claim of production-grade cryptographic security.
"""

from __future__ import annotations

import hashlib

import numpy as np

from common.config import Config


def _pair_seed(secret: str, a: int, b: int) -> int:
    """Deterministic seed for a client pair.

    Must be identical across *processes* (each client runs in its own OS
    process), so we cannot use Python's builtin ``hash()``, which is salted
    per-process with a random ``PYTHONHASHSEED``.
    """
    digest = hashlib.sha256(
        f"{secret}:{min(a, b)}:{max(a, b)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="big")


class PairwiseMaskProvider:
    """Generates per-client masks that cancel out when summed by the server."""

    def __init__(
        self,
        num_clients: int,
        secret: str,
        param_size: int,
        seed: int = 0,
    ) -> None:
        if num_clients < 2:
            raise ValueError("Secure aggregation requires at least 2 clients.")
        self.num_clients = int(num_clients)
        self.secret = secret
        self.param_size = int(param_size)
        self.seed = seed
        self._cache: dict[int, np.ndarray] = {}

    def mask_for(self, cid: int) -> np.ndarray:
        """Return the combined mask vector for client ``cid``."""
        if cid in self._cache:
            return self._cache[cid]
        total = np.zeros(self.param_size, dtype=np.float32)
        for other in range(self.num_clients):
            if other == cid:
                continue
            rng = np.random.default_rng(
                _pair_seed(self.secret, cid, other) ^ self.seed
            )
            pair_mask = rng.normal(0.0, 1.0, size=self.param_size).astype(np.float32)
            sign = 1.0 if cid < other else -1.0
            total += sign * pair_mask
        self._cache[cid] = total
        return total


def apply_mask(flat_params: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Add a mask to a flattened parameter vector."""
    if flat_params.shape != mask.shape:
        raise ValueError(
            f"Parameter vector {flat_params.shape} does not match mask {mask.shape}."
        )
    return flat_params + mask


def build_mask_provider(config: Config, param_size: int) -> PairwiseMaskProvider | None:
    """Create the shared mask provider when secure aggregation is enabled."""
    pcfg = config.section("privacy")
    if not bool(pcfg.get("secure_aggregation", False)):
        return None
    num_clients = int(config.section("partition")["num_clients"])
    secret = str(pcfg.get("mask_secret", "simulation-only"))
    seed = int(config.section("project")["random_seed"])
    return PairwiseMaskProvider(
        num_clients=num_clients, secret=secret, param_size=param_size, seed=seed
    )