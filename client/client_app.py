"""Flower client for one simulated organization (factory).

The client:
  1. loads ONLY its own private partition,
  2. receives the global model parameters from the server,
  3. trains locally (with DP-SGD when enabled),
  4. optionally masks its update for secure aggregation,
  5. returns only the model update (never raw data).
"""

from __future__ import annotations

import logging

import numpy as np
from flwr.client import NumPyClient

from client.data_loader import load_client_data
from common.config import Config
from models.federated_model import build_federated_model
from privacy.secure_aggregation import PairwiseMaskProvider, apply_mask

logger = logging.getLogger("xfl.CLIENT")


def _unflatten(flat: np.ndarray, reference: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    offset = 0
    for arr in reference:
        size = arr.size
        out.append(flat[offset : offset + size].reshape(arr.shape))
        offset += size
    return out


class PredictiveClient(NumPyClient):
    """NumPy-based Flower client holding one private data partition."""

    def __init__(
        self,
        cid: int,
        config: Config,
        mask_provider: PairwiseMaskProvider | None = None,
    ) -> None:
        self.cid = int(cid)
        self.config = config
        self.mask_provider = mask_provider
        self.data = load_client_data(config, self.cid)
        self.federated_model = build_federated_model(config, len(self.data.feature_names))
        logger.info(
            "[CLIENT %d] initialized with %d train / %d test samples.",
            self.cid, len(self.data.y_train), len(self.data.y_test),
        )

    # ------------------------------------------------------- Flower contract
    def get_parameters(self, config):
        return self.federated_model.get_parameters()

    def fit(self, parameters, config):
        self.federated_model.set_parameters(parameters)

        dp_enabled = bool(config.get("dp_enabled", False))
        result = self.federated_model.local_train(
            self.data.X_train, self.data.y_train, dp_enabled=dp_enabled
        )
        new_parameters = result.parameters

        if self.mask_provider is not None:
            flat = np.concatenate([p.ravel() for p in new_parameters]).astype(np.float32)
            flat = apply_mask(flat, self.mask_provider.mask_for(self.cid))
            new_parameters = _unflatten(flat, new_parameters)

        metrics = {
            "loss": result.metrics.get("loss", float("nan")),
            "cid": self.cid,
            "dp_enabled": int(dp_enabled),
        }
        logger.info(
            "[CLIENT %d] local training done (loss=%.4f, samples=%d)",
            self.cid, metrics["loss"], result.n_examples,
        )
        return new_parameters, result.n_examples, metrics

    def evaluate(self, parameters, config):
        self.federated_model.set_parameters(parameters)
        result = self.federated_model.local_evaluate(
            self.data.X_test, self.data.y_test
        )
        metrics = dict(result.metrics)
        metrics["cid"] = self.cid
        logger.info(
            "[CLIENT %d] local evaluation acc=%.4f (samples=%d)",
            self.cid, metrics.get("accuracy", 0.0), result.n_examples,
        )
        return float(result.loss), result.n_examples, metrics


def client_factory(cid: int, config: Config, mask_provider=None) -> PredictiveClient:
    """Construct a client for one simulated factory."""
    return PredictiveClient(cid=cid, config=config, mask_provider=mask_provider)