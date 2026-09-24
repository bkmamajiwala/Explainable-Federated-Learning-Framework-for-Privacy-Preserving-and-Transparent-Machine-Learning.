"""Federated model wrapper around the NumPy MLP.

Provides parameter exchange (``get_parameters`` / ``set_parameters``), local
training and local evaluation for a Flower ``NumPyClient``. All raw data stays
inside the client; only model parameters travel over the wire.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.config import Config
from models.ann import DPConfig, MLP, build_ann


@dataclass
class LocalTrainResult:
    parameters: list[np.ndarray]
    n_examples: int
    metrics: dict = field(default_factory=dict)


@dataclass
class LocalEvalResult:
    loss: float
    n_examples: int
    metrics: dict = field(default_factory=dict)


class FederatedModel:
    """Owns one MLP instance and exposes FL-friendly operations."""

    def __init__(self, config: Config, model: MLP | None = None) -> None:
        self.config = config
        self.model = model
        self.input_size: int | None = None

    # ------------------------------------------------------------ parameters
    def get_parameters(self) -> list[np.ndarray]:
        return self.model.get_parameters()

    def set_parameters(self, parameters) -> None:
        self.model.set_parameters(parameters)

    # ---------------------------------------------------------------- local
    def local_train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dp_enabled: bool | None = None,
        secure_aggregation: bool | None = None,
    ) -> LocalTrainResult:
        """Train the model locally on the client's private data.

        Returns the (optionally privacy-protected) new parameters.
        """
        fcfg = self.config.section("federated")
        pcfg = self.config.section("privacy")

        if dp_enabled is None:
            dp_enabled = bool(pcfg.get("dp_enabled", False))

        dp = None
        if dp_enabled:
            dp = DPConfig(
                enabled=True,
                noise_multiplier=float(pcfg.get("noise_multiplier", 1.0)),
                clipping_norm=float(pcfg.get("clipping_norm", 1.0)),
            )

        seed = int(self.config.section("project")["random_seed"])
        pos_weight = float(fcfg.get("pos_weight", 0.0)) or None
        history = self.model.train(
            X,
            y,
            epochs=int(fcfg.get("local_epochs", 10)),
            batch_size=int(fcfg.get("batch_size", 64)),
            learning_rate=float(fcfg.get("learning_rate", 0.01)),
            dp=dp,
            seed=seed,
            pos_weight=pos_weight,
        )

        return LocalTrainResult(
            parameters=self.model.get_parameters(),
            n_examples=int(len(X)),
            metrics={"loss": history["loss"]},
        )

    def local_evaluate(self, X: np.ndarray, y: np.ndarray) -> LocalEvalResult:
        loss = self.model.loss(X, y)
        preds = self.model.predict(X)

        from evaluation.metrics import compute_metrics

        metrics = compute_metrics(y, preds, prefix="")
        return LocalEvalResult(
            loss=loss, n_examples=int(len(X)), metrics=metrics
        )

    # ---------------------------------------------------------------- factory
    def ensure_model(self, input_size: int) -> MLP:
        if self.model is None:
            self.model = build_ann(input_size, self.config)
            self.input_size = input_size
        return self.model


def build_federated_model(config: Config, input_size: int) -> FederatedModel:
    fm = FederatedModel(config)
    fm.ensure_model(input_size)
    return fm