"""Custom Flower strategy (extends FedAvg).

* Tracks the federated workflow through structured logs.
* Measures communication volume per round (actual parameter bytes).
* Saves per-round global-model checkpoints.
* Evaluates the global model on the held-out federated test set each round.
* Aggregates per-client evaluation metrics (metrics only - never raw data).
* Uses weighted FedAvg, or uniform aggregation when secure aggregation is on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from common.config import Config
from evaluation.metrics import compute_metrics
from server.aggregation import (
    aggregate_uniform,
    aggregate_weighted,
    parameters_size_bytes,
)

logger = logging.getLogger("xfl.SERVER")


@dataclass
class RoundLog:
    round: int
    fit_clients: list[int] = field(default_factory=list)
    client_fit_metrics: list[dict] = field(default_factory=list)
    client_eval_metrics: list[dict] = field(default_factory=list)
    fit_weights: list[int] = field(default_factory=list)
    eval_weights: list[int] = field(default_factory=list)
    comm_bytes: int = 0
    global_metrics: dict = field(default_factory=dict)
    global_loss: float | None = None


class XFLStrategy(FedAvg):
    """FedAvg extended for logging, communication accounting and checkpoints."""

    def __init__(self, config: Config) -> None:
        fcfg = config.section("federated")
        pcfg = config.section("privacy")
        self.config = config
        num_clients = int(config.section("partition")["num_clients"])
        fraction = float(fcfg.get("client_fraction", 1.0))
        min_clients = max(1, int(round(num_clients * fraction)))
        secure_agg = bool(pcfg.get("secure_aggregation", False))

        from models.ann import build_ann
        from preprocessing.pipeline import build_from_saved

        self.artifacts = build_from_saved(config)
        self.server_model = build_ann(self.artifacts.X_train.shape[1], config)

        # Weighted FedAvg by default; uniform when masked (secure) aggregation
        # is enabled so pairwise masks cancel exactly.
        self.use_weighted = bool(fcfg.get("weighted_aggregation", True)) and not secure_agg

        super().__init__(
            fraction_fit=fraction,
            fraction_evaluate=fraction,
            min_fit_clients=min_clients,
            min_evaluate_clients=min_clients,
            min_available_clients=num_clients,
            initial_parameters=ndarrays_to_parameters(self.server_model.get_parameters()),
            evaluate_fn=self._global_evaluate,
            fit_metrics_aggregation_fn=self._agg_fit_metrics,
            evaluate_metrics_aggregation_fn=self._agg_eval_metrics,
            on_fit_config_fn=self._fit_config,
            on_evaluate_config_fn=self._eval_config,
        )

        self.round_logs: dict[int, RoundLog] = {}
        self.round_parameters: dict[int, object] = {}
        self.timings: dict[int, float] = {}

    # ------------------------------------------------------------------ utils
    def _fit_config(self, server_round: int) -> dict:
        fcfg = self.config.section("federated")
        pcfg = self.config.section("privacy")
        return {
            "local_epochs": int(fcfg.get("local_epochs", 10)),
            "batch_size": int(fcfg.get("batch_size", 64)),
            "learning_rate": float(fcfg.get("learning_rate", 0.01)),
            "dp_enabled": bool(pcfg.get("dp_enabled", False)),
            "noise_multiplier": float(pcfg.get("noise_multiplier", 1.0)),
            "clipping_norm": float(pcfg.get("clipping_norm", 1.0)),
            "secure_aggregation": bool(pcfg.get("secure_aggregation", False)),
            "server_round": int(server_round),
        }

    def _eval_config(self, server_round: int) -> dict:
        return {"server_round": int(server_round)}

    def _agg_fit_metrics(self, metrics: list[tuple[int, dict]]) -> dict:
        if not metrics:
            return {}
        total = sum(n for n, _ in metrics)
        numeric_keys = [k for k in metrics[0][1] if isinstance(metrics[0][1][k], (int, float))]
        out: dict = {}
        for key in numeric_keys:
            out[key] = sum(m.get(key, 0.0) * n for n, m in metrics) / max(total, 1)
        return out

    def _agg_eval_metrics(self, metrics: list[tuple[int, dict]]) -> dict:
        if not metrics:
            return {}
        total = sum(n for n, _ in metrics)
        out: dict = {}
        for key in ("accuracy", "precision", "recall", "f1"):
            out[key] = sum(m.get(key, 0.0) * n for n, m in metrics) / max(total, 1)
        return out

    # ------------------------------------------------------------- aggregate
    def aggregate_fit(self, server_round, results, failures):
        from flwr.common import parameters_to_ndarrays as _to_nd

        log = self.round_logs.setdefault(server_round, RoundLog(round=server_round))
        log.comm_bytes = 0
        for _, fitres in results:
            log.fit_clients.append(int(fitres.metrics.get("cid", 0)))
            log.client_fit_metrics.append(dict(fitres.metrics))
            log.fit_weights.append(int(fitres.num_examples))
            log.comm_bytes += parameters_size_bytes(fitres.parameters)

        if not results:
            logger.warning("[ROUND %d] No fit results received.", server_round)
            return super().aggregate_fit(server_round, results, failures)

        if self.use_weighted:
            updates = [
                (_to_nd(fitres.parameters), int(fitres.num_examples))
                for _, fitres in results
            ]
            aggregated = aggregate_weighted(updates)
        else:
            updates = [_to_nd(fitres.parameters) for _, fitres in results]
            aggregated = aggregate_uniform(updates)

        params = ndarrays_to_parameters(aggregated)
        self.round_parameters[server_round] = params

        for cid, (_, fitres) in zip(log.fit_clients, results):
            logger.info(
                "[ROUND %d][CLIENT %s] local fit loss=%.4f (samples=%d)",
                server_round, cid, fitres.metrics.get("loss", float("nan")),
                fitres.num_examples,
            )
        logger.info(
            "[ROUND %d] Aggregating %d client updates -> global model "
            "(comm=%.1f KB)", server_round, len(results), log.comm_bytes / 1024,
        )
        return params, self._agg_fit_metrics(
            [(int(fitres.num_examples), dict(fitres.metrics)) for _, fitres in results]
        )

    def aggregate_evaluate(self, server_round, results, failures):
        log = self.round_logs.setdefault(server_round, RoundLog(round=server_round))
        for _, evalres in results:
            log.client_eval_metrics.append(dict(evalres.metrics))
            log.eval_weights.append(int(evalres.num_examples))

        for metrics, n in zip(log.client_eval_metrics, log.eval_weights):
            logger.info(
                "[ROUND %d] client eval acc=%.4f prec=%.4f rec=%.4f f1=%.4f (%d samples)",
                server_round, metrics.get("accuracy", 0), metrics.get("precision", 0),
                metrics.get("recall", 0), metrics.get("f1", 0), n,
            )
        return super().aggregate_evaluate(server_round, results, failures)

    # ------------------------------------------------------------- evaluate
    def _global_evaluate(self, server_round, parameters, config):
        """Evaluate the aggregated global model on the held-out test set."""
        if hasattr(parameters, "tensors"):
            ndarrays = parameters_to_ndarrays(parameters)
        else:
            ndarrays = parameters
        self.server_model.set_parameters(ndarrays)
        X, y = self.artifacts.X_test, self.artifacts.y_test
        loss = float(self.server_model.loss(X, y))
        preds = self.server_model.predict(X)
        metrics = compute_metrics(y, preds)
        log = self.round_logs.setdefault(server_round, RoundLog(round=server_round))
        log.global_loss = loss
        log.global_metrics = metrics
        logger.info(
            "[ROUND %d] Global model  acc=%.4f prec=%.4f rec=%.4f f1=%.4f loss=%.4f",
            server_round, metrics["accuracy"], metrics["precision"],
            metrics["recall"], metrics["f1"], loss,
        )
        return loss, metrics