"""Runs the full federated training loop using a real Flower server and clients.

The Flower **server** runs in the main process; each **client** runs as an
independent OS subprocess (``client/client_worker.py``) and connects to the
server over gRPC on localhost. This is genuine Flower server/client
communication: raw data never leaves a client process, and only model
parameters and aggregate metrics are exchanged.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from flwr.server import ServerConfig, start_server

from common.config import Config
from common.logging_setup import setup_logging
from common.utils import model_paths, save_model, to_serializable, write_json
from evaluation.experiments import ExperimentManager
from models.ann import MLP
from server.aggregation import count_parameters
from server.strategy import XFLStrategy


def run_federated_training(config: Config) -> dict:
    logger = setup_logging(tag="SERVER", log_dir=config.resolve("results/logs"))
    fcfg = config.section("federated")
    num_clients = int(config.section("partition")["num_clients"])
    num_rounds = int(fcfg["num_rounds"])
    server_address = str(fcfg["server_address"])

    logger.info("Starting federated training: %d clients, %d rounds", num_clients, num_rounds)

    strategy = XFLStrategy(config)
    param_count = count_parameters(strategy.server_model.get_parameters())
    logger.info(
        "Global model: %d parameters (~%.1f KB per copy).",
        param_count, param_count * 4 / 1024,
    )

    if config.section("privacy").get("secure_aggregation", False):
        logger.warning(
            "Secure aggregation ENABLED - updates are masked pairwise by clients; "
            "the server only sees their sum. All clients must participate and "
            "uniform FedAvg is used."
        )

    # --- launch client subprocesses -----------------------------------------
    # Clients must see the *effective* configuration (including any in-memory
    # overrides), so we dump it to disk and pass the path to each worker.
    project_root = Path(__file__).resolve().parent.parent
    log_dir = config.resolve(config.section("paths")["logs_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    effective_config = config.dump_to(config.resolve("results/effective_config.yaml"))
    processes: list[subprocess.Popen] = []
    for cid in range(num_clients):
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "client.client_worker",
                "--cid",
                str(cid),
                "--config",
                str(effective_config),
            ],
            cwd=str(project_root),
            stdout=open(log_dir / f"client_{cid}.log", "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
        processes.append(proc)
        logger.info("[ROUND 1] Spawned client subprocess %d (pid %d).", cid, proc.pid)

    time.sleep(2.0)

    # --- run the Flower server (blocks until training completes) ------------
    server_cfg = ServerConfig(
        num_rounds=num_rounds,
        round_timeout=float(fcfg.get("round_timeout", 1800)),
    )
    t_start = time.time()
    try:
        history = start_server(
            server_address=server_address,
            config=server_cfg,
            strategy=strategy,
            grpc_max_message_length=1024 * 1024 * 64,
        )
        strategy.history = history
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
    elapsed = time.time() - t_start

    return _finalize(config, strategy, elapsed, logger)


def _finalize(config: Config, strategy: XFLStrategy, elapsed: float, logger) -> dict:
    fcfg = config.section("federated")
    num_rounds = int(fcfg["num_rounds"])
    paths = model_paths(config)

    save_model(strategy.server_model, paths["global_initial"])

    round_summaries: list[dict] = []
    accuracy_per_round: list[float] = []
    loss_per_round: list[float] = []
    comm_per_round: list[int] = []

    checkpoints = paths["checkpoints_dir"]
    checkpoints.mkdir(parents=True, exist_ok=True)
    final_ndarrays = strategy.server_model.get_parameters()

    for r in range(1, num_rounds + 1):
        log = strategy.round_logs.get(r)
        if log is None:
            continue
        round_summaries.append({
            "round": r,
            "global_loss": log.global_loss,
            "global_metrics": log.global_metrics,
            "comm_bytes": log.comm_bytes,
            "num_fit_clients": len(log.fit_clients),
        })
        loss_per_round.append(
            float(log.global_loss) if log.global_loss is not None else float("nan")
        )
        accuracy_per_round.append(log.global_metrics.get("accuracy", float("nan")))
        comm_per_round.append(int(log.comm_bytes))

        params = strategy.round_parameters.get(r)
        if params is not None:
            from flwr.common import parameters_to_ndarrays

            ndarrays = parameters_to_ndarrays(params)
            m = MLP(strategy.server_model.input_size, strategy.server_model.hidden_layers)
            m.set_parameters(ndarrays)
            save_model(m, checkpoints / f"round_{r}.pkl")
            if r == num_rounds:
                final_ndarrays = ndarrays

    final_model = MLP(strategy.server_model.input_size, strategy.server_model.hidden_layers)
    final_model.set_parameters(final_ndarrays)
    save_model(final_model, paths["final"])

    per_client: list[dict] = []
    last_log = strategy.round_logs.get(num_rounds)
    if last_log is not None:
        for metrics, n in zip(last_log.client_eval_metrics, last_log.eval_weights):
            per_client.append({"cid": metrics.get("cid"), **metrics, "n_test": n})

    final_metrics = last_log.global_metrics if last_log is not None else {}
    total_comm = sum(comm_per_round)

    run_info = {
        "num_clients": config.section("partition")["num_clients"],
        "num_rounds": num_rounds,
        "local_epochs": fcfg["local_epochs"],
        "batch_size": fcfg["batch_size"],
        "learning_rate": fcfg["learning_rate"],
        "client_fraction": fcfg["client_fraction"],
        "weighted_aggregation": strategy.use_weighted,
        "partition_strategy": config.section("partition")["strategy"],
        "privacy": {
            "dp_enabled": config.section("privacy").get("dp_enabled", False),
            "noise_multiplier": config.section("privacy").get("noise_multiplier", 1.0),
            "clipping_norm": config.section("privacy").get("clipping_norm", 1.0),
            "secure_aggregation": config.section("privacy").get("secure_aggregation", False),
        },
        "training_time_s": round(elapsed, 2),
        "communication": {
            "model_parameters": count_parameters(strategy.server_model.get_parameters()),
            "total_bytes": total_comm,
            "total_mb": round(total_comm / (1024 ** 2), 4),
            "per_round_bytes": comm_per_round,
        },
        "per_round": round_summaries,
        "per_client": per_client,
    }

    results_dir = config.resolve(config.section("paths")["results_dir"])
    write_json(results_dir / "federated_results.json", to_serializable(run_info))

    mgr = ExperimentManager(config)
    rec = mgr.start("federated-training", "federated", extra={
        "num_clients": run_info["num_clients"],
        "num_rounds": num_rounds,
        "local_epochs": run_info["local_epochs"],
        "privacy": run_info["privacy"],
        "training_time_s": run_info["training_time_s"],
        "communication_mb": run_info["communication"]["total_mb"],
        "partition": config.section("partition")["strategy"],
    })
    mgr.save(rec, final_metrics)

    from visualization.plots import (
        save_communication_overhead,
        save_training_curves,
    )

    figs = config.resolve(config.section("paths")["figures_dir"])
    save_training_curves(
        config, loss_per_round, accuracy_per_round, figs / "federated_curves.png"
    )
    save_communication_overhead(
        config, {"per_round_bytes": comm_per_round}, figs / "communication_overhead.png"
    )

    logger.info(
        "Federated training complete: acc=%.4f f1=%.4f in %.1fs (total comm %.2f MB)",
        final_metrics.get("accuracy", float("nan")),
        final_metrics.get("f1", float("nan")),
        elapsed, total_comm / (1024 ** 2),
    )
    return run_info