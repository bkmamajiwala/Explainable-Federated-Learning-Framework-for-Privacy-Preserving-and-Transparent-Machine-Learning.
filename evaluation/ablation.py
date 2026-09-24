"""Ablation studies (centralised vs. federated variants A-F).

The framework is evaluated in increasingly complex configurations so that the
contribution of each layer is measured:

    A. Centralised baseline      - ANN trained on the full central dataset.
    B. FL-only                   - federated training without any privacy layer.
    C. FL + DP                   - federated training with DP-SGD.
    D. FL + DP + XAI             - as C plus SHAP/LIME explanations.
    E. Full framework            - as D plus federated clustering.

Every number comes from an actual run (persisted as JSON/CSV + experiment
records); nothing is hard-coded.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from common.config import Config
from common.logging_setup import setup_logging
from common.utils import to_serializable, write_json


def _variant(config: Config, *, dp: bool, explain: bool, cluster: bool) -> Config:
    variant = copy.deepcopy(config)
    variant.data["privacy"]["dp_enabled"] = dp
    variant.data["privacy"]["secure_aggregation"] = False
    variant.data["explainability"]["shap_enabled"] = explain
    variant.data["explainability"]["lime_enabled"] = explain
    variant.data["clustering"]["enabled"] = cluster
    return variant


def run_all_ablations(config: Config) -> dict:
    logger = setup_logging(tag="ABLATION", log_dir=config.resolve("results/logs"))
    from evaluation.experiments import ExperimentManager, _train_ann
    from evaluation.metrics import compute_metrics
    from preprocessing.pipeline import build_from_saved

    abcfg = config.section("evaluation").get("ablation", {})
    artifacts = build_from_saved(config)
    X_train, y_train, X_test, y_test = (
        artifacts.X_train, artifacts.y_train, artifacts.X_test, artifacts.y_test
    )

    results: dict[str, dict] = {}

    def final_metrics(info: dict) -> dict:
        per_round = info.get("per_round", [])
        return dict(per_round[-1]["global_metrics"]) if per_round else {}

    # ------------------------------------------------------------------ A
    logger.info("Ablation A: centralised ANN baseline...")
    t0 = time.time()
    ann = _train_ann(config, X_train, y_train, X_test, y_test)
    results["A_centralized"] = compute_metrics(y_test, ann.predict(X_test))
    results["A_centralized"]["train_time_s"] = round(time.time() - t0, 2)

    # ------------------------------------------------------------------ B
    if abcfg.get("fl_only", True):
        logger.info("Ablation B: federated learning only (no privacy, no XAI)...")
        from server.runner import run_federated_training

        info = run_federated_training(_variant(config, dp=False, explain=False, cluster=False))
        results["B_fl_only"] = {**final_metrics(info),
                                "train_time_s": info["training_time_s"],
                                "comm_mb": info["communication"]["total_mb"]}

    # ------------------------------------------------------------------ C
    if abcfg.get("fl_dp", True):
        logger.info("Ablation C: federated learning + differential privacy...")
        from server.runner import run_federated_training

        info = run_federated_training(_variant(config, dp=True, explain=False, cluster=False))
        results["C_fl_dp"] = {**final_metrics(info),
                              "train_time_s": info["training_time_s"],
                              "comm_mb": info["communication"]["total_mb"]}

    # ------------------------------------------------------------------ D
    if abcfg.get("fl_dp_shap", True):
        logger.info("Ablation D: federated + DP + explainability (SHAP/LIME)...")
        from explainability.explain_runner import run_explanations
        from server.runner import run_federated_training

        variant = _variant(config, dp=True, explain=True, cluster=False)
        info = run_federated_training(variant)
        expl = run_explanations(variant)
        results["D_fl_dp_shap"] = {
            **final_metrics(info),
            "train_time_s": info["training_time_s"],
            "comm_mb": info["communication"]["total_mb"],
            "shap_samples": expl.get("shap", {}).get("samples_explained", 0),
            "lime_samples": expl.get("lime", {}).get("samples_explained", 0),
        }

    # ------------------------------------------------------------------ E
    if abcfg.get("full", True):
        logger.info("Ablation E: full framework (FL + DP + XAI + clustering)...")
        from clustering.federated_clustering import run_federated_clustering
        from explainability.explain_runner import run_explanations
        from server.runner import run_federated_training

        variant = _variant(config, dp=True, explain=True, cluster=True)
        info = run_federated_training(variant)
        expl = run_explanations(variant)
        clust = run_federated_clustering(variant)
        results["E_full"] = {
            **final_metrics(info),
            "train_time_s": info["training_time_s"],
            "comm_mb": info["communication"]["total_mb"],
            "shap_samples": expl.get("shap", {}).get("samples_explained", 0),
            "lime_samples": expl.get("lime", {}).get("samples_explained", 0),
            "clusters": clust["num_clusters"],
            "cluster_counts": clust["global_cluster_counts"],
        }

    _persist(config, results)
    for name, r in results.items():
        logger.info(
            "Ablation %s: acc=%.4f prec=%.4f rec=%.4f f1=%.4f",
            name, r.get("accuracy", np.nan), r.get("precision", np.nan),
            r.get("recall", np.nan), r.get("f1", np.nan),
        )
    return results


def _persist(config: Config, results: dict[str, dict]) -> None:
    from evaluation.experiments import ExperimentManager

    results_dir = config.resolve(config.section("paths")["results_dir"])
    write_json(results_dir / "ablation_results.json", to_serializable(results))

    mgr = ExperimentManager(config)
    for name, metrics in results.items():
        rec = mgr.start(f"ablation-{name}", "ablation")
        mgr.save(rec, to_serializable(metrics))

    _save_plot(config, results)


def _save_plot(config: Config, results: dict[str, dict]) -> None:
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(labels))
    width = 0.2
    palette = ["#2c6e9c", "#3d9970", "#e08a2e", "#c0392b"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for j, m in enumerate(metrics):
        vals = [results[l].get(m, 0.0) * 100 for l in labels]
        bars = ax.bar(x + (j - 1.5) * width, vals, width, label=m.capitalize(),
                      color=palette[j], edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}",
                    ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Score (%)")
    ax.set_title("Ablation study - centralised vs. federated configurations")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = Path(config.resolve(config.section("paths")["figures_dir"])) / "ablation_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)