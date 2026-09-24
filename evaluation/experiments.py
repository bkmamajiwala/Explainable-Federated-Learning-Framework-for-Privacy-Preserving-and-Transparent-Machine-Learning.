"""Experiment orchestration and tracking.

Every experiment is recorded with a unique ID, timestamp, configuration
snapshot and computed metrics. Results are persisted as JSON + CSV under
``results/`` so that no reported number is hard-coded: everything here comes
from an actual execution.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import Config
from common.logging_setup import setup_logging
from common.utils import ensure_dirs, timestamp, to_serializable, write_json

# Reference values reported in the research paper (comparison only - never used
# as the output of this implementation).
PAPER_RESULTS = {
    "ann": {"accuracy": 93.2, "precision": 77.98, "recall": 81.59, "f1": 79.39},
    "rf": {"accuracy": 95.4, "precision": 81.87, "recall": 84.41, "f1": 83.05},
    "svm": {"accuracy": 82.0, "precision": 79.79, "recall": 83.12, "f1": 81.04},
    "xfl": {"accuracy": 98.8, "precision": 84.96, "recall": 87.96, "f1": 87.08},
}

PAPER_INCONSISTENCY_NOTE = (
    "NOTE: The manuscript reports 98.15% accuracy in the abstract/conclusion but "
    "98.8% in the detailed Results section and experimental tables. This "
    "implementation uses the detailed experimental value (98.8%) as the reference, "
    "and flags the discrepancy for the reader."
)


@dataclass
class ExperimentRecord:
    experiment_id: str
    name: str
    timestamp: str
    mode: str
    config_snapshot: dict
    metrics: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "metrics": self.metrics,
            "extra": self.extra,
        }


class ExperimentManager:
    """Creates, persists and lists experiments."""

    def __init__(self, config: Config) -> None:
        self.config = config
        paths = ensure_dirs(config)
        self.results_dir: Path = paths["results_dir"]
        self.experiments_dir = self.results_dir / "experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.results_dir / "experiments.csv"
        self._records: list[ExperimentRecord] = []
        self.logger = setup_logging(tag="EXPERIMENT", log_dir=config.resolve("results/logs"))

    def start(
        self,
        name: str,
        mode: str,
        extra: dict | None = None,
    ) -> ExperimentRecord:
        now = datetime.now()
        experiment_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{name.lower().replace(' ', '_')}"
        record = ExperimentRecord(
            experiment_id=experiment_id,
            name=name,
            timestamp=now.isoformat(timespec="seconds"),
            mode=mode,
            config_snapshot=self._flatten_config(),
            extra=extra or {},
        )
        self._records.append(record)
        return record

    def save(self, record: ExperimentRecord, extra_metrics: dict | None = None) -> Path:
        if extra_metrics:
            record.metrics.update(extra_metrics)
        path = self.experiments_dir / f"{record.experiment_id}.json"
        write_json(path, to_serializable(record.to_dict()))
        self._append_csv(record)
        return path

    def _append_csv(self, record: ExperimentRecord) -> None:
        row = {"experiment_id": record.experiment_id, "name": record.name,
               "timestamp": record.timestamp, "mode": record.mode}
        row.update(to_serializable(record.metrics))
        row.update(to_serializable(record.extra))

        # Read existing rows (tolerating a previously malformed file) so that the
        # column set is the *union* of all columns ever written. Rewriting keeps
        # experiments.csv a valid, consistently-headed table.
        rows: list[dict] = []
        if self.csv_path.exists():
            try:
                df = pd.read_csv(self.csv_path, dtype=str)
                rows = df.where(pd.notna(df), None).to_dict(orient="records")
            except Exception:
                rows = []
        rows.append(row)

        columns = ["experiment_id", "name", "timestamp", "mode"]
        for r in rows:
            for k in r:
                if k not in columns:
                    columns.append(k)

        with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in columns})

    def _flatten_config(self) -> dict:
        flat: dict = {}
        for section, values in self.config.data.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    flat[f"{section}.{k}"] = v
        return flat

    def list(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.csv_path)


def _train_ann(config: Config, X_train, y_train, X_test, y_test):
    from models.ann import build_ann

    model = build_ann(X_train.shape[1], config)
    model.train(
        X_train, y_train,
        epochs=int(config.section("evaluation").get("ann_baseline_epochs", 60)),
        batch_size=int(config.section("federated")["batch_size"]),
        learning_rate=float(config.section("federated")["learning_rate"]),
        seed=int(config.section("project")["random_seed"]),
    )
    return model


def run_preparation(config: Config) -> dict:
    """Prepare the dataset, preprocess and partition it into client datasets."""
    logger = setup_logging(tag="PREP", log_dir=config.resolve("results/logs"))
    from preprocessing.pipeline import PreprocessingPipeline
    from preprocessing.partitioning import partition_data

    logger.info("Loading and preprocessing the predictive-maintenance dataset...")
    pipeline = PreprocessingPipeline(config)
    artifacts = pipeline.prepare()

    logger.info(
        "Split: %d train / %d test, %d failure samples in train, %d in test.",
        len(artifacts.y_train), len(artifacts.y_test),
        int(artifacts.y_train.sum()), int(artifacts.y_test.sum()),
    )
    logger.info("Feature names: %s", artifacts.feature_names)

    partitions = partition_data(config, artifacts)
    for p in partitions:
        logger.info(
            "Client %d: %d train rows (%d failures), %d local-test rows (%d failures)",
            p.cid, p.n_train, p.train_positives, p.n_test, p.test_positives,
        )

    mgr = ExperimentManager(config)
    rec = mgr.start("data-preparation", "prepare", extra={
        "strategy": config.section("partition")["strategy"],
        "num_clients": config.section("partition")["num_clients"],
    })
    mgr.save(rec)
    return {"artifacts": artifacts, "partitions": partitions}


def run_baselines(config: Config) -> dict:
    """Train and evaluate the centralised baselines (ANN, RF, SVM)."""
    logger = setup_logging(tag="BASELINE", log_dir=config.resolve("results/logs"))
    from preprocessing.pipeline import build_from_saved
    from visualization.plots import save_model_comparison

    artifacts = build_from_saved(config)
    X_train, y_train = artifacts.X_train, artifacts.y_train
    X_test, y_test = artifacts.X_test, artifacts.y_test

    results: dict[str, dict] = {}
    mgr = ExperimentManager(config)

    logger.info("Training centralised ANN baseline...")
    t0 = time.time()
    ann = _train_ann(config, X_train, y_train, X_test, y_test)
    ann_time = time.time() - t0
    ann_pred = ann.predict(X_test)

    from evaluation.metrics import compute_metrics

    results["ann"] = compute_metrics(y_test, ann_pred)
    results["ann"]["train_time_s"] = round(ann_time, 2)

    from common.utils import save_model, model_paths

    paths = model_paths(config)
    save_model(ann, paths["ann"])
    logger.info("ANN baseline accuracy=%.4f", results["ann"]["accuracy"])

    from models.random_forest import build_rf
    from models.svm import build_svm

    logger.info("Training centralised Random Forest baseline...")
    t0 = time.time()
    rf = build_rf(config)
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0
    results["rf"] = compute_metrics(y_test, rf.predict(X_test))
    results["rf"]["train_time_s"] = round(rf_time, 2)
    save_model(rf, paths["rf"])
    logger.info("RF baseline accuracy=%.4f", results["rf"]["accuracy"])

    logger.info("Training centralised SVM baseline...")
    t0 = time.time()
    svm = build_svm(config)
    svm.fit(X_train, y_train)
    svm_time = time.time() - t0
    results["svm"] = compute_metrics(y_test, svm.predict(X_test))
    results["svm"]["train_time_s"] = round(svm_time, 2)
    save_model(svm, paths["svm"])
    logger.info("SVM baseline accuracy=%.4f", results["svm"]["accuracy"])

    for name, m in results.items():
        rec = mgr.start(f"baseline-{name}", "baseline")
        mgr.save(rec, {**m, "model": name})

    save_model_comparison(config, results, results_path=config.resolve("results/figures"))
    return results


def run_federated_experiment(config: Config) -> dict:
    """Run the full federated (Flower) training pipeline."""
    from server.runner import run_federated_training

    return run_federated_training(config)


def run_explanation_stage(config: Config) -> dict:
    """Generate SHAP/LIME explanations for the final federated global model."""
    from explainability.explain_runner import run_explanations

    return run_explanations(config)


def run_clustering_stage(config: Config) -> dict:
    """Run the federated clustering module."""
    from clustering.federated_clustering import run_federated_clustering

    return run_federated_clustering(config)


def run_ablation_studies(config: Config) -> dict:
    """Run the centralised-vs-federated ablation experiments (A-F)."""
    from evaluation.ablation import run_all_ablations

    return run_all_ablations(config)