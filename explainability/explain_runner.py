"""End-to-end explanation stage for the final federated global model.

Loads the trained global model (``models_saved/final_global_model.pkl``), runs
SHAP and LIME explanations on held-out test samples, persists figures + JSON and
records an experiment entry. Uses only the actual federated artefact - never a
centralised model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.config import Config
from common.logging_setup import setup_logging
from common.utils import load_model, model_paths, to_serializable, write_json


def run_explanations(config: Config) -> dict:
    """Generate and persist SHAP/LIME explanations of the global model."""
    logger = setup_logging(tag="EXPLAIN", log_dir=config.resolve("results/logs"))
    from evaluation.experiments import ExperimentManager
    from explainability.lime_explainer import (
        explain_samples,
        save_lime_artifacts,
    )
    from explainability.shap_explainer import (
        compute_shap_values,
        save_shap_artifacts,
    )
    from preprocessing.pipeline import build_from_saved

    xcfg = config.section("explainability")
    ecfg = config.section("evaluation")
    seed = int(config.section("project")["random_seed"])

    artifacts = build_from_saved(config)
    X_test, y_test = artifacts.X_test, artifacts.y_test
    feature_names = artifacts.feature_names

    paths = model_paths(config)
    model_path = paths["final"]
    if not model_path.exists():
        model_path = paths["global_model"]
    if not model_path.exists():
        raise FileNotFoundError(
            "No trained global model found. Run the federated stage first "
            "(`python main.py federated`)."
        )
    model = load_model(model_path)
    logger.info("Explaining global model from %s", model_path.name)

    predict_proba = model.predict_proba

    results: dict = {}
    figure_dir = config.resolve(config.section("paths")["figures_dir"]) / "explainability"

    shap_enabled = bool(xcfg.get("shap_enabled", True))
    lime_enabled = bool(xcfg.get("lime_enabled", True))

    if shap_enabled:
        logger.info("Computing SHAP values (KernelExplainer) on the global model...")
        summary_n = min(len(X_test), int(xcfg.get("summary_samples", 50)))
        explain_n = min(len(X_test), int(xcfg.get("explain_samples", 5)))
        bg = artifacts.X_train
        shap = compute_shap_values(
            config,
            predict_proba,
            bg,
            X_test[:summary_n],
            background_samples=int(xcfg.get("background_samples", 60)),
        )
        shap["n_waterfalls"] = explain_n
        shap_paths = save_shap_artifacts(
            config, figure_dir, shap, feature_names
        )
        results["shap"] = {
            "samples_explained": int(shap["shap_values"].shape[0]),
            "base_value": shap["base_value"],
            "mean_abs_importance": {
                f: float(np.abs(shap["shap_values"]).mean(axis=0)[i])
                for i, f in enumerate(feature_names)
            },
            "figures": [str(p) for p in shap_paths],
        }
        logger.info("SHAP summary computed over %d test samples.", shap["shap_values"].shape[0])

    if lime_enabled:
        logger.info("Computing LIME explanations on the global model...")
        explain_n = min(len(X_test), int(xcfg.get("explain_samples", 5)))
        lime_samples = explain_samples(
            config,
            predict_proba,
            artifacts.X_train,
            X_test[:explain_n],
            feature_names,
            max_display_features=int(xcfg.get("max_display_features", 6)),
            n_samples=explain_n,
        )
        lime_paths = save_lime_artifacts(config, figure_dir, lime_samples)
        results["lime"] = {
            "samples_explained": len(lime_samples),
            "per_sample": to_serializable(lime_samples),
            "figures": [str(p) for p in lime_paths],
        }
        logger.info("LIME explanations computed for %d test samples.", len(lime_samples))

    # Per-sample predictions used by the dashboard.
    preds = predict_proba(X_test)
    y_pred = (preds[:, 1] >= 0.5).astype(int)
    results["predictions"] = {
        "y_true": to_serializable(y_test.astype(int)),
        "y_pred": to_serializable(y_pred),
        "prob_failure": to_serializable(preds[:, 1]),
    }

    write_json(
        config.resolve(config.section("paths")["results_dir"]) / "explainability_results.json",
        to_serializable(results),
    )

    from evaluation.metrics import compute_metrics

    metrics = compute_metrics(y_test, y_pred)
    mgr = ExperimentManager(config)
    rec = mgr.start("explainability", "explain", extra={
        "shap_enabled": shap_enabled,
        "lime_enabled": lime_enabled,
        "background_samples": xcfg.get("background_samples"),
        "explain_samples": xcfg.get("explain_samples"),
    })
    mgr.save(rec, {**metrics, "global_model": str(model_path)})
    logger.info("Explanation stage complete: %s", {k: v["samples_explained"] for k, v in results.items() if isinstance(v, dict) and "samples_explained" in v})
    return results