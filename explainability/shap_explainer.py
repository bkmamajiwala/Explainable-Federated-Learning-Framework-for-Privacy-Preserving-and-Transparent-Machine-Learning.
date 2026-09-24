"""SHAP-based explanations (KernelExplainer).

KernelExplainer is model-agnostic, so it works directly on the NumPy MLP
(``predict_proba``) without any framework-specific integration. A background
sample drawn from the (central) training data provides the expected model
output (``base_value``); each explained test sample then gets additive feature
attributions that sum to the model prediction.
"""

from __future__ import annotations

import numpy as np

from common.config import Config


def compute_shap_values(
    config: Config,
    predict_proba_fn,
    X_background: np.ndarray,
    X_to_explain: np.ndarray,
    background_samples: int = 60,
) -> dict:
    """Compute SHAP values for ``X_to_explain`` using a background sample.

    Returns
    -------
    dict with keys:
        shap_values : (n_samples, n_features) float array
        base_value  : expected model output over the background set
        explained_index : int (index of the explained sample)
    """
    import shap

    seed = int(config.section("project")["random_seed"])
    n_bg = min(int(background_samples), len(X_background))
    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(X_background), size=n_bg, replace=False)
    background = np.asarray(X_background[bg_idx], dtype=np.float32)

    def predict_fn(X):
        return np.asarray(
            predict_proba_fn(np.asarray(X, dtype=np.float32))[:, 1], dtype=float
        )

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = np.asarray(explainer.shap_values(X_to_explain), dtype=float)
    expected = explainer.expected_value
    if isinstance(expected, np.ndarray):
        expected = float(expected.ravel()[0])
    return {
        "shap_values": shap_values,
        "base_value": float(expected),
    }


def save_shap_artifacts(config: Config, output_dir, results: dict, feature_names: list[str]) -> list:
    """Persist the SHAP summary / importance and per-sample waterfall plots."""
    from pathlib import Path

    from visualization.plots import (
        save_feature_importance,
        save_shap_summary,
        save_shap_waterfall,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sv = results["shap_values"]
    base = results["base_value"]

    paths = [
        save_shap_summary(config, sv, feature_names, out / "shap_summary.png"),
    ]
    mean_abs = np.abs(sv).mean(axis=0)
    importances = {f: float(mean_abs[i]) for i, f in enumerate(feature_names)}
    paths.append(
        save_feature_importance(
            config, importances, out / "shap_feature_importance.png",
            title="SHAP mean |impact| per feature",
        )
    )

    n_waterfalls = min(results.get("n_waterfalls", 3), len(sv))
    for i in range(n_waterfalls):
        paths.append(
            save_shap_waterfall(
                config, sv[i], feature_names, base,
                out / f"shap_waterfall_{i + 1}.png",
                title=f"SHAP waterfall - test sample {i + 1}",
            )
        )
    return paths