"""LIME-based local explanations.

LIME fits a sparse linear surrogate model around each explained test sample.
The returned weights are the (feature, weight) pairs of the local surrogate,
visualised as horizontal bars.
"""

from __future__ import annotations

import numpy as np

from common.config import Config


def explain_samples(
    config: Config,
    predict_proba_fn,
    X_training: np.ndarray,
    X_to_explain: np.ndarray,
    feature_names: list[str],
    max_display_features: int = 6,
    n_samples: int = 3,
) -> list[dict]:
    """Explain ``n_samples`` rows of ``X_to_explain`` with LIME.

    Returns a list of per-sample dicts:
        index : row index within X_to_explain
        weights : list of (feature_name, weight) sorted by |weight| desc
        predicted_class : int (0/1) predicted for the sample
    """
    from lime.lime_tabular import LimeTabularExplainer

    seed = int(config.section("project")["random_seed"])
    max_display = int(max_display_features)

    explainer = LimeTabularExplainer(
        np.asarray(X_training, dtype=np.float32),
        feature_names=list(feature_names),
        mode="classification",
        random_state=seed,
        verbose=False,
    )

    def predict_fn(X):
        return np.asarray(predict_proba_fn(np.asarray(X, dtype=np.float32)), dtype=float)

    out: list[dict] = []
    for i in range(min(int(n_samples), len(X_to_explain))):
        row = np.asarray(X_to_explain[i], dtype=np.float32)
        exp = explainer.explain_instance(row, predict_fn, top_labels=1, num_features=max_display)
        pred = int(np.argmax(predict_fn(row.reshape(1, -1))))
        local_exp = exp.local_exp.get(pred, [])
        weights = [(feature_names[int(ix)], float(w)) for ix, w in local_exp]
        weights.sort(key=lambda t: -abs(t[1]))
        out.append({"index": int(i), "predicted_class": pred, "weights": weights})
    return out


def save_lime_artifacts(config: Config, output_dir, results: list[dict]) -> list:
    from pathlib import Path

    import numpy as np

    from visualization.plots import save_feature_importance, save_lime_bar

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list = []

    # Aggregate mean |weight| across the explained samples for a global view.
    feature_scores: dict[str, float] = {}
    for sample in results:
        for name, w in sample["weights"]:
            feature_scores[name] = feature_scores.get(name, 0.0) + abs(w)
    total = sum(feature_scores.values()) or 1.0
    mean_abs = {k: v / len(results) for k, v in feature_scores.items()}
    if mean_abs:
        paths.append(
            save_feature_importance(
                config, mean_abs, out / "lime_feature_importance.png",
                title="LIME mean |weight| per feature",
            )
        )

    for i, sample in enumerate(results):
        paths.append(
            save_lime_bar(config, sample["weights"], out / f"lime_sample_{i + 1}.png")
        )
    return paths