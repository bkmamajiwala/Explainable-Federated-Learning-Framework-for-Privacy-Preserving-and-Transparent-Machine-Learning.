"""Support Vector Machine baseline model (scikit-learn)."""

from __future__ import annotations

from sklearn.svm import SVC

from common.config import Config, load_config


def build_svm(config: Config | None = None) -> SVC:
    if config is None:
        config = load_config()
    seed = int(config.section("project")["random_seed"])
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,  # needed for SHAP/LIME probability explanations
        random_state=seed,
    )