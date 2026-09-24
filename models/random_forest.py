"""Random Forest baseline model (scikit-learn)."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from common.config import Config, load_config


def build_rf(config: Config | None = None, n_estimators: int = 200) -> RandomForestClassifier:
    if config is None:
        config = load_config()
    seed = int(config.section("project")["random_seed"])
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
    )