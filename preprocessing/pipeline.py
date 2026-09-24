"""Data loading, cleaning, feature engineering and scaling.

The predictive-maintenance dataset used in the research paper is the standard
``ai4i2020`` dataset (10,000 samples). The raw file may use either of two
column-name conventions; this module normalises both into a canonical schema:

    ``Type, Air temperature [K], Process temperature [K], Rotational speed
    [rpm], Torque [Nm], Tool wear [min]``  -> features
    ``target``                              -> 0/1 failure label
    ``failure_type``                        -> human-readable failure category

Preprocessing statistics (the scaler) are fitted on the central train split.
This is the *research-simulation* mode: in a true decentralized deployment each
client would fit its own scaler on its own local data (see README, section
"Data preprocessing & information leakage"). The simulation choice is flagged
and is not presented as a decentralized property.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common.config import Config

TARGET_ALIASES = ("Target", "Machine failure", "target", "machine failure")
FAILURE_FLAGS = ("TWF", "HDF", "PWF", "OSF", "RNF")
FAILURE_FLAG_NAMES = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]


@dataclass
class PipelineArtifacts:
    """Everything produced by the central preparation pipeline."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: ColumnTransformer
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    metadata: dict = field(default_factory=dict)


def load_raw(config: Config) -> pd.DataFrame:
    """Load the raw dataset from disk (or a synthetic fallback)."""
    raw_path = config.resolve(config.section("data")["raw_path"])
    if not raw_path.exists():
        if config.section("data").get("allow_synthetic_fallback", True):
            return _synthetic_fallback(config)
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. Place the CSV there or set "
            "`allow_synthetic_fallback: true` in configs/config.yaml."
        )
    df = pd.read_csv(raw_path)
    if df.empty:
        raise ValueError(f"Raw dataset at {raw_path} is empty.")
    return df


def _synthetic_fallback(config: Config) -> pd.DataFrame:
    """Faithful structural reproduction of the ai4i2020 dataset.

    Implementation decision: if the official CSV is unavailable, this generator
    reproduces the documented feature ranges and failure-generation rules of the
    predictive-maintenance dataset (Tool-Wear, Heat-Dissipation, Power,
    Overstrain and Random failure modes). It is clearly a *synthetic*
    reproduction, not the original data.
    """
    import warnings

    warnings.warn(
        "Official predictive-maintenance CSV not found - generating a synthetic "
        "structural reproduction of the dataset.",
        stacklevel=2,
    )
    n = int(config.section("data")["n_samples"])
    rng = np.random.default_rng(42)

    type_probs = [0.6, 0.3, 0.1]
    types = rng.choice(["L", "M", "H"], size=n, p=type_probs)
    serials = rng.permutation(np.arange(1, n + 1))
    product_ids = np.array([f"{t}{s}" for t, s in zip(types, serials)])
    udi = np.arange(1, n + 1)

    air = 300.0 + rng.normal(0, 2, n)
    air = np.clip(air, 295.3, 304.5)
    process = air + 10.0 + rng.normal(0, 1, n)
    process = np.clip(process, 305.7, 313.8)
    speed = 2860.0 + rng.normal(0, 1000, n)
    speed = np.clip(speed, 1168, 2886)
    torque = 40.0 + rng.normal(0, 10, n)
    torque = np.clip(torque, 3.8, 76.6)
    tool_wear = rng.integers(0, 253, n).astype(float)

    # Failure mode flags (documented rules of the original dataset).
    twf = tool_wear + rng.normal(0, 5, n) > 225
    hdf = (process - air < 8.6) & (np.abs(speed) * np.abs(torque) > 800000)  # heat dissipation
    pwf = (torque * np.sqrt(np.abs(speed)) < 3500) | (torque * np.sqrt(np.abs(speed)) > 9000)
    osf = (tool_wear * torque > 11500) & (speed < 1600)
    rnf = rng.random(n) < 0.001

    flags = np.column_stack([twf, hdf, pwf, osf, rnf])
    target = flags.any(axis=1).astype(int)
    failure_type = np.where(target == 0, "No Failure", "Failure")
    flag_labels = ["Tool Wear Failure", "Heat Dissipation Failure", "Power Failure",
                   "Overstrain Failure", "Random Failure"]
    idx = np.argmax(flags, axis=1)
    has_flag = flags.any(axis=1)
    for i, label in enumerate(flag_labels):
        failure_type[has_flag & (idx == i)] = label

    df = pd.DataFrame(
        {
            "UDI": udi,
            "Product ID": product_ids,
            "Type": types,
            "Air temperature [K]": air.round(1),
            "Process temperature [K]": process.round(1),
            "Rotational speed [rpm]": speed.round(1),
            "Torque [Nm]": torque.round(2),
            "Tool wear [min]": tool_wear,
            "Machine failure": target,
            "TWF": twf.astype(int),
            "HDF": hdf.astype(int),
            "PWF": pwf.astype(int),
            "OSF": osf.astype(int),
            "RNF": rnf.astype(int),
            "Failure Type": failure_type,
        }
    )
    return df


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw column conventions to the canonical schema."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    target_col = next((c for c in df.columns if c in TARGET_ALIASES), None)
    if target_col is None:
        raise ValueError(
            f"Target column not found. Expected one of {TARGET_ALIASES}, got {list(df.columns)}."
        )
    df = df.rename(columns={target_col: "target"})

    if "Failure Type" not in df.columns and "Failure type" not in df.columns:
        flags = [c for c in df.columns if c in FAILURE_FLAGS]
        if flags:
            arr = df[flags].to_numpy()
            has = arr.any(axis=1)
            labels = np.array([FAILURE_FLAG_NAMES[c] for c in flags])
            idx = np.argmax(arr, axis=1)
            failure = np.full(len(df), "No Failure", dtype=object)
            for i in range(len(df)):
                if has[i]:
                    failure[i] = labels[idx[i]]
            df["Failure Type"] = failure
        else:
            df["Failure Type"] = np.where(df["target"] == 1, "Failure", "No Failure")

    # Drop identifiers (UDI, Product ID) - they are record IDs, not predictive.
    for drop in ("UDI", "Product ID"):
        if drop in df.columns:
            df = df.drop(columns=[drop])

    df["target"] = df["target"].astype(int)
    missing = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].isna().sum()
    if missing.any():
        raise ValueError(f"Missing values in predictive features: {missing.to_dict()}")
    return df


class PreprocessingPipeline:
    """Encapsulates the full central preprocessing pipeline."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.feature_names: list[str] = []
        self.scaler: ColumnTransformer | None = None

    def build_preprocessor(self) -> ColumnTransformer:
        numeric = StandardScaler()
        categorical = OneHotEncoder(drop="first", sparse_output=False)
        ct = ColumnTransformer(
            transformers=[
                ("num", numeric, NUMERIC_FEATURES),
                ("cat", categorical, CATEGORICAL_FEATURES),
            ],
            remainder="drop",
        )
        return ct

    def prepare(self, df: pd.DataFrame | None = None) -> PipelineArtifacts:
        """Run load -> clean -> split -> scale and persist artifacts."""
        cfg = self.config.section("data")
        if df is None:
            df = load_raw(self.config)
        df = normalize_schema(df)

        train_size = int(cfg["train_size"])
        test_size = int(cfg["test_size"])
        if len(df) < train_size + test_size:
            raise ValueError(
                f"Dataset has {len(df)} rows; need {train_size + test_size} "
                "for the documented 8000/2000 train/test split."
            )

        from sklearn.model_selection import train_test_split

        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=self.config.section("project")["random_seed"],
            stratify=df["target"], train_size=train_size,
        )

        X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        y_train = train_df["target"].to_numpy()
        X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        y_test = test_df["target"].to_numpy()

        self.scaler = self.build_preprocessor()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.feature_names = self._feature_names(X_train.columns)

        artifacts = PipelineArtifacts(
            X_train=np.asarray(X_train_scaled, dtype=np.float32),
            X_test=np.asarray(X_test_scaled, dtype=np.float32),
            y_train=y_train.astype(int),
            y_test=y_test.astype(int),
            feature_names=self.feature_names,
            scaler=self.scaler,
            train_df=train_df,
            test_df=test_df,
            metadata={
                "n_total": int(len(df)),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "n_failures_train": int(y_train.sum()),
                "n_failures_test": int(y_test.sum()),
                "feature_names": self.feature_names,
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
            },
        )

        self._save_artifacts(artifacts)
        return artifacts

    def _feature_names(self, df_columns: list[str]) -> list[str]:
        names: list[str] = list(NUMERIC_FEATURES)
        categories = ["L", "M", "H"]
        for cat in categories[1:]:  # drop='first'
            names.append(f"Type_{cat}")
        return names

    def _save_artifacts(self, artifacts: PipelineArtifacts) -> None:
        import pickle

        cfg = self.config.section("data")
        out_dir = self.config.resolve(cfg["processed_path"])
        out_dir.mkdir(parents=True, exist_ok=True)

        with (out_dir / "scaler.pkl").open("wb") as fh:
            pickle.dump(self.scaler, fh)
        np.save(out_dir / "X_train.npy", artifacts.X_train)
        np.save(out_dir / "X_test.npy", artifacts.X_test)
        np.save(out_dir / "y_train.npy", artifacts.y_train)
        np.save(out_dir / "y_test.npy", artifacts.y_test)

        from common.utils import write_json

        write_json(out_dir / "metadata.json", artifacts.metadata)
        artifacts.train_df.to_csv(out_dir / "train.csv", index=False)
        artifacts.test_df.to_csv(out_dir / "test.csv", index=False)


def inverse_transform_features(
    config: Config, scaled: np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    """Map scaled feature rows back to original physical units (for the UI).

    The scaler is a ``ColumnTransformer`` (numeric + one-hot categorical). The
    OneHotEncoder is configured with ``drop="first"`` and therefore has no
    built-in ``inverse_transform``; we invert each block manually:
    numeric features via the fitted ``StandardScaler`` and the categorical
    columns to the closest valid one-hot pattern.
    """
    import pickle

    scaler_path = config.resolve(config.section("data")["processed_path"]) / "scaler.pkl"
    with scaler_path.open("rb") as fh:
        scaler = pickle.load(fh)

    num_scaler = scaler.named_transformers_["num"]
    cat_enc = scaler.named_transformers_["cat"]

    X = np.asarray(scaled, dtype=np.float64)
    n_num = len(NUMERIC_FEATURES)
    out = np.empty((len(X), len(feature_names)), dtype=np.float64)
    out[:, :n_num] = num_scaler.inverse_transform(X[:, :n_num])

    if cat_enc is not None and n_num < len(feature_names):
        categories = list(cat_enc.categories_[0])  # e.g. ["L", "M", "H"]
        kept = list(categories[1:])                 # kept after drop="first"
        valid_patterns = [[0.0] * len(kept)]        # first category (dropped)
        for cat in kept:                            # one-hot for each kept cat
            pattern = [0.0] * len(kept)
            pattern[kept.index(cat)] = 1.0
            valid_patterns.append(pattern)
        valid_patterns = np.asarray(valid_patterns, dtype=np.float64)

        for i in range(len(X)):
            vec = X[i, n_num:]
            dist = np.linalg.norm(valid_patterns - vec, axis=1)
            out[i, n_num:] = valid_patterns[int(np.argmin(dist))]

    return pd.DataFrame(out, columns=feature_names)


def build_from_saved(config: Config) -> PipelineArtifacts:
    """Reload a previously prepared pipeline from disk (used by dashboard/tests)."""
    import pickle

    out_dir = config.resolve(config.section("data")["processed_path"])
    metadata_path = out_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Preprocessed artifacts not found. Run the preparation step first "
            "(`python main.py prepare`)."
        )
    from common.utils import read_json

    meta = read_json(metadata_path)
    artifacts = PipelineArtifacts(
        X_train=np.load(out_dir / "X_train.npy"),
        X_test=np.load(out_dir / "X_test.npy"),
        y_train=np.load(out_dir / "y_train.npy"),
        y_test=np.load(out_dir / "y_test.npy"),
        feature_names=meta["feature_names"],
        scaler=pickle.load((out_dir / "scaler.pkl").open("rb")),
        train_df=pd.read_csv(out_dir / "train.csv"),
        test_df=pd.read_csv(out_dir / "test.csv"),
        metadata=meta,
    )
    return artifacts