"""Federated data partitioning.

Each simulated client receives its own private CSV partition. Two strategies are
supported:

* ``iid``      - stratified random partition (each client sees a balanced mix).
* ``non_iid``  - Dirichlet-based skewed partition so different clients observe
                 different class distributions (closer to real distributed
                 organizations).

Non-IID is an *implementation enhancement* - the research paper itself does not
report non-IID experiments, so this is explicitly documented as an extension,
not a claim from the paper. The default reproduction mode is ``iid``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import Config
from common.utils import write_json
from preprocessing.pipeline import PipelineArtifacts


@dataclass
class ClientPartition:
    cid: int
    n_train: int
    n_test: int
    train_positives: int
    test_positives: int
    csv_path: Path
    test_csv_path: Path


def partition_data(
    config: Config,
    artifacts: PipelineArtifacts,
    overwrite: bool = True,
) -> list[ClientPartition]:
    """Split the central train set among ``num_clients`` clients."""
    pcfg = config.section("partition")
    num_clients = int(pcfg["num_clients"])
    strategy = str(pcfg["strategy"]).lower()
    test_frac = float(pcfg["client_test_fraction"])
    seed = int(config.section("project")["random_seed"])

    out_dir = config.resolve(config.section("data")["clients_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if strategy == "iid":
        indices = _iid_indices(artifacts, num_clients, seed)
    elif strategy == "non_iid":
        alpha = float(pcfg["non_iid_alpha"])
        indices = _non_iid_indices(artifacts, num_clients, alpha, seed)
    else:
        raise ValueError(
            f"Unknown partition strategy '{strategy}'. Use 'iid' or 'non_iid'."
        )

    X = artifacts.X_train
    y = artifacts.y_train
    feature_names = artifacts.feature_names

    partitions: list[ClientPartition] = []
    per_client: dict[str, object] = {}
    for cid in range(num_clients):
        idx = indices[cid]
        Xc, yc = X[idx], y[idx]

        from sklearn.model_selection import train_test_split

        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            Xc, yc, test_size=test_frac, random_state=seed + cid, stratify=yc
        )

        train_df = pd.DataFrame(Xc_train, columns=feature_names)
        train_df["target"] = yc_train
        test_df = pd.DataFrame(Xc_test, columns=feature_names)
        test_df["target"] = yc_test

        train_path = out_dir / f"client_{cid}.csv"
        test_path = out_dir / f"client_{cid}_test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        partitions.append(
            ClientPartition(
                cid=cid,
                n_train=len(train_df),
                n_test=len(test_df),
                train_positives=int(yc_train.sum()),
                test_positives=int(yc_test.sum()),
                csv_path=train_path,
                test_csv_path=test_path,
            )
        )
        per_client[str(cid)] = {
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "train_positives": int(yc_train.sum()),
            "test_positives": int(yc_test.sum()),
            "train_failure_rate": round(float(yc_train.mean()), 4),
        }

    write_json(
        out_dir / "partition_metadata.json",
        {
            "strategy": strategy,
            "num_clients": num_clients,
            "non_iid_alpha": pcfg.get("non_iid_alpha") if strategy == "non_iid" else None,
            "total_train_rows": int(len(y)),
            "clients": per_client,
        },
    )
    return partitions


def _iid_indices(artifacts: PipelineArtifacts, num_clients: int, seed: int) -> list[np.ndarray]:
    """Stratified random partition into ``num_clients`` groups."""
    rng = np.random.default_rng(seed)
    y = artifacts.y_train
    n = len(y)
    indices = np.arange(n)

    # Stratified shuffle: keep class ratio within each client's share.
    pos = indices[y == 1]
    neg = indices[y == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    result: list[np.ndarray] = []
    for cid in range(num_clients):
        pos_share = _chunk(pos, num_clients, cid)
        neg_share = _chunk(neg, num_clients, cid)
        group = np.concatenate([pos_share, neg_share])
        rng.shuffle(group)
        result.append(group)
    return result


def _non_iid_indices(
    artifacts: PipelineArtifacts, num_clients: int, alpha: float, seed: int
) -> list[np.ndarray]:
    """Dirichlet-based non-IID partition (see module docstring)."""
    rng = np.random.default_rng(seed)
    y = artifacts.y_train
    indices = np.arange(len(y))
    result: list[np.ndarray] = [np.array([], dtype=int) for _ in range(num_clients)]

    for cls in np.unique(y):
        cls_idx = indices[y == cls]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        splits = (proportions * len(cls_idx)).astype(int)
        # Fix rounding drift.
        diff = len(cls_idx) - splits.sum()
        splits[0] += diff
        splits = np.maximum(splits, 0)

        start = 0
        for cid in range(num_clients):
            end = start + splits[cid]
            result[cid] = np.concatenate([result[cid], cls_idx[start:end]])
            start = end
    return result


def _chunk(arr: np.ndarray, num_clients: int, cid: int) -> np.ndarray:
    size = int(np.ceil(len(arr) / num_clients))
    start = cid * size
    return arr[start : start + size]