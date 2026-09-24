"""Loads a single client's private partition.

Each client thread only ever loads its own CSV. No other client's data, and no
central training data, is ever accessed by a client.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import Config


@dataclass
class ClientData:
    cid: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def load_client_data(config: Config, cid: int) -> ClientData:
    """Load the private train/test split of client ``cid``."""
    clients_dir = config.resolve(config.section("data")["clients_dir"])
    train_path = clients_dir / f"client_{cid}.csv"
    test_path = clients_dir / f"client_{cid}_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Client {cid} partition not found in {clients_dir}. "
            "Run the preparation step first (`python main.py prepare`)."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if train_df.empty:
        raise ValueError(f"Client {cid} has an empty local dataset.")

    feature_names = [c for c in train_df.columns if c != "target"]
    X_train = train_df[feature_names].to_numpy(dtype=np.float32)
    y_train = train_df["target"].to_numpy(dtype=np.int64)
    X_test = test_df[feature_names].to_numpy(dtype=np.float32)
    y_test = test_df["target"].to_numpy(dtype=np.int64)

    return ClientData(
        cid=cid,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
    )