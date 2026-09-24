"""Federated K-Means clustering.

Implements a round-based *federated* K-Means protocol:

* The server holds only the global centroids (randomly initialised in
  standardised feature space - the server never sees any raw data).
* Each round the server broadcasts the centroids; every client locally assigns
  its *private* partition rows to the nearest centroid and returns only two
  aggregates per cluster: the **sum of points** and the **count**.
* The server computes the new centroids as ``sum / count`` and repeats.

Raw feature vectors never leave a client partition - only per-cluster sums and
counts cross the wire, which is the standard secure pooling of FedKMeans-style
algorithms. This is a functional, deterministic simulation of the protocol
(see README for documented limitations).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import Config
from common.logging_setup import setup_logging


class FederatedKMeans:
    """Federated K-Means with local assignment and centroid pooling."""

    def __init__(self, num_clusters: int, seed: int = 42, d: int = 7) -> None:
        self.k = int(num_clusters)
        self.d = int(d)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.centroids: np.ndarray = self._initialize()
        self.history: list[float] = []

    def _initialize(self) -> np.ndarray:
        # Server-side random init in standardised space (mean 0, std 1).
        return self.rng.uniform(-1.0, 1.0, size=(self.k, self.d)).astype(np.float32)

    # ------------------------------------------------------------ local step
    def assign(self, X: np.ndarray) -> np.ndarray:
        """Local assignment: label of the nearest centroid for each row."""
        X = np.asarray(X, dtype=np.float32)
        diff = X[:, None, :] - self.centroids[None, :, :]
        dist = np.einsum("ijk,ijk->ij", diff, diff)
        return np.argmin(dist, axis=1)

    # ------------------------------------------------------- aggregation step
    def aggregate(self, sums: list[np.ndarray], counts: list[np.ndarray]) -> np.ndarray:
        """Merge per-client (sum, count) aggregates into new centroids."""
        total_sum = np.zeros((self.k, self.d), dtype=np.float64)
        total_count = np.zeros(self.k, dtype=np.float64)
        for s, c in zip(sums, counts):
            total_sum += s
            total_count += c
        new_centroids = np.empty_like(self.centroids, dtype=np.float32)
        for k in range(self.k):
            if total_count[k] > 0:
                new_centroids[k] = (total_sum[k] / total_count[k]).astype(np.float32)
            else:
                new_centroids[k] = self.centroids[k]
        self.history.append(float(np.mean(np.linalg.norm(new_centroids - self.centroids, axis=1))))
        self.centroids = new_centroids
        return self.centroids

    def fit(self, client_data: list[np.ndarray], rounds: int = 5) -> dict:
        """Run ``rounds`` federated rounds over the per-client data matrices."""
        for r in range(1, int(rounds) + 1):
            sums, counts = [], []
            for X in client_data:
                labels = self.assign(X)
                per_sum = np.zeros((self.k, self.d), dtype=np.float64)
                per_count = np.zeros(self.k, dtype=np.float64)
                for k in range(self.k):
                    sel = X[labels == k]
                    if len(sel) > 0:
                        per_sum[k] = sel.sum(axis=0)
                        per_count[k] = len(sel)
                sums.append(per_sum)
                counts.append(per_count)
            self.aggregate(sums, counts)
        return {
            "centroids": self.centroids,
            "movement_history": list(self.history),
        }


def run_federated_clustering(config: Config) -> dict:
    """Run the federated clustering module end-to-end and persist artefacts."""
    logger = setup_logging(tag="CLUSTER", log_dir=config.resolve("results/logs"))
    from evaluation.experiments import ExperimentManager
    from preprocessing.pipeline import build_from_saved

    ccfg = config.section("clustering")
    k = int(ccfg.get("num_clusters", 3))
    rounds = int(ccfg.get("rounds", 5))
    local_rounds = int(ccfg.get("local_cluster_rounds", 5))
    seed = int(config.section("project")["random_seed"])

    artifacts = build_from_saved(config)
    feature_names = artifacts.feature_names

    # Each "client" contributes only its own partition (kept private).
    clients_dir = config.resolve(config.section("data")["clients_dir"])
    client_data: list[np.ndarray] = []
    num_clients = int(config.section("partition")["num_clients"])
    for cid in range(num_clients):
        path = clients_dir / f"client_{cid}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Client partition not found: {path}. Run `prepare` first.")
        df = pd.read_csv(path)
        client_data.append(df[feature_names].to_numpy(dtype=np.float32))

    km = FederatedKMeans(num_clusters=k, seed=seed, d=len(feature_names))
    fitted = km.fit(client_data, rounds=rounds)

    # Per-client cluster distributions (drives the non-IID discussion).
    per_client: list[dict] = []
    for cid, X in enumerate(client_data):
        labels = km.assign(X)
        counts = np.bincount(labels, minlength=k)
        per_client.append({
            "cid": cid,
            "n": int(len(X)),
            "cluster_counts": [int(c) for c in counts],
            "cluster_fractions": [round(float(c / len(X)), 4) for c in counts],
        })

    global_labels = np.concatenate([km.assign(X) for X in client_data])
    global_counts = [int(c) for c in np.bincount(global_labels, minlength=k)]

    # Cluster profiles in original physical units for human-readable summaries.
    from preprocessing.pipeline import inverse_transform_features

    centroid_orig = inverse_transform_features(config, km.centroids, feature_names)

    logger.info(
        "Federated clustering complete: %d clusters over %d clients (%d rows) in %d rounds.",
        k, len(client_data), int(global_labels.size), rounds,
    )
    for c in range(k):
        logger.info(
            "Cluster %d: %d points | %s", c, global_counts[c],
            ", ".join(f"{f}={centroid_orig.iloc[c][f]:.2f}" for f in feature_names[:4]),
        )

    _save_figure(config, km, client_data, feature_names)
    _save_report(config, km, per_client, global_counts, centroid_orig, feature_names, rounds, local_rounds)

    mgr = ExperimentManager(config)
    rec = mgr.start("federated-clustering", "cluster", extra={
        "num_clusters": k, "rounds": rounds, "num_clients": num_clients,
        "converged_movement": round(fitted["movement_history"][-1], 5),
    })
    mgr.save(rec, {
        "global_counts": global_counts,
        "final_centroid_movement": round(fitted["movement_history"][-1], 6),
    })

    return {
        "num_clusters": k,
        "rounds": rounds,
        "centroids": km.centroids.tolist(),
        "global_cluster_counts": global_counts,
        "per_client": per_client,
        "movement_history": fitted["movement_history"],
    }


def _save_figure(config, km: FederatedKMeans, client_data, feature_names) -> None:
    from sklearn.decomposition import PCA

    from visualization.plots import save_fig

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X = np.concatenate(client_data)
    labels = km.assign(X)
    reducer = PCA(n_components=2, random_state=int(config.section("project")["random_seed"]))
    X2 = reducer.fit_transform(X)
    C2 = reducer.transform(km.centroids)

    fig, ax = plt.subplots(figsize=(8, 6))
    for k in range(km.k):
        mask = labels == k
        ax.scatter(X2[mask, 0], X2[mask, 1], s=8, alpha=0.5,
                   label=f"Cluster {k} (n={int(mask.sum())})")
    ax.scatter(C2[:, 0], C2[:, 1], marker="X", s=220, c="black", edgecolor="white",
               linewidths=1.5, zorder=5, label="Centroids")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Federated K-Means - client points and global centroids (PCA)")
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.2, linestyle="--")
    ax.set_axisbelow(True)

    out = config.resolve(config.section("paths")["figures_dir"]) / "federated_clustering.png"
    save_fig(fig, out)


def _save_report(config, km, per_client, global_counts, centroid_orig, feature_names, rounds, local_rounds) -> None:
    from common.utils import write_json

    report = {
        "num_clusters": km.k,
        "rounds": rounds,
        "local_rounds": local_rounds,
        "num_features": len(feature_names),
        "global_cluster_counts": global_counts,
        "movement_history": km.history,
        "centroid_original_units": centroid_orig.to_dict(orient="records"),
        "per_client": per_client,
    }
    write_json(
        config.resolve(config.section("paths")["results_dir"]) / "clustering_results.json",
        report,
    )