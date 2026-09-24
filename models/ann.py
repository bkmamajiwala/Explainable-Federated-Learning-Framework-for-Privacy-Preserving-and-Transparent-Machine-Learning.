"""A small fully-connected neural network (ANN) implemented with NumPy.

This is the model used both for the centralised ANN baseline and (via
:mod:`models.federated_model`) as the federated neural network. Using a small
hand-rolled MLP gives us full control over parameter exchange and enables a
*genuine* per-example-gradient Differential Privacy mechanism (DP-SGD), which
is difficult to do cleanly with a black-box scikit-learn model.

Architecture
    input -> ReLU hidden layer(s) -> sigmoid output (binary classification)

Parameters are exposed as plain NumPy arrays so federated clients can exchange
them (via Flower) without ever sharing raw data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class DPConfig:
    """Settings for per-example-gradient DP-SGD.

    noise_multiplier:
        Sigma in DP-SGD; the added Gaussian noise has standard deviation
        ``clipping_norm * noise_multiplier / batch_size`` on the averaged
        per-batch gradient.
    clipping_norm:
        Maximum L2 norm allowed for each per-example gradient.
    """

    enabled: bool = True
    noise_multiplier: float = 1.0
    clipping_norm: float = 1.0


class MLP:
    """NumPy multi-layer perceptron for binary classification."""

    def __init__(
        self,
        input_size: int,
        hidden_layers: Sequence[int] = (64, 32),
        seed: int | None = None,
    ) -> None:
        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        self.input_size = int(input_size)
        self.hidden_layers = [int(h) for h in hidden_layers]
        self.seed = seed

        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        self._rng = np.random.default_rng(seed)
        self._initialize()

    # ------------------------------------------------------------------ init
    def _initialize(self) -> None:
        sizes = [self.input_size] + self.hidden_layers + [1]
        self.W = []
        self.b = []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            scale = np.sqrt(2.0 / max(fan_in, 1))  # He init
            self.W.append(
                self._rng.normal(0.0, scale, size=(sizes[i], sizes[i + 1])).astype(
                    np.float32
                )
            )
            self.b.append(np.zeros(sizes[i + 1], dtype=np.float32))

    # ------------------------------------------------------------- parameters
    def get_parameters(self) -> list[np.ndarray]:
        """Return all trainable parameters (weights then biases)."""
        return list(self.W) + list(self.b)

    def set_parameters(self, parameters: Sequence[np.ndarray]) -> None:
        """Overwrite the network parameters from a list of arrays."""
        params = list(parameters)
        n = len(self.W)
        if len(params) != 2 * n:
            raise ValueError(f"Expected {2 * n} parameter arrays, got {len(params)}.")
        for i in range(n):
            if params[i].shape != self.W[i].shape:
                raise ValueError(
                    f"Weight {i} shape mismatch: {params[i].shape} != {self.W[i].shape}"
                )
            if params[n + i].shape != self.b[i].shape:
                raise ValueError(
                    f"Bias {i} shape mismatch: {params[n + i].shape} != {self.b[i].shape}"
                )
            self.W[i] = np.asarray(params[i], dtype=np.float32)
            self.b[i] = np.asarray(params[n + i], dtype=np.float32)

    def copy(self) -> "MLP":
        clone = MLP(self.input_size, self.hidden_layers, seed=self.seed)
        clone.set_parameters(self.get_parameters())
        return clone

    # -------------------------------------------------------------- forward
    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Return (P(class=1), [post-ReLU activations of each hidden layer])."""
        h = np.asarray(X, dtype=np.float32)
        activations: list[np.ndarray] = []
        for W, b in zip(self.W[:-1], self.b[:-1]):
            z = h @ W + b
            h = np.maximum(z, 0.0)
            activations.append(h)
        logit = activations[-1] @ self.W[-1] + self.b[-1]
        return _sigmoid(logit).ravel(), activations

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return a (n, 2) probability matrix [P(class0), P(class1)]."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        p1 = self._forward(X)[0]
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Average binary cross-entropy."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        p = np.clip(self._forward(X)[0], 1e-7, 1.0 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    # -------------------------------------------------------------- training
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 0.01,
        dp: DPConfig | None = None,
        seed: int | None = None,
        verbose: bool = False,
        pos_weight: float | None = None,
    ) -> dict:
        """Mini-batch SGD with Adam, optionally with DP-SGD.

        When ``dp`` is enabled every per-example gradient is clipped to
        ``clipping_norm`` and Gaussian noise
        ``N(0, (clipping_norm * noise_multiplier / batch_size)^2)`` is added to
        the averaged batch gradient (genuine DP-SGD).

        ``pos_weight`` re-weights the positive class in the binary cross-entropy
        loss (used to handle the strong class imbalance of the failure target).
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n_samples = len(X)
        if n_samples == 0:
            raise ValueError("Cannot train on an empty dataset.")

        rng = np.random.default_rng(seed)
        # Noise is drawn from a dedicated stream so DP noise never perturbs the
        # data-shuffling RNG (zero-noise DP then reproduces plain training).
        noise_rng = np.random.default_rng((seed or 0) + 1)
        m_w = [np.zeros_like(w) for w in self.W]
        v_w = [np.zeros_like(w) for w in self.W]
        m_b = [np.zeros_like(b) for b in self.b]
        v_b = [np.zeros_like(b) for b in self.b]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        batch_size = int(batch_size)
        history: list[float] = []

        for epoch in range(int(epochs)):
            perm = rng.permutation(n_samples)
            epoch_losses: list[float] = []
            for start in range(0, n_samples, batch_size):
                idx = perm[start : start + batch_size]
                Xb, yb = X[idx], y[idx]
                t += 1

                grads = self._batch_gradients(Xb, yb, dp, rng, pos_weight, noise_rng)

                for i in range(len(self.W)):
                    wg, bg = grads[i]
                    m_w[i] = beta1 * m_w[i] + (1 - beta1) * wg
                    v_w[i] = beta2 * v_w[i] + (1 - beta2) * wg**2
                    m_b[i] = beta1 * m_b[i] + (1 - beta1) * bg
                    v_b[i] = beta2 * v_b[i] + (1 - beta2) * bg**2

                    mhat_w = m_w[i] / (1 - beta1**t)
                    vhat_w = v_w[i] / (1 - beta2**t)
                    mhat_b = m_b[i] / (1 - beta1**t)
                    vhat_b = v_b[i] / (1 - beta2**t)

                    self.W[i] -= learning_rate * mhat_w / (np.sqrt(vhat_w) + eps)
                    self.b[i] -= learning_rate * mhat_b / (np.sqrt(vhat_b) + eps)

                epoch_losses.append(self.loss(Xb, yb))
            history.append(float(np.mean(epoch_losses)))
            if verbose:
                print(f"    epoch {epoch + 1}/{epochs} loss={history[-1]:.4f}")

        return {"loss": history[-1] if history else float("nan"), "losses": history}

    # ------------------------------------------------------------- gradients
    def _batch_gradients(
        self,
        Xb: np.ndarray,
        yb: np.ndarray,
        dp: DPConfig | None,
        rng: np.random.Generator,
        pos_weight: float | None = None,
        noise_rng: np.random.Generator | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per-layer averaged gradients (dW, db), with DP-SGD clipping/noise."""
        per_example = self._per_example_gradients(Xb, yb, pos_weight)
        if dp is not None and dp.enabled and dp.clipping_norm > 0:
            clip = float(dp.clipping_norm)
            norm = np.linalg.norm(per_example, axis=1, keepdims=True)
            scale = np.minimum(1.0, clip / np.maximum(norm, 1e-12))
            per_example = per_example * scale
        summed = per_example.sum(axis=0)

        if dp is not None and dp.enabled:
            noise_std = dp.clipping_norm * dp.noise_multiplier / len(Xb)
            if noise_std > 0:
                noise_gen = noise_rng if noise_rng is not None else rng
                summed = summed + noise_gen.normal(0.0, noise_std, size=summed.shape)

        mean = summed / len(Xb)

        grads: list[tuple[np.ndarray, np.ndarray]] = []
        weight_size = sum(w.size for w in self.W)
        w_flat = mean[:weight_size]
        b_flat = mean[weight_size:]
        w_offset = 0
        b_offset = 0
        for w, b in zip(self.W, self.b):
            wg = w_flat[w_offset : w_offset + w.size].reshape(w.shape)
            w_offset += w.size
            bg = b_flat[b_offset : b_offset + b.size].reshape(b.shape)
            b_offset += b.size
            grads.append((wg, bg))
        return grads

    def _per_example_gradients(
        self, X: np.ndarray, y: np.ndarray, pos_weight: float | None = None
    ) -> np.ndarray:
        """Vectorised per-example gradients, one row per sample.

        The columns are ordered [dW1, db1, dW2, db2, ..., dWL, dbL] flattened.
        When ``pos_weight`` is given the gradient is computed for the weighted
        binary cross-entropy loss ``-mean(w*y*log p + (1-y)*log(1-p))``.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n = len(X)
        probs, _ = self._forward(X)
        p = np.clip(probs, 1e-7, 1.0 - 1e-7)

        # Pre-activations for the ReLU masks.
        preacts: list[np.ndarray] = []
        h = X
        for W, b in zip(self.W[:-1], self.b[:-1]):
            z = h @ W + b
            preacts.append(z)
            h = np.maximum(z, 0.0)

        if pos_weight is None or pos_weight == 1.0:
            dz = (p - y).reshape(-1, 1)  # derivative of BCE w.r.t. final logit
        else:
            w = float(pos_weight)
            coeff = w * y + (1 - y)  # (n,)
            dz = (p * coeff - w * y).reshape(-1, 1)

        activations = [X] + [np.maximum(z, 0.0) for z in preacts]
        dWs: list[np.ndarray] = []
        dbs: list[np.ndarray] = []
        for layer_idx in reversed(range(len(self.W))):
            a = activations[layer_idx]
            dW = np.einsum("ni,nj->nij", a, dz).reshape(n, -1)
            db = dz.reshape(n, -1)
            dWs.append(dW)
            dbs.append(db)
            if layer_idx > 0:
                dh = dz @ self.W[layer_idx].T
                mask = (preacts[layer_idx - 1] > 0).astype(np.float32)
                dz = dh * mask
        dWs.reverse()
        dbs.reverse()
        return np.concatenate(dWs + dbs, axis=1)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def build_ann(input_size: int, config=None, seed: int | None = None) -> MLP:
    """Factory used by both the ANN baseline and the federated clients."""
    from common.config import load_config

    if config is None:
        config = load_config()
    hidden = config.section("federated").get("hidden_layers", [64, 32])
    if seed is None:
        seed = int(config.section("project")["random_seed"])
    return MLP(input_size=input_size, hidden_layers=hidden, seed=seed)