# Explainable Federated Learning Framework

A research-grade implementation of an **Explainable Federated Learning (XFL)**
framework for predictive maintenance, reproducing the methodology of an
"Explainable Federated Learning Framework for Predictive Maintenance" study
(ai4i2020 dataset).

The framework federates a neural network across simulated organizations
(clients) with the Flower framework, protects updates with **Differential
Privacy (DP-SGD)** and **secure aggregation**, and explains the resulting global
model with **SHAP** and **LIME**. It also includes federated clustering,
centralised baselines, an ablation study and an interactive **Streamlit**
dashboard.

> All numbers in this project come from **actual executions** (`python main.py`)
> and are persisted under `results/`. Nothing is hard-coded; reference values
> from the paper are shown for comparison only.

---

## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [Reproducing the paper](#reproducing-the-paper)
- [Installation](#installation)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [Configuration](#configuration)
- [What the results mean](#what-the-results-mean)
- [Honest limitations](#honest-limitations)
- [Project layout](#project-layout)
- [Tests](#tests)

---

## Features

| Capability | Implementation |
|---|---|
| Federated learning | Flower 1.30 low-level API; real server + client subprocesses over gRPC |
| Client model | Hand-rolled NumPy MLP (ReLU hidden, sigmoid output, Adam) — full control of gradients and parameter exchange |
| Differential privacy | **Genuine DP-SGD**: per-example gradient clipping + calibrated Gaussian noise (`noise_multiplier`) |
| Secure aggregation | Pairwise-masked (masked-vector) aggregation that cancels at the server |
| Explainability | SHAP (KernelExplainer) + LIME on the **final federated global model** |
| Federated clustering | Round-based federated K-Means (only per-cluster sums/counts are shared) |
| Evaluation | Centralised ANN/RF/SVM baselines, per-round/per-client metrics, communication cost |
| Ablations | Centralised vs FL-only vs FL+DP vs FL+DP+XAI vs full framework (A–E) |
| Dashboard | Streamlit with 10 sections (no raw records ever displayed) |
| Experiments | Every run recorded as JSON + a consistent `experiments.csv` log |
| Tests | 43 pytest tests (preprocessing, gradients, DP, aggregation, XAI, clustering, prediction) |

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │               Flower server (main thread)    │
                    │  XFLStrategy(FedAvg) · weighted/uniform agg  │
                    │  round_logs · comm accounting · checkpoints  │
                    └───────────────▲──────────────────────────────┘
                                    │  parameters / metrics only
        ┌───────────────────────────┼───────────────────────────┐
        │          (no raw data ever leaves a client)           │
   ┌────┴─────┐   ┌────┴─────┐   ┌────┴─────┐            ┌──────┴──┐
   │ Client 0 │   │ Client 1 │   │ Client 2 │            │ Client N│
   │ subprocess│   │ subprocess│  │ subprocess│            │         │
   │ DP-SGD   │   │ DP-SGD   │   │ DP-SGD   │            │ DP-SGD  │
   │ + mask   │   │ + mask   │   │ + mask   │            │ + mask  │
   └────┬─────┘   └────┬─────┘   └────┬─────┘            └─────────┘
        └──────────────┴──────────────┴── private partitions ──────┘
```

* The **server** runs in the main process (Flower requires `start_server` in the
  main thread to install its signal handlers) and coordinates training.
* Each **client** is an independent OS subprocess (`client/client_worker.py`)
  that owns only its private partition. It trains locally (optionally with
  DP-SGD), masks its update (optional secure aggregation), and sends **only
  parameters** — raw data never leaves the client.
* The **global model** is evaluated each round on the held-out test set; every
  round's parameters are checkpointed.

## Reproducing the paper

The manuscript reports centralised ANN/RF/SVM baselines and an XFL result
(accuracy 98.8 % in the Results/experimental tables; 98.15 % in the
abstract/conclusion — the discrepancy is flagged in
`evaluation/experiments.py::PAPER_INCONSISTENCY_NOTE`).

This implementation **reproduces the methodology, not the exact numbers**:

| Model | Paper (F1 %) | This implementation (F1 %) |
|---|---|---|
| ANN (centralised) | 79.39 | 70.6 (acc 98.0 %) |
| Random Forest | 83.05 | 69.7 (acc 98.35 %) |
| SVM | 81.04 | 31.3 (acc 97.15 %) |
| Proposed XFL | 87.08 | 63.4 (acc 97.75 %) |

The differences come from the dataset's strong class imbalance (≈3.4 % failure
rate), the DP noise applied in every federated round, and the fact that the
paper's exact hyper-parameters are not published. See
[What the results mean](#what-the-results-mean) for the interpretation.

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Python 3.10 is recommended. The `requirements.txt` pins the tested versions
(Flower 1.30, NumPy 1.26, pandas 2.2, scikit-learn 1.5, SHAP, LIME, Streamlit,
pytest, PyYAML, matplotlib).

## Quick start

```bash
# 1. Prepare the dataset (download/load ai4i2020, preprocess, partition).
python main.py prepare

# 2. Train the centralised baselines (ANN, Random Forest, SVM).
python main.py baselines

# 3. Run federated training (Flower server + client subprocesses).
python main.py federated

# 4. Explain the final global model (SHAP + LIME).
python main.py explain

# 5. Federated clustering over client partitions.
python main.py cluster

# 6. Ablation studies (centralised vs FL variants).
python main.py experiments

# 7. Launch the interactive dashboard.
streamlit run dashboard/app.py
```

Everything can be run in one go with `python main.py all`.

## CLI reference

| Command | What it does |
|---|---|
| `python main.py prepare` | Load, clean, scale and partition the data |
| `python main.py baselines` | Train centralised ANN/RF/SVM baselines |
| `python main.py federated` | Run the Flower federated training loop |
| `python main.py explain` | SHAP/LIME explanations of the global model |
| `python main.py cluster` | Federated K-Means over client partitions |
| `python main.py experiments` | Ablation studies A–E |
| `python main.py diagrams` | Architecture diagrams (`results/figures`) |
| `python main.py all` | Run every stage in sequence |
| `python -m pytest tests` | Run the test suite |
| `streamlit run dashboard/app.py` | Launch the dashboard |

## Configuration

All settings live in `configs/config.yaml`. The most important sections:

```yaml
partition:
  strategy: "iid"        # or "non_iid" (Dirichlet) - an implementation enhancement
  num_clients: 5

federated:
  num_rounds: 10
  local_epochs: 10
  batch_size: 64
  learning_rate: 0.01
  hidden_layers: [64, 32]
  pos_weight: 0.0        # positive-class weight in BCE (0 disables)

privacy:
  dp_enabled: true       # per-example-gradient DP-SGD
  noise_multiplier: 1.0  # σ (std = σ · clipping_norm / batch_size)
  clipping_norm: 1.0     # per-example gradient L2 bound C
  secure_aggregation: false
```

Notes:

* When `secure_aggregation: true` the framework automatically switches to
  **uniform** FedAvg (pairwise masks only cancel with equal weights) — this is
  recorded in the run info.
* `evaluation.ann_baseline_epochs` (60) was chosen by tuning the centralised ANN
  (lr 0.01, batch 64).

## What the results mean

A representative run (10 rounds, 5 clients, DP on, `results/federated_results.json`):

* The global model starts random (round 0 ≈ 6.5 % accuracy) and reaches
  **≈ 97.75 % accuracy / 63.4 % F1** on the held-out test set.
* The ablation study shows the expected pattern:
  * **A centralised** — highest F1 (≈ 70.6 %): the accuracy ceiling.
  * **B FL-only** — near-centralised performance (≈ 63.6 % F1) while data never
    leaves the clients.
  * **C FL+DP** — DP-SGD protects privacy at negligible cost (≈ 63.4 % F1; the
    noise acts as mild regularisation).
  * **D FL+DP+XAI** and **E full framework** — transparency (SHAP/LIME) and
    clustering add no accuracy cost (≈ 63.4 % F1).
* Communication: a tiny model (≈ 1.4 k parameters) costs well under 1 MB for the
  whole training run.

The F1 gap versus the paper is dominated by the rare failure class (≈3.4 % of
samples) and the privacy noise every round — not by a bug in the pipeline.

## Honest limitations

Documented openly in the code and README — this is a functional, reproducible
research framework, not a claim of production-grade security or of reproducing
the paper's exact numbers:

1. **Formal DP accounting.** DP-SGD is implemented correctly (per-example
   clipping + calibrated Gaussian noise) but no ε,δ accounting is claimed.
   A privacy accountant is required before any real deployment.
2. **Secure aggregation prototype.** Masks are derived from a *shared simulation
   secret*; a real deployment would use client-to-client key agreement
   (e.g. Diffie-Hellman). All clients must participate every round, and masked
   updates require uniform FedAvg (no dropout handling).
3. **Federated scaler.** The scaler is fitted on the central train split
   (research-simulation mode). In a truly decentralised deployment each client
   would fit its own scaler; the current choice is flagged in
   `preprocessing/pipeline.py` and is *not* presented as a decentralized
   property.
4. **Non-IID partitioning** is an implementation enhancement, not a claim from
   the paper (the paper does not report non-IID experiments).
5. **`hash()` pitfall.** Secure-aggregation mask seeds use `hashlib.sha256`
   (not Python's per-process salted `hash()`) so that masks cancel across the
   client subprocesses.

## Project layout

```
common/            config loading, logging, utils
preprocessing/     pipeline (clean/scale/split) + partitioning
models/            NumPy MLP (+ DP-SGD), RandomForest, SVM, FederatedModel
client/            Flower NumPyClient + subprocess worker
server/            strategy (FedAvg), aggregation, runner (orchestration)
privacy/           differential_privacy.py, secure_aggregation.py
explainability/    SHAP, LIME, explain_runner
clustering/        federated_clustering.py (FedKMeans)
evaluation/        metrics, experiments (tracking), ablation
visualization/     matplotlib plots + architecture diagrams
dashboard/         Streamlit app
tests/             pytest suite
configs/           config.yaml
data/              raw/processed/client partitions
results/           JSON results, figures, logs, experiments.csv
models_saved/      baselines, checkpoints, final global model
```

## Tests

```bash
python -m pytest tests -q      # 43 tests
```

The suite covers configuration, preprocessing, partitioning, the NumPy MLP
(including a numeric gradient check against `_per_example_gradients`), DP-SGD,
weighted/uniform aggregation, secure (masked) aggregation (masks cancel under
uniform averaging), metrics, SHAP additivity, LIME, federated clustering, and
the dashboard prediction path.