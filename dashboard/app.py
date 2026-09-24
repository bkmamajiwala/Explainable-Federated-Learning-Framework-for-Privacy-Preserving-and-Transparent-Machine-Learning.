"""Interactive Streamlit dashboard for the Explainable Federated Learning framework.

Run from the project root::

    streamlit run dashboard/app.py

Every value shown here is loaded from the artefacts produced by ``main.py``
(results JSON/CSV, saved models, generated figures) - nothing is hard-coded.
Raw dataset records are never displayed; the Prediction page only shows
user-entered inputs and model outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json  # noqa: E402
import pickle  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common.config import load_config  # noqa: E402
from preprocessing.pipeline import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)

st.set_page_config(page_title="Explainable Federated Learning", page_icon="🏭", layout="wide")

CSS = """
<style>
.block-container { padding-top: 2rem; }
[data-testid="stSidebar"] { background-color: #0e1a2b; }
h1, h2, h3 { letter-spacing: -0.3px; }
div[data-testid="stMetric"] {
    background: #f7f9fc; border: 1px solid #e3e8ef; border-radius: 8px; padding: 12px 16px;
}
.kpi { text-align: center; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

CONFIG = load_config()
ROOT = PROJECT_ROOT
RESULTS = Path(CONFIG.resolve("results"))
FIGURES = Path(CONFIG.resolve("results/figures"))
MODELS = Path(CONFIG.resolve("models_saved"))


# --------------------------------------------------------------------------- loaders
@st.cache_resource(show_spinner=False)
def load_json(rel: str) -> dict | None:
    p = RESULTS / rel
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_resource(show_spinner=False)
def load_global_model():
    path = MODELS / "final_global_model.pkl"
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    from preprocessing.pipeline import build_from_saved

    try:
        return build_from_saved(CONFIG)
    except FileNotFoundError:
        return None


def figure(rel: str) -> Path | None:
    p = FIGURES / rel
    return p if p.exists() else None


def section_heading(text: str, sub: str | None = None) -> None:
    st.markdown(f"## {text}")
    if sub:
        st.caption(sub)
    st.divider()


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("🏭 Explainable FL")
st.sidebar.caption("Predictive maintenance · ai4i2020")
PAGE = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Federated Training",
        "Client View",
        "Prediction",
        "Explainability",
        "Privacy",
        "Clustering",
        "Comparison",
        "Ablations",
        "Experiments",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    "Framework v1.0 · Flower + NumPy MLP + DP-SGD + SHAP/LIME\n"
    "All numbers are produced by actual runs (`main.py`)."
)

fed = load_json("federated_results.json")
expl = load_json("explainability_results.json")
clust = load_json("clustering_results.json")
abl = load_json("ablation_results.json")
partition_meta_path = Path(CONFIG.resolve("data/clients/partition_metadata.json"))
partition_meta = json.loads(partition_meta_path.read_text(encoding="utf-8")) if partition_meta_path.exists() else None

EXPERIMENTS_CSV = RESULTS / "experiments.csv"
experiments_df = pd.read_csv(EXPERIMENTS_CSV) if EXPERIMENTS_CSV.exists() else pd.DataFrame()


def page_dashboard() -> None:
    section_heading("Dashboard", "System overview, KPIs and the framework architecture.")

    if fed:
        fin = fed.get("per_round", [])[-1] if fed.get("per_round") else {}
        m = fin.get("global_metrics", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Global accuracy", f"{m.get('accuracy', 0):.2%}")
        c2.metric("F1 score", f"{m.get('f1', 0):.2%}")
        c3.metric("FL rounds", fed.get("num_rounds", "—"))
        c4.metric("Communication", f"{fed.get('communication', {}).get('total_mb', 0):.2f} MB")

    b1, b2, b3 = st.columns(3)
    b1.metric("Clients", CONFIG["partition"]["num_clients"])
    b2.metric("DP-SGD", "ON" if CONFIG["privacy"]["dp_enabled"] else "OFF")
    b3.metric("Secure aggregation", "ON" if CONFIG["privacy"]["secure_aggregation"] else "OFF")

    st.markdown("#### Pipeline stages")
    stages = [
        ("Data preparation", "Load → clean → scale → partition", "prepare"),
        ("Baselines", "Centralised ANN / RF / SVM", "baselines"),
        ("Federated training", "Flower clients + server, DP-SGD", "federated"),
        ("Explainability", "SHAP + LIME on the global model", "explain"),
        ("Federated clustering", "FedKMeans over client partitions", "cluster"),
        ("Ablations", "Centralised vs FL+DP+XAI+clustering", "experiments"),
    ]
    for title, desc, _cmd in stages:
        st.markdown(f"- **{title}** — {desc}")

    diag = figure("architecture.png")
    if diag:
        st.image(str(diag), use_container_width=True)


def page_federated() -> None:
    section_heading("Federated Training", "Server/client training dynamics and communication cost.")
    if not fed:
        st.info("No federated run found. Run `python main.py federated` first.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Rounds", fed["num_rounds"])
    c2.metric("Local epochs", fed["local_epochs"])
    c3.metric("Training time", f"{fed.get('training_time_s', 0):.1f} s")

    st.markdown("#### Training curves")
    col1, col2 = st.columns(2)
    cur = figure("federated_curves.png")
    if cur:
        col1.image(str(cur), use_container_width=True)
    comm = figure("communication_overhead.png")
    if comm:
        col2.image(str(comm), use_container_width=True)

    st.markdown("#### Per-round global evaluation")
    if fed.get("per_round"):
        rows = [
            {
                "Round": r["round"],
                "Loss": round(r.get("global_loss", 0), 4),
                "Accuracy": round(r["global_metrics"].get("accuracy", 0), 4),
                "Precision": round(r["global_metrics"].get("precision", 0), 4),
                "Recall": round(r["global_metrics"].get("recall", 0), 4),
                "F1": round(r["global_metrics"].get("f1", 0), 4),
                "Comm (MB)": round(r.get("comm_bytes", 0) / (1024 ** 2), 4),
            }
            for r in fed["per_round"]
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Per-client evaluation (last round)")
    if fed.get("per_client"):
        rows = []
        for pc in fed["per_client"]:
            rows.append({
                "Client": pc.get("cid", "—"),
                "Test samples": pc.get("n_test", 0),
                "Accuracy": round(pc.get("accuracy", 0), 4),
                "Precision": round(pc.get("precision", 0), 4),
                "Recall": round(pc.get("recall", 0), 4),
                "F1": round(pc.get("f1", 0), 4),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Configuration snapshot")
    st.json({
        "federated": CONFIG["federated"],
        "privacy": CONFIG["privacy"],
    })


def page_clients() -> None:
    section_heading("Client View", "Per-client data partitions (privacy-preserving simulation).")
    if not partition_meta:
        st.info("No partition metadata. Run `python main.py prepare` first.")
        return

    clients = partition_meta["clients"]
    rows = []
    for cid in sorted(clients, key=int):
        c = clients[cid]
        rows.append({
            "Client": cid,
            "Train samples": c["n_train"],
            "Local test": c["n_test"],
            "Failures (train)": c["train_positives"],
            "Failure rate": f"{c['train_failure_rate']:.2%}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    df = pd.DataFrame(rows)
    if len(df):
        st.markdown("#### Failure rate across clients")
        st.bar_chart(df.set_index("Client")["Failure rate"].str.rstrip("%").astype(float))

    st.caption(f"Partition strategy: **{partition_meta['strategy']}** · "
               f"{partition_meta['num_clients']} clients · "
               f"{partition_meta['total_train_rows']} training rows in total.")


def _transform_input(features: dict) -> np.ndarray:
    artifacts = load_artifacts()
    row = {**features}
    df = pd.DataFrame([row], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    return artifacts.scaler.transform(df).astype(np.float32)


def _custom_shap_figure(model, scaled_row, artifacts, sample_idx: int = 1):
    from explainability.shap_explainer import compute_shap_values

    shap = compute_shap_values(
        CONFIG, model.predict_proba, artifacts.X_train, scaled_row,
        background_samples=int(CONFIG["explainability"]["background_samples"]),
    )
    return shap


def page_prediction() -> None:
    section_heading("Prediction", "Interactive failure prediction on the federated global model.")
    model = load_global_model()
    artifacts = load_artifacts()
    if model is None or artifacts is None:
        st.info("Global model not found. Run `python main.py federated` first.")
        return

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        air = col1.slider("Air temperature [K]", 295.0, 305.0, 300.0, 0.1)
        proc = col2.slider("Process temperature [K]", 305.0, 314.0, 310.0, 0.1)
        rpm = col3.slider("Rotational speed [rpm]", 1100.0, 2900.0, 1500.0, 10.0)
        col4, col5, col6 = st.columns(3)
        torque = col4.slider("Torque [Nm]", 3.0, 77.0, 40.0, 0.5)
        wear = col5.slider("Tool wear [min]", 0, 253, 200, 1)
        mtype = col6.selectbox("Product type", ["L", "M", "H"])
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        features = {
            "Air temperature [K]": air,
            "Process temperature [K]": proc,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": float(wear),
            "Type": mtype,
        }
        scaled = _transform_input(features)
        prob = float(model.predict_proba(scaled)[0, 1])
        label = "FAILURE" if prob >= 0.5 else "No failure"

        st.markdown("#### Prediction")
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", label)
        c2.metric("P(failure)", f"{prob:.2%}")
        c3.metric("Confidence", f"{max(prob, 1 - prob):.2%}")

        with st.spinner("Computing SHAP explanation for this sample..."):
            try:
                shap = _custom_shap_figure(model, scaled, artifacts)
                sv = shap["shap_values"][0]
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                names = artifacts.feature_names
                order = np.argsort(-np.abs(sv))
                fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(names))))
                cum = shap["base_value"] + np.cumsum(sv[order])
                ys = np.arange(len(sv))[::-1]
                ax.barh(ys, np.abs(sv[order]), left=cum - sv[order],
                        color=["#c0392b" if v > 0 else "#3d9970" for v in sv[order]],
                        edgecolor="white", height=0.7)
                ax.axvline(shap["base_value"], color="gray", linestyle="--", lw=1)
                ax.set_yticks(ys)
                ax.set_yticklabels([f"{names[i]}\n({sv[i]:+.3f})" for i in order])
                ax.set_xlabel("Model output (P(failure))")
                ax.set_title(f"SHAP waterfall - prediction {prob:.2%}")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                st.pyplot(fig)
            except Exception as exc:  # noqa: BLE001 - keep the page usable
                st.warning(f"SHAP explanation unavailable: {exc}")

        with st.spinner("Computing LIME explanation for this sample..."):
            try:
                from explainability.lime_explainer import explain_samples

                lime = explain_samples(
                    CONFIG, model.predict_proba, artifacts.X_train, scaled,
                    artifacts.feature_names,
                    max_display_features=int(CONFIG["explainability"]["max_display_features"]),
                    n_samples=1,
                )
                if lime and lime[0]["weights"]:
                    wdf = pd.DataFrame(lime[0]["weights"], columns=["Feature", "LIME weight"])
                    wdf = wdf.sort_values("LIME weight")
                    st.markdown("#### LIME surrogate")
                    st.bar_chart(wdf.set_index("Feature"))
            except Exception as exc:  # noqa: BLE001
                st.warning(f"LIME explanation unavailable: {exc}")


def page_explainability() -> None:
    section_heading("Explainability", "SHAP and LIME explanations of the final global model.")
    if not expl:
        st.info("No explanations found. Run `python main.py explain` first.")
        return

    c1, c2 = st.columns(2)
    c1.metric("SHAP samples", expl.get("shap", {}).get("samples_explained", 0))
    c2.metric("LIME samples", expl.get("lime", {}).get("samples_explained", 0))

    st.markdown("#### SHAP")
    col1, col2 = st.columns(2)
    s1 = figure("explainability/shap_summary.png")
    s2 = figure("explainability/shap_feature_importance.png")
    if s1:
        col1.image(str(s1), use_container_width=True, caption="SHAP summary (beeswarm)")
    if s2:
        col2.image(str(s2), use_container_width=True, caption="Mean |SHAP| per feature")

    imp = expl.get("shap", {}).get("mean_abs_importance")
    if imp:
        st.markdown("#### SHAP feature ranking")
        idf = pd.DataFrame(list(imp.items()), columns=["Feature", "Mean |SHAP|"]).sort_values(
            "Mean |SHAP|", ascending=False
        )
        st.dataframe(idf, use_container_width=True, hide_index=True)

    st.markdown("#### Per-sample SHAP waterfalls")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        p = figure(f"explainability/shap_waterfall_{i + 1}.png")
        if p:
            col.image(str(p), use_container_width=True, caption=f"Sample {i + 1}")

    st.markdown("#### LIME")
    col1, col2 = st.columns(2)
    l1 = figure("explainability/lime_feature_importance.png")
    l2 = figure("explainability/lime_sample_1.png")
    if l1:
        col1.image(str(l1), use_container_width=True, caption="LIME mean |weight|")
    if l2:
        col2.image(str(l2), use_container_width=True, caption="LIME sample 1")

    samples = expl.get("lime", {}).get("per_sample")
    if samples:
        st.markdown("#### LIME weights (samples)")
        rows = []
        for i, s in enumerate(samples):
            for name, w in s["weights"]:
                rows.append({"Sample": i + 1, "Predicted": "Failure" if s["predicted_class"] == 1 else "No failure",
                             "Feature": name, "Weight": round(w, 4)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_privacy() -> None:
    section_heading("Privacy", "Differential privacy and secure aggregation.")
    p = CONFIG["privacy"]

    c1, c2, c3 = st.columns(3)
    c1.metric("DP-SGD", "Enabled" if p["dp_enabled"] else "Disabled")
    c2.metric("Noise multiplier σ", p["noise_multiplier"])
    c3.metric("Clipping norm C", p["clipping_norm"])

    st.markdown("### Differential Privacy (DP-SGD)")
    st.write(
        "Every client computes **per-example gradients** during local training. Each "
        "per-example gradient is clipped to an L2 norm of "
        f"**{p['clipping_norm']}**, and Gaussian noise with standard deviation "
        f"**σ·C/B = {p['noise_multiplier']}×{p['clipping_norm']}/batch** is added to the "
        "averaged batch gradient before the parameter update. This is genuine DP-SGD on "
        "the model updates exchanged with the server."
    )
    st.info(
        "This framework does **not** claim a formal ε,δ accounting. The privacy budget "
        "is presented qualitatively (per-example clipping + calibrated Gaussian noise). "
        "A formal accountant would be required before any production deployment."
    )

    st.markdown("### Secure Aggregation")
    if p["secure_aggregation"]:
        st.success("Secure aggregation is **enabled**.")
    else:
        st.warning("Secure aggregation is **disabled** (the server sees unmasked updates).")

    st.write(
        "The masked-vector (pairwise mask) scheme adds a private mask to each client "
        "update; masks cancel exactly when the server sums them, so the server learns "
        "only the aggregate update, never an individual client's contribution. "
    )
    st.caption(
        "Documented limitations: all clients must participate every round, aggregation "
        "must be uniform, and the masks are derived from a simulation secret - a real "
        "deployment would use client-to-client key agreement (see README)."
    )


def page_clustering() -> None:
    section_heading("Federated Clustering", "Round-based federated K-Means over client partitions.")
    if not clust:
        st.info("No clustering results. Run `python main.py cluster` first.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters", clust["num_clusters"])
    c2.metric("Rounds", clust["rounds"])
    c3.metric("Points clustered", sum(clust["global_cluster_counts"]))

    img = figure("federated_clustering.png")
    if img:
        st.image(str(img), use_container_width=True, caption="Client points + global centroids (PCA)")

    st.markdown("#### Global cluster sizes")
    st.bar_chart(pd.DataFrame({
        "Cluster": [f"Cluster {i}" for i in range(len(clust["global_cluster_counts"]))],
        "Points": clust["global_cluster_counts"],
    }).set_index("Cluster"))

    st.markdown("#### Centroid profiles (original units)")
    profiles = clust.get("centroid_original_units")
    if profiles:
        st.dataframe(pd.DataFrame(profiles), use_container_width=True, hide_index=True)

    st.markdown("#### Per-client cluster distribution")
    if clust.get("per_client"):
        rows = []
        for pc in clust["per_client"]:
            rows.append({"Client": pc["cid"], **{f"C{k}": v for k, v in enumerate(pc["cluster_counts"])}})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _model_metrics_df() -> pd.DataFrame:
    rows = []
    if not experiments_df.empty:
        for _, r in experiments_df.iterrows():
            name = str(r.get("name", ""))
            mode = str(r.get("mode", ""))
            if mode == "baseline" and "model" in r:
                rows.append({"model": r["model"], "accuracy": r.get("accuracy"), "precision": r.get("precision"),
                             "recall": r.get("recall"), "f1": r.get("f1")})
    if fed:
        fin = fed.get("per_round", [])[-1] if fed.get("per_round") else {}
        m = fin.get("global_metrics", {})
        rows.append({"model": "xfl", "accuracy": m.get("accuracy"), "precision": m.get("precision"),
                     "recall": m.get("recall"), "f1": m.get("f1")})
    return pd.DataFrame(rows)


def page_comparison() -> None:
    section_heading("Comparison", "Centralised baselines vs. the proposed XFL framework.")
    df = _model_metrics_df()
    if df.empty:
        st.info("No model results found. Run `python main.py baselines` and `python main.py federated` first.")
        return

    LABELS = {"ann": "ANN (centralised)", "rf": "Random Forest (centralised)",
              "svm": "SVM (centralised)", "xfl": "Proposed XFL (federated)"}
    df["label"] = df["model"].map(LABELS)
    view = df[["label", "accuracy", "precision", "recall", "f1"]].set_index("label")
    st.dataframe(view.style.format("{:.3f}"), use_container_width=True)

    img = figure("model_comparison.png")
    if img:
        st.image(str(img), use_container_width=True, caption="Model comparison")

    st.markdown("### Reference values from the research paper")
    from evaluation.experiments import PAPER_INCONSISTENCY_NOTE, PAPER_RESULTS

    paper_df = pd.DataFrame(PAPER_RESULTS).T
    st.dataframe(paper_df.style.format("{:.2f}"), use_container_width=True)
    st.caption(PAPER_INCONSISTENCY_NOTE)
    st.caption(
        "Note: paper values are percentages reported in the manuscript; this "
        "implementation reproduces the framework (FL + DP + XAI) and reports its own "
        "measured metrics - it does not claim to reproduce the paper's exact numbers."
    )


def page_ablations() -> None:
    section_heading("Ablation Studies", "Centralised vs. federated configurations (A-E).")
    if not abl:
        st.info("No ablation results. Run `python main.py experiments` first.")
        return

    img = figure("ablation_comparison.png")
    if img:
        st.image(str(img), use_container_width=True)

    rows = []
    for name, m in abl.items():
        rows.append({
            "Configuration": name,
            "Accuracy": round(m.get("accuracy", 0), 4),
            "Precision": round(m.get("precision", 0), 4),
            "Recall": round(m.get("recall", 0), 4),
            "F1": round(m.get("f1", 0), 4),
            "Train time (s)": round(m.get("train_time_s", 0), 1),
            "Comm (MB)": round(m.get("comm_mb", 0), 2),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown(
        "**Reading the results.** The centralised ANN (A) is the accuracy ceiling. "
        "Federated learning (B) reaches comparable accuracy with data never leaving "
        "the clients. DP-SGD (C) protects privacy at negligible cost. XAI (D) and "
        "clustering (E) add transparency and insight without degrading predictive "
        "performance."
    )


def page_experiments() -> None:
    section_heading("Experiment Log", "Every experiment is recorded with its configuration snapshot.")
    if experiments_df.empty:
        st.info("No experiments recorded yet.")
        return

    cols = ["experiment_id", "name", "mode", "timestamp"]
    cols += [c for c in experiments_df.columns if c not in cols]
    st.dataframe(experiments_df[cols], use_container_width=True, hide_index=True)

    exp_dir = RESULTS / "experiments"
    if exp_dir.exists():
        files = sorted(exp_dir.glob("*.json"), reverse=True)[:5]
        if files:
            st.markdown("#### Latest experiment details")
            for f in files:
                with f.open("r", encoding="utf-8") as fh:
                    rec = json.load(fh)
                with st.expander(f"{rec.get('experiment_id', f.name)} — {rec.get('name')}"):
                    st.json(rec)


PAGES = {
    "Dashboard": page_dashboard,
    "Federated Training": page_federated,
    "Client View": page_clients,
    "Prediction": page_prediction,
    "Explainability": page_explainability,
    "Privacy": page_privacy,
    "Clustering": page_clustering,
    "Comparison": page_comparison,
    "Ablations": page_ablations,
    "Experiments": page_experiments,
}

PAGES[PAGE]()