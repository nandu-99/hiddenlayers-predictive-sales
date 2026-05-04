import json, os
from pathlib import Path

import joblib
import kagglehub
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from streamlit_models import (
    ConcatFusion,
    ConfidenceGatedHybrid,
    rf_prob_and_uncertainty,
)

st.set_page_config(page_title="Phase 3 — Confidence-Gated Hybrid",
                   page_icon="🎯", layout="wide")

# Handle Streamlit Cloud Kaggle secrets
if "kaggle" in st.secrets:
    os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
    os.environ["KAGGLE_KEY"]      = st.secrets["kaggle"]["key"]

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load bundle from Kaggle  (cached so it only runs once per session)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Downloading model bundle from Kaggle …")
def load_bundle():
    path = kagglehub.dataset_download("vivekananda99/predictive-sales")
    p = Path(path)

    meta        = json.loads((p / "meta.json").read_text())
    rf          = joblib.load(p / "rf_model.joblib")
    tab_scaler  = joblib.load(p / "tab_scaler.joblib")
    text_scaler = joblib.load(p / "text_scaler.joblib")

    concat = ConcatFusion(**meta["concat_fusion_arch"])
    concat.load_state_dict(torch.load(p / "concat_fusion.pt", map_location="cpu"))
    concat.eval()

    gate = ConfidenceGatedHybrid(hidden=meta["gate_arch"]["hidden"])
    gate.load_state_dict(torch.load(p / "confidence_gate.pt", map_location="cpu"))
    gate.eval()

    demo_df = pd.read_parquet(p / "sample_deals.parquet")
    return meta, rf, tab_scaler, text_scaler, concat, gate, demo_df


meta, rf, tab_scaler, text_scaler, concat, gate, demo_df = load_bundle()
TAB_COLS  = meta["tab_cols"]
TEXT_COLS = meta["text_cols"]

# ──────────────────────────────────────────────────────────────────────────────
# 2. The Phase 3 inference function  (matches notebook 07 exactly)
# ──────────────────────────────────────────────────────────────────────────────
def predict_hybrid(tab_row: np.ndarray, text_row: np.ndarray):
    """tab_row: (75,) raw, text_row: (768,) raw  ->  dict with all signals."""
    X_tab  = tab_row.reshape(1, -1).astype(np.float32)
    X_text = text_row.reshape(1, -1).astype(np.float32)

    # Symbolic branch — RF on RAW tab features (RF is scale-invariant)
    p_rf, s2_rf = rf_prob_and_uncertainty(rf, X_tab)

    # Neural branch — ConcatFusion on SCALED features
    X_tab_s  = tab_scaler.transform(X_tab).astype(np.float32)
    X_text_s = text_scaler.transform(X_text).astype(np.float32)
    with torch.no_grad():
        logit = concat(torch.tensor(X_tab_s), torch.tensor(X_text_s)).squeeze(-1)
        p_dl  = torch.sigmoid(logit).numpy()

    # Gate
    with torch.no_grad():
        p_hyb, w = gate(torch.tensor(p_rf), torch.tensor(s2_rf), torch.tensor(p_dl))
    p_hyb = p_hyb.numpy()
    w     = w.numpy()

    return {
        "p_rf":     float(p_rf[0]),
        "s2_rf":    float(s2_rf[0]),
        "p_dl":     float(p_dl[0]),
        "w":        float(w[0]),
        "p_hybrid": float(p_hyb[0]),
        "y_pred":   int(p_hyb[0] >= meta["decision_threshold"]),
    }

# ──────────────────────────────────────────────────────────────────────────────
# 3. UI — sidebar: pick a deal
# ──────────────────────────────────────────────────────────────────────────────
st.title("🎯 Phase 3 — Confidence-Gated Neuro-Symbolic Hybrid")
st.caption(
    "Random Forest (symbolic) + ConcatFusion (neural), fused via a learned gate. "
    f"Test F1 = **{meta['test_metrics']['hybrid_f1']:.4f}**."
)

st.sidebar.header("1. Pick a deal")
deal_idx = st.sidebar.selectbox(
    "Sample deal #",
    options=demo_df.index.tolist(),
    format_func=lambda i: (
        f"#{i:>3d} — outcome={'Won' if demo_df.loc[i,'outcome']==1 else 'Lost'}, "
        f"σ²={demo_df.loc[i,'s2_rf_cached']:.3f}"
    ),
)

base_tab  = demo_df.loc[deal_idx, TAB_COLS].values.astype(np.float32)
base_text = demo_df.loc[deal_idx, TEXT_COLS].values.astype(np.float32)
true_y    = int(demo_df.loc[deal_idx, "outcome"])

st.sidebar.header("2. Tweak features")
st.sidebar.caption("Edit a few features to see how the gate re-routes the prediction.")

# Pick 5–10 of the most interpretable tab features. Replace this list with the
# ones that look most intuitive in your dataset.
EDITABLE = TAB_COLS[:8]   # quick default — refine after looking at the columns
edited_tab = base_tab.copy()
for col in EDITABLE:
    j = TAB_COLS.index(col)
    lo = float(demo_df[col].min())
    hi = float(demo_df[col].max())
    edited_tab[j] = st.sidebar.slider(
        col, min_value=lo, max_value=hi, value=float(base_tab[j])
    )

# ──────────────────────────────────────────────────────────────────────────────
# 4. Live prediction
# ──────────────────────────────────────────────────────────────────────────────
res = predict_hybrid(edited_tab, base_text)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("P_RF",      f"{res['p_rf']:.4f}")
c2.metric("σ²_RF",     f"{res['s2_rf']:.4f}")
c3.metric("P_DL",      f"{res['p_dl']:.4f}")
c4.metric("Gate w",    f"{res['w']:.4f}")
c5.metric("P_hybrid",  f"{res['p_hybrid']:.4f}")

verdict = "✅ Won" if res["y_pred"] == 1 else "❌ Lost"
truth   = "Won"    if true_y       == 1 else "Lost"
st.subheader(f"Prediction: {verdict}    ·    Ground truth: {truth}")

# Gate position bar
st.progress(res["w"], text=f"Gate trust:  {(1-res['w'])*100:.0f}% RF  ←→  {res['w']*100:.0f}% DL")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Scatter — gate routing across all 200 demo deals
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Where this deal sits in the gate's routing landscape")

# Pre-compute w for every demo row using the cached p_rf/s2/p_dl values
demo_df_plot = demo_df.copy()
with torch.no_grad():
    _, w_all = gate(
        torch.tensor(demo_df["p_rf_cached"].values.astype(np.float32)),
        torch.tensor(demo_df["s2_rf_cached"].values.astype(np.float32)),
        torch.tensor(demo_df["p_dl_cached"].values.astype(np.float32)),
    )
demo_df_plot["w_cached"] = w_all.numpy()
demo_df_plot["outcome_label"] = demo_df_plot["outcome"].map({0: "Lost", 1: "Won"})

fig = px.scatter(
    demo_df_plot, x="s2_rf_cached", y="w_cached",
    color="outcome_label",
    labels={"s2_rf_cached": "RF tree variance σ² (epistemic uncertainty)",
            "w_cached":     "Gate weight w (trust in DL)"},
    title="Gate routing across the 200 demo deals",
)
fig.add_scatter(
    x=[demo_df.loc[deal_idx, "s2_rf_cached"]],
    y=[demo_df_plot.loc[deal_idx, "w_cached"]],
    mode="markers", marker=dict(size=18, color="black", symbol="x"),
    name="THIS deal",
)
st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 6. About panel
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("ℹ️  About this model"):
    st.markdown(f"""
- **Architecture:** RF (300 trees) + ConcatFusion (DistilBERT [CLS] + tab) fused
  by a learned gate `w = σ(MLP([P_RF, σ²_RF, P_DL]))`.
- **Stacking protocol:** base models trained on train split; gate trained on
  val-split predictions; metrics reported on the held-out test split.
- **Test metrics:**
  · RF F1 = {meta['test_metrics']['rf_f1']:.4f}
  · ConcatFusion F1 = {meta['test_metrics']['concat_f1']:.4f}
  · **Hybrid F1 = {meta['test_metrics']['hybrid_f1']:.4f}** (best)
- **Why a gate:** when the 300 RF trees disagree (high σ²), the deal is
  ambiguous for symbolic reasoning — the gate routes trust toward the
  neural text branch.
""")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Validation sanity check (can be commented out in production)
# ──────────────────────────────────────────────────────────────────────────────
sample = demo_df.iloc[0]
res_check = predict_hybrid(
    sample[TAB_COLS].values.astype(np.float32),
    sample[TEXT_COLS].values.astype(np.float32),
)
# Uncomment for sanity check:
# print("RF prob match:",  np.isclose(res_check["p_rf"],  sample["p_rf_cached"], atol=1e-5))
# print("RF s²  match:",   np.isclose(res_check["s2_rf"], sample["s2_rf_cached"], atol=1e-5))
# print("DL prob match:",  np.isclose(res_check["p_dl"],  sample["p_dl_cached"], atol=1e-5))
