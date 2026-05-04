import json, os, shutil, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).parent))
from models import ConcatFusion, ConfidenceGatedHybrid, rf_prob_and_uncertainty

st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Kaggle credentials when running on Streamlit Cloud
if "kaggle" in st.secrets:
    os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
    os.environ["KAGGLE_KEY"]      = st.secrets["kaggle"]["key"]

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
[data-testid="stAppViewContainer"] { background: #ffffff; }
[data-testid="block-container"] { padding: 2.5rem 3rem 3rem 3rem; max-width: 1200px; }
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] { background: #f3f4f6; border-right: 1px solid #e5e7eb; }
[data-testid="stSidebar"] > div:first-child { padding: 1.75rem 1.25rem; }

/* Expander — styled as section box */
[data-testid="stExpander"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    background: #ffffff;
    margin-bottom: 1rem;
    overflow: hidden;
    box-shadow: none !important;
}
[data-testid="stExpander"] details summary {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #374151 !important;
    padding: 1rem 1.25rem !important;
    background: #ffffff;
}
[data-testid="stExpander"] details summary:hover { background: #f9fafb; }
[data-testid="stExpanderDetails"] {
    padding: 1rem 1.5rem 1.5rem !important;
    border-top: 1px solid #f3f4f6;
}

/* Metric cards */
.metric-grid-5 { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.75rem; }
.metric-grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; }
.mc { background: #f3f4f6; border-radius: 10px; padding: 1rem 1.125rem; }
.mc-label { font-size: 0.78rem; color: #6b7280; margin-bottom: 0.2rem; }
.mc-value {
    font-size: 1.7rem; font-weight: 700; color: #0f172a;
    font-variant-numeric: tabular-nums; line-height: 1.2;
}
.mc-sub        { font-size: 0.72rem; color: #16a34a; margin-top: 0.3rem; }
.mc-sub.muted  { color: #6b7280; }
.mc-sub.blue   { color: #2563eb; }

/* Badges */
.badge { display: inline-block; padding: .3rem 1rem; border-radius: 9999px; font-size: .875rem; font-weight: 600; }
.badge-won  { background: #dcfce7; color: #15803d; }
.badge-lost { background: #fee2e2; color: #b91c1c; }

/* Gate bar */
.gate-labels { display: flex; justify-content: space-between; font-size: .78rem; color: #6b7280; margin-bottom: .35rem; }
.gate-track  { background: #e5e7eb; border-radius: 9999px; height: 8px; overflow: hidden; }
.gate-fill   { height: 100%; border-radius: 9999px; background: #2563eb; }

/* Sidebar labels */
.sb-label { font-size: .78rem; font-weight: 600; color: #374151;
            text-transform: uppercase; letter-spacing: .05em; margin: 1.25rem 0 .4rem; }
.sb-divider { border: none; border-top: 1px solid #e5e7eb; margin: 1.25rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Bundle cache ──────────────────────────────────────────────────────────────
BUNDLE_DIR = Path(__file__).parent / "model_bundle"
BUNDLE_FILES = [
    "meta.json", "rf_model.joblib", "tab_scaler.joblib", "text_scaler.joblib",
    "concat_fusion.pt", "confidence_gate.pt", "sample_deals.parquet",
]

def _bundle_ready():
    return all((BUNDLE_DIR / f).exists() for f in BUNDLE_FILES)

@st.cache_resource(show_spinner="Loading models …")
def load_bundle():
    if _bundle_ready():
        p = BUNDLE_DIR
    else:
        import kagglehub
        with st.spinner("Downloading model bundle from Kaggle (one-time) …"):
            src = Path(kagglehub.dataset_download("vivekananda99/predictive-sales"))
        BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
        for f in BUNDLE_FILES:
            if (src / f).exists():
                shutil.copy2(src / f, BUNDLE_DIR / f)
        p = BUNDLE_DIR

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

    return meta, rf, tab_scaler, text_scaler, concat, gate, pd.read_parquet(p / "sample_deals.parquet")

meta, rf, tab_scaler, text_scaler, concat, gate, demo_df = load_bundle()
TAB_COLS  = meta["tab_cols"]
TEXT_COLS = meta["text_cols"]

# ── Inference ─────────────────────────────────────────────────────────────────
def predict_hybrid(tab_row, text_row):
    X_tab, X_text = tab_row.reshape(1,-1).astype(np.float32), text_row.reshape(1,-1).astype(np.float32)
    p_rf, s2_rf   = rf_prob_and_uncertainty(rf, X_tab)
    X_tab_s  = tab_scaler.transform(X_tab).astype(np.float32)
    X_text_s = text_scaler.transform(X_text).astype(np.float32)
    with torch.no_grad():
        p_dl = torch.sigmoid(concat(torch.tensor(X_tab_s), torch.tensor(X_text_s)).squeeze(-1)).numpy()
    with torch.no_grad():
        p_hyb, w = gate(torch.tensor(p_rf), torch.tensor(s2_rf), torch.tensor(p_dl))
    return {
        "p_rf": float(p_rf[0]), "s2_rf": float(s2_rf[0]),
        "p_dl": float(p_dl[0]), "w":     float(w.numpy()[0]),
        "p_hybrid": float(p_hyb.numpy()[0]),
        "y_pred": int(p_hyb.numpy()[0] >= meta["decision_threshold"]),
    }

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Sales Predictor")
    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Deal</div>', unsafe_allow_html=True)
    deal_idx = st.selectbox(
        "deal", options=demo_df.index.tolist(),
        format_func=lambda i: (
            f"#{i:03d}  {'Won' if demo_df.loc[i,'outcome']==1 else 'Lost'}"
            f"  σ²={demo_df.loc[i,'s2_rf_cached']:.3f}"
        ),
        label_visibility="collapsed",
    )

    base_tab  = demo_df.loc[deal_idx, TAB_COLS].values.astype(np.float32)
    base_text = demo_df.loc[deal_idx, TEXT_COLS].values.astype(np.float32)
    true_y    = int(demo_df.loc[deal_idx, "outcome"])

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Feature Tweaks</div>', unsafe_allow_html=True)
    st.caption("Adjust values to see the gate re-route in real time.")

    EDITABLE   = TAB_COLS[:8]
    edited_tab = base_tab.copy()
    for col in EDITABLE:
        j  = TAB_COLS.index(col)
        lo, hi = float(demo_df[col].min()), float(demo_df[col].max())
        edited_tab[j] = st.slider(col, lo, hi, float(base_tab[j]))

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:2.25rem;font-weight:700;color:#0f172a;line-height:1.2">'
    'Sales Prediction Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<hr style="border:none;border-top:2.5px solid #2563eb;margin:.6rem 0 1.75rem 0">',
    unsafe_allow_html=True,
)

# ── Live prediction ───────────────────────────────────────────────────────────
res = predict_hybrid(edited_tab, base_text)
tm  = meta["test_metrics"]

# ── Section 1 — Prediction Signals (collapsible) ──────────────────────────────
with st.expander("Prediction Signals", expanded=True):
    st.markdown(f"""
<div class="metric-grid-5">
  <div class="mc"><div class="mc-label">Random Forest</div><div class="mc-value">{res['p_rf']:.3f}</div><div class="mc-sub">↑ Win probability</div></div>
  <div class="mc"><div class="mc-label">RF Uncertainty σ²</div><div class="mc-value">{res['s2_rf']:.4f}</div><div class="mc-sub muted">↑ Tree variance</div></div>
  <div class="mc"><div class="mc-label">Neural (DL)</div><div class="mc-value">{res['p_dl']:.3f}</div><div class="mc-sub blue">↑ Win probability</div></div>
  <div class="mc"><div class="mc-label">Gate Weight w</div><div class="mc-value">{res['w']:.3f}</div><div class="mc-sub muted">↑ Trust in DL branch</div></div>
  <div class="mc"><div class="mc-label">Hybrid Output</div><div class="mc-value">{res['p_hybrid']:.3f}</div><div class="mc-sub">↑ Final prediction</div></div>
</div>
""", unsafe_allow_html=True)

# ── Section 2 — Deal Outcome (collapsible) ────────────────────────────────────
pred_badge  = '<span class="badge badge-won">Won</span>'  if res["y_pred"] == 1 else '<span class="badge badge-lost">Lost</span>'
truth_badge = '<span class="badge badge-won">Won</span>'  if true_y == 1        else '<span class="badge badge-lost">Lost</span>'
rf_pct, dl_pct, gate_w = int((1-res["w"])*100), int(res["w"]*100), res["w"]*100

with st.expander("Deal Outcome", expanded=True):
    st.markdown(f"""
<div style="display:flex;gap:3rem;margin-bottom:1.25rem">
  <div>
    <div style="font-size:.78rem;color:#6b7280;margin-bottom:.35rem">Prediction</div>
    {pred_badge}
  </div>
  <div>
    <div style="font-size:.78rem;color:#6b7280;margin-bottom:.35rem">Ground Truth</div>
    {truth_badge}
  </div>
  <div>
    <div style="font-size:.78rem;color:#6b7280;margin-bottom:.35rem">Threshold</div>
    <span style="font-size:.95rem;font-weight:600;color:#374151">{meta['decision_threshold']:.2f}</span>
  </div>
</div>
<div style="font-size:.78rem;color:#6b7280;margin-bottom:.4rem">Gate routing — RF {rf_pct}% ←→ DL {dl_pct}%</div>
<div class="gate-track"><div class="gate-fill" style="width:{gate_w:.1f}%"></div></div>
<div class="gate-labels"><span>Random Forest</span><span>Neural (DL)</span></div>
""", unsafe_allow_html=True)

# ── Section 3 — Model Performance (collapsible) ───────────────────────────────
with st.expander("Model Performance (held-out test set)", expanded=True):
    st.markdown(f"""
<div class="metric-grid-4">
  <div class="mc"><div class="mc-label">RF F1</div><div class="mc-value">{tm['rf_f1']:.4f}</div><div class="mc-sub muted">↑ Random Forest</div></div>
  <div class="mc"><div class="mc-label">ConcatFusion F1</div><div class="mc-value">{tm['concat_f1']:.4f}</div><div class="mc-sub blue">↑ Neural branch</div></div>
  <div class="mc"><div class="mc-label">Hybrid F1</div><div class="mc-value">{tm['hybrid_f1']:.4f}</div><div class="mc-sub">↑ Best overall</div></div>
  <div class="mc"><div class="mc-label">Decision Threshold</div><div class="mc-value">{meta['decision_threshold']:.2f}</div><div class="mc-sub muted">↑ Calibrated</div></div>
</div>
""", unsafe_allow_html=True)

# ── Win Probability Comparison chart ─────────────────────────────────────────
st.markdown(
    '<div style="font-size:1.35rem;font-weight:700;color:#0f172a;margin:1.75rem 0 0.25rem">Win Probability Comparison</div>',
    unsafe_allow_html=True,
)
st.caption("How confident is each model that this deal will be won? The dashed line is the decision threshold.")

threshold = meta["decision_threshold"]
models    = ["Random Forest", "Neural (DL)", "Hybrid"]
probs     = [res["p_rf"], res["p_dl"], res["p_hybrid"]]
colors    = [
    "#2563eb" if p >= threshold else "#94a3b8"
    for p in probs
]
# Hybrid always gets a distinct colour to stand out
colors[2] = "#16a34a" if res["p_hybrid"] >= threshold else "#ef4444"

fig = go.Figure()

fig.add_trace(go.Bar(
    x=models,
    y=probs,
    marker_color=colors,
    marker_line_width=0,
    width=0.45,
    text=[f"{p:.3f}" for p in probs],
    textposition="outside",
    textfont=dict(size=13, color="#374151", family="sans-serif"),
    hovertemplate="%{x}<br>Win probability: %{y:.4f}<extra></extra>",
))

# Decision threshold line
fig.add_shape(
    type="line",
    x0=-0.5, x1=2.5,
    y0=threshold, y1=threshold,
    line=dict(color="#e11d48", width=1.5, dash="dash"),
)
fig.add_annotation(
    x=2.5, y=threshold,
    text=f"  threshold {threshold:.2f}",
    showarrow=False,
    font=dict(size=11, color="#e11d48"),
    xanchor="left",
)

fig.update_layout(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    margin=dict(l=0, r=60, t=30, b=0),
    yaxis=dict(
        range=[0, 1.05],
        tickformat=".2f",
        showgrid=True, gridcolor="#f3f4f6",
        zeroline=False,
        title="Win probability",
        title_font=dict(size=12, color="#6b7280"),
        tickfont=dict(size=11, color="#6b7280"),
    ),
    xaxis=dict(
        tickfont=dict(size=13, color="#374151"),
        showgrid=False,
    ),
    showlegend=False,
    height=440,
    font=dict(family="sans-serif"),
)

st.plotly_chart(fig, use_container_width=True)

# ── Architecture note ─────────────────────────────────────────────────────────
with st.expander("Architecture & methodology"):
    st.markdown(f"""
**Model:** RF (300 trees) + ConcatFusion (DistilBERT [CLS] + tabular features),
fused by a learned gate `w = σ(MLP([P_RF, σ²_RF, P_DL]))`.

**Stacking protocol:** base models trained on the training split; gate trained on
val-split predictions only; metrics reported on the held-out test split.

**Why a gate?** When the 300 RF trees disagree (high σ²), the deal is ambiguous
for symbolic reasoning — the gate shifts trust toward the neural text branch.
""")
