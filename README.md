# Predictive Sales Analytics Engine

## 1. Project Title & Overview
**Project Name:** Predictive Sales Analytics Engine 

**Phase 3 Live Demo:** [https://hiddenlayers-predictive-sales.streamlit.app/](https://hiddenlayers-predictive-sales.streamlit.app/)

**Description:** An AI-driven machine learning system that predicts B2B SaaS sales deal outcomes by intelligently fusing structured CRM data with unstructured conversational dialogue. This predictive engine empowers sales organizations to forecast pipeline accuracy and capture missed revenue opportunities based on data-driven signals rather than human intuition.

**Problem Statement:** Accurately predicting binary sales outcomes (Won/Lost) by modelling the complex interplay between unstructured text data (conversation transcripts, emails) and tabular business constraints (deal size, engagement metrics, stage duration).

## 2. Motivation
- **Business Impact:** Sales organizations frequently rely on gut feeling or simple heuristics to forecast deal success. Accurately predicting these outcomes algorithmically optimizes resource allocation, provides structurally reliable revenue forecasts, and explicitly guides representatives on high-value interactions.
- **Why Combine Text + Tabular Data:** Tabular data (e.g., deal size, timeline) provides the rigid structural constraints of the deal, while text data (e.g., transcripts) reveals the underlying sentiment, nuanced objections, and conversational engagement levels. Fusing both modalities theoretically yields a richer, highly predictive representation of the customer's true purchasing intent.

## 3. Dataset
- **Source:** [DeepMostInnovations/saas-sales-conversations](https://huggingface.co/datasets/DeepMostInnovations/saas-sales-conversations)
- **Description:** A comprehensive open-source dataset containing B2B SaaS sales interactions between representatives and prospective clients, definitively paired with matching CRM metadata.
- **Key Features:**
  - **Text:** Conversation transcripts, dialogue exchanges, and emails.
  - **Numerical & Categorical:** User engagement metrics, interaction length, deal stage, and company size.
- **Target Variable:** Deal Outcome (e.g., Won / Lost).

## 4. Approach: An End-to-End Modular Pipeline

The project is delivered in three course phases. **Phase 1** establishes a classical + shallow-DL baseline; **Phase 2** replaces the PCA-compressed text pathway with a frozen DistilBERT encoder and introduces a novel Gated Cross-Modal Attention (GCMA) fusion block; **Phase 3** fuses the two best models into a differentiable neuro-symbolic hybrid gated by RF epistemic uncertainty.

### Phase 1 — Classical ML + Shallow Deep Learning

- **EDA & Data Quality:** Exhaustive exploratory analysis of feature distributions, structural relationships, and temporal engagement variations; all features screened for target leakage.
- **Feature Engineering:** TF-IDF + Truncated SVD and MiniLM neural embeddings, reduced via PCA-20 to dense numerical representations. Composite interaction features (sentiment trajectory, engagement velocity) capture dialogue evolution.
- **Modeling:** Class-Weighted Random Forest, Tabular MLP, Text-only MLP on PCA embeddings, Domain-Modified DL (threshold tuning + weighted loss), and a Hybrid Late-Fusion ensemble.
- **Key finding:** PCA-20 was the bottleneck — the text pathway discarded most of the language signal, which is what motivated Phase 2.

### Phase 3 — Confidence-Gated Neuro-Symbolic Hybrid

- **Core idea:** The Random Forest computes *epistemic uncertainty* (σ²) as the variance of class-1 probability across its 300 trees. When trees disagree (high σ²), the sample is ambiguous for symbolic reasoning — so a learned gate routes trust to the neural text branch instead.
- **Fusion rule (differentiable):** `P_hybrid = w · P_DL + (1−w) · P_RF` where `w = σ( MLP([P_RF, σ²_RF, P_DL]) )`.  The gate MLP (3 → 32 → 16 → 1) is trained end-to-end on held-out val-split predictions (stacking protocol — no leakage).
- **Ablation proof of synergy:** On RF-uncertain test samples (σ² above median) the Hybrid achieves F1 = 0.9606 vs. RF = 0.9518 and DL = 0.9593. The hybrid fixes 17 errors from either branch and introduces **zero new errors** (36.2% error recovery rate).
- **Architecture diagram:** [`docs/architecture_diagram.png`](docs/architecture_diagram.png)
- **Notebook:** [`notebooks/07_phase3_hybrid.ipynb`](notebooks/07_phase3_hybrid.ipynb)
- **Interactive Dashboard:** [Streamlit Live Demo](https://hiddenlayers-predictive-sales.streamlit.app/) — A web app visualizing the Phase 3 Hybrid model's real-time gating mechanism, comparing RF epistemic uncertainty against Neural (DL) probabilities.

### Phase 2 — Transformer Text Encoding + Gated Cross-Modal Attention

- **Frozen DistilBERT Text Encoder** ([`04_phase2_text_encoding.ipynb`](notebooks/04_phase2_text_encoding.ipynb)): `distilbert-base-uncased` with `[CLS]` pooling produces a 768-dim per-conversation representation. No fine-tuning — the encoder is used as a feature extractor, preserving the full semantic content that PCA-20 previously destroyed.
- **Dual-Stream Models** ([`05_phase2_models.ipynb`](notebooks/05_phase2_models.ipynb)): Tabular MLP, Text-only MLP over DistilBERT features, a Concat-Fusion baseline, and the novel **GCMAFusion** — a tabular branch queries K learned text views via scaled dot-product attention, then a learned sigmoid gate decides per-sample how much to trust the text branch versus the tabular projection.
- **Ablation & Interpretability** ([`06_phase2_validation.ipynb`](notebooks/06_phase2_validation.ipynb)): Component-wise ablations (attention off, gate off, `K=4`) plus three data-efficiency levers — MixUp (VRM), Curriculum Learning by `conversation_length`, and SSL pretraining via Masked Tabular Feature Modeling. Interpretability artifacts include attention heatmaps over the 8 text views, gate-value distributions per outcome class, and per-sample Won/Lost case studies.

## 5. Final Results & Model Benchmarks

### Phase 1 — Baselines and Late Fusion

| Architecture Type | Feature Blocks | F1-Score | Accuracy |
|---------|---------|----------|----------|
| **Baseline ML (RF)** | Tabular Focus | 0.9638 | 0.9638 |
| **Tabular DL (MLP)** | Tabular Focus | 0.9655 | 0.9656 |
| **Standard DL (No Mod)** | PCA Embeddings | 0.5332 | 0.5119 |
| **Domain-Modified DL** | PCA Embeddings | 0.6630 | 0.4962 |
| **Hybrid Late Fusion** | Tabular + PCA Vectors | **0.9710** | **0.9712** |

### Phase 2 — DistilBERT + GCMA (Test Split, seed 42)

| Model | Feature Blocks | F1-Score | Accuracy |
|---------|---------|----------|----------|
| **TextMLP (DistilBERT only)** | 768-dim `[CLS]` | 0.7048 | 0.6992 |
| **TabularMLP** | Tabular Focus | 0.9664 | 0.9667 |
| **GCMAFusion (full)** | Tabular + DistilBERT + Cross-Modal Attention | 0.9688 | 0.9692 |
| **ConcatFusion** | Tabular + DistilBERT (concat) | **0.9749** | **0.9750** |

**Phase 2 ablations (ΔF1 vs. Full GCMA):** `− Attention` +0.0060, `+ MixUp` +0.0051, `+ SSL pretrain` +0.0043, `K=4` +0.0040, `+ Curriculum` +0.0017, `− Gate` +0.0002. The small deltas indicate the fusion is saturated by the tabular signal on this dataset; the text branch alone recovers F1 ≈ 0.70 (vs. 0.55 in Phase 1), confirming the PCA-20 bottleneck was the Phase 1 ceiling. Interpretability outputs (attention heatmap, gate distributions) are produced inline in [`06_phase2_validation.ipynb`](notebooks/06_phase2_validation.ipynb).

### Phase 3 — Comparison Table (70/15/15 split, seed=42)

| Phase | Model | F1-Score | Accuracy |
|-------|-------|----------|----------|
| Phase 3 | **Confidence-Gated Hybrid** | **0.9749** | **0.9750** |
| Phase 2 | ConcatFusion | 0.9740 | 0.9742 |
| Phase 2 | GCMAFusion | 0.9688 | 0.9692 |
| Phase 2 | TabularMLP | 0.9664 | 0.9667 |
| Phase 3 | RF (symbolic baseline) | 0.9691 | 0.9692 |
| Phase 2 | TextMLP (DistilBERT) | 0.7048 | 0.6992 |

Hybrid gain: **ΔF1 = +0.0057 over RF** and **+0.0009 over the best Phase 2 DL model**.

## 6. Project Structure (Modular & Clean)

The codebase strictly follows industry best practices for modularity, neatly organizing data storage, exploratory logic, execution flows, and theoretical references.

```text
hiddenlayers-predictive-sales/
│
├── data/                     # Data storage — raw/ and processed/ silos
├── docs/
│   ├── architecture_diagram.png          # Phase 3 — Confidence-Gated Hybrid (publication-ready)
│   ├── gcma_architecture.png             # Phase 2 — GCMA Fusion diagram
│   └── ...                              # Literature review, writeups
├── notebooks/
│   ├── 01_eda_saas_sales_conversations.ipynb
│   ├── 02_feature_engineering.ipynb                # Phase 1 — TF-IDF + PCA features
│   ├── 02b_feature_engineering_pretrained_embeddings.ipynb
│   ├── 03_model_application.ipynb                  # Phase 1 — RF, MLPs, Hybrid Late Fusion
│   ├── 03b_model_application_pretrained_embeddings.ipynb
│   ├── 04_phase2_text_encoding.ipynb               # Phase 2 — frozen DistilBERT encoding
│   ├── 05_phase2_models.ipynb                      # Phase 2 — TextMLP, TabularMLP, Concat, GCMAFusion
│   ├── 06_phase2_validation.ipynb                  # Phase 2 — ablations, attention/gate interpretability
│   └── 07_phase3_hybrid.ipynb                      # Phase 3 — Confidence-Gated Neuro-Symbolic Hybrid
├── streamlit/                # Phase 3 Interactive Web Dashboard
├── .github/workflows/ci.yml  # GitHub Actions CI — import smoke test + Docker build
├── presentations/            # Phase 1–3 slide decks / demo PDFs
├── reports/                  # LaTeX + PDF reports for each phase
├── research_papers/          # Literature review references
├── Dockerfile                # Containerised environment (python:3.10-slim)
├── setup.sh                  # Turn-key local setup script
├── requirements.txt          # Python dependencies
└── README.md
```

## 7. Installation & Reproducible Setup

All framework backends (PyTorch, Scikit-Learn) are configured with deterministic algorithms and a fixed random seed (`seed=42`).

**Option A — automated setup script (recommended)**
```bash
git clone https://github.com/nandu-99/hiddenlayers-predictive-sales
cd hiddenlayers-predictive-sales
bash setup.sh           # creates .venv, installs deps, runs import smoke test
source .venv/bin/activate
```

**Option B — Docker (fully isolated)**
```bash
docker build -t hiddenlayers-predictive-sales .
docker run --rm -it hiddenlayers-predictive-sales bash
```

**Option C — manual**
```bash
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 8. How to Run

Execute the notebooks in order. Phase 1 (01 → 03) produces the classical + shallow-DL baselines; Phase 2 (04 → 06) adds the transformer encoder and GCMA fusion.

**Phase 1 — Baselines**
1. **Data Exploration:** [`notebooks/01_eda_saas_sales_conversations.ipynb`](notebooks/01_eda_saas_sales_conversations.ipynb) — visualize data quality, distribution bounds, and target representation.
2. **Feature Engineering:** [`notebooks/02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb) — generate PCA-compressed text embeddings and composite features.
3. **Model Training & Comparison:** [`notebooks/03_model_application.ipynb`](notebooks/03_model_application.ipynb) — benchmark Random Forest, MLP, and Hybrid Late-Fusion architectures.

**Phase 2 — Transformer + GCMA**

4. **Text Encoding:** [`notebooks/04_phase2_text_encoding.ipynb`](notebooks/04_phase2_text_encoding.ipynb) — produce frozen DistilBERT `[CLS]` features (8000 × 768) and cache to `saas_features_pretrained.parquet`. GPU recommended (~3 min on a T4).
5. **Model Training:** [`notebooks/05_phase2_models.ipynb`](notebooks/05_phase2_models.ipynb) — train TextMLP, TabularMLP, ConcatFusion, and GCMAFusion; writes `phase2_results.csv`.
6. **Validation & Ablations:** [`notebooks/06_phase2_validation.ipynb`](notebooks/06_phase2_validation.ipynb) — component ablations, MixUp / Curriculum / SSL variants, confusion matrices, attention heatmaps; writes `phase2_ablation_results.csv`.

**Phase 3 — Confidence-Gated Neuro-Symbolic Hybrid**

7. **Hybrid Training & Ablations:** [`notebooks/07_phase3_hybrid.ipynb`](notebooks/07_phase3_hybrid.ipynb) — retrains RF + ConcatFusion on identical split, trains ConfidenceGatedHybrid meta-learner on val predictions (stacking), runs uncertain-zone ablation and error decomposition. GPU recommended (~5 min on a T4). Upload `saas_features_pretrained.parquet` to `/content/` before running in Colab/Kaggle.
8. **Interactive Dashboard:** Run the Streamlit app locally via `streamlit run streamlit/app.py` or view the [Live Demo](https://hiddenlayers-predictive-sales.streamlit.app/) to interactively explore the Phase 3 gate behavior.

## 9. Development Methodology & Code Quality

This repository adheres to strict, industry-ready development standards designed to ensure stable scaling and team collaborative transparency:
- **Consistent & Meaningful Version Control:** Continuous tracking driven by highly logical, semantic Git commits showing a steady, methodical progression from setup to final outcomes. 
- **Clean Modularity:** Machine learning components are heavily abstracted and directory-enforced. Separating uncleaned inputs from engineered outputs protects against data contamination and subsequent target leakage.
- **Reproducibility Focus:** Documentation and scripts are constructed such that any developer can check out this code, run `pip install`, and train identically accurate models seamlessly over time.

## 10. Limitations & Future Work
- **Domain Data Bias:** Deep models may inadvertently learn the specific conversational parlance of highly represented representatives natively within the initial dataset bounds.
- **Leakage Prevention:** Real-world API deployments require rigorous time-based cutoff mechanisms explicitly constructed to prevent late-stage CRM duration knowledge leaking into early prediction horizons.
- **Future Improvements:** Introduce large multi-industry deployment datasets to boost generalization, and experiment with zero-shot sequence inferences via Large Language Models (LLMs) native API calls.

## 11. Contributors
- **Vivekananda** (230077)
- **Manasa** (230078)
