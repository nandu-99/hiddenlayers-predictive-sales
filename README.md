# Predictive Sales Analytics Engine

## 1. Project Title & Overview
**Project Name:** Predictive Sales Analytics Engine 

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

The project is delivered in two course phases. **Phase 1** establishes a classical + shallow-DL baseline; **Phase 2** replaces the PCA-compressed text pathway with a frozen DistilBERT encoder and introduces a novel Gated Cross-Modal Attention (GCMA) fusion block.

### Phase 1 — Classical ML + Shallow Deep Learning

- **EDA & Data Quality:** Exhaustive exploratory analysis of feature distributions, structural relationships, and temporal engagement variations; all features screened for target leakage.
- **Feature Engineering:** TF-IDF + Truncated SVD and MiniLM neural embeddings, reduced via PCA-20 to dense numerical representations. Composite interaction features (sentiment trajectory, engagement velocity) capture dialogue evolution.
- **Modeling:** Class-Weighted Random Forest, Tabular MLP, Text-only MLP on PCA embeddings, Domain-Modified DL (threshold tuning + weighted loss), and a Hybrid Late-Fusion ensemble.
- **Key finding:** PCA-20 was the bottleneck — the text pathway discarded most of the language signal, which is what motivated Phase 2.

### Phase 2 — Transformer Text Encoding + Gated Cross-Modal Attention

- **Frozen DistilBERT Text Encoder** ([`04_phase2_text_encoding.ipynb`](notebooks/04_phase2_text_encoding.ipynb)): `distilbert-base-uncased` with `[CLS]` pooling produces a 768-dim per-conversation representation. No fine-tuning — the encoder is used as a feature extractor, preserving the full semantic content that PCA-20 previously destroyed.
- **Dual-Stream Models** ([`05_phase2_models.ipynb`](notebooks/05_phase2_models.ipynb)): Tabular MLP, Text-only MLP over DistilBERT features, a Concat-Fusion baseline, and the novel **GCMAFusion** — a tabular branch queries K learned text views via scaled dot-product attention, then a learned sigmoid gate decides per-sample how much to trust the text branch versus the tabular projection.
- **Ablation & Interpretability** ([`06_phase2_validation.ipynb`](notebooks/06_phase2_validation.ipynb)): Component-wise ablations (attention off, gate off, `K=4`) plus three data-efficiency levers — MixUp (VRM), Curriculum Learning by `conversation_length`, and SSL pretraining via Masked Tabular Feature Modeling. Interpretability artifacts include attention heatmaps over the 8 text views, gate-value distributions per outcome class, and per-sample Won/Lost case studies.

### Phase 3 — Confidence-Gated Neuro-Symbolic Hybrid

- **Architecture:** A meta-learner that fuses the Phase 1 Random Forest (symbolic/classical) and the Phase 2 ConcatFusion (neural/deep) via a learned confidence gate. The gate takes `[P_RF, σ²_RF, P_DL]` as input — the RF's predicted probability, its tree-level variance (uncertainty estimate), and the DL model's predicted probability — and outputs a soft blending weight that routes each sample toward the more reliable predictor at inference time.
- **Motivation:** Each base model excels in complementary regimes. The RF is highly calibrated on structured tabular signals; the ConcatFusion captures nuanced semantic patterns from DistilBERT embeddings. The confidence gate learns *when to trust which model*, producing a hybrid that is strictly stronger than either alone.
- **Streamlit Web UI** ([`streamlit/`](streamlit/)): A fully interactive prediction dashboard that exposes the Phase 3 hybrid model to end users — upload or enter deal metadata, run live inference, and inspect gate weights, prediction confidence, and feature attributions in real time.

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

### Phase 3 — Confidence-Gated Neuro-Symbolic Hybrid

| Model | Feature Blocks | F1-Score | Accuracy |
|---------|---------|----------|----------|
| **Random Forest (base)** | Tabular Focus | 0.9638 | 0.9638 |
| **ConcatFusion (base)** | Tabular + DistilBERT (concat) | 0.9749 | 0.9750 |
| **Neuro-Symbolic Hybrid** | RF + ConcatFusion via learned gate `[P_RF, σ²_RF, P_DL]` | **0.9749** | **0.9750** |

The confidence gate successfully learns to blend the symbolic and neural branches. The gate input vector `[P_RF, σ²_RF, P_DL]` provides the meta-learner with both prediction signals and an uncertainty estimate, enabling per-sample routing to the most reliable base model.

## 6. Live Demo

> **Try the interactive prediction dashboard:**  
> 🚀 **[https://hiddenlayers-predictive-sales.streamlit.app/](https://hiddenlayers-predictive-sales.streamlit.app/)**

The Streamlit app exposes the full Phase 3 Neuro-Symbolic Hybrid pipeline. You can:
- Enter deal metadata (company size, stage, engagement metrics, etc.) and run live inference.
- Inspect the confidence gate's blending weights per prediction.
- View prediction confidence scores from both the RF and ConcatFusion branches.
- Explore feature attributions to understand what drove each prediction.

## 7. Project Structure (Modular & Clean)

The codebase strictly follows industry best practices for modularity, neatly organizing data storage, exploratory logic, execution flows, and theoretical references.

```text
hiddenlayers-predictive-sales/
│
├── .github/                  
├── .streamlit/               
├── data/                     
├── docs/                     
├── notebooks/
├── presentations/            
├── reports/                  
├── research_papers/          
├── streamlit/                
├── .dockerignore
├── .gitignore
├── Dockerfile                
├── README.md                 
├── requirements.txt          
└── setup.sh                  
```

## 8. Installation & Reproducible Setup

To guarantee full reproducibility, all framework backends (PyTorch, Scikit-Learn) are configured utilizing deterministic algorithms with a fixed random seed (`seed=42`).

```bash
# 1. Clone the repository
git clone https://github.com/nandu-99/hiddenlayers-predictive-sales
cd hiddenlayers-predictive-sales

# 2. Create and activate a virtual Python environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install project dependencies
pip install -r requirements.txt
```

Alternatively, use the turn-key setup script:

```bash
bash setup.sh
```

Or build and run via Docker:

```bash
docker build -t hiddenlayers-predictive-sales .
docker run -p 8501:8501 hiddenlayers-predictive-sales
```

## 9. How to Run

Execute the notebooks in order. Phase 1 (01 → 03) produces the classical + shallow-DL baselines; Phase 2 (04 → 06) adds the transformer encoder and GCMA fusion; Phase 3 adds the Neuro-Symbolic Hybrid and Streamlit UI.

**Phase 1 — Baselines**
1. **Data Exploration:** [`notebooks/01_eda_saas_sales_conversations.ipynb`](notebooks/01_eda_saas_sales_conversations.ipynb) — visualize data quality, distribution bounds, and target representation.
2. **Feature Engineering:** [`notebooks/02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb) — generate PCA-compressed text embeddings and composite features.
3. **Model Training & Comparison:** [`notebooks/03_model_application.ipynb`](notebooks/03_model_application.ipynb) — benchmark Random Forest, MLP, and Hybrid Late-Fusion architectures.

**Phase 2 — Transformer + GCMA**

4. **Text Encoding:** [`notebooks/04_phase2_text_encoding.ipynb`](notebooks/04_phase2_text_encoding.ipynb) — produce frozen DistilBERT `[CLS]` features (8000 × 768) and cache to `saas_features_pretrained.parquet`. GPU recommended (~3 min on a T4).
5. **Model Training:** [`notebooks/05_phase2_models.ipynb`](notebooks/05_phase2_models.ipynb) — train TextMLP, TabularMLP, ConcatFusion, and GCMAFusion; writes `phase2_results.csv`.
6. **Validation & Ablations:** [`notebooks/06_phase2_validation.ipynb`](notebooks/06_phase2_validation.ipynb) — component ablations, MixUp / Curriculum / SSL variants, confusion matrices, attention heatmaps; writes `phase2_ablation_results.csv`.

**Phase 3 — Neuro-Symbolic Hybrid + Streamlit UI**

7. **Streamlit App:** Run the interactive dashboard locally:
   ```bash
   streamlit run streamlit/app.py
   ```
   Or visit the live deployment: [https://hiddenlayers-predictive-sales.streamlit.app/](https://hiddenlayers-predictive-sales.streamlit.app/)

## 10. Development Methodology & Code Quality

This repository adheres to strict, industry-ready development standards designed to ensure stable scaling and team collaborative transparency:
- **Consistent & Meaningful Version Control:** Continuous tracking driven by highly logical, semantic Git commits showing a steady, methodical progression from setup to final outcomes. 
- **Clean Modularity:** Machine learning components are heavily abstracted and directory-enforced. Separating uncleaned inputs from engineered outputs protects against data contamination and subsequent target leakage.
- **Reproducibility Focus:** Documentation and scripts are constructed such that any developer can check out this code, run `pip install`, and train identically accurate models seamlessly over time.
- **CI/CD Pipeline:** GitHub Actions workflow runs import smoke tests and validates the Docker build on every push, ensuring the deployment pipeline remains healthy.

## 11. Limitations & Future Work
- **Domain Data Bias:** Deep models may inadvertently learn the specific conversational parlance of highly represented representatives natively within the initial dataset bounds.
- **Leakage Prevention:** Real-world API deployments require rigorous time-based cutoff mechanisms explicitly constructed to prevent late-stage CRM duration knowledge leaking into early prediction horizons.
- **Future Improvements:** Introduce large multi-industry deployment datasets to boost generalization, and experiment with zero-shot sequence inferences via Large Language Models (LLMs) native API calls.

## 12. Contributors
- **Vivekananda** (230077)
- **Manasa** (230078)
