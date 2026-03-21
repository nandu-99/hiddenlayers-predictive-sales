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

### Phase 1: Exploratory Data Analysis & Data Quality
- Conducted exhaustive exploratory data analysis to understand feature distributions, structural relationships, and temporal engagement variations.
- Screened all features for target leakage to ensure no posterior proxies of the final deal state were inadvertently included.

### Phase 2: Feature Engineering
- **Text Embeddings:** Applied TF-IDF coupled with Truncated SVD, as well as neural embeddings (e.g., BERT-based), subsequently reduced via PCA to extract meaningful, dense numerical representations from transcripts.
- **Interaction Features:** Crafted composite metrics (e.g., sentiment trajectory, engagement velocity) accurately capturing the evolution of the dialogue constraints.

### Phase 3: Modeling & Ablation Studies
- **Baseline ML:** Implemented robust classical models (Class-Weighted Random Forest) utilizing explicitly engineered tabular parameters.
- **Tabular DL:** A Multi-Layer Perceptron (MLP) trained exclusively on the tabular dataset.
- **Standard Deep Learning:** Neural architectures built solely on standardized PCA-compressed text embeddings to capture semantic similarity.
- **Domain-Modified DL:** Deep Learning models dynamically tuned with Precision-Recall optimization thresholds and weighted training loss functions to confront severe class imbalances natively.
- **Hybrid Fusion (Ensemble):** A sophisticated network seamlessly synergizing the structured robustness of tree-based inputs alongside the nuanced comprehension of Deep Learning embeddings.

## 5. Final Results & Model Benchmarks

Proving mathematical utility strictly through empirical separation, the ablation study definitively shows that tabular behavioral features carry the overwhelming majority of predictive signal for conversion outcomes in this specific SaaS domain, while the Hybrid Fusion architecture successfully balances both modalities.

| Architecture Type | Feature Blocks | F1-Score | Accuracy |
|---------|---------|----------|----------|
| **Baseline ML (RF)** | Tabular Focus | 0.9638 | 0.9638 |
| **Tabular DL (MLP)** | Tabular Focus | 0.9655 | 0.9656 |
| **Standard DL (No Mod)** | PCA Embeddings | 0.5332 | 0.5119 |
| **Domain-Modified DL** | PCA Embeddings | 0.6630 | 0.4962 |
| **Hybrid Fusion (Full Integration)** | Tabular Logics + Vectors | **0.9710** | **0.9712** |

*(Metrics reflect aggregate validation benchmarking as derived directly from the application notebook)*

## 6. Project Structure (Modular & Clean)

The codebase strictly follows industry best practices for modularity, neatly organizing data storage, exploratory logic, execution flows, and theoretical references.

```text
hiddenlayers-predictive-sales/
│
├── data/                     # Data storage separated into 'raw/' and 'processed/' silos
├── docs/                     # Supplemental operational documentation
├── notebooks/                # Sequential files: EDA, Feature Engineering, Model Application
├── reports/                  # Generated reports on outcomes and performance metrics
├── research_papers/          # Theoretical literature review and methodology references
├── requirements.txt          # Locked Python environments and required dependencies
└── README.md                 # Primary system documentation
```

## 7. Installation & Reproducible Setup

To guarantee full reproducibility, all framework backends (PyTorch, Scikit-Learn) are configured utilizing deterministic algorithms with a fixed random seed (`seed=42`).

```bash
# 1. Clone the repository
git clone <repo_url>
cd hiddenlayers-predictive-sales

# 2. Create and activate a virtual Python environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install project dependencies
pip install -r requirements.txt
```

## 8. How to Run

Follow these logically ordered workflow steps to execute the robust data pipeline from ingestion to interpretation:

1. **Data Exploration:** Open and execute `notebooks/01_eda_saas_sales_conversations.ipynb` to visualize data quality, distribution bounds, and target representation.
2. **Feature Engineering:** Execute `notebooks/02_feature_engineering.ipynb` to generate PCA embeddings, establish composite text features, and compile output vectors.
3. **Model Training & Comparison:** Open `notebooks/03_model_application.ipynb` to benchmark Baseline, DL, and Hybrid architectures, validating utility via the rigorous ablation study.

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
