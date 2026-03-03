# Predictive Sales Analytics Engine

## Problem Statement

Sales organizations struggle to make informed decisions about deal outcomes due to the complex interplay between structured CRM data (deal size, stage, duration) and unstructured conversational data (call transcripts, emails). Without predictive tooling, sales teams rely on gut feeling, leading to poor pipeline forecasting and missed revenue opportunities. This project addresses the need for an AI-driven system that fuses text and tabular business data to predict deal outcomes accurately.

## Project Objectives

- Build predictive models that combine **text and numerical business data** (text–tabular fusion)
- Progress through four modeling tiers: Baseline ML → Classical NLP → Deep Learning → Hybrid/Ensemble
- Evaluate models using **business-focused metrics** (F1-score, AUC-ROC, Precision-Recall)
- Explore **explainability** to surface actionable sales insights from model predictions
- Analyze potential biases in predictions across deal segments or sales rep groups

## Project Overview

This project is structured as a multi-phase machine learning pipeline applied to sales conversation and CRM data:

| Phase | Approach | Description |
|-------|----------|-------------|
| **Baseline ML** | Text stats + numerical features | Simple word count / TF-IDF statistics combined with structured CRM features |
| **Advanced ML** | Classical NLP + ML classifiers | Richer NLP representations (TF-IDF, n-grams) with SVM, Random Forest classifiers |
| **Deep Learning** | Neural language models | End-to-end models (LSTM, BERT) for sequence-level outcome prediction |
| **Hybrid / Edge** | Ensemble & fusion models | Combining multiple model outputs for improved accuracy and explainability |

The project follows a standard ML workflow: data collection → EDA → feature engineering → model training → evaluation → explainability analysis.

## Repository Structure

```
hiddenlayers-predictive-sales/
│
├── data/
│   ├── raw/                  # Original, unprocessed datasets
│   └── processed/            # Cleaned and feature-engineered datasets
│
├── notebooks/                # Jupyter notebooks for EDA, modeling, and analysis
│
├── reports/                  # Generated reports, figures, and evaluation summaries
│
├── docs/                     # Project documentation and references
│
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

## Team Members

| Name | Role |
|------|------|
| Vivekananda | ML Engineering, Model Development |
| Manasa | Data Analysis, Feature Engineering |
