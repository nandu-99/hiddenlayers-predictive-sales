# Literature Review

## Predictive Sales Analytics — Phase 2: Conversational-Tabular Fusion

---

## 1. Introduction

Phase 1 of this project established an interpretable performance baseline using classical text statistics (BoW/TF-IDF) combined with structured CRM features, drawing on the findings of Perelman & Grinberg (2013) and the ensemble insights of Shi et al. (2021). That phase confirmed that text and tabular signals each contribute independently to conversion prediction — but treated them as parallel, non-interacting feature streams fed into a single classifier.

Phase 2 moves beyond this. The central question is no longer *whether* text and tabular data are both useful, but *how* they should interact at the architectural level to maximise predictive power. Two recent papers define the state of the art on each side of this problem and together motivate the Phase 2 architecture directly.

---

## 2. Categorised Literature Review

### 2.1 Conversation-Only Sales Prediction: SalesRLAgent

**Nandakishor M. (2025). SalesRLAgent: A Reinforcement Learning Approach for Real-Time Sales Conversion Prediction and Optimization. arXiv:2503.23303. Deepmost Innovations.**

SalesRLAgent is the most directly relevant prior work to this project. It was authored by the same team that produced the `DeepMostInnovations/saas-sales-conversations` dataset used here, uses identical Azure OpenAI 3072-dimensional embeddings, and targets the same binary conversion prediction task. Its existence as a published baseline makes the gap analysis in this review precise rather than speculative.

The paper reframes conversion prediction as a **sequential decision problem** rather than a static classification task. Each conversation turn is a state $s_t$, a conversion probability estimate is the action $a_t$, and prediction accuracy is the reward $r_t$. A PPO-trained policy network learns to track how each exchange shifts the probability of a successful outcome — capturing the pivoting, non-linear dynamics of sales dialogue that static classifiers miss.

The state representation combines recency-weighted conversation history embeddings, turn-level features (sentiment, question density, speaking time), and customer engagement signals (response latency, message length). An additional meta-learning module — inspired by MAML (Finn et al., 2017) — estimates prediction confidence by measuring the similarity of the current state to training-set states, allowing the model to express uncertainty on unfamiliar conversation patterns rather than producing overconfident wrong predictions.

**Results.** SalesRLAgent achieves **96.7% accuracy** and **AUC-ROC 0.98**, compared to 75% for the best hybrid LLM+ML baseline and 62% for GPT-4 few-shot. Inference latency is **85 ms on CPU** versus 3,450 ms for GPT-4, making genuine real-time deployment feasible. An ablation study confirms that the sequential RL framework is the highest-value component: collapsing to a basic ML baseline loses 28.7 percentage points (96.7% → 68%).

**Critical gap.** SalesRLAgent operates exclusively on conversational text. It neither ingests nor exploits structured CRM metadata — deal size, pipeline stage, days-in-stage, engagement scores, product category — that accumulates alongside conversations in real sales pipelines. The paper acknowledges this explicitly under future work. This gap is the precise opening that the Phase 2 hybrid architecture targets.

---

### 2.2 Multimodal Tabular-Text Fusion: Tabular-Text Transformer (TTT)

**Bonnier, T. (2024). Revisiting Multimodal Transformers for Tabular Data with Text Fields. Findings of ACL 2024, pp. 1481–1500.**

Where SalesRLAgent defines the conversation-side SOTA, Bonnier (2024) defines the tabular-text fusion SOTA. The Tabular-Text Transformer (TTT) introduces three novel components that are directly applicable to Phase 2.

**Distance-to-quantile embedding for numerical features.** Standard linear projections of scalar features (Gorishniy et al., 2021) lose distributional information — they cannot express that a value is an outlier or that it sits in a high-density region of the training distribution. TTT instead embeds each numerical value $v$ as a weighted sum of trainable quantile embeddings:

$$z_{num} = \omega^T S$$

where $S \in \mathbb{R}^{s \times d}$ stacks embeddings of $s$ empirical quantiles and $\omega_k \propto 1/|v - q_k|$. This encoding naturally reflects order and handles outliers — directly applicable to CRM variables such as deal size and engagement rate, which are right-skewed with meaningful extreme values.

**Overall attention (simultaneous self- and cross-modal attention).** TTT's key architectural innovation is its *overall attention* module. Standard cross-modal attention (Tsai et al., 2019) computes attention from one modality's queries to the other modality's keys and values — keeping the two streams informationally separate between layers. TTT instead defines keys and values as projections of the **concatenated** sequence $[Z_{text} \| Z_{tab}]$, while queries are modality-specific:

$$\text{OverAtt}_\alpha(Z_\alpha, Z_{\alpha\beta}) = \text{softmax}\!\left(\frac{Q_\alpha K_{\alpha\beta}^T}{\sqrt{d_k}}\right) V_{\alpha\beta}$$

This means each token in the tabular stream can attend to both other tabular tokens (self-attention) and text tokens (cross-modal attention) in a single operation per layer. The ablation study confirms this is TTT's most valuable component: replacing overall attention with pure self-attention (Ablation 2) is the most damaging modification across nearly all eight benchmarks.

**Bimodal Shapley uncertainty explanation.** TTT leverages its dual-stream design for principled uncertainty quantification: when the two streams produce different predicted labels (stream disagreement), the prediction is flagged as uncertain. Feature contributions to uncertainty are computed via a sampling-based Shapley approximation (Štrumbelj & Kononenko, 2010) adapted to the bimodal context, using Jensen-Shannon divergence between the two streams' softmax distributions as the value function. Pearson correlations between these Shapley estimates and Kernel SHAP range from 0.86–0.99, confirming reliability.

**Results.** TTT achieves top or tied-top accuracy on all eight classification benchmarks against six baselines, including AllTextBERT (66.4M parameters) and LateFuseBERT (81.3M parameters), using only 8.2M parameters in its TTT-SRP variant.

**Critical gap.** TTT's benchmarks are entirely generic: Airbnb price prediction, clothing review sentiment, pet adoption speed, wine variety classification. None involve conversational data. The text fields in TTT's evaluations are static product descriptions or user reviews — documents without the sequential, multi-turn, speaker-alternating structure that defines a sales conversation. TTT has never been evaluated in the sales domain, and its fixed positional encoding is designed for document-style inputs, not turn-level dialogue dynamics.

---

## 3. Comparative Analysis

The progression from Phase 1 to Phase 2 mirrors a broader arc in the literature:

| Approach | Text Handling | Tabular Handling | Fusion | Interpretability |
|---|---|---|---|---|
| Perelman & Grinberg (2013) | BoW — flat | None | None | High |
| Shi et al. (2021) | Transformer embeddings | GBT ensemble | Stack ensemble | Low |
| Koval et al. (2024) | Document encoder | Time-series | Multi-stage | Low |
| **SalesRLAgent (2025)** | **Sequential RL on 3072-dim embeddings** | **None** | **None** | **Medium (meta-learning confidence)** |
| **TTT / Bonnier (2024)** | **Static text transformer** | **Distance-to-quantile** | **Overall attention** | **High (Shapley)** |
| **Phase 2 (this project)** | **Turn-aware conversational encoder** | **Distance-to-quantile + CRM features** | **Cross-modal attention** | **High (bimodal Shapley)** |

A consistent theme across all reviewed work: no single modality is sufficient, and the quality of fusion matters as much as the quality of each unimodal representation. The Phase 1 late-fusion baseline confirmed this empirically on this dataset. Phase 2 addresses the fusion quality problem directly.

---

## 4. Identified Gaps

**Gap 1 — No principled fusion for conversational sales data.**
SalesRLAgent achieves SOTA on conversations but ignores CRM metadata. TTT achieves SOTA on tabular-text fusion but has never been applied to conversational data. No published work fuses both for the sales conversion prediction task.

**Gap 2 — Static text assumptions in tabular-text models.**
TTT and all reviewed tabular-text transformers treat text as a bag of tokens from a static document. Sales conversations have sequential turn structure, speaker roles, and temporal dynamics that static positional encodings do not capture. Adapting TTT's architecture to conversational inputs requires replacing the text branch.

**Gap 3 — Interpretability absent from SOTA conversation models.**
SalesRLAgent's meta-learning module quantifies *how confident* a prediction is but provides no feature attribution explanation for *why* — it cannot identify which conversation turns or CRM signals drove the prediction. The bimodal Shapley framework from TTT directly fills this gap when adapted to the sales domain.

**Gap 4 — Temporal leakage risk unaddressed.**
None of the reviewed works rigorously address temporal leakage in conversational settings — the risk that post-outcome language (confirmations, follow-ups recorded after the deal closes) leaks the target label into the training features. This project addresses it through strict turn-cutoff preprocessing.

---

## 5. Proposed Approach Justification

The Phase 2 architecture synthesises SalesRLAgent and TTT to fill every gap identified above:

- **Text branch:** A turn-aware conversational encoder (DistilBERT fine-tuned with turn-level positional tokens and speaker-role embeddings) replaces TTT's static document encoder — capturing the sequential dynamics that SalesRLAgent demonstrated are the highest-value signal component.

- **Tabular branch:** CRM features (deal size, pipeline stage, days-in-stage, engagement rate, product category) are encoded using TTT's distance-to-quantile embedding for continuous variables and standard categorical embeddings for discrete ones — directly addressing the distributional skew of deal-size and engagement variables observed in EDA.

- **Fusion:** TTT's overall attention (simultaneous self- and cross-modal attention) replaces the concatenation-based late fusion of Phase 1 — creating direct gradient pathways between modalities and allowing CRM metadata to selectively attend to the conversation turns most relevant to each feature.

- **Interpretability:** Bimodal Shapley attribution (from TTT) is applied to the sales domain — when the two streams disagree, Shapley values identify which conversation turns and which CRM signals are driving the uncertain prediction, providing actionable output for sales managers.

This progressive strategy — from Phase 1's interpretable baseline to Phase 2's dual-stream fusion — allows direct empirical measurement of the accuracy lift attributable to cross-modal attention over late fusion, using the SalesRLAgent 96.7% figure as the conversation-only ceiling to beat.

---

## References

1. Nandakishor M. (2025). SalesRLAgent: A Reinforcement Learning Approach for Real-Time Sales Conversion Prediction and Optimization. *arXiv:2503.23303*. Deepmost Innovations.
2. Bonnier, T. (2024). Revisiting Multimodal Transformers for Tabular Data with Text Fields. *Findings of ACL 2024*, pp. 1481–1500.
3. Perelman, A. & Grinberg, J. (2013). Nail the Sale: Predicting Sales Outcomes with Textual Features. *Stanford CS229*.
4. Shi, X., Mueller, J., Erickson, N., Li, M., & Smola, A. J. (2021). Benchmarking Multimodal AutoML for Tabular Data with Text Fields. *arXiv:2111.02705*.
5. Koval, R., Andrews, N., & Yan, X. (2024). Financial Forecasting from Textual and Tabular Time Series. *Findings of EMNLP 2024*, pp. 8289–8300.
6. Tsai, Y. et al. (2019). Multimodal Transformer for Unaligned Multimodal Language Sequences. *ACL 2019*.
7. Gorishniy, Y., Rubachev, I., & Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data. *NeurIPS 2021*.
8. Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. *ICML 2017*.
9. Štrumbelj, E. & Kononenko, I. (2010). An Efficient Explanation of Individual Classifications Using Game Theory. *JMLR*, 11(1):1–18.
