# Theoretical Rigor

## Phase 2 — Conversational-Tabular Fusion for Sales Conversion Prediction

---

## 1. Problem Formulation

Let $\mathcal{D} = \{(x_t^{(i)}, x_s^{(i)}, y^{(i)})\}_{i=1}^{N}$ be a corpus of $N$ i.i.d. samples drawn from a joint distribution $\mathcal{P}(X_t, X_s, Y)$. Each sample comprises:

- **Transcript tensor:** $x_t \in \mathbb{R}^{L \times d_{\text{text}}}$ — a sequence of $L$ token embeddings of dimension $d_{\text{text}} = 768$ (DistilBERT hidden size).
- **Tabular CRM vector:** $x_s \in \mathbb{R}^{d_{\text{tab}}}$ — structured deal metadata: deal size, pipeline stage, days-in-stage, engagement rate, product category.
- **Label:** $y \in \{0, 1\}$ — $1$ = conversion (closed-won), $0$ = no conversion.

The learning objective is to find parameters $\theta^*$ minimising expected binary cross-entropy risk:

$$\theta^* = \arg\min_{\theta} \; \mathbb{E}_{(x_t, x_s, y) \sim \mathcal{P}} \left[ \mathcal{L}_{\text{BCE}}\!\left( f_\theta(x_t, x_s),\; y \right) \right] \tag{1}$$

where $f_\theta : \mathbb{R}^{L \times d_{\text{text}}} \times \mathbb{R}^{d_{\text{tab}}} \to [0,1]$ is the dual-stream fusion model. In practice, expectation over $\mathcal{P}$ is approximated by empirical risk minimisation over $\mathcal{D}_{\text{train}}$, with $\mathcal{D}_{\text{val}}$ used strictly for early stopping and $\mathcal{D}_{\text{test}}$ held out until final evaluation — no hyperparameter selection is performed on $\mathcal{D}_{\text{test}}$.

**Assumptions:**
- **i.i.d.:** Individual deals are treated as independent samples from $\mathcal{P}$. Violation risk from temporal deal correlation is acknowledged in Section 8.
- **Stationarity:** The conditional distribution $P(y \mid x_t, x_s)$ does not shift between train, validation, and test windows.
- **No leakage:** $x_t$ is truncated to turns occurring strictly before the deal outcome is recorded; post-outcome confirmations are excluded from the feature space.

---

## 2. Text Encoder: Transformer Mathematics

### 2.1 Scaled Dot-Product Self-Attention

Given token embeddings $X \in \mathbb{R}^{L \times d_{\text{model}}}$, the model projects into query, key, and value matrices:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V \tag{2}$$

The attention output is:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V \tag{3}$$

**Why scale by $1/\sqrt{d_k}$?** The dot product $QK^T$ has variance $\mathcal{O}(d_k)$ when $Q, K$ have unit-variance entries. Without scaling, the softmax input grows in magnitude, pushing the function into near-zero-gradient saturation regions and causing gradient vanishing during early training. Dividing by $\sqrt{d_k}$ restores variance to $\mathcal{O}(1)$, preserving stable gradient flow (Vaswani et al., 2017).

### 2.2 Multi-Head Attention

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W_O \tag{4}$$

where $\text{head}_i = \text{Attn}(QW_Q^i, KW_K^i, VW_V^i)$. Running $h$ parallel heads with independent projections allows different heads to specialise on different relational subspaces simultaneously — in the sales context, one head can attend to objection language while another tracks commitment signals across turns.

### 2.3 Positional Encoding

Self-attention is permutation-equivariant: $\text{Attn}(PX, PX, PX) = P\,\text{Attn}(X,X,X)$ for any permutation matrix $P$. Without positional information, the model cannot distinguish turn 2 from turn 14, making temporal dynamics in the conversation invisible. Sinusoidal positional encodings:

$$\text{PE}(\text{pos}, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right), \quad \text{PE}(\text{pos}, 2i{+}1) = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right) \tag{5}$$

are added to token embeddings before the first layer. In Phase 2, speaker-role tokens (`[BUYER]`, `[SELLER]`) are prepended to each turn to encode dialogue structure beyond positional order.

### 2.4 Why DistilBERT

DistilBERT (Sanh et al., 2019) is a knowledge-distilled variant of BERT (Devlin et al., 2019): 6 layers, $d_{\text{model}} = 768$, ~66M parameters. Knowledge distillation trains the student to match the teacher's temperature-scaled soft probability distributions, preserving **97% of BERT's GLUE benchmark performance at 40% parameter reduction** and ~60% faster inference. For this project — operating on an ~8K-sample dataset on standard academic compute — DistilBERT hits the optimal accuracy-vs-compute Pareto point. Full BERT (110M params) would increase fine-tuning time by ~2.5× with negligible gain at this dataset scale.

---

## 3. Cross-Modal Attention: The Novel Fusion Block

### 3.1 Motivation and Inductive Bias

Phase 1 used **late fusion**: the conversational encoder and tabular encoder trained independently; their final representations were concatenated before the classification head. Late fusion prevents either modality from conditioning its encoding on the other — gradients from the loss cannot flow from the text branch into the tabular branch or vice versa.

This is suboptimal for sales data because the predictive signal in CRM metadata is **context-dependent**: a large deal size is a strong positive predictor only if the conversation contains genuine commitment language. The same deal size paired with objection-heavy dialogue may predict loss. Cross-modal attention allows the CRM tabular branch to dynamically attend to the conversation turns semantically most relevant to each structured feature — precisely the inductive bias the task requires.

### 3.2 Formal Definition

Let $H_t \in \mathbb{R}^{L \times d}$ be the per-token hidden states from the DistilBERT encoder. Let $x_s \in \mathbb{R}^{d_{\text{tab}}}$ be the CRM feature vector. First project $x_s$ to the shared model dimension $d$:

$$x_s^{\text{proj}} = W_{\text{proj}}\, x_s + b_{\text{proj}}, \quad x_s^{\text{proj}} \in \mathbb{R}^d \tag{6}$$

Compute query from the tabular branch, keys and values from the transcript:

$$Q_s = x_s^{\text{proj}}\, W_Q^s \in \mathbb{R}^{1 \times d_k} \tag{7a}$$

$$K_t = H_t\, W_K^t \in \mathbb{R}^{L \times d_k} \tag{7b}$$

$$V_t = H_t\, W_V^t \in \mathbb{R}^{L \times d_v} \tag{7c}$$

The cross-modal fusion output is:

$$z_{\text{cross}} = \text{softmax}\!\left(\frac{Q_s K_t^T}{\sqrt{d_k}}\right) V_t, \quad z_{\text{cross}} \in \mathbb{R}^{1 \times d_v} \tag{8}$$

The softmax over $L$ transcript positions produces an **attention distribution across turns**. Tokens semantically related to the CRM context receive high attention weight; irrelevant filler turns are down-weighted. This is the key inductive bias: structured deal metadata selectively reads from the conversational record rather than treating all turns equally.

The cross-modal gradient pathway is: $\partial \mathcal{L} / \partial x_s^{\text{proj}}$ flows through $W_Q^s$, and $\partial \mathcal{L} / \partial H_t$ flows through $W_K^t$ and $W_V^t$ — both branches are updated jointly by the same classification loss. This is the defining capability advantage of Phase 2 over Phase 1's independent-branch training.

### 3.3 Overall Attention (TTT-style)

Following Bonnier (2024), the full model uses **overall attention** — each stream's keys and values are computed from the concatenated sequence $[Z_\alpha \| Z_\beta]$:

$$\text{OverAtt}_\alpha(Z_\alpha, Z_{\alpha\beta}) = \text{softmax}\!\left(\frac{Q_\alpha\, K_{\alpha\beta}^T}{\sqrt{d_k}}\right) V_{\alpha\beta} \tag{9}$$

where $Z_{\alpha\beta} = [Z_\alpha \| Z_\beta]$. This allows each tabular token to attend to both other tabular tokens (self-attention) and text tokens (cross-modal attention) in a single operation per layer. Bonnier's ablation study shows replacing overall attention with pure self-attention is the most damaging modification across nearly all benchmarks — confirming this is the architecturally critical choice.

### 3.4 Gated Fusion Variant

To prevent the model from collapsing to a single modality when one is uninformative:

$$g = \sigma\!\left(W_g\, [x_s^{\text{proj}};\, \text{CLS}_t] + b_g\right) \tag{10a}$$

$$z_{\text{fused}} = g \odot z_{\text{cross}} + (1-g) \odot \text{CLS}_t \tag{10b}$$

where $\text{CLS}_t$ is the `[CLS]` token output from DistilBERT, $\sigma$ is the sigmoid, and $\odot$ is element-wise product. The gate $g \in [0,1]$ is learned end-to-end: when CRM metadata is noisy or missing, $g \to 0$ and $z_{\text{fused}}$ recovers the text-only representation. When structured features are strongly predictive, $g \to 1$. This adaptive weighting is superior to a fixed concatenation, which cannot respond to sample-level modality quality variation.

---

## 4. Loss Function

### 4.1 Binary Cross-Entropy

The model outputs a scalar logit $z \in \mathbb{R}$, converted to probability via $\hat{y} = \sigma(z)$. The per-sample BCE loss is:

$$\mathcal{L}_{\text{BCE}} = -\left[ y \log \hat{y} + (1-y) \log(1-\hat{y}) \right] \tag{11}$$

Averaged over $N$ training samples:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i) \right] \tag{12}$$

This is derived from Maximum Likelihood Estimation under a Bernoulli likelihood model — minimising BCE is equivalent to maximising $\prod_i P(y_i \mid x_t^{(i)}, x_s^{(i)};\, \theta)$.

### 4.2 Gradient w.r.t. Logit

Differentiating Eq. 11 with respect to the logit $z$:

$$\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y \tag{13}$$

This result is notable: the gradient is simply the **prediction error**, bounded in $(-1, 1)$. When $\hat{y} \approx y$ (confident correct prediction), the gradient magnitude is near zero and the model stops updating that sample — naturally focusing learning on uncertain or misclassified cases. This eliminates gradient explosion risk common to squared-error losses.

### 4.3 Focal Loss (Hedging Variant)

Although the `DeepMostInnovations/saas-sales-conversations` dataset is approximately balanced (56/44 split), borderline conversations — deals with mixed signals — can dominate gradient updates. Focal loss (Lin et al., 2017) addresses this:

$$\mathcal{L}_{\text{focal}} = -(1 - \hat{y})^\gamma \log \hat{y} \tag{14}$$

The modulating factor $(1 - \hat{y})^\gamma$ reduces the loss contribution of easy examples exponentially. At $\gamma = 2$, a sample with $\hat{y} = 0.9$ receives $100\times$ less gradient contribution than a sample with $\hat{y} = 0.5$. BCE ($\gamma = 0$) is the primary loss for this project; focal loss is evaluated as a regularisation experiment when the model shows overconfidence on clearly non-converting conversations.

---

## 5. Optimisation

### 5.1 Adam vs SGD

SGD with momentum applies a single global learning rate to all parameters. For transformer training, attention matrices $W_Q, W_K, W_V$ operate on different gradient scales than the classification head, and earlier layers receive smaller gradients due to backpropagation depth. Adam (Kingma & Ba, 2014) maintains per-parameter adaptive rates through first and second moment estimates:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \tag{15a}$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \tag{15b}$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \tag{15c}$$

$$\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \tag{15d}$$

Settings: $\beta_1 = 0.9,\; \beta_2 = 0.999,\; \epsilon = 10^{-8}$.

### 5.2 AdamW: Decoupled Weight Decay

Standard L2 regularisation adds $\lambda\|\theta\|^2$ to the loss; in Adam this is scaled by the adaptive denominator, so parameters with large gradients receive less regularisation. AdamW (Loshchilov & Hutter, 2019) decouples weight decay from the gradient update:

$$\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha \lambda \theta_t \tag{16}$$

The decay term is applied uniformly regardless of gradient magnitude — critical for transformer fine-tuning where embedding layers and LayerNorm parameters should receive reduced or zero decay relative to dense projection matrices.

### 5.3 LR Schedule: Warmup + Linear Decay

At initialisation, attention matrices are random and softmax distributions are nearly uniform. A large learning rate at this stage produces high-variance gradient steps that destabilise the pretrained DistilBERT representations. Linear warmup over $W_{\text{warmup}}$ steps (6% of total training steps) ramps $\text{lr}$ from $0$ to $\alpha_{\max}$ linearly, then decays linearly to near zero:

$$\text{lr}(t) = \alpha_{\max} \cdot \min\!\left(\frac{t}{W_{\text{warmup}}},\; 1 - \frac{t - W_{\text{warmup}}}{T - W_{\text{warmup}}}\right) \tag{17}$$

---

## 6. Regularisation and Why It Works

### 6.1 Dropout

During training, each activation is zeroed with probability $p$ (0.1 for transformer layers, 0.2 for the classification head). This approximates training an ensemble of $2^n$ thinned networks sharing weights (Srivastava et al., 2014). At inference, scaling by $(1-p)$ approximates the ensemble average. Dropout reduces the effective VC dimension of the network, tightening the generalisation bound $\mathcal{O}(\sqrt{\log(1/\delta)/n})$.

### 6.2 Layer Normalisation

BatchNorm normalises across the batch dimension — for variable-length sequences, padding tokens skew batch statistics and statistics computed during training may not match inference-time statistics if batch composition changes. LayerNorm (Ba et al., 2016) normalises per-sample across the feature dimension:

$$\text{LN}(x) = \gamma \cdot \frac{x - \mu_x}{\sqrt{\sigma_x^2 + \epsilon}} + \beta \tag{18}$$

where $\mu_x, \sigma_x^2$ are computed per-token per-sample. This is batch-size-independent and immune to padding-induced distribution shift. Pre-LN (normalise before the attention/FFN sub-layer, not after) is used throughout Phase 2 for improved training stability.

### 6.3 Residual Connections

Each transformer sub-layer wraps its output in a residual connection:

$$H^{(l+1)} = H^{(l)} + \text{SubLayer}\!\left(\text{LN}(H^{(l)})\right) \tag{19}$$

The gradient of the loss with respect to layer $l$ is:

$$\frac{\partial \mathcal{L}}{\partial H^{(l)}} = \frac{\partial \mathcal{L}}{\partial H^{(l+1)}} \cdot \left(I + J_{\text{SubLayer}}\right)$$

The identity term $I$ guarantees a direct gradient path from the loss to every layer regardless of depth — preventing the multiplicative gradient decay that causes vanishing gradients in plain deep networks (He et al., 2016).

### 6.4 Early Stopping

Training halts when validation loss fails to decrease for `patience = 5` epochs. The checkpoint at minimum validation loss — not the final iterate — is retained. This is implicit regularisation: it selects the model iterate with minimum expected test error before the model begins memorising training noise.

---

## 7. Bias–Variance in the Deep Learning Regime

Classical statistical learning theory predicts a U-shaped bias-variance tradeoff with a single optimal capacity. Modern deep learning invalidates this for over-parameterised models: Belkin et al. (2019) demonstrate a **double-descent** curve where test error decreases again beyond the interpolation threshold as model capacity continues to grow. Under the classical theory, DistilBERT at ~66M parameters on ~8K training samples appears grossly over-parameterised.

However, **fine-tuning a pretrained model is not training from scratch**. The pretrained weights encode a strong inductive bias over English language structure acquired from billions of tokens. Fine-tuning updates only a small fraction of parameters significantly — primarily the classification head and top attention layers. The Rademacher complexity of the fine-tuned function class is bounded by the $\ell_2$ distance from the pretrained initialisation in parameter space, which is small after a small number of gradient steps on 8K samples. The effective degrees of freedom consumed by the data is far less than 66M — the prior from pretraining dominates the posterior, keeping effective model complexity low and generalisation tight.

---

## 8. Assumptions and Limitations

- **Temporal correlation breaks i.i.d.** Deals within the same company, industry, or quarter may be correlated. A random train/test split could produce an optimistic accuracy estimate. A time-based split — train on earlier deals, test on later — provides a more operationally valid evaluation and will be used in Phase 2 ablations.

- **DistilBERT domain mismatch.** DistilBERT was pretrained on BookCorpus and English Wikipedia — general-domain text. Sales-specific terminology ("champion", "MEDDIC", "expansion ARR", "multi-threaded") is underrepresented. The tokeniser may fragment domain terms into semantically poor subword units. Domain-adaptive continued pre-training (masked language modelling on unlabelled sales transcripts) is a planned mitigation.

- **Attention weights ≠ feature importance.** A well-known finding (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019) shows that attention weights do not reliably identify which input tokens caused a prediction — alternative attention distributions can produce identical model outputs. The bimodal Shapley attribution layer (Bonnier, 2024) adopted in Phase 2 addresses this by computing marginal feature contributions rather than reading attention weights directly.

- **Synthetic dataset limitations.** The `DeepMostInnovations/saas-sales-conversations` dataset is GPT-4O-generated. Synthetic conversations may lack the disfluencies, implicit shared context, and domain-specific negotiation patterns of real enterprise sales calls. Performance on real-world transcripts should be validated before production deployment.

---

## References

1. Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
2. Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT 2019*.
3. Sanh, V. et al. (2019). DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter. *arXiv:1910.01108*.
4. Kingma, D. P. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*.
5. Lin, T. Y. et al. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*.
6. Ba, J. L. et al. (2016). Layer Normalization. *arXiv:1607.06450*.
7. He, K. et al. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
8. Bonnier, T. (2024). Revisiting Multimodal Transformers for Tabular Data with Text Fields. *ACL Findings 2024*.
9. Nandakishor M. (2025). SalesRLAgent. *arXiv:2503.23303*.
10. Jain, S. & Wallace, B. C. (2019). Attention is not Explanation. *NAACL 2019*.
11. Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*.
12. Belkin, M. et al. (2019). Reconciling Modern Machine Learning Practice and the Bias-Variance Trade-off. *PNAS 2019*.
13. Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. *ICML 2017*.
14. Štrumbelj, E. & Kononenko, I. (2010). An Efficient Explanation of Individual Classifications Using Game Theory. *JMLR*, 11(1):1–18.
