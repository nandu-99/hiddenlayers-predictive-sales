"""Model classes + helpers — copied verbatim from notebook 07."""
import numpy as np
import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """Tab(D_tab→128) || Text(D_text→128) → Head(256→128→1). Identical to notebook 05."""
    def __init__(self, d_tab, d_text, d_hidden=128, p_drop=0.3):
        super().__init__()
        self.tab_enc  = nn.Sequential(nn.Linear(d_tab,  d_hidden), nn.ReLU(), nn.Dropout(p_drop))
        self.text_enc = nn.Sequential(nn.Linear(d_text, d_hidden), nn.ReLU(), nn.Dropout(p_drop))
        self.head = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x_tab, x_text):
        return self.head(torch.cat([self.tab_enc(x_tab), self.text_enc(x_text)], dim=-1))


class ConfidenceGatedHybrid(nn.Module):
    """Phase 3 meta-learner.
    Input  : [P_RF, sigma2_RF, P_DL]  -> (N, 3)
    Output : (P_hybrid, w) where P_hybrid = w*P_DL + (1-w)*P_RF
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(3, hidden),            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),  nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, p_rf, sigma2_rf, p_dl):
        x = torch.stack([p_rf, sigma2_rf, p_dl], dim=1)     # (N, 3)
        w = torch.sigmoid(self.gate_net(x)).squeeze(-1)      # (N,)
        p_hybrid = w * p_dl + (1.0 - w) * p_rf              # (N,)
        return p_hybrid, w


def rf_prob_and_uncertainty(rf_model, X):
    """Mean and variance of class-1 probability across all RF trees.
    sigma2 is the epistemic uncertainty signal that drives the gate.
    """
    tree_probs = np.stack(
        [t.predict_proba(X)[:, 1] for t in rf_model.estimators_], axis=0
    )  # (n_estimators, N)
    return tree_probs.mean(0).astype(np.float32), tree_probs.var(0).astype(np.float32)
