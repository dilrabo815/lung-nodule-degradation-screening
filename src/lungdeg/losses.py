"""Training objectives compared in the ablation.

All methods share the weighted BCE on the clean (augmented) crop.

baseline     BCE(clean)                                 - official recipe
aug          BCE on a crop that is degraded with prob p  - "ordinary augmentation"
symmetric    BCE(clean) + BCE(deg) + lam * (sigmoid(z) - sigmoid(z~))^2
             penalises *any* change of the prediction under degradation
directional  BCE(clean) + BCE(deg)
             + lam       * y     * relu(sg(z) - z~ - m)^2      malignant drifting DOWN
             + lam_b     * (1-y) * relu(z~ - sg(z) - m)^2      benign drifting UP (small)
             i.e. only the *harmful* direction is penalised strongly: a malignant
             nodule whose score falls when detail is lost is exactly a potential
             false reassurance. sg = stop-gradient: the clean prediction is the
             teacher, so the model cannot satisfy the term by lowering clean scores.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bce(logit, y, w=None):
    l = F.binary_cross_entropy_with_logits(logit.float(), y.float(), reduction="none")
    if w is not None:
        return (l * w).sum() / w.sum().clamp_min(1e-8)
    return l.mean()


def symmetric_consistency(z, z_deg, w=None):
    d = (torch.sigmoid(z.float()) - torch.sigmoid(z_deg.float())) ** 2
    return (d * w).sum() / w.sum() if w is not None else d.mean()


def directional(z, z_deg, y, margin=0.0, lam_benign=0.1, w=None):
    z_t = z.detach().float()
    z_d = z_deg.float()
    down = F.relu(z_t - z_d - margin) ** 2          # malignant: degraded score fell
    up = F.relu(z_d - z_t - margin) ** 2            # benign: degraded score rose
    term = y * down + lam_benign * (1 - y) * up
    return (term * w).sum() / w.sum() if w is not None else term.mean()


def total_loss(method, z, y, z_deg=None, w=None, lam=1.0, margin=0.0, lam_benign=0.1):
    parts = {"bce_clean": bce(z, y, w)}
    if method in ("symmetric", "directional"):
        parts["bce_deg"] = bce(z_deg, y, w)
        if method == "symmetric":
            parts["reg"] = lam * symmetric_consistency(z, z_deg, w)
        else:
            parts["reg"] = lam * directional(z, z_deg, y, margin, lam_benign, w)
    loss = sum(parts.values())
    return loss, {k: float(v.detach()) for k, v in parts.items()}
