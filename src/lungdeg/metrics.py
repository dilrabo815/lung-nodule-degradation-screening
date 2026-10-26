"""Metrics, calibration, threshold selection, referral analysis and bootstrap CIs.

Pure numpy / scikit-learn so it is unit-tested without a GPU.

Key quantity (proposal objective 2)
-----------------------------------
A case is *automated* unless the referral policy sends it to a radiologist.
A **false low-risk (FLR)** result is a malignant nodule that is automated AND
scored below the low-risk threshold t: the system would have falsely reassured.
    FLR rate        = #FLR / #malignant
    benign rule-out = #(benign, automated, p < t) / #benign
Policies are compared at matched coverage (default 80 % automated).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, roc_auc_score


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def safe_auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def safe_ap(y, p):
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")


def sens_spec(y, p, t):
    y, pred = np.asarray(y), np.asarray(p) >= t
    sens = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
    spec = (~pred & (y == 0)).sum() / max((y == 0).sum(), 1)
    return float(sens), float(spec)


def basic_metrics(y, p, t):
    s, sp = sens_spec(y, p, t)
    return {"auroc": safe_auc(y, p), "auprc": safe_ap(y, p), "sensitivity": s, "specificity": sp,
            "n": int(len(y)), "n_malignant": int(np.sum(y))}


def fit_temperature(logits, y) -> float:
    """Temperature scaling (Guo et al. 2017) on the calibration partition."""
    z, y = np.asarray(logits, float), np.asarray(y, float)

    def nll(log_t):
        p = np.clip(sigmoid(z / np.exp(log_t)), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def threshold_for_sensitivity(y, p, target=0.95) -> float:
    """Largest threshold whose sensitivity is still >= target ("low risk" = p < t)."""
    y, p = np.asarray(y), np.asarray(p)
    mal = np.sort(p[y == 1])
    if len(mal) == 0:
        return 0.5
    k = int(np.floor((1 - target) * len(mal)))  # malignant cases allowed below t
    return float(mal[k]) if k < len(mal) else float(mal[-1])


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def referral_outcome(y, p, score, t, coverage=0.8):
    """Refer the (1-coverage) fraction with the highest referral score."""
    y, p, score = map(np.asarray, (y, p, score))
    n = len(y)
    n_ref = int(round((1 - coverage) * n))
    order = np.argsort(-score, kind="stable")
    referred = np.zeros(n, bool)
    referred[order[:n_ref]] = True
    auto = ~referred
    low = p < t
    flr = int(((y == 1) & auto & low).sum())
    rule_out = int(((y == 0) & auto & low).sum())
    return {
        "coverage": float(auto.mean()),
        "false_low_risk": flr,
        "flr_rate": flr / max(int((y == 1).sum()), 1),
        "benign_rule_out": rule_out / max(int((y == 0).sum()), 1),
        "malignant_referred": int(((y == 1) & referred).sum()),
    }


def confidence_score(logit, t_logit):
    """Confidence-only referral: refer cases closest to the decision threshold."""
    return -np.abs(np.asarray(logit) - t_logit)


def mahalanobis_fit(train_feats, shrink=0.1):
    mu = train_feats.mean(0)
    c = np.cov(train_feats, rowvar=False)
    c = (1 - shrink) * c + shrink * np.trace(c) / c.shape[0] * np.eye(c.shape[0])
    return mu, np.linalg.inv(c)


def mahalanobis_score(feats, mu, prec):
    d = feats - mu
    return np.einsum("ij,jk,ik->i", d, prec, d)


def patient_bootstrap(groups, fn, n_boot=1000, seed=0):
    """95 % CI of fn(index_array) resampling *patients* with replacement."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        v = fn(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
