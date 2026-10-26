"""Collect all runs into the tables and figures used on the progress slides.

    python -m lungdeg.report --root /content/luna25 --out /content/luna25/report

Writes:
  curves_<metric>.png        train / dev curves per method
  auroc_by_condition.png     test AUROC, clean vs each held-out degradation
  flr_at_coverage.png        false-low-risk count at 80 % coverage, method x policy
  summary_metrics.csv        one row per run x condition
  summary_referral.csv       one row per run x condition x policy
  headline.json              numbers quoted on the results slide
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["baseline", "aug", "symmetric", "directional"]
LABEL = {"baseline": "Baseline (official recipe)", "aug": "+ degradation augmentation",
         "symmetric": "+ symmetric consistency", "directional": "+ directional loss (ours)"}
COLOR = {"baseline": "#6b7280", "aug": "#2563eb", "symmetric": "#d97706", "directional": "#059669"}
POL_LABEL = {"none": "No referral", "confidence": "Confidence", "consistency": "TTA consistency",
             "ood": "OOD (Mahalanobis)", "error_head": "Error head (ours)"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def load_runs(runs_dir: Path):
    runs = {}
    for d in sorted(Path(runs_dir).glob("*_s*")):
        if not (d / "history.json").exists():
            continue
        method, seed = d.name.rsplit("_s", 1)
        runs[d.name] = {"dir": d, "method": method, "seed": int(seed),
                        "hist": pd.DataFrame(json.load(open(d / "history.json")))}
    return runs


def plot_curves(runs, out: Path):
    panels = [("train_loss", "dev_loss", "Loss (BCE)"), ("dev_auroc", "dev_auroc_deg", "Dev AUROC")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (a, b, title) in zip(axes, panels):
        for r in runs.values():
            h, c = r["hist"], COLOR.get(r["method"], "k")
            if a in h:
                ax.plot(h.epoch, h[a], color=c, lw=2, label=f"{r['method']} " + ("train" if "train" in a else "clean"))
            if b in h:
                ax.plot(h.epoch, h[b], color=c, lw=2, ls="--",
                        label=f"{r['method']} " + ("dev" if "loss" in b else "degraded"))
        ax.set_title(title); ax.set_xlabel("epoch"); _style(ax)
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    axes[1].legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(out / "curves.png", dpi=180); plt.close(fig)


def summarize(runs, out: Path):
    mets, refs = [], []
    for name, r in runs.items():
        d = r["dir"]
        if (d / "test_metrics.csv").exists():
            m = pd.read_csv(d / "test_metrics.csv"); m["run"], m["method"] = name, r["method"]; mets.append(m)
        if (d / "test_referral.csv").exists():
            f = pd.read_csv(d / "test_referral.csv"); f["run"], f["method"] = name, r["method"]; refs.append(f)
    met = pd.concat(mets) if mets else pd.DataFrame()
    ref = pd.concat(refs) if refs else pd.DataFrame()
    if len(met):
        met.to_csv(out / "summary_metrics.csv", index=False)
    if len(ref):
        ref.to_csv(out / "summary_referral.csv", index=False)
    return met, ref


def plot_auroc(met, out: Path):
    if met.empty:
        return
    conds = list(dict.fromkeys(met["condition"]))
    methods = [m for m in ORDER if m in set(met["method"])]
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, m in enumerate(methods):
        g = met[met.method == m].groupby("condition")[["auroc", "auroc_ci_lo", "auroc_ci_hi"]].mean().reindex(conds)
        x = np.arange(len(conds)) + i * w
        ax.bar(x, g["auroc"], w, color=COLOR[m], label=LABEL[m])
        ax.errorbar(x, g["auroc"], yerr=[g["auroc"] - g["auroc_ci_lo"], g["auroc_ci_hi"] - g["auroc"]],
                    fmt="none", ecolor="k", lw=0.8, capsize=2)
    ax.axhline(0.80, color="crimson", lw=1, ls=":", label="target 0.80")
    ax.set_xticks(np.arange(len(conds)) + w * (len(methods) - 1) / 2)
    ax.set_xticklabels(conds, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("Test AUROC (95% CI)"); _style(ax)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "auroc_by_condition.png", dpi=180); plt.close(fig)


def plot_flr(ref, out: Path):
    if ref.empty:
        return
    deg = ref[ref.condition != "clean"]
    tab = deg.groupby(["method", "policy"])["false_low_risk"].mean().unstack()
    tab = tab.reindex([m for m in ORDER if m in tab.index])
    pols = [p for p in POL_LABEL if p in tab.columns]
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.8 / len(pols)
    cmap = plt.get_cmap("viridis", len(pols))
    for i, p in enumerate(pols):
        ax.bar(np.arange(len(tab)) + i * w, tab[p], w, label=POL_LABEL[p], color=cmap(i))
    ax.set_xticks(np.arange(len(tab)) + w * (len(pols) - 1) / 2)
    ax.set_xticklabels([LABEL[m] for m in tab.index], fontsize=8)
    ax.set_ylabel("Malignant accepted as low risk\n(mean over held-out degradations)")
    ax.set_title("False low-risk results at 80% automated coverage (lower is better)")
    _style(ax); ax.legend(fontsize=8, frameon=False, ncol=5)
    fig.tight_layout(); fig.savefig(out / "flr_at_coverage.png", dpi=180); plt.close(fig)
    tab.to_csv(out / "flr_table.csv")


def headline(met, ref, runs) -> dict:
    h = {"runs": sorted(runs)}
    if not met.empty:
        clean = met[met.condition == "clean"].groupby("method")[["auroc", "auprc", "sensitivity", "specificity",
                                                                 "auroc_ci_lo", "auroc_ci_hi"]].mean()
        h["clean_test"] = clean.round(3).to_dict(orient="index")
        deg = met[met.condition != "clean"].groupby("method")["auroc"].mean()
        h["mean_auroc_heldout_degradations"] = deg.round(3).to_dict()
    if not ref.empty:
        deg = ref[ref.condition != "clean"]
        t = deg.groupby(["method", "policy"])["false_low_risk"].mean()
        h["flr_mean_heldout"] = {f"{m}|{p}": round(float(v), 2) for (m, p), v in t.items()}
        try:
            a = t[("aug", "confidence")]
            b = t[("directional", "error_head")]
            h["relative_flr_reduction_vs_aug_confidence"] = round(float((a - b) / a), 3) if a > 0 else None
        except KeyError:
            pass
    for name, r in runs.items():
        hh = r["hist"]
        h.setdefault("best_dev_auroc", {})[name] = round(float(hh.dev_auroc.max()), 3)
        h.setdefault("epoch_seconds", {})[name] = round(float(hh.train_seconds.mean()), 1)
    return h


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/luna25")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    out = Path(a.out) if a.out else root / "report"
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(root / "runs")
    plot_curves(runs, out)
    met, ref = summarize(runs, out)
    plot_auroc(met, out)
    plot_flr(ref, out)
    h = headline(met, ref, runs)
    json.dump(h, open(out / "headline.json", "w"), indent=2)
    print(json.dumps(h, indent=2))


if __name__ == "__main__":
    main()
