"""Inference, post-hoc fitting (temperature, threshold, OOD, error head) and the
degradation / referral evaluation for one trained run.

    python -m lungdeg.evaluate --root /content/luna25 --run runs/directional_s2025

Order of operations (no test label is used for any choice):
  1. temperature T      <- dev 'calib' patients, clean crops
  2. low-risk threshold <- dev 'thresh' patients, clean, sensitivity 0.95
  3. OOD Gaussian       <- training-split embeddings (subset), clean
  4. error-referral head<- dev 'head_fit' patients, train-family degradations
  5. test: clean + every HELD-OUT degradation; metrics, referral at 80 % coverage,
     patient-level bootstrap CIs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Paths, TrainConfig
from .data.augment import center, clip_and_scale, resample
from .data.dataset import CropDataset, make_loader
from .data.degrade import HELD_OUT, random_train_degradation
from .metrics import (basic_metrics, confidence_score, ece, fit_temperature, mahalanobis_fit,
                      mahalanobis_score, patient_bootstrap, referral_outcome, sigmoid,
                      threshold_for_sensitivity)
from .models.net import ErrorReferralHead, MalignancyNet, detail_stats

POLICIES = ("confidence", "consistency", "ood", "error_head")


@torch.no_grad()
def infer(model, loader, cfg: TrainConfig, device, transform=None, tta: int = 0, seed: int = 0):
    """Returns logits, 1024-d features, detail stats and (optional) TTA std of p."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    out = {k: [] for k in ("logit", "feat", "stats", "tta_std", "y", "idx")}
    for b in loader:
        x72 = b["x"].to(device, non_blocking=True)
        x = center(x72, cfg.size_px)
        if transform is not None:
            x = transform(x, gen)
        inp = clip_and_scale(x, cfg.hu_min, cfg.hu_max)
        with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
            z, f = model.forward_with_features(inp)
        out["logit"].append(z.float().cpu()); out["feat"].append(f.float().cpu())
        out["stats"].append(detail_stats(inp).float().cpu())
        if tta:
            ps = [torch.sigmoid(z.float())]
            for _ in range(tta):
                xa = resample(x72, cfg.size_px, rotation_deg=10, translation_mm=1.0, generator=gen)
                if transform is not None:
                    xa = transform(xa, gen)
                with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
                    ps.append(torch.sigmoid(model(clip_and_scale(xa, cfg.hu_min, cfg.hu_max)).float()))
            out["tta_std"].append(torch.stack(ps).std(0).cpu())
        out["y"].append(b["y"]); out["idx"].append(b["idx"])
    res = {k: torch.cat(v).numpy() for k, v in out.items() if v}
    return res


def fit_error_head(feats, logits, stats, y, t_logit, epochs=300, seed=0, tau=0.5):
    """Soft target: P(malignant and scored below threshold) for each degraded view."""
    torch.manual_seed(seed)
    f = torch.tensor(feats, dtype=torch.float32)
    z = torch.tensor(logits, dtype=torch.float32)
    s = torch.tensor(stats, dtype=torch.float32)
    yy = torch.tensor(y, dtype=torch.float32)
    target = yy * torch.sigmoid((t_logit - z) / tau)
    head = ErrorReferralHead(f.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    pos_w = torch.tensor(max(1.0, float((1 - target).sum() / target.sum().clamp_min(1.0))))
    for _ in range(epochs):
        head.train()
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(f, z, s), target, pos_weight=pos_w)
        loss.backward()
        opt.step()
    head.eval()
    return head


def head_score(head, d):
    with torch.no_grad():
        return head(torch.tensor(d["feat"]), torch.tensor(d["logit"]), torch.tensor(d["stats"])).numpy()


def run_eval(run_dir: Path, paths: Paths, split_csv: Path, tta: int = 4, n_boot: int = 1000,
             head_views: int = 6, coverage: float = 0.8, cache_px: int = 72, workers: int = 2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig(**{k: v for k, v in json.load(open(run_dir / "config.json")).items()
                         if k in TrainConfig.__dataclass_fields__})
    model = MalignancyNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pth", map_location=device))
    df = pd.read_csv(split_csv)
    idx = pd.read_csv(paths.cache / f"crops_{cache_px}_index.csv")
    df = df[df["annotation"].astype(str).isin(set(idx.loc[idx["ok"], "annotation"].astype(str)))]
    npy, idxp = paths.cache / f"crops_{cache_px}.npy", paths.cache / f"crops_{cache_px}_index.csv"

    def loader(frame):
        ds = CropDataset(npy, idxp, frame)
        return ds, make_loader(ds, 64, train=False, workers=workers)

    rng = np.random.default_rng(cfg.seed)
    # 1-2: temperature + threshold (clean)
    _, l_cal = loader(df[df.dev_part == "calib"])
    cal = infer(model, l_cal, cfg, device)
    T = fit_temperature(cal["logit"], cal["y"])
    _, l_thr = loader(df[df.dev_part == "thresh"])
    thr = infer(model, l_thr, cfg, device)
    t = threshold_for_sensitivity(thr["y"], sigmoid(thr["logit"] / T), 0.95)
    t_logit = float(np.log(t / (1 - t))) * T          # threshold on the raw-logit scale
    # 3: OOD reference = up to 1500 training observations
    tr_frame = df[df.split == "train"]
    tr_frame = tr_frame.iloc[rng.permutation(len(tr_frame))[:1500]]
    _, l_tr = loader(tr_frame)
    mu, prec = mahalanobis_fit(infer(model, l_tr, cfg, device)["feat"])
    # 4: error head on head_fit patients, clean + train-family degraded views
    _, l_hf = loader(df[df.dev_part == "head_fit"])
    views = [infer(model, l_hf, cfg, device)]
    for v in range(head_views):
        views.append(infer(model, l_hf, cfg, device, seed=100 + v,
                           transform=lambda x, g: random_train_degradation(x, g, sigma_mm=(0.3, 1.5),
                                                                           noise_hu=(10, 70))))
    cat = {k: np.concatenate([d[k] for d in views]) for k in ("feat", "logit", "stats", "y")}
    head = fit_error_head(cat["feat"], cat["logit"], cat["stats"], cat["y"], t_logit, seed=cfg.seed)

    # 5: test
    test_frame = df[df.split == "test"].reset_index(drop=True)
    _, l_te = loader(test_frame)
    groups = test_frame["patient"].astype(str).to_numpy()
    conditions = {"clean": None, **HELD_OUT}
    rows, ref_rows = [], []
    pred_dir = run_dir / "preds"
    pred_dir.mkdir(exist_ok=True)
    for ci, (name, tf) in enumerate(conditions.items()):
        d = infer(model, l_te, cfg, device, transform=tf, tta=tta, seed=500 + ci)
        p = sigmoid(d["logit"] / T)
        y = d["y"].astype(int)
        np.savez_compressed(pred_dir / f"test_{name}.npz", p=p, logit=d["logit"], y=y, groups=groups,
                            tta_std=d.get("tta_std", np.zeros_like(p)), stats=d["stats"])
        m = basic_metrics(y, p, t)
        m["ece"] = ece(y, p)
        lo, hi = patient_bootstrap(groups, lambda ii: basic_metrics(y[ii], p[ii], t)["auroc"], n_boot, seed=ci)
        m.update(condition=name, auroc_ci_lo=lo, auroc_ci_hi=hi)
        opn = test_frame["one_per_nodule"].to_numpy().astype(bool)
        m["auroc_one_per_nodule"] = basic_metrics(y[opn], p[opn], t)["auroc"]
        rows.append(m)
        scores = {
            "confidence": confidence_score(d["logit"], t_logit),
            "consistency": d["tta_std"] if tta else np.zeros_like(p),
            "ood": mahalanobis_score(d["feat"], mu, prec),
            "error_head": head_score(head, d),
        }
        none = referral_outcome(y, p, np.zeros_like(p), t, coverage=1.0)
        ref_rows.append({"condition": name, "policy": "none", **none})
        for pol, sc in scores.items():
            r = referral_outcome(y, p, sc, t, coverage)
            lo, hi = patient_bootstrap(groups, lambda ii: referral_outcome(y[ii], p[ii], sc[ii], t, coverage)["false_low_risk"],
                                       n_boot, seed=ci)
            ref_rows.append({"condition": name, "policy": pol, **r, "flr_ci_lo": lo, "flr_ci_hi": hi})
        print(f"[{name}] AUROC {m['auroc']:.3f} ({m['auroc_ci_lo']:.3f}-{m['auroc_ci_hi']:.3f})  sens {m['sensitivity']:.3f} "
              f"spec {m['specificity']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(run_dir / "test_metrics.csv", index=False)
    pd.DataFrame(ref_rows).to_csv(run_dir / "test_referral.csv", index=False)
    json.dump({"temperature": T, "threshold": t, "threshold_logit": t_logit, "coverage": coverage,
               "dev_calib_n": int(len(cal["y"])), "dev_thresh_n": int(len(thr["y"])),
               "head_fit_views": int(len(cat["y"]))}, open(run_dir / "posthoc.json", "w"), indent=2)
    return run_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/luna25")
    ap.add_argument("--run", required=True)
    ap.add_argument("--split_csv", default=None)
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    paths = Paths(Path(a.root))
    split = Path(a.split_csv) if a.split_csv else paths.splits / "split_seed2025.csv"
    run_eval(Path(a.run), paths, split, tta=a.tta, n_boot=a.n_boot, workers=a.workers)


if __name__ == "__main__":
    main()
