"""Train one model.

    python -m lungdeg.train --root /content/luna25 --method directional --seed 2025

Every epoch logs (to history.json): train loss and its parts, development
loss / AUROC on clean crops and on train-family degraded crops (fixed seed),
and the mean malignant logit drop under degradation. The checkpoint with the
best clean development AUROC is kept, exactly like the baseline.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import METHODS, Paths, TrainConfig
from .data.augment import center, clip_and_scale, resample
from .data.dataset import CropDataset, make_loader
from .data.degrade import random_train_degradation, train_family_fixed
from .losses import bce, total_loss
from .metrics import safe_auc
from .models.net import MalignancyNet


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def prep(x_hu72, cfg: TrainConfig, train: bool, gen=None):
    """72^3 HU -> 64^3 HU (augmented when training)."""
    if train:
        return resample(x_hu72, cfg.size_px, cfg.rotation_deg, cfg.translation_mm,
                        cfg.size_mm / cfg.size_px, gen)
    return center(x_hu72, cfg.size_px)


def to_input(x_hu, cfg):
    return clip_and_scale(x_hu, cfg.hu_min, cfg.hu_max)


@torch.no_grad()
def evaluate_dev(model, loader, cfg, device, degrade_level=2):
    model.eval()
    zs, zds, ys = [], [], []
    gen = torch.Generator().manual_seed(1234)
    deg = train_family_fixed(degrade_level)
    for b in loader:
        x = prep(b["x"].to(device, non_blocking=True), cfg, train=False)
        xd = deg(x, gen)
        with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
            z = model(to_input(x, cfg))
            zd = model(to_input(xd, cfg))
        zs.append(z.float().cpu()); zds.append(zd.float().cpu()); ys.append(b["y"])
    z, zd, y = torch.cat(zs), torch.cat(zds), torch.cat(ys)
    out = {
        "dev_loss": float(bce(z, y)),
        "dev_auroc": safe_auc(y.numpy(), z.numpy()),
        "dev_loss_deg": float(bce(zd, y)),
        "dev_auroc_deg": safe_auc(y.numpy(), zd.numpy()),
    }
    m = y.numpy() == 1
    out["dev_malignant_logit_drop"] = float((z.numpy()[m] - zd.numpy()[m]).mean()) if m.any() else float("nan")
    return out


def run(cfg: TrainConfig, paths: Paths, split_csv: Path, out_dir: Path, max_steps: int | None = None,
        pretrained: bool = True, cache_px: int | None = None):
    assert cfg.method in METHODS, cfg.method
    seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(split_csv)
    px = cache_px or cfg.cache_px
    cache_npy, cache_idx = paths.cache / f"crops_{px}.npy", paths.cache / f"crops_{px}_index.csv"
    ok = pd.read_csv(cache_idx)
    bad = set(ok.loc[~ok["ok"], "annotation"].astype(str))
    if bad:
        print(f"excluding {len(bad)} observations whose block could not be read")
        df = df[~df["annotation"].astype(str).isin(bad)]
    tr = CropDataset(cache_npy, cache_idx, df[df.split == "train"])
    dv = CropDataset(cache_npy, cache_idx, df[df.split == "dev"])
    tl = make_loader(tr, cfg.batch_size, train=True, workers=cfg.num_workers, seed=cfg.seed)
    dl = make_loader(dv, cfg.batch_size, train=False, workers=cfg.num_workers)
    print(f"train {len(tr)} obs ({int(tr.labels.sum())} malignant) | dev {len(dv)} obs "
          f"({int(dv.labels.sum())} malignant) | device {device}")

    model = MalignancyNet(paths.i3d_weights if pretrained else None, pretrained=pretrained).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    gen = torch.Generator().manual_seed(cfg.seed)

    hist, best, best_ep, bad_epochs = [], -1.0, -1, 0
    json.dump(cfg.as_dict(), open(out_dir / "config.json", "w"), indent=2, default=str)
    for ep in range(1, cfg.epochs + 1):
        model.train()
        t0, parts_sum, n = time.time(), {}, 0
        for step, b in enumerate(tl):
            if max_steps and step >= max_steps:
                break
            x = prep(b["x"].to(device, non_blocking=True), cfg, train=True, gen=gen)
            y = b["y"].to(device)
            # build the network inputs in fp32 (degradations operate in HU)
            if cfg.method == "aug":
                use = (torch.rand(x.shape[0], generator=gen) < cfg.p_degrade).to(device)
                if use.any():
                    x = x.clone()
                    x[use] = random_train_degradation(x[use], gen)
                inp = to_input(x, cfg)
            elif cfg.method in ("symmetric", "directional"):
                # clean and degraded view of the *same* augmented crop
                inp = torch.cat([to_input(x, cfg), to_input(random_train_degradation(x, gen), cfg)])
            else:
                inp = to_input(x, cfg)
            with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
                zz = model(inp)
            z, zd = (zz[: x.shape[0]], zz[x.shape[0]:]) if inp.shape[0] > x.shape[0] else (zz, None)
            loss, parts = total_loss(cfg.method, z, y, zd, None, cfg.lam, cfg.margin, cfg.lam_benign)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v
            parts_sum["loss"] = parts_sum.get("loss", 0.0) + float(loss.detach())
            n += 1
        rec = {"epoch": ep, **{f"train_{k}": v / max(n, 1) for k, v in parts_sum.items()},
               "train_seconds": round(time.time() - t0, 1)}
        rec.update(evaluate_dev(model, dl, cfg, device))
        hist.append(rec)
        json.dump(hist, open(out_dir / "history.json", "w"), indent=2)
        print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
        if rec["dev_auroc"] > best:
            best, best_ep, bad_epochs = rec["dev_auroc"], ep, 0
            torch.save(model.state_dict(), out_dir / "best.pth")
        else:
            bad_epochs += 1
            if bad_epochs > cfg.patience:
                break
    json.dump({"best_dev_auroc": best, "best_epoch": best_ep}, open(out_dir / "best.json", "w"))
    print(f"done: best dev AUROC {best:.4f} at epoch {best_ep}")
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/luna25")
    ap.add_argument("--split_csv", default=None)
    ap.add_argument("--method", default="baseline", choices=METHODS)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=None, help="smoke tests only")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    paths = Paths(Path(a.root))
    cfg = TrainConfig(method=a.method, seed=a.seed, epochs=a.epochs, batch_size=a.batch_size,
                      lam=a.lam, num_workers=a.workers)
    split = Path(a.split_csv) if a.split_csv else paths.splits / "split_seed2025.csv"
    out = Path(a.out) if a.out else paths.runs / f"{a.method}_s{a.seed}"
    run(cfg, paths, split, out, a.max_steps, pretrained=not a.no_pretrained)


if __name__ == "__main__":
    main()
