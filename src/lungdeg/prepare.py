"""Audit the annotations, make the patient-level split and build the crop cache.

    python -m lungdeg.prepare --root /content/luna25 [--seed 2025] [--no_cache]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Paths
from .data.cache import build_cache
from .data.download import find_block_root
from .data.splits import audit, check_no_leakage, load_annotations, make_splits, save_split, split_summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/luna25")
    ap.add_argument("--csv", default=None, help="override annotation CSV path")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--px", type=int, default=72)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no_cache", action="store_true")
    a = ap.parse_args(argv)
    paths = Paths(Path(a.root))
    df, cols = load_annotations(a.csv or paths.csv)
    print("columns used:", cols)
    blocks = find_block_root(paths.blocks)
    rep = audit(df, blocks)
    rep["columns"] = cols
    paths.splits.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(paths.splits / "data_audit.json", "w"), indent=2)
    print(json.dumps(rep, indent=2))
    df = make_splits(df, seed=a.seed)
    check_no_leakage(df)
    p = save_split(df, paths.splits, a.seed)
    print(split_summary(df).to_string())
    print(df[df.split == "dev"].groupby("dev_part").agg(obs=("label", "size"), malignant=("label", "sum")))
    print("split written to", p)
    if not a.no_cache:
        if blocks is None:
            raise SystemExit("nodule blocks not found - run lungdeg.data.download first")
        build_cache(blocks, df["annotation"].astype(str).tolist(), paths.cache, a.px, a.workers)


if __name__ == "__main__":
    main()
