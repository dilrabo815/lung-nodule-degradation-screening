"""Pre-extract one fixed crop per observation into a single memory-mapped array.

Why: the baseline resamples every patch on the CPU with scipy on each epoch.
Colab gives 2 CPU cores, which would starve the GPU. We therefore run the
baseline's own ``extract_patch`` once per observation (no augmentation) at the
baseline voxel spacing (50/64 mm) but on a slightly larger 72^3 grid, store the
raw HU values as float16, and do rotation/translation on the GPU later
(``augment.py``). The final 64^3 / 50 mm field of view and the HU window are
unchanged.

Output: <cache>/crops_<px>.npy  (N, px, px, px) float16 HU
        <cache>/crops_<px>_index.csv  (row -> AnnotationID)
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import extract_patch


def crop_one(block_dir: Path, annotation_id: str, px: int = 72, size_mm: float = 50.0,
             size_px: int = 64) -> np.ndarray:
    img = np.load(block_dir / "image" / f"{annotation_id}.npy", mmap_mode="r")
    meta = np.load(block_dir / "metadata" / f"{annotation_id}.npy", allow_pickle=True).item()
    sp = size_mm / size_px
    patch = extract_patch(
        CTData=img,
        coord=tuple(np.array(img.shape) // 2),    # baseline: centre of the 64x128x128 block
        srcVoxelOrigin=meta["origin"],
        srcWorldMatrix=meta["transform"],
        srcVoxelSpacing=meta["spacing"],
        output_shape=(px, px, px),
        voxel_spacing=(sp, sp, sp),
        rotations=None,
        translations=None,
        coord_space_world=False,
        mode="3D",
    )[0]
    return np.clip(patch, -1500, 3000).astype(np.float16)


def _worker(args):
    block_dir, aid, px = args
    try:
        return aid, crop_one(Path(block_dir), aid, px), None
    except Exception as e:  # missing / corrupt file -> reported, not fatal
        return aid, None, repr(e)


def build_cache(block_dir, annotation_ids, out_dir, px: int = 72, workers: int = 2) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arr_path = out_dir / f"crops_{px}.npy"
    idx_path = out_dir / f"crops_{px}_index.csv"
    ids = [str(a) for a in annotation_ids]
    if arr_path.exists() and idx_path.exists():
        done = pd.read_csv(idx_path)
        if len(done) == len(ids) and done["ok"].all():
            print(f"[skip] cache complete: {arr_path}")
            return arr_path
    arr = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.float16, shape=(len(ids), px, px, px))
    ok, err = np.zeros(len(ids), bool), [""] * len(ids)
    pos = {a: i for i, a in enumerate(ids)}
    jobs = [(str(block_dir), a, px) for a in ids]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, (aid, patch, e) in enumerate(ex.map(_worker, jobs, chunksize=16)):
            i = pos[aid]
            if patch is not None:
                arr[i] = patch
                ok[i] = True
            else:
                arr[i] = -1000
                err[i] = e
            if n % 500 == 0:
                print(f"  cached {n}/{len(ids)}", flush=True)
    arr.flush()
    pd.DataFrame({"annotation": ids, "ok": ok, "error": err}).to_csv(idx_path, index=False)
    print(f"cache written: {arr_path}  ok={ok.sum()}  failed={(~ok).sum()}")
    return arr_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--px", type=int, default=72)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    ids = pd.read_csv(a.split_csv)["annotation"].astype(str).tolist()
    build_cache(a.blocks, ids, a.out, a.px, a.workers)
