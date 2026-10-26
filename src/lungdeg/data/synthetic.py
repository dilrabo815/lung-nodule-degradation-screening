"""Tiny synthetic stand-in for LUNA25 so the whole pipeline can be smoke-tested
in a minute without downloading 6.8 GB. Malignant 'nodules' are larger, with a
lobulated border; benign ones are small and smooth. NOT used for any result.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic(out_dir, n_patients=60, px=72, seed=0):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in range(n_patients):
        malignant_patient = rng.random() < 0.25
        for nod in range(rng.integers(1, 3)):
            label = int(malignant_patient and nod == 0)
            for obs in range(rng.integers(1, 3)):
                rows.append(dict(patient=f"P{p:04d}", series=f"S{p}_{obs}", study_date=2000 + obs,
                                 annotation=f"A{p:04d}_{nod}_{obs}", nodule=nod, label=label))
    df = pd.DataFrame(rows)
    arr = np.lib.format.open_memmap(out_dir / f"crops_{px}.npy", "w+", np.float16, (len(df), px, px, px))
    g = np.indices((px, px, px)).astype(np.float32) - px / 2
    r = np.sqrt((g ** 2).sum(0))
    theta = np.arctan2(g[1], g[2])
    for i, lab in enumerate(df["label"]):
        rad = rng.uniform(7, 11) if lab else rng.uniform(3, 6)
        border = rad * (1 + (0.25 * np.sin(5 * theta) if lab else 0))
        vol = np.where(r < border, 40.0, -850.0) + rng.normal(0, 25, r.shape)
        arr[i] = vol.astype(np.float16)
    arr.flush()
    pd.DataFrame({"annotation": df["annotation"], "ok": True, "error": ""}).to_csv(
        out_dir / f"crops_{px}_index.csv", index=False)
    csv = out_dir / "synthetic_annotations.csv"
    df.rename(columns={"patient": "PatientID", "series": "SeriesInstanceUID", "study_date": "StudyDate",
                       "annotation": "AnnotationID", "nodule": "NoduleID"}).to_csv(csv, index=False)
    return csv
