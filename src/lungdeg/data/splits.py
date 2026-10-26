"""Patient-level splitting, sample weights and the data audit.

Split design (proposal §8):
  * patients -> 70 % train / 15 % development / 15 % test, stratified on
    "patient has >= 1 malignant observation" so every split keeps ~9 % malignancy;
  * development patients -> head_fit (50 %), calibration (25 %), threshold (25 %);
  * all observations of a patient stay together, so repeated scans of the same
    nodule (NLST follow-up) can never leak across splits.

Sample weights: each observation gets 1 / (#observations of its NoduleID), so a
nodule seen at three time points counts once in total ("inverse observation
frequency"). The training sampler multiplies this by the class-balance weight
used by the official baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# canonical name -> accepted spellings (case-insensitive)
_ALIASES = {
    "patient": ["PatientID", "patient_id", "pid"],
    "series": ["SeriesInstanceUID", "series_uid", "seriesuid"],
    "annotation": ["AnnotationID", "annotation_id"],
    "nodule": ["NoduleID", "nodule_id", "LesionID"],
    "label": ["label", "Label", "malignancy"],
    "study_date": ["StudyDate", "study_date"],
    "age": ["Age_at_StudyDate", "Age", "age"],
    "gender": ["Gender", "Sex", "gender"],
}


def resolve_columns(df: pd.DataFrame) -> dict:
    low = {c.lower(): c for c in df.columns}
    out = {}
    for key, names in _ALIASES.items():
        for n in names:
            if n.lower() in low:
                out[key] = low[n.lower()]
                break
    missing = [k for k in ("patient", "annotation", "nodule", "label") if k not in out]
    if missing:
        raise KeyError(f"CSV is missing required columns {missing}; has {list(df.columns)}")
    return out


def load_annotations(csv_path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)
    cols = resolve_columns(df)
    df = df.rename(columns={v: k for k, v in cols.items()})
    df["label"] = df["label"].astype(int)
    # NoduleID is only unique within a patient in some exports -> make it global
    df["nodule_key"] = df["patient"].astype(str) + "/" + df["nodule"].astype(str)
    return df, cols


def _stratified_patient_split(pat: pd.DataFrame, fracs, names, rng) -> dict:
    """pat: one row per patient with column 'any_malig'. Returns patient->split."""
    assign = {}
    for _, grp in pat.groupby("any_malig"):
        ids = grp.index.to_numpy().copy()
        rng.shuffle(ids)
        cuts = np.round(np.cumsum(fracs)[:-1] * len(ids)).astype(int)
        for name, part in zip(names, np.split(ids, cuts)):
            for p in part:
                assign[p] = name
    return assign


def make_splits(df: pd.DataFrame, seed: int = 2025,
                fracs=(0.70, 0.15, 0.15), dev_fracs=(0.50, 0.25, 0.25)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pat = df.groupby("patient")["label"].max().rename("any_malig").to_frame()
    top = _stratified_patient_split(pat, fracs, ["train", "dev", "test"], rng)
    df = df.copy()
    df["split"] = df["patient"].map(top)
    dev_pat = pat.loc[[p for p, s in top.items() if s == "dev"]]
    sub = _stratified_patient_split(dev_pat, dev_fracs, ["head_fit", "calib", "thresh"], rng)
    df["dev_part"] = df["patient"].map(sub).fillna("")
    n_obs = df.groupby("nodule_key")["annotation"].transform("count")
    df["w_obs"] = 1.0 / n_obs
    # secondary analysis: one observation per nodule (latest study, else last row)
    order = df.sort_values(["nodule_key", "study_date"] if "study_date" in df else ["nodule_key"])
    last = order.groupby("nodule_key").tail(1).index
    df["one_per_nodule"] = df.index.isin(last)
    return df


def check_no_leakage(df: pd.DataFrame) -> None:
    per_patient = df.groupby("patient")["split"].nunique()
    assert (per_patient == 1).all(), "a patient appears in more than one split"
    per_nodule = df.groupby("nodule_key")["split"].nunique()
    assert (per_nodule == 1).all(), "a nodule appears in more than one split"


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("split")
    s = pd.DataFrame({
        "patients": g["patient"].nunique(),
        "nodules": g["nodule_key"].nunique(),
        "observations": g.size(),
        "malignant": g["label"].sum(),
    })
    s["malignant_%"] = (100 * s["malignant"] / s["observations"]).round(1)
    return s.loc[[x for x in ("train", "dev", "test") if x in s.index]]


def audit(df: pd.DataFrame, block_dir: Path | None = None, max_meta: int = 6163) -> dict:
    """Data checks reported on the 'Data work' slide."""
    rep: dict = {}
    rep["observations"] = int(len(df))
    rep["malignant"] = int(df["label"].sum())
    rep["benign"] = int((df["label"] == 0).sum())
    rep["patients"] = int(df["patient"].nunique())
    rep["series"] = int(df["series"].nunique()) if "series" in df else None
    rep["nodules"] = int(df["nodule_key"].nunique())
    rep["duplicate_annotation_ids"] = int(df["annotation"].duplicated().sum())
    obs = df.groupby("nodule_key").size()
    rep["obs_per_nodule"] = {str(k): int(v) for k, v in obs.value_counts().sort_index().items()}
    rep["nodules_with_repeats"] = int((obs > 1).sum())
    flips = df.groupby("nodule_key")["label"].nunique()
    rep["nodules_with_conflicting_labels"] = int((flips > 1).sum())
    per_pat = df.groupby("patient")["label"].agg(["max", "min"])
    rep["patients_with_both_classes"] = int(((per_pat["max"] == 1) & (per_pat["min"] == 0)).sum())
    rep["imbalance_ratio_benign_per_malignant"] = round(rep["benign"] / max(rep["malignant"], 1), 2)
    rep["rows_with_missing_values"] = int(df.isna().any(axis=1).sum())
    if "age" in df:
        rep["age_mean_by_label"] = {str(k): round(float(v), 1) for k, v in df.groupby("label")["age"].mean().items()}
    if block_dir is not None and Path(block_dir).exists():
        bd = Path(block_dir)
        have = {p.stem for p in (bd / "image").glob("*.npy")}
        missing = df.loc[~df["annotation"].astype(str).isin(have), "annotation"]
        rep["missing_block_files"] = int(len(missing))
        rep["missing_block_examples"] = missing.astype(str).head(5).tolist()
        sp, shapes, nzs = [], {}, []
        for aid in df["annotation"].astype(str).head(max_meta):
            mp = bd / "metadata" / f"{aid}.npy"
            if not mp.exists():
                continue
            m = np.load(mp, allow_pickle=True).item()
            sp.append(np.asarray(m["spacing"], float))
            img = np.load(bd / "image" / f"{aid}.npy", mmap_mode="r")
            shapes[str(tuple(img.shape))] = shapes.get(str(tuple(img.shape)), 0) + 1
            nzs.append(img.shape[0])
        if sp:
            sp = np.stack(sp)
            rep["block_shapes"] = shapes
            rep["slice_spacing_mm_quantiles"] = np.quantile(sp[:, 0], [0, .25, .5, .75, 1]).round(2).tolist()
            rep["inplane_spacing_mm_quantiles"] = np.quantile(sp[:, 1], [0, .25, .5, .75, 1]).round(2).tolist()
            rep["thick_slices_ge_2mm"] = int((sp[:, 0] >= 2.0).sum())
            # the 72-px cache crop spans 56.25 mm; shorter blocks get zero-padded (HU 0)
            rep["blocks_shorter_than_crop_in_z"] = int((sp[:, 0] * np.asarray(nzs) < 56.25).sum())
    return rep


def save_split(df: pd.DataFrame, out_dir: Path, seed: int) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = [c for c in ["patient", "series", "study_date", "annotation", "nodule", "nodule_key", "label",
                        "split", "dev_part", "w_obs", "one_per_nodule"] if c in df.columns]
    p = out_dir / f"split_seed{seed}.csv"
    df[keep].to_csv(p, index=False)
    json.dump(split_summary(df).reset_index().to_dict(orient="records"),
              open(out_dir / f"split_seed{seed}_summary.json", "w"), indent=2)
    return p
