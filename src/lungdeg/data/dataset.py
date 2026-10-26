"""Torch dataset over the cached crops plus the baseline's balanced sampler."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class CropDataset(Dataset):
    """Returns raw-HU 72^3 crops; augmentation/degradation happen on the GPU."""

    def __init__(self, cache_npy: Path, cache_index_csv: Path, frame: pd.DataFrame):
        self.arr_path = str(cache_npy)
        self.arr = None  # opened lazily so each worker gets its own memmap
        idx = pd.read_csv(cache_index_csv)
        pos = {str(a): i for i, a in enumerate(idx["annotation"].astype(str))}
        self.frame = frame.reset_index(drop=True)
        self.rows = np.array([pos[str(a)] for a in self.frame["annotation"]], dtype=np.int64)
        self.labels = self.frame["label"].to_numpy().astype(np.float32)
        self.w_obs = self.frame["w_obs"].to_numpy().astype(np.float32) if "w_obs" in self.frame else np.ones(len(self.frame), np.float32)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        if self.arr is None:
            self.arr = np.load(self.arr_path, mmap_mode="r")
        x = torch.from_numpy(np.asarray(self.arr[self.rows[i]], dtype=np.float32))[None]
        return {"x": x, "y": torch.tensor(self.labels[i]), "w": torch.tensor(self.w_obs[i]),
                "idx": torch.tensor(i)}


def balanced_sampler(labels: np.ndarray, w_obs: np.ndarray | None = None, seed: int = 0):
    """Baseline class-balanced sampler, optionally times inverse observation frequency."""
    labels = np.asarray(labels).astype(int)
    n = len(labels)
    cnt = np.bincount(labels, minlength=2)
    w = n / cnt[labels].astype(float)
    if w_obs is not None:
        w = w * np.asarray(w_obs)
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=n,
                                 replacement=True, generator=g)


def make_loader(ds: CropDataset, batch_size=32, train=False, workers=2, seed=0, use_obs_weights=True):
    if train:
        sampler = balanced_sampler(ds.labels, ds.w_obs if use_obs_weights else None, seed)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=workers,
                          pin_memory=True, drop_last=True, persistent_workers=workers > 0)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
