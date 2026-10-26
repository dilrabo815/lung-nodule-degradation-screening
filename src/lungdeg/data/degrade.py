"""Image-degradation operators, applied in HU on the 64^3 / 0.78 mm crop.

Two disjoint families keep the robustness test honest:

TRAIN family (used for training and for fitting the error-referral head)
    blur  : isotropic Gaussian, sigma 0.3-0.9 mm  (detail loss)
    noise : white Gaussian, 10-40 HU             (dose reduction, crude)
HELD-OUT family (test time only - never seen in training, head fitting,
calibration or threshold selection)
    thick   : slice thickening - box average of k slices along z, re-sampled
    lowres  : in-plane resolution loss - k x k box down-sample, re-sampled
    lowdose : spatially correlated noise (low-dose CT texture)
    smooth  : soft reconstruction kernel - in-plane blur + mild noise

These are simulations on reconstructed images, not physics-based CT
re-simulation; that limitation is stated on the slides.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F

VOXEL_MM = 50 / 64


def _gauss1d(sigma_vox: float, device) -> torch.Tensor:
    r = max(1, int(math.ceil(3 * sigma_vox)))
    t = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma_vox) ** 2)
    return k / k.sum()


def blur(x: torch.Tensor, sigma_mm=(1.0, 1.0, 1.0)) -> torch.Tensor:
    """Separable Gaussian blur. sigma_mm = (z, y, x); 0 skips that axis."""
    out = x
    for axis, s in zip((2, 3, 4), sigma_mm):
        if s <= 0:
            continue
        k = _gauss1d(s / VOXEL_MM, x.device).to(x.dtype)
        shape = [1, 1, 1, 1, 1]
        shape[axis] = -1
        pad = [0, 0, 0, 0, 0, 0]
        p = (k.numel() - 1) // 2
        pad[2 * (4 - axis)] = pad[2 * (4 - axis) + 1] = p
        out = F.conv3d(F.pad(out, pad, mode="replicate"), k.view(shape))
    return out


def white_noise(x, sigma_hu, gen=None):
    return x + torch.randn(x.shape, generator=gen, device="cpu").to(x) * sigma_hu


def correlated_noise(x, sigma_hu, corr_mm=0.8, gen=None):
    n = torch.randn(x.shape, generator=gen, device="cpu").to(x)
    n = blur(n, (corr_mm, corr_mm, corr_mm))
    n = n / n.flatten(1).std(dim=1).view(-1, 1, 1, 1, 1).clamp_min(1e-6)
    return x + n * sigma_hu


def thick_slices(x, k: int):
    """Simulate thicker slices: average k neighbouring z-slices, keep every k-th, re-sample."""
    d = x.shape[2]
    pooled = F.avg_pool3d(x, kernel_size=(k, 1, 1), stride=(k, 1, 1), ceil_mode=True)
    return F.interpolate(pooled, size=(d, x.shape[3], x.shape[4]), mode="trilinear", align_corners=False)


def lowres_inplane(x, k: int):
    pooled = F.avg_pool3d(x, kernel_size=(1, k, k), stride=(1, k, k), ceil_mode=True)
    return F.interpolate(pooled, size=x.shape[2:], mode="trilinear", align_corners=False)


# ---------------------------------------------------------------- train family
def random_train_degradation(x: torch.Tensor, gen: torch.Generator | None = None,
                             sigma_mm=(0.3, 0.9), noise_hu=(10.0, 40.0)) -> torch.Tensor:
    """Per-sample random mild blur and/or noise (at least one of the two)."""
    b = x.shape[0]
    u = torch.rand(b, 4, generator=gen)
    use_blur = u[:, 0] < 0.5
    use_noise = (u[:, 1] < 0.5) | ~use_blur
    out = []
    for i in range(b):
        xi = x[i:i + 1]
        if use_blur[i]:
            s = sigma_mm[0] + float(u[i, 2]) * (sigma_mm[1] - sigma_mm[0])
            xi = blur(xi, (s, s, s))
        if use_noise[i]:
            n = noise_hu[0] + float(u[i, 3]) * (noise_hu[1] - noise_hu[0])
            xi = white_noise(xi, n, gen)
        out.append(xi)
    return torch.cat(out)


def train_family_fixed(level: int) -> Callable:
    """Deterministic train-family severities for validation curves (level 1..3)."""
    s, n = {1: (0.4, 15), 2: (0.7, 30), 3: (1.0, 45)}[level]
    return lambda x, gen=None: white_noise(blur(x, (s, s, s)), n, gen)


# ------------------------------------------------------------- held-out family
HELD_OUT: dict[str, Callable] = {
    "thick_k3": lambda x, gen=None: thick_slices(x, 3),        # ~2.3 mm slices
    "thick_k5": lambda x, gen=None: thick_slices(x, 5),        # ~3.9 mm slices
    "lowres_k2": lambda x, gen=None: lowres_inplane(x, 2),
    "lowres_k3": lambda x, gen=None: lowres_inplane(x, 3),
    "lowdose_60": lambda x, gen=None: correlated_noise(x, 60, gen=gen),
    "lowdose_100": lambda x, gen=None: correlated_noise(x, 100, gen=gen),
    "smooth_kernel": lambda x, gen=None: white_noise(blur(x, (0, 1.4, 1.4)), 20, gen),
    "thick_k3+lowdose_60": lambda x, gen=None: correlated_noise(thick_slices(x, 3), 60, gen=gen),
}
