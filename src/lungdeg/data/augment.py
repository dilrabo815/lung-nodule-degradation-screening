"""GPU versions of the baseline's geometric augmentation and HU windowing.

Baseline behaviour reproduced:
  * random rotation about each of the three axes, uniform in [-20, 20] degrees;
  * random translation: a random direction, radius uniform in [0, 2.5] mm;
  * 64^3 output grid at 50/64 mm, trilinear interpolation;
  * HU clipped to [-1000, 400] and scaled to [0, 1].
The input is the cached 72^3 crop (same voxel spacing), so a rotated 64^3 grid
always has real image context; with zero rotation and translation the output is
exactly the central 64^3 voxels (verified in tests/test_augment.py).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _rot(ax: torch.Tensor, ay: torch.Tensor, az: torch.Tensor) -> torch.Tensor:
    cx, sx, cy, sy, cz, sz = ax.cos(), ax.sin(), ay.cos(), ay.sin(), az.cos(), az.sin()
    one, zero = torch.ones_like(ax), torch.zeros_like(ax)
    rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], -1).view(-1, 3, 3)
    ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], -1).view(-1, 3, 3)
    rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], -1).view(-1, 3, 3)
    return rx @ ry @ rz


def resample(x: torch.Tensor, out_px: int = 64, rotation_deg: float = 0.0,
             translation_mm: float = 0.0, spacing_mm: float = 50 / 64,
             generator: torch.Generator | None = None) -> torch.Tensor:
    """x: (B,1,N,N,N) HU on the cache grid -> (B,1,out,out,out)."""
    b, n = x.shape[0], x.shape[-1]
    dev = x.device
    scale = out_px / n
    if rotation_deg > 0:
        lim = math.radians(rotation_deg)
        ang = (torch.rand(b, 3, generator=generator, device="cpu") * 2 - 1) * lim
        r = _rot(ang[:, 0], ang[:, 1], ang[:, 2])
    else:
        r = torch.eye(3).expand(b, 3, 3)
    theta = torch.zeros(b, 3, 4)
    theta[:, :, :3] = r * scale
    if translation_mm > 0:
        d = torch.randn(b, 3, generator=generator)
        d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-8)
        rad = torch.rand(b, 1, generator=generator) * translation_mm
        half_extent_mm = n * spacing_mm / 2
        theta[:, :, 3] = d * rad / half_extent_mm
    grid = F.affine_grid(theta.to(dev, x.dtype), (b, 1, out_px, out_px, out_px), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def center(x: torch.Tensor, out_px: int = 64) -> torch.Tensor:
    n = x.shape[-1]
    o = (n - out_px) // 2
    return x[..., o:o + out_px, o:o + out_px, o:o + out_px]


def clip_and_scale(x: torch.Tensor, hu_min: float = -1000.0, hu_max: float = 400.0) -> torch.Tensor:
    return ((x - hu_min) / (hu_max - hu_min)).clamp_(0.0, 1.0)
