"""Model wrappers.

MalignancyNet  - the official-baseline I3D (Kinetics-400 initialised, frozen BN,
                 single-logit head). Used unchanged by every training method so
                 differences come from the loss / data, not the architecture.
ErrorReferralHead - small MLP fitted *after* the classifier is trained, on the
                 development 'head_fit' patients only. Input: the 1024-d image
                 embedding of the (possibly degraded) crop, the malignancy logit
                 and two image-detail statistics. Output: probability that this
                 case is a *false low-risk* result (malignant but scored below the
                 low-risk threshold) - i.e. the harm we want to refer.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .i3d import I3D


class MalignancyNet(nn.Module):
    def __init__(self, weights_path=None, pretrained=True):
        super().__init__()
        self.backbone = I3D(num_classes=1, input_channels=3, pre_trained=pretrained,
                            freeze_bn=True, weights_path=weights_path)

    def forward(self, x):
        return self.backbone(x)

    def forward_with_features(self, x):
        return self.backbone.forward_with_features(x)


def detail_stats(x01: torch.Tensor) -> torch.Tensor:
    """Two cheap image-quality cues on the [0,1] crop: high-frequency energy
    (Laplacian) and a robust noise estimate in the lung-parenchyma window."""
    lap = (-6 * x01
           + x01.roll(1, 2) + x01.roll(-1, 2) + x01.roll(1, 3)
           + x01.roll(-1, 3) + x01.roll(1, 4) + x01.roll(-1, 4))
    hf = lap.abs().flatten(1).mean(1)
    dz = (x01[:, :, 1:] - x01[:, :, :-1]).abs().flatten(1)
    noise = dz.median(dim=1).values
    return torch.stack([hf, noise], 1)


class ErrorReferralHead(nn.Module):
    def __init__(self, feat_dim=1024, hidden=128, dropout=0.2):
        super().__init__()
        self.feat_norm = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + 3, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, feats, logit, stats):
        z = torch.cat([self.feat_norm(feats), logit[:, None], stats * 10.0], 1)
        return self.mlp(z).squeeze(1)
