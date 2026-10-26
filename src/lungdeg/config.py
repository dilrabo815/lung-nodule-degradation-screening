"""Single place for paths and hyper-parameters.

Values that mirror the official LUNA25 3D baseline (experiment_config.py) are
marked BASELINE so they are not changed by accident.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Paths:
    root: Path = Path("/content/luna25")          # Colab default; override locally
    @property
    def raw(self) -> Path: return self.root / "raw"            # zip parts + CSV
    @property
    def blocks(self) -> Path: return self.root / "nodule_blocks"  # image/ metadata/
    @property
    def csv(self) -> Path: return self.raw / "LUNA25_Public_Training_Development_Data.csv"
    @property
    def cache(self) -> Path: return self.root / "cache"         # pre-extracted crops
    @property
    def splits(self) -> Path: return self.root / "splits"
    @property
    def runs(self) -> Path: return self.root / "runs"
    @property
    def i3d_weights(self) -> Path: return self.raw / "model_rgb.pth"


@dataclass
class TrainConfig:
    method: str = "baseline"          # baseline | aug | symmetric | directional
    seed: int = 2025                  # BASELINE seed; we add 2026, 2027 later
    epochs: int = 10                  # BASELINE
    batch_size: int = 32              # BASELINE
    lr: float = 1e-4                  # BASELINE
    weight_decay: float = 5e-4        # BASELINE
    patience: int = 20                # BASELINE
    size_px: int = 64                 # BASELINE crop: 64^3 voxels ...
    size_mm: float = 50.0             # BASELINE ... covering 50 mm
    cache_px: int = 72                # cached crop (same spacing) so rotations have context
    rotation_deg: float = 20.0        # BASELINE +/-20 deg about each axis
    translation_mm: float = 2.5       # BASELINE random shift radius
    hu_min: float = -1000.0           # BASELINE clip_and_scale
    hu_max: float = 400.0             # BASELINE clip_and_scale
    # --- degradation-aware training ---
    p_degrade: float = 0.5            # 'aug': prob. a sample is degraded
    lam: float = 1.0                  # weight of consistency / directional term
    margin: float = 0.0               # logit margin before the directional penalty starts
    lam_benign: float = 0.1           # small penalty on benign upward drift (directional)
    amp: bool = True
    num_workers: int = 2
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)


METHODS = ("baseline", "aug", "symmetric", "directional")
