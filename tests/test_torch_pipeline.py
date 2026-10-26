"""GPU/torch tests - run in Colab (cell 2 of the notebook) before real training.

The end-to-end test trains each method for a few steps on synthetic crops,
evaluates it and builds the report, so a broken pipeline fails in ~2 minutes
instead of after an hour of training.
"""
import json

import pytest

torch = pytest.importorskip("torch")

from lungdeg.config import Paths, TrainConfig  # noqa: E402
from lungdeg.data.augment import center, clip_and_scale, resample  # noqa: E402
from lungdeg.data.degrade import HELD_OUT, blur, random_train_degradation, thick_slices  # noqa: E402
from lungdeg.losses import directional, symmetric_consistency, total_loss  # noqa: E402
from lungdeg.models.net import ErrorReferralHead, MalignancyNet, detail_stats  # noqa: E402


def test_resample_identity_equals_center_crop():
    x = torch.randn(2, 1, 72, 72, 72)
    a = resample(x, 64, rotation_deg=0, translation_mm=0)
    assert torch.allclose(a, center(x, 64), atol=1e-4)


def test_augmented_shape_and_range():
    x = torch.randn(3, 1, 72, 72, 72) * 300 - 500
    y = clip_and_scale(resample(x, 64, 20, 2.5))
    assert y.shape == (3, 1, 64, 64, 64) and float(y.min()) >= 0 and float(y.max()) <= 1


def test_blur_preserves_constant_and_reduces_detail():
    c = torch.full((1, 1, 32, 32, 32), 50.0)
    assert torch.allclose(blur(c, (1, 1, 1)), c, atol=1e-3)
    x = torch.randn(1, 1, 32, 32, 32)
    assert blur(x, (1, 1, 1)).std() < x.std()


def test_degradations_keep_shape():
    x = torch.randn(2, 1, 64, 64, 64) * 200 - 600
    g = torch.Generator().manual_seed(0)
    assert random_train_degradation(x, g).shape == x.shape
    for name, f in HELD_OUT.items():
        assert f(x, g).shape == x.shape, name
    assert torch.allclose(thick_slices(torch.full_like(x, 7.0), 3), torch.full_like(x, 7.0), atol=1e-4)


def test_directional_only_penalises_harmful_direction():
    z = torch.tensor([2.0, 2.0, -1.0, -1.0])
    y = torch.tensor([1.0, 1.0, 0.0, 0.0])
    z_down = z - 1.0       # every score drops under degradation
    z_up = z + 1.0
    # drop hurts malignant (y=1) only; rise hurts benign only, with smaller weight
    assert directional(z, z_down, y, lam_benign=0.1) > 0
    assert directional(z, z_up, y, lam_benign=0.0) == 0
    assert symmetric_consistency(z, z_down) > 0 and symmetric_consistency(z, z_up) > 0


def test_directional_does_not_backprop_into_clean():
    z = torch.tensor([1.0], requires_grad=True)
    zd = torch.tensor([0.0], requires_grad=True)
    directional(z, zd, torch.tensor([1.0])).backward()
    assert z.grad is None or float(z.grad.abs().sum()) == 0
    assert float(zd.grad.abs().sum()) > 0


def test_model_forward_shapes():
    m = MalignancyNet(pretrained=False).eval()
    x = torch.rand(2, 1, 64, 64, 64)
    with torch.no_grad():
        z, f = m.forward_with_features(x)
        assert torch.allclose(z, m(x), atol=1e-5)
    assert z.shape == (2,) and f.shape == (2, 1024)
    h = ErrorReferralHead(1024)
    assert h(f, z, detail_stats(x)).shape == (2,)


def test_total_loss_methods():
    z, zd, y = torch.randn(4), torch.randn(4), torch.tensor([1., 0., 1., 0.])
    for m in ("baseline", "aug"):
        l, parts = total_loss(m, z, y)
        assert set(parts) == {"bce_clean"}
    for m in ("symmetric", "directional"):
        l, parts = total_loss(m, z, y, zd)
        assert set(parts) == {"bce_clean", "bce_deg", "reg"}


@pytest.mark.slow
def test_end_to_end_on_synthetic(tmp_path):
    from lungdeg import evaluate, report, train
    from lungdeg.data.splits import load_annotations, make_splits, save_split
    from lungdeg.data.synthetic import make_synthetic
    paths = Paths(tmp_path)
    csv = make_synthetic(paths.cache, n_patients=120, px=72)
    df = make_splits(load_annotations(csv)[0], seed=0)
    split = save_split(df, paths.splits, 0)
    for method in ("baseline", "directional"):
        cfg = TrainConfig(method=method, epochs=2, batch_size=8, num_workers=0, seed=1)
        run = train.run(cfg, paths, split, paths.runs / f"{method}_s1", max_steps=3, pretrained=False)
        assert (run / "best.pth").exists()
        evaluate.run_eval(run, paths, split, tta=1, n_boot=20, head_views=1, workers=0)
        assert (run / "test_referral.csv").exists()
    report.main(["--root", str(tmp_path)])
    h = json.load(open(tmp_path / "report" / "headline.json"))
    assert "clean_test" in h and (tmp_path / "report" / "curves.png").exists()
