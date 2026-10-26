"""CPU-only tests (no torch needed): splitting, leakage, weights, metrics, cache geometry."""
import numpy as np
import pytest

from lungdeg.data.splits import audit, check_no_leakage, load_annotations, make_splits, split_summary
from lungdeg.data.synthetic import make_synthetic
from lungdeg.metrics import (basic_metrics, fit_temperature, patient_bootstrap, referral_outcome, sigmoid,
                             threshold_for_sensitivity)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    csv = make_synthetic(d, n_patients=200, px=24)
    df, _ = load_annotations(csv)
    return make_splits(df, seed=1)


def test_patient_level_split_has_no_leakage(synth):
    check_no_leakage(synth)
    assert set(synth.split) == {"train", "dev", "test"}
    assert set(synth.loc[synth.split == "dev", "dev_part"]) == {"head_fit", "calib", "thresh"}
    assert (synth.loc[synth.split != "dev", "dev_part"] == "").all()


def test_split_fractions_roughly_70_15_15(synth):
    frac = synth.groupby("split")["patient"].nunique() / synth["patient"].nunique()
    assert abs(frac["train"] - 0.70) < 0.05 and abs(frac["test"] - 0.15) < 0.05


def test_every_split_has_malignant_cases(synth):
    s = split_summary(synth)
    assert (s["malignant"] > 0).all()


def test_inverse_observation_weights_sum_to_one_per_nodule(synth):
    w = synth.groupby("nodule_key")["w_obs"].sum()
    assert np.allclose(w, 1.0)
    assert synth.groupby("nodule_key")["one_per_nodule"].sum().eq(1).all()


def test_audit_counts(synth):
    rep = audit(synth)
    assert rep["observations"] == len(synth)
    assert rep["malignant"] + rep["benign"] == len(synth)
    assert rep["duplicate_annotation_ids"] == 0
    assert rep["nodules_with_conflicting_labels"] == 0


def test_threshold_hits_target_sensitivity():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(100), np.zeros(900)].astype(int)
    p = np.r_[rng.beta(5, 2, 100), rng.beta(2, 5, 900)]
    t = threshold_for_sensitivity(y, p, 0.95)
    assert basic_metrics(y, p, t)["sensitivity"] >= 0.95


def test_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < sigmoid(z)).astype(int)
    assert abs(fit_temperature(z * 3, y) - 3) < 0.25


def test_referral_removes_highest_scores():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([.1, .9, .1, .1, .1, .1, .1, .1, .1, .1])   # case 0 is a false low-risk
    perfect = np.array([9, 0, 0, 0, 0, 0, 0, 0, 0, 0.])
    r = referral_outcome(y, p, perfect, t=0.5, coverage=0.9)
    assert r["false_low_risk"] == 0 and abs(r["coverage"] - 0.9) < 1e-9
    r0 = referral_outcome(y, p, np.zeros(10), t=0.5, coverage=1.0)
    assert r0["false_low_risk"] == 1 and r0["flr_rate"] == 0.5


def test_patient_bootstrap_ci_contains_point():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(100), 3)
    y = (rng.random(300) < 0.3).astype(int)
    p = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
    est = basic_metrics(y, p, 0.5)["auroc"]
    lo, hi = patient_bootstrap(g, lambda i: basic_metrics(y[i], p[i], 0.5)["auroc"], n_boot=200)
    assert lo <= est <= hi


def test_cache_crop_uses_baseline_geometry(tmp_path):
    from lungdeg.data.cache import crop_one
    blk = tmp_path / "blocks"
    (blk / "image").mkdir(parents=True); (blk / "metadata").mkdir()
    img = np.full((64, 128, 128), -900, np.float32)
    img[30:34, 62:66, 62:66] = 100            # bright cube at the block centre
    np.save(blk / "image" / "A1.npy", img)
    np.save(blk / "metadata" / "A1.npy", {"origin": np.zeros(3), "spacing": np.array([1.0, 0.7, 0.7]),
                                          "transform": np.eye(3)})
    c = crop_one(blk, "A1", px=72)
    assert c.shape == (72, 72, 72) and c.dtype == np.float16
    assert c[36, 36, 36] > 0 and c[0, 36, 36] < -800
