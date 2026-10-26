# Degradation-Aware Lung Nodule Malignancy Screening on CT

Deep Learning course project — New Uzbekistan University.
Team: Mumtozmahal Salimova · Dilrabo Orziqulova · Baxodir Abdug'afforov · Kamoliddin Saloxutdinov

**Question.** When a low-dose CT loses image detail (thicker slices, lower resolution, more noise),
does a malignancy classifier become *confidently wrong in the benign direction* — and can we train
and refer so that those cases go to a radiologist instead of producing a false "low risk" result?

This is a retrospective decision-support prototype for already-localised nodules. It is not a
diagnostic device and not a whole-scan detector.

## Status (progress presentation)

| part | state |
|---|---|
| Data download, audit, patient-level split, crop cache | implemented, tested |
| Official LUNA25 3D baseline (I3D, Kinetics init, baseline hyper-parameters) | implemented |
| Degradation operators (train family vs held-out family) | implemented, tested |
| Training objectives: `baseline`, `aug`, `symmetric`, `directional` | implemented |
| Post-hoc: temperature, threshold @ 95 % sensitivity, OOD control, error-referral head | implemented |
| Evaluation: AUROC/AUPRC/sens/spec, patient bootstrap CIs, false-low-risk at 80 % coverage | implemented |
| 3 seeds, LUNGx external test, CT viewer, model card | planned (weeks 11–16) |

## Quick start (Google Colab, GPU)

Open `notebooks/progress_run_colab.ipynb` in Colab with a GPU runtime and run all cells.
It tests the code, downloads the data (~6.8 GB), trains and evaluates every method, and writes the
figures used on the slides to Google Drive.

Locally (any CUDA machine):

```bash
pip install -e .
python -m lungdeg.data.download --root DATA            # annotations, nodule blocks, I3D weights
python -m lungdeg.prepare      --root DATA            # audit, split, crop cache
python -m lungdeg.train        --root DATA --method baseline
python -m lungdeg.train        --root DATA --method directional
python -m lungdeg.evaluate     --root DATA --run DATA/runs/directional_s2025
python -m lungdeg.report       --root DATA
python -m pytest -q                                   # torch tests skip automatically without torch
```

## Repository layout

```
src/lungdeg/
  config.py            paths + hyper-parameters (BASELINE-marked values mirror the official config)
  data/download.py     Zenodo + baseline-weights download, split-zip extraction
  data/splits.py       column resolution, data audit, patient-level split, observation weights
  data/geometry.py     baseline patch-extraction code (vendored, Apache-2.0)
  data/cache.py        one-off 72^3 HU crop cache (memory-mapped float16)
  data/augment.py      baseline rotation/translation + HU window, on the GPU
  data/degrade.py      train-family and held-out-family degradations
  data/dataset.py      dataset + class-balanced x inverse-observation-frequency sampler
  data/synthetic.py    synthetic crops for smoke tests only
  models/i3d.py        official baseline I3D (vendored, + feature output)
  models/net.py        malignancy network and error-referral head
  losses.py            BCE, symmetric consistency, directional loss
  train.py  evaluate.py  metrics.py  report.py  prepare.py
notebooks/progress_run_colab.ipynb
tests/                 CPU tests (data, metrics) + torch tests (augment, losses, end-to-end)
docs/                  method notes, progress slides
```

## Data and licences

* LUNA25 imaging (nodule blocks) — Zenodo 10.5281/zenodo.14223624, CC BY 4.0
* LUNA25 annotations — Zenodo 10.5281/zenodo.14673658, CC BY-NC 4.0 → non-commercial research use, attributed
* Baseline code and I3D weights — DIAGNijmegen/luna25-baseline-public, Apache 2.0
  (`src/lungdeg/models/i3d.py`, `src/lungdeg/data/geometry.py`; licence in `third_party_LICENSE_luna25_baseline.txt`)

No imaging data is stored in this repository; split manifests list only public identifiers.
