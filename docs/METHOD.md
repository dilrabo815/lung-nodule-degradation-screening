# Method notes (progress week)

## Pipeline

```
LUNA25 nodule block (64x128x128, native spacing)
   │  baseline extract_patch, 50/64 mm iso, no augmentation      (once, data/cache.py)
   ▼
72^3 HU crop, float16 memmap
   │  GPU: random rotation ±20° per axis, shift ≤2.5 mm → 64^3    (data/augment.py)
   │  [degraded view: train-family blur / noise]                 (data/degrade.py)
   │  clip HU [-1000, 400] → [0, 1]
   ▼
I3D (Kinetics-400 init, frozen BN) ── logit z ── malignancy p = σ(z / T)
   │
   └─ 1024-d embedding ─┐
detail stats (HF energy, noise) ─┼─► error-referral head ─► P(false low risk)
logit z ────────────────┘
```

## Training objectives (same network, same hyper-parameters)

| method | loss |
|---|---|
| baseline | BCE(z, y) |
| aug | BCE on a crop degraded with probability 0.5 |
| symmetric | BCE(z) + BCE(z̃) + λ (σ(z) − σ(z̃))² |
| directional | BCE(z) + BCE(z̃) + λ·y·relu(sg(z) − z̃)² + 0.1·λ·(1−y)·relu(z̃ − sg(z))² |

`sg` = stop-gradient: the clean prediction is the teacher, so the term cannot be satisfied by lowering
clean malignant scores. Only the harmful direction (malignant score falling when detail is lost) is
penalised at full weight.

## Post-hoc steps (development patients only)

| dev part | share | used for |
|---|---|---|
| head_fit | 50 % | error-referral head (clean + 6 train-family degraded views per case) |
| calib | 25 % | temperature scaling |
| thresh | 25 % | low-risk threshold t at 95 % sensitivity |

The whole dev split also gives the epoch curves and best-epoch selection (clean AUROC, as in the baseline).

## Referral policies compared at 80 % automated coverage

| policy | score (refer highest 20 %) |
|---|---|
| confidence | −&#124;z − z_t&#124; (closest to the threshold) |
| consistency | std of p over 4 test-time augmentations |
| OOD | Mahalanobis distance of the embedding to training embeddings |
| error head (ours) | learned P(malignant ∧ p < t) |

Primary outcome: malignant nodules automated with p < t ("false low risk"), averaged over the held-out
degradations, `directional + error head` vs `aug + confidence`. Target: ≥ 20 % relative reduction
(hypothesis, not a promise).

## Degradations

Train family (training + head fitting): Gaussian blur σ 0.3–0.9 mm, white noise 10–40 HU.
Held-out family (test only): slice thickening ×3/×5, in-plane down-sampling ×2/×3, correlated noise
60/100 HU, soft in-plane kernel, thickening + noise. These act on reconstructed images; they are a
proxy for acquisition changes, not a physics simulation.

## Changes since the proposal

1. Crops are cached once and augmented on the GPU (same geometry and HU window as the baseline);
   the baseline's per-epoch CPU resampling would leave a Colab GPU idle.
2. The official released baseline weights are **not** used as our baseline: they were trained on
   LUNA25 patients that may fall in our test split. We retrain the identical recipe on our split.
3. All methods, including the baseline, use the class-balanced × inverse-observation-frequency
   sampler so the comparison isolates the loss.
4. Development set is also used for epoch selection (the proposal did not name a separate validation
   split; carving one out would leave too few malignant cases for the post-hoc steps).
