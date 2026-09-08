# SoM2026 — results log

Recorded 2026-09-08.

## Leaderboard

| submission | task 1 score | approach |
|---|---|---|
| first | 0.72 | WiFo2 `tinypro` encoder + `Linear(24576,2)` head, the repo baseline |
| second | **0.75** | physics features + WiFo2 mean-pool, logistic regression |

Delta **+0.03** (described in session as ~0.04).

Task 2 not submitted — persistence baseline ready, NMSE 0.0100 locally.
Task 3 not submitted — `dataset/Task3/` absent locally.

### Metric — the challenge uses BINARY F1, not macro

The Task 1 spec defines TP as *correctly predicts the LoS class*, so precision,
recall and F1 are computed with **LoS as the positive class**. `train.py:78`
computes macro F1 instead, and every result recorded here before 2026-09-08 used
macro because it followed the repo.

The two metrics rank approaches identically on this data, so earlier conclusions
stand. What changes is the floor and the baseline:

| predictor | binary F1 (LoS) | macro F1 |
|---|---|---|
| all NLoS (majority class) | **0.000** | 0.375 |
| all LoS (trivial positive) | **0.571** | 0.286 |
| WiFo2 linear probe — repo baseline | **0.000** | 0.375 |

The repo's baseline head never predicts LoS once (confusion `[[6,0],[4,0]]`), so
it scores **binary F1 0.000** — no true positives at all. Macro F1 reported 0.375
for the same predictions because it credits the NLoS class it gets right for
free. Accuracy is 0.60 for *every* variant tested and is useless here.

Floor to beat is **0.571**, not 0.375.

**Confirmed from the challenge page** (Evaluation Metrics, Task 1):

```
Precision = TP/(TP+FP),  Recall = TP/(TP+FN),  F1 = 2*P*R/(P+R),  score_1 = F1
TP: correctly predicts LoS;  FP: predicts LoS when actually NLoS;
FN: predicts NLoS when actually LoS
```

So the leaderboard reports **binary F1, LoS positive**. Both 0.72 and 0.75 are
that number. Higher is better.

### Local CV does not predict the private ranking

| approach | local binary F1 (out-of-fold) | private leaderboard |
|---|---|---|
| WiFo2 linear probe — repo baseline | 0.169 | **0.72** |
| WiFo2 mean-pool probe (SGD head) | 0.475 | not submitted |
| WiFo2 mean-pool + logistic regression (k=3) | 0.679 | not submitted |
| physics + WiFo2 mean-pool, k=3 | 0.846 | **0.75** |
| gap, baseline to best | +0.68 | **+0.03** |

Only the first and last rows have private scores; the two mean-pool variants were
never submitted. Given the baseline's 0.169 -> 0.72 jump, their private scores are
not predictable from the local column — mean-pooling is a confirmed local gain
(+0.31 binary F1 over the baseline head, winning on 9 of 10 seeds) with unknown
private value.

The baseline scores 0.169 locally and 0.72 privately — a 4x jump. Two reasons,
neither of them a contradiction:

1. **Out-of-fold CV fits on 8 samples; the submitted model fits on all 10.** The
   submitted baseline predicted 15 NLoS / 5 LoS on the test set, not the
   all-NLoS collapse the CV folds produced. Two extra training samples changed
   its behaviour qualitatively — that is what 10-sample training looks like.
2. The private set is larger and its class balance is unknown.

Consequence: **the local CV ranks approaches only weakly and predicts absolute
private scores not at all.** A +0.68 local gain bought +0.03 privately.

Useful reference point: an all-LoS submission scores F1 = 2p/(1+p) where p is the
private LoS fraction — about 0.57 at 40% LoS, 0.67 at 50%. The current 0.75 beats
that, but not by a wide margin.

### Local CV badly overestimates

| | baseline | mean-pool (SGD) | mean-pool + logreg | physics + wifo | baseline -> best |
|---|---|---|---|---|---|
| local 5-fold CV, macro F1 | 0.428 | 0.561 | 0.748 | 0.880 | +0.45 |
| local 5-fold CV, binary F1 (the real metric) | 0.169 | 0.475 | 0.679 | 0.846 | +0.68 |
| private leaderboard (binary F1) | 0.72 | — | — | 0.75 | **+0.03** |

A 0.45 local gain bought 0.03 on the private set — **15x smaller**. Two things
follow, and both matter for how this branch is run:

1. **The local CV is nearly useless as a magnitude estimate.** 10 samples, folds
   of 2. Use it to rank approaches, never to predict the leaderboard.
2. **The private set is much easier than the local CV suggests** — the baseline
   scores 0.72 there against 0.428 locally. The 10 training samples are either
   unrepresentative or simply too few to estimate anything.

Practical consequence: a local gain under ~0.1 is not worth a submission slot,
and even a large one may move the leaderboard by a few hundredths.

## Architecture used

WiFo2, `size=tinypro`, encoder only. No decoder, no ResNet-18, no FastDepthV2 in
the Task 1 path.

```
(B,2,24,8,128) -> patchify -> Linear(128,64) -> +SinCos_3D pos -> 6x Block
               -> LayerNorm -> flatten(24576) -> Linear(24576,2) -> (B,2)
```

| setting | value |
|---|---|
| `embed_dim` | 64 |
| `depth` | 6 encoder blocks |
| `num_heads` | 4 (head_dim 16) |
| `mlp_ratio` | 1 |
| `patch_size` / `t_patch_size` | 4 / 4 |
| tokens | 384 = 6 time x 2 antenna x 32 subcarrier |
| `pos_emb` | `SinCos_3D` |
| `MoE` | False |
| masking | `fre` at `mask_ratio=0.0` (off) |
| pretrained | `weights/model_best.pkl`, backbone only, 183 tensors / 323,584 params |
| head | `Fine_Tune_Layer_LoS_NLoS` = `Linear(24576, 2)`, random init |
| trainable | 49,154 of 15,713,661 (0.313%) |

`tinypro` is the only preset whose shapes match `model_best.pkl`. Every other
size raises `RuntimeError: size mismatch` on load — see [[dataset]].

## Noise floor — read this before trusting any single number

10 samples, 5 folds of 2. Each macro F1 comes from 10 predictions, so it moves
in large jumps. Seed-averaged over 10 seeds, sigma is **0.06-0.12**. Any
single-seed change below ~0.1 is unmeasurable.

Two results that only survive averaging:

- mean-pooling is real: 0.428 -> 0.561, wins on 9 of 10 seeds
- weight decay is **not**: a single seed showed 0.375 -> 0.524, but averaged it
  is 0.428 -> 0.395, slightly worse. That was noise

Every number below is a 10-seed mean unless labelled LOOCV.

## Best result so far — physics features (branch `task1`)

`notebook/task1_physics.ipynb`, `physics_features.py`.

Scored under the challenge metric (binary F1, LoS positive); macro shown because
it is what the earlier notebooks reported.

| approach | binary F1 | macro F1 |
|---|---|---|
| all NLoS (majority class) | 0.000 | 0.375 |
| WiFo2 linear probe — the repo baseline | 0.169 | 0.428 |
| WiFo2 mean-pool probe (SGD head) | 0.475 | 0.561 |
| all LoS (trivial positive) | 0.571 | 0.286 |
| WiFo2 mean-pool + logistic regression | 0.679 | 0.748 |
| physics features alone (k=1) | 0.771 | 0.811 |
| **physics + WiFo2 mean-pool, k=3** | **0.846** | **0.880** |

Best model LOOCV: precision 1.000, recall 0.750, binary F1 0.857, confusion
`[[6,0],[1,3]]` — zero false positives, one LoS sample missed of four.

(The 0.169 for the linear probe is the 10-seed mean; at seed 42 alone it is
0.000.)

Two independent gains stack. Swapping the SGD-trained linear head for
`StandardScaler + SelectKBest + LogisticRegression` moved the *same* WiFo2
features from 0.561 to 0.748 — the frozen representation was better than the
baseline head could exploit. Physics features then added 0.13 more.

LOOCV confusion at 0.890: `[[6,0],[1,3]]` — one LoS sample missed, nothing else.

### The features that matter

Selected in 10/10 leave-one-out folds:

| feature | class 0 | class 1 | t-stat |
|---|---|---|---|
| `first_tap_frac` — power in delay tap 0 | 0.028 | 0.491 | 3.85 |
| `rms_delay` — RMS delay spread | 20.55 | 15.55 | 3.86 |
| `temporal_corr` — correlation across time slots | 0.560 | 0.743 | 2.77 |

### Label mapping, settled by the physics

**Class 1 = LoS, class 0 = NLoS.** Class 1 puts ~49% of its energy in the first
delay tap and has the shorter delay spread — that is a direct path. The repo
never states the mapping.

### Selection leakage, measured

Picking features by t-stat over all 10 samples gives 0.890; doing it inside each
training fold gives 0.811. **The gap is 0.08** — worth remembering whenever a
tuning result looks good here.

## Earlier cross-validation (baseline head only)

5-fold stratified, seed 42, out-of-fold macro F1. `notebook/task1_train.ipynb`.
Superseded by the table above; kept because it is what the 0.72 leaderboard
submission was built on.

| variant | trainable params | accuracy | macro F1 |
|---|---|---|---|
| linear probe (24576->2) — the baseline head | 49,154 | 0.60 | **0.375** |
| linear probe + weight decay 1e-2 | 49,154 | 0.60 | 0.524 |
| mean-pool probe (64->2) | 130 | 0.60 | **0.583** |
| mean-pool + weight decay 1e-2 | 130 | 0.60 | 0.583 |
| unfreeze last encoder block + head | 74,370 | 0.60 | 0.524 |
| majority-class baseline | — | 0.60 | 0.375 |

**The baseline head scores exactly the majority-class baseline.** Its confusion
matrix is `[[6,0],[4,0]]` — it predicted class 0 for every held-out sample.
Train accuracy was 1.00 throughout, i.e. pure memorisation of 10 points across
24,576 features.

The 130-param mean-pool head beat it with 378x fewer parameters. With 10
samples every number here is noise; rerun with a different seed to see the
spread.

## Task 2 baseline (measured, not submitted)

Persistence — predict the next channel as the previous one. Scored on all 500
labelled train samples:

| prediction | NMSE |
|---|---|
| copy `X_prev` (persistence) | **0.0100** |
| untrained WiFo2 task-2 path | 0.8729 |
| zeros | 1.0000 |

Per-sample spread: min 0.0085, median 0.0100, max 0.0125.

Persistence beats the model 87x because Task 2 was never trained —
`pos_align_layer` and `res18_align_layer` are randomly initialised. Over a short
horizon the channel barely moves, so "next = previous" is the standard channel
prediction baseline. Produced by `make_submission.py`.

## Submission format

The grader does `json.load(f)` on one file and reads all three task keys
unconditionally, flattening each against ground truth.

```json
{"task1": [20 ints], "task2": [[[[...]]]], "task3": [...]}
```

| shape | element count |
|---|---|
| task1 | 20 |
| task2 | 20 x 2 x 128 x 64 = 327,680 |
| task3 | unknown — dataset absent locally |

Task 2 layout assumed `(N, 2, 128, 64)`, real channel before imag, matching
`DataLoader.LoadBatch`. Not independently verified against the grader's own
ordering — if Task 2 scores near 1.0 instead of near 0.01, flip that first.

### Errors hit, in order

Each traceback leaked the next requirement:

1. CSV upload → `JSONDecodeError: Expecting value: line 1 column 1 (char 0)` — grader wants JSON
2. `task1`-only JSON → `KeyError: 'task2'` — all three keys must exist
3. `"task2": []` → `ValueError: operands could not be broadcast together with shapes (327680,) (0,)` — revealed Task 2's exact element count

Expect the same for Task 3: submit it blank and the error names the count.

`np.int64` is not JSON-serialisable — cast with `.tolist()` or `int()`.

## Negative results — tested, did not work

Recording these so they are not retried.

### Augmentation makes things worse

Pool of 10 originals + 32 augmented copies each, augmented copies confined to the
training fold. Macro F1:

| features | no aug | with aug |
|---|---|---|
| WiFo2 mean-pool (k=3) | 0.748 | **0.421** |
| physics (k=1) | 0.811 | **0.735** |
| physics + WiFo2 (k=3) | 0.880 | **0.667** |

Consistent across every feature set and every k. Two reasons:

1. **The physics features are invariant to most of it by construction.** Measured
   change from a global phase rotation, amplitude scaling, or antenna
   permutation: **0.0%** on `first_tap_frac`, `rms_delay`, `temporal_corr`,
   `K_max`. Only AWGN moves them (7% on `rms_delay`, 14% on `K_max`) — and that
   movement is corruption, shifting training features away from the clean
   validation distribution.
2. **WiFo2 was never trained to be invariant** to phase or antenna order, so
   augmented copies land in a region of feature space no real sample occupies.

Augmentations tried: global phase, amplitude scale 0.5-2x, antenna permutation,
AWGN 15-30 dB, time roll.

### Doppler, angular spread and per-antenna K-spread add nothing

Added 11 features to `physics_features.py` (17 -> 28): `K_ant_std/range/cv`,
`doppler_dc_frac/spread/entropy/peak_ratio`,
`angular_spread/entropy/peak_ratio/top_frac`.

Binary F1, seed-averaged:

| feature set | k=1 | k=2 | k=3 | k=5 |
|---|---|---|---|---|
| physics OLD (17) | 0.771 | 0.760 | 0.757 | 0.749 |
| **the 11 NEW alone** | **0.076** | **0.082** | **0.076** | 0.183 |
| physics ALL (28) | 0.771 | 0.760 | 0.757 | 0.705 |
| OLD + wifo | 0.700 | 0.808 | **0.846** | 0.720 |
| ALL + wifo | 0.700 | 0.808 | **0.846** | 0.720 |

The new features alone score near zero (LOOCV F1 0.000 at k=1,2,3 — no true
positives at all). Adding them to the existing set changes **nothing**: identical
scores to three decimals, identical std. `SelectKBest` never picks one.

Best new t-stat is 1.58 (`doppler_spread`) against 3.86 for `rms_delay`.

Why each fails, measured:

- **Doppler.** The channel decorrelates fast across the 24 slots — correlation
  against slot 0 falls 1.00 -> 0.60 by slot 7 and ends near 0.40. Time-domain
  K-factor is 0.0156, so `|time-mean|^2 << var`: Rayleigh-like along time for both
  classes. Power is spread across Doppler bins with the DC bin holding 0.006,
  *below* the 1/24 = 0.042 a uniform spectrum would give. No LoS-vs-NLoS contrast
  survives.
- **Angular spread.** Only 8 antennas, so beamspace has 8 independent bins.
  Zero-padding the FFT to 64 interpolates but adds no resolution. Array geometry
  is also unstated — without confirmed half-wavelength ULA spacing the
  FFT-beamspace reading is not even the right transform.
- **Per-antenna K spread.** K is ~0.006-0.01 everywhere, so both classes look
  Rayleigh by this estimator. The spread of a near-zero quantity is noise. This is
  also why every K-based feature is weak while the delay-domain ones are strong.

Features kept in the module (they cost nothing at inference and the private set
behaves nothing like these 10 samples), but marked as measured-neutral.

### Threshold tuning does not help

The best model sits at precision 1.000, recall 0.750, so lowering the decision
threshold below 0.5 looked like free recall. Swept 0.20-0.80, 10 seeds:

| threshold | binary F1 | std | precision | recall |
|---|---|---|---|---|
| 0.35 | 0.823 | 0.125 | 0.840 | 0.825 |
| 0.45 | 0.856 | 0.094 | 0.935 | 0.800 |
| **0.50 (default)** | **0.846** | **0.032** | 0.975 | 0.750 |
| 0.60 | 0.808 | 0.078 | 0.975 | 0.700 |

Best is +0.010 at threshold 0.45, well inside the noise. More telling: std is
0.032 at the default and rises to 0.094-0.125 as the threshold drops. The default
is both near-best and by far the most stable. LOOCV peaks at 0.889 (threshold
0.35, TP=4 FP=1 FN=0) but 5-fold gives 0.823 +/- 0.125 there — the protocols
disagree by more than the gain, the signature of tuning noise.

High precision is also the right operational choice: a false LoS call breaks the
precoding rank assumption and can drop the link, while a false NLoS call only
wastes spatial degrees of freedom.

## Next

- ~~settle the metric direction~~ — resolved, higher is better (macro F1)
- ~~submit the physics+WiFo2 predictions~~ — done, 0.72 -> 0.75
- ~~augmentation~~ — tested, hurts. See negative results
- ~~threshold tuning~~ — tested, +0.01 inside noise. See negative results
- more physics: per-antenna K-factor spread, Doppler from the time axis, angular spread via spatial FFT
- MAE pretraining on the 20 unlabelled test samples — self-supervised, adapts the backbone with no labels
- submit Task 2 persistence — 0.0100 NMSE for zero training
- get `dataset/Task3/`, or submit blank to learn its shape
- `L_test.mat` does not ship, so `data_load_task_1` (`DataLoader.py:70`) raises; `main.py --task_id 1` cannot run as-is
