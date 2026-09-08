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

### Metric direction — resolved

Higher is better, so the leaderboard reports **macro F1**, not the `1 - F1` that
`train.py:78` computes internally. Evidence: the physics model is clearly better
by local CV (0.880 vs 0.428) and scored *higher* on the private set (0.75 vs
0.72). Under a `1 - F1` column a better model would have scored lower.

So 0.75 macro F1, against a 0.375 majority-class floor.

### Local CV badly overestimates

| | baseline | physics | delta |
|---|---|---|---|
| local 5-fold CV (10 samples) | 0.428 | 0.880 | **+0.45** |
| private leaderboard | 0.72 | 0.75 | **+0.03** |

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

| approach | macro F1 (5-fold, 10 seeds) | LOOCV |
|---|---|---|
| majority class | 0.375 | — |
| WiFo2 linear probe — the repo baseline | 0.428 | — |
| WiFo2 mean-pool probe (SGD head) | 0.561 | — |
| WiFo2 mean-pool + logistic regression | 0.748 | 0.762 |
| physics features alone (k=1) | 0.811 | 0.792 |
| **physics + WiFo2 mean-pool, k=3** | **0.880** | **0.890** |

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

## Next

- ~~settle the metric direction~~ — resolved, higher is better (macro F1)
- ~~submit the physics+WiFo2 predictions~~ — done, 0.72 -> 0.75
- augmentation: global phase rotation, AWGN at known SNR, antenna permutation. All label-preserving, turns 10 samples into thousands
- more physics: per-antenna K-factor spread, Doppler from the time axis, angular spread via spatial FFT
- MAE pretraining on the 20 unlabelled test samples — self-supervised, adapts the backbone with no labels
- submit Task 2 persistence — 0.0100 NMSE for zero training
- get `dataset/Task3/`, or submit blank to learn its shape
- `L_test.mat` does not ship, so `data_load_task_1` (`DataLoader.py:70`) raises; `main.py --task_id 1` cannot run as-is
