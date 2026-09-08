# SoM2026 — results log

Recorded 2026-09-08.

## Leaderboard

| task | score | notes |
|---|---|---|
| Task 1 — LoS/NLoS | **0.72** | private test set. **Metric direction unconfirmed** — could be macro F1 (higher better) or the challenge's `1 - F1` (lower better). See below |
| Task 2 — channel prediction | not submitted | persistence baseline prepared, NMSE 0.0100 locally |
| Task 3 — depth estimation | not submitted | `dataset/Task3/` absent locally |

**Resolve the metric direction before trusting the 0.72.** `train.py:78` scores
`1 - f1_macro` internally, so both readings are live:

- macro F1 = 0.72 → good, well above the 0.375 majority-class baseline
- `1 - F1` = 0.72 → macro F1 = 0.28, *below* the majority baseline

Check the leaderboard column header or submit a majority-class-only file: it
scores macro F1 0.375, so whichever side of that the result lands on settles it.

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

## Local cross-validation (10 labelled samples)

5-fold stratified, seed 42, out-of-fold macro F1. `notebook/task1_train.ipynb`.

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

- settle the metric direction on the 0.72
- swap the flattened head for mean-pooling; it won CV and unbinds the model from a 384-token input
- submit Task 2 persistence — 0.0100 NMSE for zero training
- get `dataset/Task3/`, or submit blank to learn its shape
- `L_test.mat` does not ship, so `data_load_task_1` (`DataLoader.py:70`) raises; `main.py --task_id 1` cannot run as-is
