# Task1 Dataset — Layer-Wise Breakdown

```
H (10, 24, 8, 128)   sample=10, time=24, antenna=8, subcarrier=128
L (1, 10)            labels 0/1, six 0s + four 1s
CSV rows = 10 × 24 × 8 × 128 = 245,760
```

`.mat` files carry no metadata — only keys `H` and `label`.

## Address, not relationship

The four index columns are not four things that relate to each other. They are
four **coordinates** of one measurement, like lat/lon/altitude/time for a
weather reading.

```
   ADDRESS (where)                          VALUE (what)
┌────────┬──────┬─────────┬────────────┐  ┌──────────────────┐
│ sample │ time │ antenna │ subcarrier │→ │ real + imag  (h) │
└────────┴──────┴─────────┴────────────┘  └──────────────────┘
```

245,760 complex numbers, each tagged with where it came from.

## Video analogy

| video | CSI dataset |
|---|---|
| 10 clips | 10 **samples** |
| 24 frames per clip | 24 **time** slots |
| 8 pixel rows per frame | 8 **antennas** |
| 128 pixel columns per row | 128 **subcarriers** |
| pixel value (R,G,B) | channel value (real, imag) |
| 1 label per clip | 1 **label** per sample |

One sample = one clip. One time slot = one frame. One frame = an 8×128 image.

## The five layers

**1 — one number.** sample 0, time 0, antenna 0, subcarrier 0 → `h = 0.1269 + 0.3570j`.
Signal on that frequency, at that antenna, at that instant, scaled by
`|h| = 0.379` and phase-rotated by `∠h = 1.229 rad`. Whole content of one row.

**2 — sweep subcarrier (128).** Channel across frequency. Some frequencies fade,
some come through strong. The *frequency response*.

**3 — add antenna (8 × 128 = 1024).** Each antenna at a different physical point,
so each sees a different frequency response. A 2D snapshot:

```
              subcarrier →
            0    1    2   ...  127
   ant 0 [ h    h    h   ...   h  ]
   ant 1 [ h    h    h   ...   h  ]   ← one time slot
    ...  [ ...                     ]     = one 8×128 "image"
   ant 7 [ h    h    h   ...   h  ]
```

**4 — add time (24 × 8 × 128 = 24,576).** 24 snapshots in sequence. Room is not
frozen — someone moves, the grid changes shape frame to frame. A **video of the
channel**. This is one sample.

**5 — add sample (10 × 24,576 = 245,760).** Ten recordings, each with a label.
Whole dataset.

## What varies along each axis

| axis | what changes | what it reveals |
|---|---|---|
| **subcarrier** | probe frequency | **delay** — multipath echoes ripple across frequency |
| **antenna** | position in space | **direction** — phase shifts across array encode arrival angle |
| **time** | when measured | **motion** — Doppler, movement, gesture dynamics |
| **sample** | which recording | nothing physical. Just separate training examples |

`sample` differs in kind. The other three are physical dimensions *inside* one
measurement; `sample` separates one measurement from the next. Hence the label
attaches to `sample` alone.

## Label

Describes the **whole clip**, not any frame or pixel. Model reads all 24,576
numbers of a sample, outputs 0 or 1.

CSV repeats the label on all 24,576 rows of a sample only because flat CSV
cannot express "this value belongs to a group." No extra information.

## CSV columns

`export_csv.py` → `export/H_train.csv`.

| column | meaning |
|---|---|
| `sample` | 0..9. One full CSI recording. Row group key |
| `time` | 0..23. Snapshot inside that sample |
| `antenna` | 0..7. Array element. Spatial axis |
| `subcarrier` | 0..127. OFDM subcarrier. Frequency axis |
| `label` | class from `L_train.mat`, copied to every row of same sample |
| `real` | Re{h}. In-phase |
| `imag` | Im{h}. Quadrature |
| `magnitude` | `abs(h)`. Channel gain |
| `phase` | `angle(h)` rad, (-π, π]. Wrapped |

`magnitude`/`phase` derived from `real`/`imag`. Redundant, convenience only.

## Row ordering

Odometer — rightmost column spins fastest.

| level | rows per block | ticks every |
|---|---|---|
| subcarrier | 1 | row |
| antenna | 128 | 128 rows |
| time | 1024 | 1024 rows |
| sample | 24576 | 24576 rows |

```
row = sample*24576 + time*1024 + antenna*128 + subcarrier
```

## H is the channel, not the signal

```
y = H·x + noise   →   H = y / x     (x = known pilot, so H solvable)
```

CSV stores `H`, never `x` or `y`. Signal already divided out.

8 antennas = 8 *different observations* of the same reference signal, not the
same value eight times. Each antenna sits at a different spot (usually
half-wavelength apart), so each sees a different sum of reflections. That
variation is the entire information content of the antenna axis.

Which side has the 8 (Tx or Rx) is not stated by dataset or code —
`mask_strategy.py:180` just treats `(antenna, subcarrier)` as an 8×128 grid.
Call it "8 array elements" until challenge docs say otherwise.

## Time slots are not seconds

| SCS | slot length | 24 slots |
|---|---|---|
| 15 kHz | 1 ms | 24 ms |
| 30 kHz | 0.5 ms | 12 ms |
| 60 kHz | 0.25 ms | 6 ms |
| 120 kHz | 0.125 ms | 3 ms |

Milliseconds, not seconds. Datasets often snapshot at a CSI-RS period
(5/10/20 ms) rather than every slot, so 24 snapshots might span 120–480 ms —
still sub-second.

Nothing in the code uses a physical time unit. `time` is axis 1, 24 ordered
uniformly spaced snapshots, positional embedding indexes 0..23. Unit matters
only for Doppler→velocity or cross-dataset comparison, and needs the challenge
spec sheet (SCS + CSI periodicity). Data alone cannot tell you.

## What the model receives

`DataLoader.py` (`LoadBatch`) splits real/imag into 2 channels:

```
(batch, 2, 24, 8, 128)
        │   │  │   └── subcarrier
        │   │  └────── antenna
        │   └───────── time
        └───────────── real & imag
```

`Embed.py:15` runs a `Conv3d` over the last three axes — each sample treated as
a small 3D video volume, patterns spanning time, space and frequency at once.
`mask_strategy.py:180` (`antenna_masking`) hides whole antenna rows in training
and asks the model to reconstruct them, forcing it to learn how the axes
correlate.

## Summary

**Sample = which recording. Time/antenna/subcarrier = where inside that
recording. Together they name one complex number. The label names the
recording, not the number.**
