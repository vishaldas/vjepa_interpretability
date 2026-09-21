# Setup findings — all measured on this machine, not assumed

Machine: Apple M2 Max, 12 cores, 32 GB unified memory, macOS 23.6. No CUDA GPU.

## It fits, comfortably

**You do have GPU acceleration** — the M2 Max integrated GPU works through PyTorch's
MPS backend. It just isn't CUDA.

| | per clip | full 4,572-clip pass |
|---|---:|---:|
| CPU fp32 | 2.84 s | ~3.6 h |
| **MPS fp32** | **0.30 s** | **~26 min** |
| MPS fp16 | 0.26 s | ~19 min |

We use **MPS + fp32**: only ~20 % slower than fp16 and numerically safer.
Batch size makes almost no difference (0.325 s/clip at B=1 vs 0.307 at B=8); B=4 is used.

**Peak memory well inside budget.** Model is 326 M params (1.30 GB fp32). Resident set
during extraction ≈ 0.9 GB plus a ~2 GB in-RAM output buffer. Nothing swaps.

**MPS is numerically sound.** Against CPU fp32 on the same clip, cosine similarity of
mean-pooled features is ≥ 0.9999998 at every layer. Max relative deviation grows with
depth (4.7e-7 at layer 0 → 1.6e-3 at layer 24) — ordinary reduction-order
non-determinism, irrelevant for probing.

> Because those reductions are **not run-to-run deterministic**, the cache is treated as
> immutable once written. Re-extracting mid-project would make probe numbers drift
> slightly between experiments and cost hours chasing ghosts.

Video decoding is negligible: 5.6 ms/clip, ~0.4 min for all 4,572. Not worth caching frames.

## Model facts (read off the loaded model, not the paper)

- `facebook/vjepa2-vitl-fpc64-256`: 24 layers, d=1024, patch 16, tubelet 2.
- 16 frames → **8 temporal × 16×16 spatial = 2,048 tokens**.
- **Token order is `t*256 + h*16 + w`** (verified by zeroing frames 8–15, then by
  zeroing the left half of every frame).
- **The checkpoint ships the predictor**: 199 tensors, 12 layers, d=384, alongside the
  388 encoder tensors. Part 2 can therefore use a real forward-prediction behaviour
  channel rather than an activation-to-activation proxy.
- **Predictor is driveable.** Context = first 4 temporal blocks, target = last 4:
  `cos(pred, true) = 0.597` vs `cos(pred, shuffled) = 0.482`. Informative, but the gap is
  modest because ~98 % of tokens are background — for Part 2, restrict the readout to
  object tokens.

### `hidden_states[24]` ≠ `last_hidden_state`

The encoder has a final LayerNorm. `hidden_states[24]` is **pre**-norm (‖·‖ = 6987),
`last_hidden_state` is **post**-norm (‖·‖ = 4278). Had we conflated them, layer 24 would
have looked anomalous on the layerwise curve.

**We store 26 states**: indices 0–24 are the residual stream at each block boundary
(pre-final-norm, mutually consistent), index 25 is the post-LayerNorm output.

### Clip length: `fpc64` vs our 16 frames

The checkpoint was trained at 64 frames per clip; we feed 16. Both run:

| input | tokens | per clip | full pass |
|---|---:|---:|---:|
| 16 frames (native) | 2,048 | 0.32 s | 25 min |
| upsampled to 64 | 8,192 | 2.46 s | 188 min |

We extract at **16 frames**. The 64-frame variant is affordable as an *ablation on a
stratified subset* (~384 clips ≈ 16 min) — that turns decision #2 in PLAN.md from a
documented assumption into a measurement.

**Preprocessing choice:** the HF processor resizes the shortest edge to 292 then
centre-crops 256, which would discard ~12 % of each frame's border. Our clips are already
exactly 256×256, so we bypass it and feed frames directly.

## Data facts

- **The disk is orange, not blue** (the supplied dataset reference says blue).
- Constant apparent area ~335 px (⌀ ≈ 20.6 px ≈ 1.3 patches). **Never leaves the frame**
  at any speed — no occlusion or scale confound.
- Scale: **~20 px total displacement per m/s** over the 16 frames (≈1.33 px/frame per m/s).
- **The bottom of the speed range is below the encoder's resolution.** At magnitude 0.43,
  measured per-frame steps were 0.00–0.02 px — the rendering is pixel-quantised. Report
  metrics stratified by decile; this is a finding, not a caveat.
- Nuisance variables are decorrelated from targets (|r| < 0.07), so the design is clean.
- `speed` has accel ≡ 0; `acceleration` starts from rest. Disjoint motion regimes — you
  cannot test speed-probe invariance to acceleration within either.
- `direction` mixes both regimes (750/750) and theta spans all 64 values in **all three**
  datasets, so a direction probe trained on `direction` can be tested cross-dataset.

## Pooling: the `sal` design, and why the first attempt failed

Three poolings are cached. Each collapses the 256 spatial patches within a temporal
token, leaving the 8 temporal tokens intact (global mean is derivable downstream).

| name | what | leak? |
|---|---|---|
| `tmean` | uniform spatial mean — the paper's pooling | none |
| `sal` | mean over the **top-8 most salient** patches | **none** |
| `obj` | mean over **tracker-selected** patches | **yes — oracle** |

**`obj` has a label leak.** The mask comes from the disk trajectory, which *is* the
label; tokens carry positional information, so selecting at the disk's positions yields
features that vary with the trajectory even for an uninformative encoder. Only `obj`'s
**excess over the mask-only baseline** (`experiments.mask_baseline`, a probe on tracker
coordinates with no activations at all) is evidence about the representation.

**`sal` is the leak-free replacement.** Salience = a patch's deviation from its own
timestep's mean token. The encoder's own ranking recovers the true object tokens with
recall **1.00 through layer 12** and **≥0.89 at layer 24** — itself a presentable result
about where object identity stays spatially localised.

> **First attempt failed and was rejected.** Soft deviation-proportional weights gave
> `cos(tmean, sal) = 1.000`: the weight distribution flattens with depth (object weight
> share 0.61 at L1 → 0.03 at L20), so the weighted mean collapses onto the uniform mean.
> Top-K keeps the pooling sharp at every depth — after the fix, `cos(tmean, sal)` is
> 0.78–0.97 and `cos(sal, obj)` is 0.84–0.99.

## Why pooling is a confound, not a detail

The disk is ~1.3 of 256 patches per frame, so a global mean dilutes it to well under 1 %
of the pooled vector. That will artificially flatten early layers and make the emergence
transition look sharper and later than it is. **Always report the layerwise curve under
at least two poolings**, or the "Physics Emergence Zone" is confounded with pooling
dilution.

## A trap worth remembering

While unit-testing the INLP random control, the control collapsed exactly like real INLP.
Cause was the *test*, not the code: `default_rng(0)`'s first `(64,1)` draw generated both
the synthetic signal direction and the "random" direction, so the control was removing
precisely the signal. The control seed is now 12345 and the synthetic test uses an
independent stream. Real INLP collapses to R²≈0 after one round while the random control
holds at 0.96.

## The tracker-only model is an oracle *ceiling*, not just a leak floor

`experiments.mask_baseline()` probes the disk trajectory with **no encoder features at
all**. Given the nonlinear features a linear probe cannot form for itself (displacement
magnitudes, their differences, unit-normalised displacement components), it scores:

| variable | tracker-only | best encoder probe (held-out values) |
|---|---:|---:|
| speed | **MAE 0.011**, R² 1.000 | MAE 0.053 |
| direction | **1.19°**, R² 0.999 | 2.57° |

Two consequences, both worth a slide:

1. **`obj` pooling is not evidence about the encoder.** The mask alone beats every
   encoder probe, so any `obj`-pooled score is explained by the mask. Select layers with
   the leak-free `sal` pooling.
2. **It bounds the task.** The gap between the oracle and the encoder probe is what
   V-JEPA's representation *loses* relative to perfect object tracking — roughly 5× on
   speed and 2× on direction.

> A first version of this baseline used raw (x, y) coordinates and scored R² ≈ 0 on
> speed. That was an artefact: speed is the *norm* of a displacement, and a linear map
> cannot take a norm. The weak baseline would have made the floor flatteringly low and
> the comparison meaningless.

## Decile stratification needs the `clip` axis

The `value` axis holds out only ~7 distinct test values, so deciles are degenerate. The
resolution analysis uses the **`clip`** axis (384 test clips spanning all 64 values).
Say which axis a decile plot came from.

Speed, layer 16, `sal`, clip axis — absolute error is roughly flat while **relative**
error degrades sharply at the low end, exactly as the sub-pixel motion predicts:

| label (m/s) | 0.40 | 0.76 | 1.14 | 1.92 | 2.66 | 3.43 | 3.82 |
|---|---|---|---|---|---|---|---|
| MAE | .070 | .059 | .072 | .063 | .075 | .068 | .075 |
| relative | **17.6 %** | 7.7 % | 6.3 % | 3.3 % | 2.8 % | 2.0 % | 2.0 % |

## First results (preliminary — one quick sweep, not yet the full protocol)

Direction, held-out values, circular MAE (chance = 90°):

| layer | 1 | 4 | 8 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|---|
| `tmean` | 12.25° | 5.31° | 4.34° | 3.27° | 3.04° | 2.98° | 2.91° |
| `sal` | 11.35° | 4.84° | 3.59° | 3.67° | 2.57° | 3.39° | 3.35° |

Speed, held-out values, MAE (m/s): R² 0.978 already at layer 1, 0.993 by layer 16.

**Worth flagging for the write-up:** the paper reports that speed appears early while
direction emerges only at a sharp mid-depth transition. Here *both* are strongly
decodable from layer 1 (direction already at 12° vs 90° chance), improving smoothly with
depth rather than stepping. That is a genuine difference from the paper and most likely
reflects how much simpler this dataset is — one high-contrast disk on an empty
background, where direction is nearly readable off the raw patch embeddings. Verify
against the full protocol before putting it on a slide.

## Control: the layer-1 direction result is real, not positional

Direction is decodable at circular MAE 11.35° from **layer 1** — one attention block.
Before concluding this contradicts the paper's "direction emerges at a mid-depth
transition", we checked whether the probe is reading *direction* or merely *position*
(the disk's location correlates with its travel direction once it has moved).

| features (no encoder at all) | circular MAE | R² |
|---|---:|---:|
| frame-0 centroid only | **86.27°** | −0.015 |
| final-frame centroid only | 52.20° | +0.310 |
| displacement (x, y) | 8.21° | +0.667 |
| unit displacement | 6.71° | +0.937 |

**Frame-0 position alone is at chance** (86° vs 90°), confirming start positions are
independent of theta. So the layer-1 encoder probe at 11.35° is genuinely reading motion.
The observation stands: in *this* dataset direction is available almost immediately and
improves smoothly with depth, rather than stepping in at one third of the encoder. Most
likely a dataset-simplicity effect — one high-contrast disk on an empty background.

Layer 0 (raw patch embeddings, no attention) sits at 66.87°, so the first attention block
is what makes direction linearly available.

## INLP: superseded — see FINDINGS.md F3

This section previously reported that INLP never reached a floor (direction R² 0.992 →
0.886 after 298 dims). **That was a bug, not a property of the representation** — the
projection was stalling because `W/sd` exploded along already-removed directions. See
FINDINGS.md C1.

After the fix the curve terminates cleanly: direction R² 0.992 → 0.01 over 398 dims
(50 % at 142, 10 % at 266) while the random control stays flat at 0.991. Direction needs
~2× the dimensions of speed (69 / 114) and acceleration (67 / 103).

What remains valid from the original analysis: **the top 2 PCs carry only R² = 0.44**, so
the direction manifold is a closed curve in a high-dimensional subspace, not a planar
circle. Fit the periodic spline in ~64-D.

## Part 2 API note: the linear baseline must differ in the PATH

`Manifold.steer_at(X, targets, t, mode)` applies steering a fraction `t` of the way.
At **t = 1 the manifold and linear modes coincide by construction** — same endpoint. The
entire Goodfire comparison lives at intermediate `t`, where the linear chord leaves the
manifold. An endpoint-only evaluation would report "no difference between the methods".

> An earlier `steer_linear()` returned exactly what `steer()` returned — it computed the
> source centroid and discarded it. The synthetic test passed only because it exercised
> `path(mode='linear')`, which was correct, rather than the steering function itself.
