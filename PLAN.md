# V-JEPA physics take-home — problem analysis & plan

> **Status.** Stage 0 (environment + smoke tests) and Stage 1 (extraction + cache) are
> **done**. Measured facts now supersede the estimates below — see
> [notes/SETUP_FINDINGS.md](notes/SETUP_FINDINGS.md) for what was verified on this
> machine and [notes/CACHE.md](notes/CACHE.md) for how to use the cache.
>
> Headline corrections to the original estimates:
> - A full pass is **~23 min on MPS**, not 1–2 h. Your M2 Max *does* have usable GPU
>   acceleration (Metal, not CUDA).
> - `hidden_states[24]` ≠ `last_hidden_state` (pre- vs post-final-LayerNorm). We store
>   **26** states, not 25.
> - The checkpoint **ships the predictor** (12 layers, d=384) and it is driveable, so
>   Part 2 gets a real behaviour channel.
> - Three poolings are cached, including a **leak-free** `sal` pooling that replaces the
>   originally-planned object pooling as the basis for layer selection.
> - Full token grids and the 64-frame variant are **deliberately not cached** — cheaper
>   to re-extract on demand than to manage 57 GB.

## 1. What is actually being asked

One sentence: **does V-JEPA 2 carry direction / speed / acceleration as recoverable
structure in its residual stream, where does that structure live, how many dimensions
does it occupy, and can you causally control it by intervening on that structure?**

The encoder stays frozen throughout. Everything you build — probes, nullspace
projections, steering bases, splines — is fitted *on top of* extracted activations.
No training of V-JEPA itself.

The deliverable is a ~15-minute presentation, so the unit of work is
"defensible result + figure", not "clean codebase".

### Part 1 is a reproduction, in three escalating steps

| Step | Question | Output |
|---|---|---|
| 1. Layerwise probing | *Where* is each variable available? | probe metric vs layer, ×3 variables |
| 2. Iterative nullspace probing | *How many dimensions*, and how redundant? | metric vs #directions-removed |
| 3. Multi-probe subspace steering | Is that subspace *causal*? | steered-vs-target readout |

These are a chain, not three independent tasks: step 1 picks the layer for step 2, and
step 2 tells you how many probes step 3 needs in its basis.

### Part 2 is an open-ended extension

Replace Part 1's **linear** subspace with a **curved manifold** (spline) fitted to the
representation, and compare steering along the curve vs steering along the straight line.

## 2. What the papers actually do

Both verified from arXiv (abstracts verbatim; method details below came from an
automated extraction over the HTML — **treat specific numbers as unverified until you
read the papers yourself**, and in particular derive your own emergence layer rather
than assuming theirs).

### Joseph et al., *Interpreting Physics in Video World Models* (arXiv 2602.07050)

Authors include Quentin Garrido, Randall Balestriero, Matthew Kowal, Thomas Fel,
Blake Richards, Mike Rabbat.

Core claim from the abstract: video models do **not** use factorized, physics-engine-like
representations; they use **distributed** ones that are nonetheless sufficient for
accurate prediction. They identify a **"Physics Emergence Zone"** — a sharp
intermediate-depth transition where physical variables become accessible. **Speed appears
early; direction emerges only at the transition and has high-dimensional circular
structure.**

Reported method details (unverified):
- V-JEPA 2 L/H/G, frozen. Probe the residual stream at every layer.
- Two probe families: linear probes on **mean-pooled space-time patches**, and
  **patch-preserving attentive-MLP probes**.
- Linear probes `f(h) = Wh + b`, MSE. Direction as `(sin θ, cos θ)`.
- INLP: fit probe → QR on `W` → `X ← X − X Q Qᵀ` → refit, until metric floors.
- Steering basis: `V, _ = QR([W₁ᵀ … W_Kᵀ])`, then `c = Vᵀx`, `x⊥ = x − Vc`, solve for
  target `c*`, reconstruct `x* = Vc* + x⊥`.
- Emergence zone ≈ 1/3 depth (layers 7–8 of 24 for ViT-L). Direction reportedly needs
  ~40–50 directions to erase. Steering evaluated with held-out probes (70/30 probe split).

**Read priority: the appendix.** The INLP procedure, the QR steering construction and the
probe train/test split are what you need to reproduce; the main text is framing.

### Wurgaft et al. (Goodfire), *Manifold Steering* (arXiv 2605.05115)

Core idea: fit an **activation manifold** `M_h` to representations *and* a **behavior
manifold** `M_y` to outputs. Steering **along** `M_h` produces behavior that follows
`M_y`; **linear** steering cuts through off-manifold regions and produces unnatural
outputs. Tagline: *steering is not about finding the right direction, it's about finding
the right geometry.*

Method (unverified specifics):
- PCA activations to ~64 dims; compute a **centroid per concept value**; fit a **cubic
  spline through the centroids** (thin-plate splines for 2D intrinsic geometry).
- Linear path:      `π_lin(t) = (1−t)h₀ + t h₁`
- Manifold path:    `π_m(t) = s((1−t)u₀ + t u₁)`,  `uᵢ = s⁻¹(hᵢ)`
  where `s` maps intrinsic coordinates → activation space.
- Metrics: off-manifold "energy" of the path; **isometry correlation** (geodesic distance
  on `M_h` vs on `M_y`); pullback R².

**Read priority: the centroid-spline fitting and Eqs. 1–2.** Skip the LM experiments.

## 3. What the supplied data actually is (measured, not assumed)

```
direction     1500 clips   64 thetas × ~23-24 reps   speed ∈ {0..7}, accel ∈ {0..10} varied
                           motion: 750 velocity / 750 acceleration   <- mixed regime
speed         1536 clips   64 magnitudes × 24 reps   0.25 – 4.0 m/s    accel ≡ 0
acceleration  1536 clips   64 magnitudes × 24 reps   0.25 – 10.0 m/s²  speed ≡ 0 (starts at rest)
```
4,572 clips total, 52 MB. 16 frames, 24 fps, 256×256.

**Verified properties:**
- Nuisance variables are decorrelated from targets (|r| < 0.07 for theta, start position).
  The dataset is cleanly designed — say so, it justifies simple probes.
- The disk is **orange**, not blue as DATA.md claims. Constant apparent area (~335 px,
  ⌀≈20 px ≈ 1.3 patches). **Never leaves the frame** at any speed — no occlusion or
  scale confound.
- Scale: total displacement over 16 frames ≈ **20 px per m/s**. So 1 m/s ≈ 1.33 px/frame.
- **The bottom of the speed range is below the encoder's resolution.** At magnitude 0.43
  the measured per-frame step was 0.00–0.02 px (pixel-quantized rendering); total
  displacement 8 px over the whole clip. Tubelet size is 2, so effective temporal
  sampling is 8 steps.

> **Treat that last point as a finding, not a caveat.** Report speed and acceleration
> metrics **stratified by label decile**. Error will concentrate at the low end, and
> "here is where the representation runs out of resolution" is a genuinely interesting
> slide.

**Free transfer test:** theta spans all 64 values in *all three* datasets. A direction
probe trained on `direction` can be evaluated on `speed` and `acceleration` clips —
cross-dataset and cross-motion-regime. Best generalization test the data affords.

**Note the asymmetry:** `speed` has accel ≡ 0 and `acceleration` starts from rest, so the
two magnitude datasets are disjoint motion regimes. You cannot test speed-probe
invariance to acceleration within either one.

## 4. The evaluation protocol — three independent held-out axes

This is what DATA.md means by "a fair test of generalization", and getting it wrong
silently invalidates both parts.

The data is **64 discrete values × 24 reps**. A random clip split puts magnitude=2.15 in
both train and test, so a probe can memorize value→activation and R² tests nothing about
continuous structure — which is precisely what Parts 1 and 2 claim.

1. **Held-out VALUES** — *the headline split.* Train on ~48 of the 64 values, evaluate on
   the 16 unseen ones. The honest interpolation test. **Mandatory for Part 2**: Goodfire
   fits splines through per-value centroids, so if you fit on all 64 you have nothing to
   steer *to*. Fit on 48 centroids, steer to the 16 held out.
2. **Held-out CLIPS within value** — secondary. Tests nuisance generalization (start
   position, theta). Report it; don't lead with it.
3. **Held-out PROBES** — Part 1 step 3 only. Build the steering basis from K probes,
   evaluate with a probe never in the basis (the paper's 70/30 probe split).

Enumerating these three cleanly is most of what makes the presentation defensible.

## 5. Compute & extraction

Hardware: M2 Max, 32 GB, MPS. Environment is currently **bare** — no torch, transformers,
or video decoder installed.

Model: `facebook/vjepa2-vitl-fpc64-256`, 24 layers, d=1024, patch 16, tubelet 2, ~1.3 GB fp32.
16 frames → 8 temporal × 16×16 spatial = **2,048 tokens/clip**.

**Verified from the safetensors header: the checkpoint ships the predictor too**
(199 tensors, 12 layers, d=384) alongside the encoder (388 tensors). This matters — see
Part 2 below.

**`fpc64` means the checkpoint was trained at 64 frames per clip; you are feeding 16.**
It will run, but decide deliberately whether to feed 16 as-is (8 temporal tokens,
off-distribution temporal extent) or interpolate up. Smoke-test one clip first.

### Extract once. Store more than you think you need.

A full pass over 4,572 clips on MPS is plausibly 1–2 hours. You do not want a second one.

- **Per-timestep spatial mean, all 24 layers, all clips** — 8×1024 per layer per clip,
  ≈1.8 GB fp16. Global mean pooling is derivable by averaging over the 8, so this
  strictly dominates storing only the global mean.
- **Full token grid (2048×1024) for 2–3 candidate layers on a ~600-clip subset** —
  ≈2.5 GB/layer. Needed for the attentive/patch-preserving probe and token-level steering.

> **Pooling is a confound, not a detail.** The disk is ~1.3 patches of 256 per frame, so a
> global mean over 2,048 tokens dilutes the object to well under 1% of the pooled vector.
> That won't kill mid-layer probes but it *will* artificially flatten early layers and make
> the emergence transition look sharper and later than it really is. **Report the layerwise
> curve under at least two poolings** (global mean vs per-timestep mean, or object-token
> selected) or the "Physics Emergence Zone" is confounded with pooling dilution.

Everything derived goes outside `data/` (`.gitignore` already expects `artifacts/`).

## 6. Staged plan

Ordered so that **any prefix is presentable.** Write artifacts at every stage.

**Stage 0 — environment & smoke test** (~half day)
Install torch + transformers + a video decoder. Load the model, confirm from the loaded
object: layer count, hidden dim, tubelet size, token count. Run one clip end to end.
Confirm the predictor is loadable and can roll a latent forward. Fix the preprocessing
convention (the processor resizes shortest edge to 292 then center-crops 256 — since the
videos are already 256×256, decide whether to bypass that and feed them directly, and
document the choice).

**Stage 1 — one extraction pass** (~2 hrs compute)
All 4,572 clips, all 24 layers, per-timestep spatial mean, fp16, cached to disk.
Plus the full-token subset for 2–3 candidate layers. Checkpoint incrementally so a crash
doesn't cost the whole pass.

**Stage 2 — layerwise probing** → *Part 1.1*
Ridge probes per layer. Speed/accel: scalar, MAE + R², **stratified by decile**.
Direction: `(sin θ, cos θ)` → circular MAE. Under ≥2 poolings. Held-out-values split.
Add the cross-dataset direction transfer test — it's nearly free.
*Expect: speed readable early and broadly, direction emerging at a sharper mid-depth
transition. If you reproduce that ordering, the qualitative result is reproduced.*

**Stage 3 — iterative nullspace probing** → *Part 1.2*
At the layer chosen in Stage 2. Fit → QR → project out → refit, logging the metric each
round. Run all three variables on the same axes.
**Run a shuffled-label control**: removing random directions also degrades performance
eventually, and without that control your curve doesn't mean what you'll claim.
*Expect: direction needs many more rounds than speed — the paper's "high-dimensional
circular structure" claim.*

**Stage 4 — multi-probe subspace steering** → *Part 1.3*
Build `V` by QR over K probe weight matrices; project, retarget, reconstruct.
Evaluate with **held-out probes** *and* **held-out values**. Sweep K, and show the
K-vs-accuracy curve — that's the money figure, and it connects directly to Stage 3's
dimensionality result.

**Stage 5 — spline / manifold steering** → *Part 2*
Per-value centroids from the 48 training values → PCA → fit splines:
- **speed, acceleration:** open cubic spline in arclength.
- **direction:** **periodic/closed** spline — this is the point the README flags.
  First check *whether* the 64 direction centroids order by angle around a ring in the top
  2 PCs, or whether the ring is genuinely higher-order. Look at the explained-variance
  profile before you commit to a 2D spline.

Steer to the 16 held-out values along the spline vs along the straight line, and compare:
path off-manifold energy, isometry correlation (representation geodesic vs readout),
held-out-probe accuracy, and naturalness.

**For the behavior manifold, use the predictor.** Steer at the emergence layer, roll the
predictor forward, and read the predicted future latent with a frozen probe. That is a
genuine behavioral measure and is much closer to Goodfire's setup than an
activation-to-activation comparison would be. (Fallback if the predictor proves awkward
to drive: a held-out-probe readout at a later layer — but say plainly in the deck that
this is weaker, because it never leaves representation space.)

## 7. The spine that ties the two parts together

Don't present Part 1 and Part 2 as two disconnected halves. The connection is:

> Part 1's nullspace iteration measures **how many dimensions** direction occupies.
> Part 2's manifold measures **the geometry of that same subspace**.

If the nullspace curve says direction needs tens of dimensions, then the direction
manifold is **a closed curve embedded in a high-dimensional subspace, not a planar
circle** — and the periodic spline must be fitted accordingly. Lead with that and you have
one coherent story: *where* → *how many dimensions* → *what shape* → *can we drive it*.

The natural headline: **linear multi-probe steering needs K probes to approximate a
structure that one correctly-shaped closed spline captures directly** — or it doesn't, and
that negative result is equally presentable.

## 8. Decisions to make consciously (and put on a slide)

1. Pooling: global mean vs per-timestep vs object-token-selected.
2. 16 frames as-is vs interpolated to the checkpoint's 64.
3. Probe family: ridge only, or ridge + attentive.
4. Emergence layer: derive from your own curves; do **not** hard-code the paper's.
5. Behavior channel for Part 2: predictor rollout (preferred) vs held-out probe.
6. Spline: intrinsic dimensionality, knot count, periodic vs open, PCA dim.

## 9. Known risks

- **Low-speed floor.** Bottom decile of speed is below pixel resolution. Mitigate by
  stratifying, not by hiding.
- **Pooling dilution** artificially sharpening the emergence transition. Mitigate with ≥2
  poolings.
- **Value leakage** across the train/test split. Mitigate with the held-out-values protocol.
- **Nullspace curve without a control** is uninterpretable. Mitigate with shuffled labels.
- **Off-distribution clip length** (16 vs 64 frames). Mitigate by documenting, and
  optionally by an interpolated-input ablation.
- **MPS flakiness / dtype issues** on a long extraction pass. Mitigate by checkpointing.
