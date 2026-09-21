# Findings

Running log of results, each with its evidence and status. Numbers come from cached runs
in `artifacts/results/`; every claim names the command that reproduces it.

**Protocol for everything below unless stated otherwise:** V-JEPA 2 ViT-L/16-256, frozen,
16-frame clips, residual stream at each block boundary, `sal` (leak-free top-8 salient
patch) pooling, spatial pooling then mean over the 8 temporal tokens, ridge probe with
alpha chosen on validation, **held-out *values*** (interleaved, interior — an
interpolation test, not a clip reshuffle). Setup details in
[notes/SETUP_FINDINGS.md](notes/SETUP_FINDINGS.md), cache usage in
[notes/CACHE.md](notes/CACHE.md).

Status key: **[solid]** verified with a control · **[preliminary]** measured, control
pending · **[open]** not yet investigated.

---

## F1. All three variables are strongly linearly decodable

| variable | layer 0 | layer 1 | best layer | best | R² | tracker oracle |
|---|---:|---:|:--:|---:|---:|---:|
| direction (circ MAE) | 66.9° | 11.4° | **16** | **2.57°** | 0.992 | 1.19° |
| speed (MAE m/s) | 0.822 | 0.103 | **14** | **0.056** | 0.994 | 0.011 |
| acceleration (MAE m/s²) | 2.137 | 0.303 | **14** | **0.153** | 0.994 | 0.036 |

Chance for direction is 90°. Layer 0 is the raw patch embedding (no attention).

**Evidence:** `run_part1.py`; `artifacts/figures/layer_sweep_all.png` (all three on one
axis set) and `layer_sweep_<variable>.png` (per variable, R² and native metric, three
poolings).

> **On "where each variable becomes available":** R² is a poor threshold here — it clears
> 0.95 at layer 1 for both scalars and layer 4 for direction, then saturates while real
> improvement continues (direction still carries ~2× its eventual error at layer 4). On the
> native metric, the scalars are within 2× of their best at **layer 1** and direction at
> **layer 4**; within 1.2× at layers **11 / 11 / 13**. Same ordering, but it says where each
> variable is genuinely finished.

**Reading:** performance rises steeply from layer 0→1, then improves gradually and
plateaus in the middle third. The encoder retains most but not all of the recoverable
signal — it sits ~2× off the tracker oracle on direction, ~5× on speed and acceleration.

---

## F2. Direction is available from layer 1 — not at a mid-depth transition

The paper reports direction emerging only at a sharp intermediate-depth transition (the
"Physics Emergence Zone", ~⅓ depth). Here one attention block suffices: 66.9° → **11.4°**
between layers 0 and 1, then a smooth glide to 2.6°. **No step.**

**The control that makes this a finding rather than an artefact.** The worry is that the
probe reads *position* rather than *direction* — a disk that has moved is displaced in
its travel direction from a random start. Probing the tracker output directly, with no
encoder features at all:

| features | circ MAE | R² |
|---|---:|---:|
| **frame-0 centroid only** | **86.27°** | −0.015 |
| final-frame centroid only | 52.20° | +0.310 |
| displacement (x, y) | 8.21° | +0.667 |
| unit displacement | 6.71° | +0.937 |

Frame-0 position alone is **at chance** (86° vs 90°), confirming start positions are
independent of theta (also |r| < 0.07 in the metadata). So the layer-1 probe at 11.4° is
genuinely reading motion.

**Interpretation:** most likely a dataset-simplicity effect — one high-contrast disk on
an empty background, where motion is nearly readable off the first attention block. 

---

## F3. All three variables are equally redundant; direction's readout is 2-dimensional

Iterative nullspace probing in the full 1024-d space: fit a probe, orthonormalise its readout,
project that subspace out of every split, refit. The random-subspace control is what makes the
curve mean anything — removing *any* directions eventually hurts a probe.

| variable | R²₀ | dims to halve | **rounds to halve** | dims to 10 % | fully exhausted | random control |
|---|---:|---:|---:|---:|---:|---:|
| **direction** (sin, cos) | 0.992 | **142** | **71** | 266 | 398 | **0.991** (flat) |
| speed | 0.994 | 69 | **69** | 114 | 199 | 0.993 (flat) |
| acceleration | 0.994 | 67 | **67** | 103 | 199 | 0.993 (flat) |

Circular MAE for direction runs 2.6° → **80°** (chance 90°) while the random control holds at
2.6° throughout. **That separation is the finding**: probe-directed removal destroys the signal,
removal of the same number of random directions does not. All three variables are carried by a
highly redundant population code.

### The dimension counts are not comparable across variables

Each INLP round removes **one probe's readout subspace**, whose rank is set by the *target*, not
by the representation: 2 for direction's (sin, cos), 1 for a scalar. So direction spends
dimensions twice as fast per round purely as bookkeeping.

**Rounds — the number of independent readouts extractable before the signal dies — are
essentially equal: 71 / 69 / 67.**

The decisive control: give direction a **1-dimensional** target on the same activations, layer
and splits.

| target | rank | dims to halve | rounds |
|---|---:|---:|---:|
| direction, sin θ alone | 1 | **76** | 76 |
| direction, cos θ alone | 1 | **54** | 54 |
| direction, (sin, cos) | 2 | 142 | 71 |
| *speed* | 1 | *69* | *69* |
| *acceleration* | 1 | *67* | *67* |

A 1-d direction target costs **54–76** dimensions — squarely in the scalars' range. The 142 is
71 rounds × 2 dims.

**Evidence:** `E.nullspace_curve(ds, L, pooling='sal', n_iter=200)`;
`artifacts/figures/nullspace_combined.png`.

### What can and cannot be claimed

**Can:** all three variables are highly redundant, with probe-directed removal separating cleanly
from the random control; direction's readout is inherently 2-dimensional, so controlling it
costs twice the dimensions per readout.

**Cannot:** "direction is twice as redundant" or "occupies twice the dimensions" as a statement
about the *representation*. It is true of the dimension count and false of the redundancy.

There is a **real** residual asymmetry, at the tail rather than the midpoint: reaching 10 % of
the original R² takes 133 rounds for direction against 114 and 103 for the scalars — about
1.2×, not 2×. Direction's signal persists slightly longer into the low-signal regime.

The same run inside a PCA-reduced space (basis fitted on train rows only):

| space | direction 50 % / 10 % | speed | acceleration |
|---|---|---|---|
| PCA-32 | 8 / 16 | 5 / 7 | 4 / 6 |
| PCA-64 | 12 / 28 | 8 / 12 | 7 / 11 |
| PCA-128 | 22 / 50 | 13 / 20 | 11 / 18 |
| full-1024 | 142 / 266 | 69 / 114 | 67 / 103 |

> **Caveat:** in the reduced spaces the random control is *not* a valid comparison near
> exhaustion (removing 126 of 128 random dims also destroys the probe: control R² 0.363). The
> control is only meaningful while dims-removed ≪ space dimension, which is why the full-1024
> run is the headline.

## F4. The direction manifold is not a planar circle

Probe R² on held-out values using only the top-k principal components (layer 16, `sal`):

| k | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|
| test R² | **0.441** | 0.854 | 0.959 | 0.963 | 0.972 | 0.980 | 0.987 | 0.991 |

**The top 2 PCs carry only R² = 0.44.** The 64 direction centroids do not lie on a plane.

Not overfitting: at layer 16 in the full space, train R² 0.9976 vs test 0.9921.

**Consequence for Part 2:** the periodic spline must be fitted in ~64-D PCA space, not
2-D. A 2-D fit would be visually appealing and quantitatively wrong. This is the hinge
between Part 1 and Part 2: *how many dimensions* (F3) → *what shape* (F4) → *can we drive
it* (F8–F12).

### F4b. The ring itself is low-dimensional; separating it from nuisance is not

Two different statements, easily conflated:

| quantity | cum. explained variance @2 PCs | @4 |
|---|---:|---:|
| all training clips | 0.486 | 0.683 |
| **per-value centroids only** | **0.733** | **0.956** |

Averaging within each value removes the nuisance variance (start position, and for
`direction` the speed/acceleration regime) and the **closed ring appears cleanly** — see
the middle panel of `artifacts/figures/manifold_direction.png`, where the 64 centroids
order smoothly round a loop that closes.

So the *shape* is roughly 2–4 dimensional, while *reading direction off a single clip*
needs tens of dimensions (F3) because the probe must first separate signal from nuisance.
The left panel of the same figure — the ring as it looks in the raw clip-PCA basis, a
messy scribble — is the honest version of the tidy circle that interpretability papers
usually show, and is worth a slide on its own.

---

## F5. Pooling changes the curve, so it must be reported explicitly

The disk is ~1.3 of 256 patches per frame, so a uniform mean dilutes it below 1 % of the
pooled vector.

| pooling | direction best | speed best | acceleration best | leak? |
|---|---|---|---|---|
| `tmean` uniform mean | 2.75° (L14) | 0.063 (L19) | 0.169 (L20) | none |
| `sal` top-8 salient | **2.57° (L16)** | **0.056 (L14)** | **0.153 (L14)** | none |
| `obj` tracker-selected | 2.12° (L15) | 0.049 (L23) | 0.154 (L18) | **yes** |

`sal` beats `tmean` everywhere and shifts the apparent best layer by up to 6 layers
(speed: L19 → L14). **Reporting a single pooling would confound the emergence claim with
pooling dilution.**

The encoder's own salience ranking recovers the true object tokens with recall **1.00
through layer 12**, decaying to **0.89 at layer 24** — a result in its own right about
where object identity stays spatially localised.

`obj` is excluded from interpretation — see F6.

---

## F6. The tracker-only model is an oracle ceiling, and it invalidates `obj` pooling

A probe on the disk trajectory with **no encoder features at all** (given the nonlinear
terms a linear probe cannot form for itself: displacement magnitudes, their differences,
unit-normalised components):

| variable | tracker-only | best encoder probe |
|---|---:|---:|
| speed | **MAE 0.011**, R² 1.000 | 0.056 |
| direction | **1.19°**, R² 0.999 | 2.57° |
| acceleration | **MAE 0.036**, R² 1.000 | 0.153 |

Two consequences:

1. **`obj` pooling proves nothing about the encoder.** Its mask is built from the
   trajectory, i.e. from the label, and the mask alone beats every encoder probe. It is
   drawn faint and flagged in the figures; layer selection uses `sal`.
2. **It bounds the task.** The gap is what V-JEPA's representation *loses* relative to
   perfect object tracking: ~2× on direction, ~5× on speed, ~4× on acceleration.

---

## F7. The low end of the speed range is below the encoder's resolution

Rendering is pixel-quantised and the scale is ~20 px total displacement per m/s, so at
magnitude 0.43 the measured per-frame step is 0.00–0.02 px.

Speed, layer 16, `sal`, **clip axis** (the value axis holds out only ~7 distinct values,
too few for deciles):

| label (m/s) | 0.40 | 0.76 | 1.14 | 1.92 | 2.66 | 3.43 | 3.82 |
|---|---|---|---|---|---|---|---|
| MAE | .070 | .059 | .072 | .063 | .075 | .068 | .075 |
| **relative** | **17.6 %** | 7.7 % | 6.3 % | 3.3 % | 2.8 % | 2.0 % | 2.0 % |

Absolute error is flat; **relative** error degrades ~9× at the bottom. The encoder has
roughly constant absolute precision, so the low end is where the representation runs out.
Report stratified, not aggregate.

---

## F8. Steering direction needs a *multi*-probe subspace; the scalars do not

Multi-probe subspace steering at the selected layer, evaluated with a **held-out probe**
(never one of the K that built the basis, and fitted on disjoint clips) on **held-out
values**. Floor = what the probe reads on unsteered clips vs the target; ceiling = the
probe's accuracy on unmodified clips.

| K | direction (circ MAE) | % gap closed | speed (MAE) | % | acceleration (MAE) | % |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **61.4°** | 28 % | 0.334 | 74 % | 0.849 | 74 % |
| 2 | 10.7° | 91 % | 0.243 | 84 % | 0.603 | 84 % |
| 4 | 6.1° | 97 % | 0.150 | 93 % | **0.331** | 95 % |
| 8 | 11.0° | 91 % | **0.067** | 103 % | 0.406 | 93 % |
| 16 | 5.3° | 99 % | 0.103 | 102 % | 0.452 | 92 % |
| 32 | **4.8°** | 101 % | 0.116 | 105 % | 0.559 | 92 % |

Floors: 84.0° / 1.07 / 2.79. Ceilings: 3.0–5.3° / 0.07–0.16 / 0.16–0.38.

### The optimisation actually solved

Write the stacked probe readouts as `W ∈ R^{M×D}` (rows = the M readout directions of the
K probes, M = 2K for direction, K for a scalar) with intercept `b`, so a probe reads
`Wx + b`. The intervention solves

> **minimise ‖Δx‖₂²  subject to  W(x + Δx) + b = y_target**

and returns `x* = x + Δx`. **This is the Moore–Penrose solution**

> **Δx = W⁺ ( y_target − (Wx + b) )**

The code does not compute `W⁺` directly: it forms an orthonormal
basis `V = QR([W₁ᵀ … W_Kᵀ])`, splits `x = VVᵀx + x_⊥`, and least-squares-solves
`(VᵀW ᵀ)ᵀ c* = y − b − Wᵀx_⊥` before recomposing `x* = Vc* + x_⊥`. That is the paper's
construction, and it is *algebraically* the pseudo-inverse whenever the stacked readouts
are full column rank — the constraint then has a unique solution inside `span(V)`, and the
minimum-norm solution of `WΔx = r` lies in `span(Wᵀ) = span(V)` too, so the two coincide.

Numerically, against `Δx = W⁺(y − Wx − b)` computed independently:

| K | M | rank(W ᵀ) | cond(VᵀW ᵀ) | mean ‖Δx‖ | max abs difference |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 1.1 | 2.0008 | 2.7e−14 |
| 4 | 8 | 8 | 2.7 | 4.6256 | 2.6e−14 |
| 16 | 32 | 32 | 7.0 | 8.4842 | 2.7e−14 |
| 32 | 64 | 64 | 14.0 | 12.6047 | 3.1e−14 |

The readouts stay full rank and well conditioned at every K, so the two formulations agree
to machine precision. Three further checks: the constraint holds to 4e−15; `Δx` lies in
`span(V)` to 2e−14; and adding any null-space vector `n` with `Wn = 0` keeps the constraint
satisfied while **increasing** ‖Δx‖ in 30/30 trials — minimality confirmed directly rather
than only by the equivalence argument.

**Why this matters for Part 2.** The intervention is by construction the *smallest Euclidean
step* that satisfies K linear constraints. Euclidean smallness is measured in the ambient
activation space, which knows nothing about where the data actually lives — so the solution
is free to cut straight across the manifold, and on a closed geometry the shortest chord
passes through the ring's interior. F11 quantifies that shortcut (80–120× off-manifold),
F12 shows it lands in a semantically empty region, and F15 shows the resulting edit is the
one attention discards (25 % retained vs 76 % for the on-manifold edit). **The failure mode
is not a bug in the solver — it is exactly what minimising the wrong norm means.**

**Evidence:** `vjp/steering.py`; `artifacts/figures/steering_*.png`.

**The contrast is the finding.** A *single* probe steers direction barely better than not
steering at all (61° against a floor of 84° — 28 % of the gap), but closes 74 % for both
scalars. Direction needs K ≥ 2, and is essentially solved by K ≈ 4. This is the
causal counterpart of F3: the variable that occupies ~2× the dimensions is also the one
that cannot be driven along a single direction. It reproduces the paper's claim that
direction "requires coordinated multi-feature intervention to control".

**Guard built in:** `evaluate_steering` asserts the basis probes hit their target to
< 1e-6 after the intervention. Without that check a broken solve looks exactly like a
representation that resists steering — which is how C6 below went unnoticed at first.

*Gap-closed above 100 % means steered error fell below the ceiling: the ceiling carries
the probe's own label noise on real clips, while steered activations are set exactly on
target within the basis subspace.*

*Accuracy degrades past K ≈ 8 for the scalars because the training clips are split K+1
ways — at K=32 each probe sees ~36 clips for a 1024-d ridge. That is a data limit of this
dataset, not a property of the representation.*

### F8c. Steering rotates the representation — it does not inject noise

The control that separates "the intervention changed the encoding" from "the intervention
found an adversarial direction that fools the probe": as the error to the **target** angle
falls, the error to each clip's **own** angle must rise by the same amount, and the readout
must stay confident.

| K | MAE → target | MAE → own label | certainty |
|---:|---:|---:|---:|
| unsteered | 80.7° | **4.6°** | 0.934 |
| 1 | 61.4° | 23.4° | 0.699 |
| 2 | 10.7° | 74.3° | 0.825 |
| 4 | 6.1° | 80.3° | 0.928 |
| 16 | 5.3° | 82.4° | 0.951 |
| **32** | **4.8°** | **83.2°** | 0.928 |

The two curves cross cleanly and **certainty holds at ~0.93 throughout**. Noise injection
would drive certainty toward 0 — as it does in F12b, where an off-manifold state collapses to
0.18. The same trade-off holds for speed (0.74 → 1.06) and acceleration (1.95 → 3.26).

**Evidence:** `evaluate_steering` now records `steered_vs_own_label` and `certainty`;
`artifacts/figures/steering_combined.png`.

### Reproduction scorecard against Joseph et al.

| their result | ours | verdict |
|---|---|---|
| Coordinated steering succeeds — 20 probes → 11.9° from an 82.9° baseline | K=16 → **5.3°**, K=32 → **4.8°**, from an 84.0° floor | ✅ reproduced, somewhat stronger |
| True representation shift — error to ground truth rises 6° → ~71° as target error falls | **4.6° → 83.2°** | ✅ reproduced |
| Single-probe steering fails — MAE stays **> 80°**, essentially unchanged | K=1 → **61.4°**, i.e. 28 % of the gap closed | ⚠️ **partial** |

**On the single-probe claim.** The ordering reproduces emphatically — one probe closes 28 % of
the gap for direction against 74 % for either scalar — but in our data a single probe *does*
move the representation, where the paper reports essentially no change. Most plausibly the
same dataset-simplicity effect behind F2 (direction readable at layer 1): one high-contrast
disk on an empty field is a far easier stimulus.

**On their takeaway.** They frame motion direction as a *uniquely* high-dimensional population
code, unlike LLM concepts steerable along a single 1-D vector. Our data supports the **steering**
half of that: direction needs K ≥ 2 while either scalar is steerable at K = 1. It does **not**
support the uniqueness: per F3, speed and acceleration require 69 and 67 independent readouts
against direction's 71 — all three are high-dimensional population codes. What actually
distinguishes direction is that its readout is **2-dimensional and cyclic**, which is why one
probe cannot drive it and why a chord through the ring lands in a semantically empty region
(F12). That is a sharper claim than the paper's, and it is the one this data licenses.

### F8b. Cross-variable leakage grows with the steering subspace

**The 3×3 matrix cannot be completed, and that is itself a finding about the data.**
Speed and acceleration are **mutually exclusive in every supplied dataset** — no clip has
both non-zero (`speed`: accel ≡ 0; `acceleration`: speed ≡ 0; `direction`: velocity clips
have accel = 0, acceleration clips have speed = 0). So the kinematic coupling
*v = v₀ + at* that would make speed↔acceleration leakage interesting has been **designed
out of the stimulus set**. Those two cells have no clips that could reveal them, and no
probe can be fitted for a variable that never varies.

Theta spans all 64 values in every dataset, so the direction↔scalar cells *are* estimable:

| steered ↓ / watched → | direction | speed | acceleration |
|---|---|---|---|
| **direction** | *on-target* | **1.02×** | **1.90×** |
| **speed** | **0.76×** | *on-target* | not estimable |
| **acceleration** | **0.45×** | not estimable | *on-target* |

Drift in the watched variable ÷ that probe's own MAE on unmodified clips, at matched
steering-subspace dimension 8. Below 1.0 the disturbance is smaller than the probe's own
noise. `artifacts/figures/leakage_matrix.png`.

**Specificity degrades monotonically as the subspace grows** — the robust result, present
in every cell:

| subspace dim | speed→dir | dir→speed | accel→dir | dir→accel |
|---:|---:|---:|---:|---:|
| 4 | **0.28×** | 0.47× | 0.72× | 1.30× |
| 8 | 0.76× | 1.02× | 0.45× | 1.90× |
| 16 | 1.32× | 1.12× | 1.21× | 1.45× |

By dim 16 **every** cell leaks. This is the real cost of the F8 result: direction needs a
multi-probe subspace to be controllable at all, and that same enlargement is what destroys
specificity. Dim ≈ 4–8 is the usable operating point.

**On the apparent asymmetry.** At dim 4–8, steering *direction* disturbs the scalars more
than the reverse (1.3–4.2× as much). That is tempting to attribute to direction's larger
footprint (F3), and it may be — but it **does not survive to dim 16** (0.8× and 1.2×),
where everything leaks and the ordering washes out. Reported as a tendency at small
subspaces, not a law.

> Two methodological corrections were needed to get here, and both changed the conclusion.
> (i) The readout probe must be fitted on a **fixed-size** fold; letting it shrink as
> train/(K+1) made its baseline degrade with K and the ratio improve for free — which
> produced an initial, wrong reading that steering became *more* specific at high K.
> (ii) Cells must be compared at matched **subspace dimension**, not matched K: a direction
> probe spans 2K dimensions (sin and cos) against K for a scalar, so equal K is a 2×
> handicap.

---

## F9. The intervention is causal, but the network undoes it within ~4 blocks

F8 steers a pooled feature and reads it back at the *same* layer. This injects the edit
into the live residual stream at layer 16 and lets the remaining blocks actually run,
reading downstream with probes fitted on **unmodified** activations at that layer.

Direction, circular MAE (deg), floor ≈ 79°, chance 90°:

| read at | blocks downstream | K=1 | K=4 | **K=16** | floor | effect retained (K=16) |
|---|---:|---:|---:|---:|---:|---:|
| L17 | 1 | 70.3 | 32.1 | **17.7** | 78.4 | **77 %** |
| L18 | 2 | 72.2 | 60.9 | 44.5 | 79.0 | 44 % |
| L20 | 4 | 77.4 | 74.1 | 67.3 | 79.3 | 15 % |
| L24 | 8 | 78.6 | 76.9 | 72.9 | 79.9 | 9 % |

For reference, F8's same-layer feature-space steering reaches **5.3°** at K=16.

**Evidence:** `run_causal.py`; `vjp/causal.py`; `artifacts/figures/causal_direction_L16.png`.

**Two things follow, and they pull in opposite directions.**

1. *The subspace is genuinely causal.* One block downstream the edit still carries 77 % of
   the effect (17.7° against a 78.4° floor), and the control is flat. The multi-probe
   subspace is not merely decodable — it changes what the next block computes, and the
   K-ordering (K=1 barely works, K=16 works) survives propagation.
2. *But the network largely re-normalises it away.* By four blocks only 15 % remains. A
   Euclidean edit that is exactly correct in the pooled readout does **not** persist as
   the model keeps computing.

**The localisation control rules out the obvious explanation.** Adding the delta to only
the 8 salient tokens per timestep — 1/32 the perturbation energy, and exactly the same
pooled shift — propagates *slightly worse*, not better (24.3° vs 17.7° at L17, K=16). So
the decay is not an artefact of smearing the edit across background tokens.

**This is the motivation for Part 2.** Wurgaft et al.'s thesis is precisely that Euclidean
steering "cuts through off-manifold regions and hence produces unnatural outputs", while
on-manifold steering yields trajectories the model follows naturally. F9 is a quantified
baseline for that claim on this model: *if* spline steering retains more of its effect
downstream, that is the strongest possible version of the Part 2 result. If it does not,
that is an equally publishable negative.

*Caveat: read-out probes are fitted per layer on the same held-out fold; the floor drifts
by <2° across layers, so the decay is not a probe-quality artefact.*


---

## F13. The ring carries a Fourier ladder — but only m ∈ {1, 2, 4}, and it is an *axis* code

The 64 directions are equally spaced, so a DFT of the centroids in θ reads the harmonic
content off directly. Noise floor from splitting the ~24 reps per angle in half.

| harmonic m | 1 | **2** | 3 | **4** | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| share of centroid energy | 73.2 % | **18.7 %** | 1.0 % | **3.0 %** | 0.2 % | 0.5 % | 0.1 % | 0.4 % |
| signal / noise | 626 | **103** | 3.4 | **20** | 0.8 | 1.6 | 0.4 | 2.6 |

**Each principal component is almost purely one harmonic**, which is the structure the
Fourier hypothesis predicts:

| PC | % var | harmonic | R² of that harmonic | 180° flip correlation |
|---|---:|:--:|---:|---:|
| PC1 | 50.4 | **m=1** | 0.989 | **−0.966** |
| PC2 | 23.6 | **m=1** | 0.967 | — |
| PC3 | 18.0 | **m=2** | 0.955 | **+0.947** |
| PC4 | 3.3 | **m=4** | 0.870 | **+0.971** |
| PC5 | 0.9 | m=2 | 0.628 | — |
| PC6+ | <1 | mixed | <0.4 | — |

So PC1–PC2 = (cos θ, sin θ), PC3 = 2θ, PC4 = 4θ. **But the ladder is not 1, 2, 3, 4 —
m=3 is at the noise floor (SNR 3.4) while m=4 is clearly real (SNR 20).**

### It is generated by the network, not inherited from the stimulus

| m | pixel trajectory (no V-JEPA) | layer 0 (no attention) | layer 16 |
|---|---:|---:|---:|
| 1 | **97.0 %** | 7.9 % | 73.2 % |
| 2 | 0.02 % | 3.1 % | **18.7 %** |
| ≥2 total | **3.0 %** | 92.1 % (unstructured) | 26.8 % |

The stimulus is an essentially pure first harmonic. Layer 0's flat spectrum is noise — it
sits at 66.9° circular MAE, near chance. The second harmonic **grows monotonically with
depth**: 3.1 % (L0) → 10.7 % (L1) → 15.0 % (L8) → 18.7 % (L16) → **23.2 % (L24)**, while
m=1 peaks near L12 and declines.

*(This rules out the natural suspicion that m=2/m=4 are pixel-lattice artefacts — the
square patch grid would produce exactly such harmonics, but then they would be present in
the pixel trajectory, and they are not.)*

### What the higher harmonics actually are: motion **axis**, not motion direction

PC1 **anti**-correlates with itself under a 180° flip (−0.966) — a true direction code.
PC3 and PC4 are **invariant** under the flip (+0.947, +0.971): they encode the *line of
motion*, not which way along it.

This is the direction-selective vs orientation-selective distinction from visual
neuroscience, appearing unprompted in a video world model. V-JEPA represents "moving along
the horizontal axis" separately from "moving left rather than right", and the axis
component strengthens with depth.

### But this does **not** explain why direction needs many dimensions

The higher harmonics carry no direction information on their own and add almost nothing to
the readout:

| basis | circular MAE |
|---|---:|
| PC1–2 (m=1) | 12.57° |
| PC3 alone (m=2) | **99.80°** — worse than chance |
| PC4 alone (m=4) | 99.07° |
| PC1–2 + PC3 | 11.73° |
| PC1–4 | 11.12° |
| PC1–16 | 2.92° |
| full 1024-d | **2.57°** |

A 180°-ambiguous code *cannot* carry direction, so this is expected — but it settles the
causal question. Going from the 4 harmonic PCs (11.12°) to the full space (2.57°) is a
**4× improvement that the harmonics contribute essentially none of.**

**So the answer to "why is direction high-dimensional" is not "higher Fourier harmonics".**
Per F4b the *shape* is 2–4 dimensional and the harmonic ladder describes it exactly; the
tens of dimensions in F3 are spent **suppressing within-value nuisance** — start position,
speed, and motion regime — so that the m=1 component can be read cleanly off a single
clip. Shape and readout dimensionality are different quantities, and the harmonic
structure explains the first, not the second.

### What this does justify for Part 2

A closed periodic spline is still the right object, and now for a measured reason rather
than an assumed one: the centroid curve is a **genuine closed loop dominated by m=1 with a
real m=2 component**, so it is a circle deformed into a 2-lobed figure — not a plane
curve, and not something an open spline or a 2-D circle fit would capture. It also
explains F12 directly: the m=2 component is what makes the interior of the ring
semantically empty, and hence why a chord through it collapses the readout.

**Evidence:** DFT over the 64 equally spaced centroids at layer 16, `sal` pooling; noise
floor from half-split reps; pixel control from `tracks_direction.npz`.


---

## F14. Why the network undoes it: attention rotates the edit out of the readout subspace

F9 showed a Euclidean edit at L16 decays to ~15 % effect by four blocks. Reading the
residual stream at **half-block resolution** (pre-norm block: `in` → `attn` → `out`) with
probes fitted per position on unmodified activations separates the candidate mechanisms.

| position | steered | control | ‖delta‖ | aligned | LN ‖·‖ ratio | LN std ratio |
|---|---:|---:|---:|---:|---:|---:|
| L16.in (injection) | 9.1° | 80.5° | 1.000 | 0.316 | 1.0012 | 1.0012 |
| L17.in | 9.1° | 80.7° | 0.881 | 0.331 | 1.0026 | 1.0027 |
| L18.in | 27.0° | 80.7° | 0.783 | 0.173 | 1.0010 | 1.0010 |
| L19.in | 57.3° | 80.6° | 0.704 | 0.201 | 0.9992 | 0.9992 |
| L20.out | 69.6° | 80.4° | **0.687** | **0.149** | 0.9986 | 0.9986 |

`aligned` = fraction of the surviving perturbation lying in that position's 2-D
direction-readout subspace. A random vector would score 0.044.

### H1 — LayerNorm rescaling: **refuted**

- Token ‖·‖ and channel-std ratios between steered and control stay within **0.5 % of 1.0**
  at every position. LayerNorm is not squashing anything.
- Only **0.60 %** of the delta's energy lies along the all-ones direction, the component LN
  deletes when it subtracts each token's channel mean. (The delta is 2× more aligned with
  all-ones than a random vector would be, but the absolute effect is negligible.)

### The edit is not removed — it is rotated

This is the substantive result. **69 % of the perturbation's magnitude survives four
blocks**, yet the readout degrades from 9.1° to 69.6°. What changes is *where it points*:

| | at injection | after 4 blocks | retained |
|---|---:|---:|---:|
| total ‖delta‖ | 1.000 | 0.687 | **69 %** |
| readout-aligned component | 0.316 | 0.102 | **32 %** |

**The readout-relevant component decays 2.2× faster than the perturbation itself.** The
network does not erase the intervention; it turns it into something that no longer means
*direction*.

### H2 — attention, not the MLP

Degrees of readout error added by each half-block:

| block | attention | MLP |
|---|---:|---:|
| 16 | −2.5° | +2.5° |
| 17 | **+15.2°** | +2.7° |
| 18 | **+26.2°** | +4.1° |
| 19 | +7.2° | +3.1° |
| 20 | +0.2° | +1.9° |
| **total** | **+46.3°** | **+14.2°** |

**Attention accounts for 77 % of the decay**, concentrated in blocks 17–18. The MLP
contributes steadily but modestly.

So the MLP is **not** acting as an autoassociative memory that projects off-manifold states
back onto the learned distribution — that hypothesis predicts the opposite split. The
decay is *attention dilution*: the edit is added uniformly to all 2,048 tokens, and
attention computes token **interactions**, so it redistributes a spatially uniform
perturbation into token-relative structure that the direction readout does not track.

This also explains F9's localisation control, which otherwise looks paradoxical: injecting
into only the 8 salient tokens propagated slightly *worse*. Neither edit matches how the
representation actually distributes direction across tokens, and attention re-mixes both.

**Evidence:** `run_mechanism.py`; `vjp/mechanism.py`. Position probes fitted on 600
training clips disjoint from the K=16 basis probes.

### What this means for Part 2

F12 showed manifold steering survives mid-path where linear steering collapses. F14 says
the reason a Euclidean edit fails is that **attention rotates it out of the readout
subspace**, not that any norm rejects it. The natural follow-up — not yet run — is whether
the on-manifold edit's readout-aligned component decays more slowly through the same
blocks. That would convert F12's behavioural result into a mechanistic one.


---

# Part 2 — spline / manifold steering

Manifolds are fitted as cubic splines through **per-value centroids** in PCA space, the
PCA and the centroids both computed on **train values only**, so reaching a held-out
value is genuine interpolation along the curve. Direction uses a **periodic** spline;
speed and acceleration use open ones.

## F10. Spline steering is *worse* than multi-probe at hitting a target

Same held-out probe, same targets, same held-out values as F8:

| variable | spline | multi-probe (K=16) | floor | ceiling |
|---|---:|---:|---:|---:|
| direction | 7.02° | **5.32°** | 83.9° | 4.62° |
| speed | 0.224 | **0.103** | 1.074 | 0.121 |
| acceleration | 0.664 | **0.452** | 2.808 | 0.253 |

Both beat the floor comfortably; multi-probe wins. Stable across `n_pca` ∈ {8 … 128}
(direction 8.2° → 7.0°), so it is not a dimensionality-of-fit problem.

**This is expected, and it is not the claim being tested.** Multi-probe steering *solves*
for the coordinates that make probe readouts equal the target — it optimises precisely the
quantity being scored. The spline simply moves to the target value's centroid. Endpoint
accuracy was never Wurgaft et al.'s claim; the path is (F11, F12).

## F11. The linear path is 74–130× further off-manifold — for **all three** variables

Mean distance from the interpolation path to the manifold, over 150 random value pairs,
endpoints identical by construction:

| variable | manifold | linear | ratio | linear worse in |
|---|---:|---:|---:|---:|
| direction | 0.037 | 4.307 | **117×** | 100 % of pairs |
| speed | 0.019 | 1.626 | 87× | 100 % of pairs |
| acceleration | 0.016 | 1.322 | 80× | 100 % of pairs |

> **Read this metric sceptically.** The manifold path lies on the manifold *by
> construction*, so its energy is ~0 almost tautologically. What this shows is that the
> chord genuinely leaves the data region — not yet that leaving it matters. F12 is the
> test that does.

## F12. Off-manifold does not imply off-behaviour — it depends on the topology

The behavioural test: inject the edit into the live residual stream at a fraction `t`
along each route, propagate, and read one block downstream. At t=0 and t=1 the two routes
coincide exactly, so any difference is purely the route.

**Direction** (edit L16, read L17), steering to the *furthest* held-out value:

| t | manifold certainty | manifold alignment | linear certainty | linear alignment |
|---|---:|---:|---:|---:|
| 0.00 | 1.003 | **+0.985** | 1.003 | **+0.985** |
| 0.25 | 0.933 | +0.918 | 0.517 | +0.341 |
| **0.50** | **0.900** | **+0.892** | **0.175** | **+0.024** |
| 0.75 | 0.941 | +0.919 | 0.540 | +0.363 |
| 1.00 | 1.029 | +1.014 | 1.029 | +1.014 |

`certainty` = mean ‖(sin, cos)‖ of the prediction (~1 confident, ~0 = no direction at all);
`alignment` = mean ⟨prediction, unit vector of the intended value⟩, which is +1 for a
perfect readout and 0 for no information.

At mid-path the linear route's readout is **entirely dead**: alignment **+0.024**, i.e.
zero information about any direction, against **+0.892** for the manifold route. The
inverted-U bottoms out exactly at t = 0.5, the chord's closest approach to the ring's
centre — precisely what the geometry predicts. Endpoints coincide to 3 decimals, confirming
the comparison isolates the route.

> **Metric correction (C7).** This table originally reported circular MAE (manifold 11.0°
> vs linear 23.1°). That understated the effect and was not trustworthy: when certainty
> collapses toward 0, the predicted (sin, cos) vector is near-zero and its *angle* is pure
> noise, so circular MAE becomes arbitrary — two equally valid probes scored the same
> linear state at 13° and 91°. Certainty and alignment are well behaved at zero and are
> used throughout. The corrected result is **much stronger** than the original.

### F12b. What the off-manifold state actually *is*: axis and speed without direction

Reading other probes at that same t = 0.5 linear state settles what kind of state it is.
Run on the `speed` dataset, where theta and speed are decorrelated (|r| = 0.001) so the
speed probe is clean. Errors are measured against the **source** clip's own angle;
direction is mod 360° (chance 90°), axis is mod 180° (chance 45°).

| state | direction (m=1) certainty / err | **axis (m=2)** certainty / err | speed read |
|---|---:|---:|---:|
| unsteered (t=0) | 0.981 / 3.2° | 0.973 / 3.7° | 2.104 m/s |
| **linear t=0.5** | **0.181 / 75.8°** | **0.914 / 5.9°** | **2.136 m/s** |

**Speed survives untouched** (2.104 → 2.136; population mean 2.12, and per-clip error stays
near the probe's own baseline). So this is the *paradox* outcome, not the tidy
"left + right = stationary" one — the centre of the direction ring is **not** the
zero-speed state.

**And the axis survives too, at full strength.** The state is not vaguely incoherent: it
encodes *"an object moving along this specific line, at 2.1 m/s, with no fact about which
of the two ways."* The direction bit has been surgically annihilated while everything else
is intact.

This follows exactly from the harmonic structure of F13, and is its sharpest confirmation:

- m=1 is **anti**-symmetric under a 180° flip, so the two chord endpoints carry opposite
  m=1 components and the midpoint has **zero** — certainty collapses;
- m=2 is **invariant** under the flip, so both endpoints carry the *same* axis component
  and it is preserved along the entire chord — 5.9° error against the source angle;
- speed lives in neither harmonic and rides along in the residual, untouched.

It also explains F12's inverted-U directly: t = 0.5 is where m=1 cancellation is exactly
complete, which is why the collapse bottoms out there and recovers on either side.

> **Why this matters for the "unnatural outputs" claim.** Wurgaft et al. argue that
> Euclidean steering produces states the model would never occupy. This is a stronger
> statement than "off-manifold": the state is not noise, it is **structured and physically
> impossible** — a well-formed speed and a well-formed axis with the direction bit removed.
> No real clip can be in it, which is precisely why the ring's interior is semantically
> empty (F12) and why the resulting edit is what attention discards (F15).

**The open-manifold controls** — three of them now, and none shows a penalty. Certainty never
collapses for a scalar, so MAE is safe here:

| control | manifold | linear | penalty |
|---|---:|---:|---:|
| speed, edit L14 → read L15 | 0.223 | **0.204** | 0.92× |
| speed, edit L14 → read L16 *(2 blocks)* | 0.297 | **0.282** | 0.95× |
| acceleration, edit L14 → read L15 | 0.601 | **0.581** | 0.97× |

Linear steering is marginally *better* in all three. The effect is specific to direction, holds
one and two blocks downstream, and holds for both scalars — so it is not an artefact of a single
layer pair or a single variable.

Per-t detail for the first control:

| t | manifold err | linear err |
|---|---:|---:|
| 0.25 | 0.198 | 0.193 |
| 0.50 | 0.223 | 0.204 |
| 0.75 | 0.236 | 0.217 |

**No difference — linear is marginally *better*.**

**Evidence:** `vjp/part2.py::causal_path`; `artifacts/figures/causal_path_*.png`.

### The refinement this forces on the Goodfire claim

The obvious explanation — "direction's manifold is more curved" — **does not survive
measurement**. Arclength/chord is 6.59 for direction but also **4.68 for speed** and 4.81
for acceleration. All three are strongly curved; all three have linear paths ~80–120× off
manifold (F11). Yet the behavioural penalty appears **only for direction**.

What distinguishes direction is **topology, not curvature**. Its manifold is a *closed
loop*, so a chord must cross the ring's interior — a region that corresponds to no
direction at all, which is why certainty collapses there. Speed's manifold is an *open*
curve: its chord leaves the manifold geometrically, but every point along it still decodes
to a sensible intermediate speed, so the model is untroubled.

So: **off-manifold energy is necessary but not sufficient for behavioural degradation.
What matters is whether the off-manifold region is semantically empty.** Manifold steering
is worth its cost for variables with closed or otherwise non-convex geometry, and buys
essentially nothing for monotone scalars — where the simpler, more accurate multi-probe
method (F10) is strictly preferable.

## Comparing the two methods

| | multi-probe subspace (Part 1) | spline / manifold (Part 2) |
|---|---|---|
| endpoint accuracy | **better** (5.3° vs 7.0°) | worse |
| needs probes? | yes, K of them | no — only per-value centroids |
| needs labels? | yes | yes (to form centroids) |
| intermediate states | off-manifold; certainty collapses on closed geometry | stay on the data manifold |
| cost | K probe fits + a least-squares solve | one PCA + one spline fit |
| where it wins | monotone scalars; any endpoint-only task | closed/cyclic variables; anything needing valid intermediate states |
| main limitation | optimises the probe metric, so it can satisfy probes without moving the representation (see C6) | ignores probe geometry, so it undershoots the target |


---

## F15. The on-manifold edit survives propagation 3× better

The open question left by F14: a Euclidean edit gets rotated out of the readout subspace by
attention — does an *on-manifold* edit resist that? Both methods steered to the **same
held-out target values**, injected at L16, propagated through the same blocks, scored with
the same non-degenerate metric.

| | alignment at injection | after 4 blocks | **retained** | ‖delta‖ retained |
|---|---:|---:|---:|---:|
| multi-probe K=16 (Part 1) | +1.329 | +0.332 | **25 %** | 69 % |
| **spline / manifold (Part 2)** | +1.030 | **+0.783** | **76 %** | 92 % |
| linear chord at t=0.5 | +0.024 | +0.011 | *(never carried any)* | 92 % |

**Evidence:** `run_open2.py`; `artifacts/figures/retention_direction.png`.

### Control: it is not a magnitude effect

The two edits differ in size (‖Δ‖ = 8.5 for multi-probe, 22.1 for the spline), so retention
could in principle be measuring *bigger edit survives better*. Rescaling the multi-probe edit —
same direction, different size — settles it:

| multi-probe edit | ‖Δ‖ | alignment at injection | after 4 blocks | **retained** |
|---|---:|---:|---:|---:|
| ×0.5 | 4.2 | +0.725 | +0.226 | **31.2 %** |
| ×1.0 | 8.5 | +1.329 | +0.332 | 25.0 % |
| ×2.0 | 17.0 | +2.537 | +0.552 | 21.7 % |
| **×2.6** | **22.1** | +3.261 | +0.686 | **21.0 %** |
| **spline (on-manifold)** | **22.1** | +1.030 | **+0.783** | **76.0 %** |

**Retention falls as the Euclidean edit grows.** The confound runs the *opposite* way: the
spline's larger size was a handicap, not an advantage. At **matched perturbation norm** the gap
widens from 3× to **3.6×** (76.0 % against 21.0 %).

The sharpest statement available: hand the Euclidean edit **three times the initial alignment**
(+3.26 against +1.03) *and* the same perturbation budget, and it still ends up **below** the
spline four blocks later — +0.686 against +0.783. What survives is not the size of the edit but
whether it points somewhere the network maintains.

### This reverses the verdict of F10

F10 found spline steering *worse* at hitting the target (7.02° vs 5.32°) — measured in
feature space, at the layer of the intervention. F15 measures the same two edits four
blocks later, where the model is actually computing, and the ordering flips hard:

> **The multi-probe method is more accurate where you measure it; the manifold method is
> more durable where the model computes.**

Multi-probe steering *solves* for probe readouts, and it overshoots to do so (alignment
1.329, well past the unit norm of a natural representation). That excess is exactly the
off-manifold component attention discards. The spline edit lands at a natural magnitude
(1.030) and is largely preserved.

Most telling: the spline edit puts **less** of itself in the readout subspace at injection
(0.092 vs 0.316) yet produces a comparable readout and survives far better. It is not
pushing harder along the probe direction — it is moving the representation the way the
network's own geometry moves it.

### The linear chord is a different failure mode

Its alignment is **+0.024 at the injection point** — it never carried direction information
at all. Attention is not destroying anything there; the chord midpoint simply lands in the
ring's interior, which is semantically empty (F12). So there are two distinct failure
modes, and F14 and F12 describe different ones:

| | starts aligned? | survives? | mechanism |
|---|---|---|---|
| multi-probe (Euclidean, valid target) | yes (+1.33) | no — 25 % | **attention rotates it out** (F14) |
| linear chord (t=0.5) | **no** (+0.02) | n/a | lands in a semantically empty region (F12) |
| spline (on-manifold) | yes (+1.03) | **yes — 76 %** | moves along directions the network maintains |

This is the strongest statement the project supports for Wurgaft et al.'s thesis: geometry
is not merely descriptive of the representation — an intervention that respects it is
**three times more durable** under the model's own forward computation.


---

## F16. Is the off-manifold state "within the margin of error"? No — but not for the obvious reason

The objection worth pre-empting: maybe the chord midpoint is a perfectly plausible state and we
are reading noise in a wobbly spline fit. Three tests, and **the intuitive one fails**.

**Method.** Resample the 1,148 training clips with replacement, 100 times. Each resample refits
the *whole* pipeline — PCA, per-value centroids, periodic spline. Because every bootstrap has its
own PCA basis (arbitrary component signs, possible axis swaps), each curve is mapped back to the
1024-d activation space and re-projected into one **reference** basis before anything is compared.
Comparing them in their native coordinates would measure basis churn, not fit uncertainty.

### Test 1 — the bootstrap tube: passes, modestly

| | |
|---|---:|
| 95 % tube radius (median over the ring) | **6.27** |
| chord distance from the manifold at t = 0.5 | **13.01** |
| worst point (t = 0.48) | **2.4× outside the tube** |
| fraction of the chord outside the tube | **51 %** |

Real, but not overwhelming — the tube is ~38 % of the ring radius, so the manifold is not that
tightly pinned by 1,148 clips.

### Test 2 — distance from the curve: **fails, and it is the test people reach for first**

| | |
|---|---:|
| mean distance of a *real clip* from the centroid curve | **17.20** |
| chord midpoint's distance from the curve | **13.01** |
| real clips sitting farther from the curve than the midpoint | **77 %** |

**The midpoint is closer to the manifold than a typical real clip is.** Anyone who argues "the
off-manifold state is only 13 units out, and real data is 17 units out" is making a correct
observation. Distance-from-the-curve does *not* establish that the state is unreachable, and
claiming it does would be overselling.

### Test 3 — the radial shell: decisive

The resolution is that in 64 dimensions the data occupies a **shell**, not a ball. The ring's
interior is empty not because it is far away, but because nothing lives there.

| distance from the ring centre | |
|---|---:|
| 1,148 real clips — min / p1 / median / max | **15.81** / 16.83 / 23.69 / 39.44 |
| chord midpoint | **5.10** |
| **real clips closer to the centre than the midpoint** | **0 of 1,148** |
| midpoint radius ÷ closest real clip's radius | **0.32×** |
| z-score of the midpoint's radius | **−4.1** |

Nearest-neighbour test agrees: the midpoint's nearest real clip is **16.06** away, the **99.5th
percentile** of the real clip-to-clip nearest-neighbour distribution (median 10.41), and only 6 of
1,148 real clips are that isolated.

**Evidence:** `vjp/bootstrap.py`; `artifacts/figures/bootstrap_tube.png`.

### What this licenses saying — and what it does not

**Licensed:** the chord midpoint occupies a region of activation space that **no training clip
occupies**, at 0.32× the radius of the closest one, 2.2 standard deviations of the radial spread
below the nearest real data. Combined with F12b — speed and axis intact, direction annihilated —
the state is not a noisy variant of a real state. It is structurally distinct.

**Not licensed:** "it is far outside the manifold's confidence tube." It is 2.4× outside, which is
a weaker claim than it sounds, and the naive distance-from-curve framing actively contradicts it.
Lead with the **radial shell** result (0 of 1,148), not the tube.

> In high dimensions, an off-distribution point
> is often **closer to the mean** than real data is, not farther from it. Distance-to-the-fit is the
> wrong diagnostic; occupancy of the region is the right one.


---

## F17. Continuous feature clamping transfers to a world model — and a single on-manifold edit beats it

Borrowed directly from LLM interpretability: rather than asking a one-shot edit to survive, assert
the feature at **every** block. F14 said attention rotates a Euclidean edit out of the readout
subspace; this stops asking it to survive and re-asserts it at blocks 16–19.

Two things get conflated, and only the second is clamping:

- **re-injection** — add the same Δx again at each block. The perturbation *accumulates*.
- **clamping** — at each block, read the probe and apply the **minimum-norm correction** that makes
  it read the target. Self-limiting: if the representation has barely drifted, the correction is small.

Alignment with the intended direction, held-out clips and held-out targets:

| arm | L17 (+1) | L18 (+2) | **L20 (+4)** | L24 (+8) | Σ‖Δ‖ per clip |
|---|---:|---:|---:|---:|---:|
| no edit (control) | +0.12 | +0.11 | +0.12 | +0.11 | 0 |
| one-shot Euclidean (F14) | +0.89 | +0.66 | **+0.36** | +0.22 | 8.5 |
| re-inject same Δx ×4 | +0.89 | **+1.30** | **+1.29** | +0.60 | **33.9** |
| **clamp K=16 across 16–19** | +0.89 | +0.89 | **+0.79** | +0.35 | 20.2 |
| clamp K=1 across 16–19 | +0.33 | +0.68 | +0.54 | +0.22 | 7.5 |
| **one on-manifold spline** | **+1.10** | **+1.22** | **+0.88** | **+0.69** | 22.1 |

**Evidence:** `run_clamp.py`; `vjp/clamp.py`; `artifacts/figures/clamping.png`.

### Clamping works

At four blocks downstream, clamping more than **doubles** the retained effect over the one-shot edit
(+0.79 vs +0.36). The technique transfers from language models to a video world model without
modification: repeatedly asserting a linear feature does hold it in place against the attention
rotation of F14.

### Re-injection is not the same thing, and it overshoots

Adding the same Δx four times drives alignment to **+1.30** — past the unit norm of any natural
representation — at **4× the perturbation cost**, and it still decays to +0.60 by L24. It buys the
readout by brute force and leaves the state over-driven.

### The comparison has to match K, or it is rigged

Clamping with a **single** probe reaches only +0.54 at L20 against +0.79 for K=16. F8 already showed
a single probe barely moves direction, so a K=1 clamp competing against a K=16 one-shot edit would
have made clamping look worse than it is. Both arms assert the same 16-probe subspace, refitted at
each clamped layer.

### The punchline: one on-manifold edit beats four clamped ones, at the same cost

| | clamp K=16 | on-manifold spline |
|---|---:|---:|
| interventions | **4** (blocks 16–19) | **1** (block 16) |
| alignment at L20 | +0.79 | **+0.88** |
| alignment at L24 | +0.35 | **+0.69** |
| total perturbation | 20.2 | 22.1 |

Essentially identical cost, better retention at every depth, and **one** intervention rather than
four. By eight blocks the gap is 2×.

> There are two ways to keep a physical variable under control. Either
> forcibly clamp a linear feature at every layer to fight the attention mechanism — the LLM
> interpretability move, and it does work here — or make one edit at one layer that respects the
> representation's geometry, which the network then sustains on its own. The second is not cheaper in
> raw perturbation; it is cheaper in *interventions*, and it ends up in a more natural state.

*Worth noting against the "smaller edits are better" intuition: the spline edit is the LARGER
one-shot perturbation (22.1 vs 8.5). It is not about moving less — it is about moving along a
direction the network maintains.*


---

## Corrections — bugs found and fixed

Two of these changed results materially. Recorded because they affect how much to trust
the numbers above.

**C1. INLP was silently stalling (fixed; changed F3 completely).**
`readout = W/sd`. Once a direction had been projected out, its feature variance sat on
the 1e-6 floor, so `W/sd` exploded *along already-dead directions* and QR kept returning
them. The projection became a no-op while `dims_removed` kept counting. Symptom: rank
froze at 30 after one iteration, and 72 "dimensions" were removed from a 32-d space.
Fixed by zeroing spent features in `readout` and deflating each new basis against the
cumulative removed span, with an early stop when the probe direction lies inside it.
*Before the fix:* direction R² 0.992 → 0.886 after "298 dims", no bend, no dimensionality
number. *After:* 0.992 → 0.01 over 398 dims with a clean bend. **The earlier
full-space numbers were wrong and have been discarded.**

**C2. `steer_linear` was a no-op duplicate of manifold steering (fixed, pre-empting Part 2).**
It computed the source centroid and discarded it, returning exactly what `steer()`
returned. Part 2's headline comparison would have reported "no difference between
methods". Replaced with `steer_at(X, targets, t, mode)`: the two modes now differ at
intermediate `t` and coincide at `t=1` *by construction*, since Eqs. 1 and 2 differ in the
**path**, not the endpoint. An endpoint-only evaluation cannot distinguish them.

**C3. The mask-only baseline was too weak (fixed; created F6).**
The first version used raw (x, y) coordinates and scored R² ≈ 0 on speed — an artefact,
since speed is the *norm* of a displacement and a linear map cannot take a norm. It would
have made the leak floor flatteringly low and the `obj` comparison meaningless.

**C4. Salience pooling initially collapsed onto the uniform mean (fixed before use).**
Soft deviation-proportional weights gave `cos(tmean, sal) = 1.000`: the weight
distribution flattens with depth (object weight share 0.61 at L1 → 0.03 at L20). Replaced
with top-K selection.

**C5. PCA for the reduced-space INLP is fitted on train rows only.** An earlier diagnostic
SVD used all rows, which would have leaked held-out values into the basis.

**C8. "Direction occupies 2× the dimensions" conflated readout rank with redundancy (fixed; reframed F3).**
INLP removes `k` dimensions per round, where `k` is the rank of the probe's readout — 2 for a
(sin, cos) target, 1 for a scalar. Reporting *dimensions removed* therefore built a factor of two
into the comparison before any property of the representation was measured. Counting **rounds**
instead gives 71 / 69 / 67 — essentially equal. Confirmed decisively by re-running direction with a
1-d target (sin θ alone: 76 dims; cos θ alone: 54), which lands in the scalars' range. The random
control and the redundancy claim are unaffected; only the cross-variable *ratio* was wrong. A real
~1.2× asymmetry survives at the 10 % threshold.

**C7. Circular MAE is degenerate when the readout collapses (fixed; strengthened F12).**
Circular MAE takes `arctan2` of the predicted (sin, cos) pair. When an intervention drives
the readout toward zero norm, that angle is noise and the metric is arbitrary — the same
linear-steered state scored **13°** under one probe and **91°** under another, both
legitimately fitted. Detected when the half-block diagnosis reported 93° at a position
where F12 had reported 23°. All steering results that can drive the readout off-manifold
now report **certainty** (‖pred‖) and **alignment** (⟨pred, unit(intended)⟩), which are
well behaved at zero. F12's corrected numbers are considerably stronger than the originals;
F8's are unaffected, as certainty never collapses there.

**C6. Steering dropped the per-probe mean term (fixed; F8 did not exist before this).**
A probe standardises before reading, so `predict(x) = (x - mu) @ readout + ym` — the
intercept is `ym - mu @ readout`. `steer()` used `ym` alone, biasing every solve.
*Symptom:* the basis probes missed their own target by MAE 1.06 on speed, so "steering"
did nothing and speed appeared to get **worse** with more probes (0.757 → 1.572). The
error was near-invisible for direction because (sin, cos) targets are near zero-mean,
which is exactly how it survived the synthetic test. Fixed via `probe_intercept()`;
residuals are now ~1e-15 and an assertion enforces it.

---

## Open

- **Deeper propagation for the path experiment.** F12 reads one block downstream; F9
  showed the linear effect decays to 15 % by four blocks. Whether the manifold advantage
  *widens* with depth is the natural follow-up and needs no new machinery.
- **Deeper open-manifold controls.** Run to two blocks; whether the null result holds at four
  is untested, though F12's mechanism predicts it should.
- **Deeper propagation.** F17 now reaches L24 (+8 blocks), where the spline still leads 2×.
  Whether that holds to the final layer is a direct extension.
- **Clamping the spline.** F17 clamps only the linear subspace. Clamping the on-manifold target at
  each block is the obvious next arm and needs no new machinery.
- **Does F15 hold for the scalars?** F12 predicts little difference for speed, since its
  off-manifold region is not semantically empty — worth confirming.
- **Stage 5 — spline / manifold steering.** `Manifold` verified on a synthetic circle
  (periodic fit recovers it; linear path has 238× the off-manifold energy); never fitted
  to real activations. F4 says fit in ~64-D.
- **Behaviour channel for Part 2.** The predictor ships and is driveable
  (cos 0.597 vs 0.482 shuffled) but the margin is modest because ~98 % of tokens are
  background — restrict the readout to object tokens.
- **Cross-dataset direction transfer.** Theta spans all 64 values in all three datasets;
  a direction probe trained on `direction` can be tested on `speed` / `acceleration`
  clips. Cheap, and the best generalisation test the data affords. Not yet run.
- **64-frame ablation.** The checkpoint is `fpc64`; we feed 16. ~16 min on a stratified
  subset would turn this documented assumption into a measurement.
