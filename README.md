# Interpreting physical variables in V-JEPA 2

A study of how a frozen **V-JEPA 2 ViT-L/16-256** encoder represents the direction, speed
and acceleration of a moving object — and of what that structure implies for steering it.

Response to the World Mechanics take-home ([task and data](https://anonymous.4open.science/r/vjepa-physics-takehome-4E00/README.md)).
Part 1 reproduces the probing / nullspace / steering progression from Joseph et al.;
Part 2 extends it with spline-based manifold steering after Wurgaft et al., plus a
mechanistic account of *why* linear steering fails and a comparison against continuous
feature clamping.

---

## Start here

| | |
|---|---|
| **[FINDINGS.md](FINDINGS.md)** | 18 findings, each with its evidence table, the command that reproduces it, and a status marker. |
| **[CORRECTIONS.md](CORRECTIONS.md)** | Ten errors found and fixed during the work — four of which changed a published result. Kept deliberately: the path to a number is part of the evidence for it. |
| **[artifacts/deck/](artifacts/deck/)** | The presentation (`.pptx`, 12 slides incl. one backup), with speaker notes. |
| **[notes/SETUP_FINDINGS.md](notes/SETUP_FINDINGS.md)** | Everything measured about the model and data rather than assumed. |
| **[notes/CACHE.md](notes/CACHE.md)** | Feature cache layout and the train/val/test protocol. |
| **[PLAN.md](PLAN.md)** | The original plan, kept for the record; superseded in places by what the measurements showed. |

---

## What was found

**The two scalars are ordinary; direction is a ring.** That difference decides whether the
variable can be steered.

#### Part 1 — what is represented, and where

- All three variables are strongly linearly decodable: direction **2.57°** circular MAE
  (chance 90°), speed **0.056** MAE, acceleration **0.153**, on held-out label values.
- **Direction is available from layer 1** (11.4°), not at a mid-depth transition. A
  position-only control sits at chance (86.3°), confirming the probe reads motion rather
  than location. This diverges from the paper and is most likely a dataset-simplicity effect.
- A **tracker-only oracle** beats every encoder probe (1.19° / 0.011 / 0.036), bounding how
  much of the recoverable signal the representation actually keeps.
- **All three variables are carried by a highly redundant population code**: probe-directed
  removal drives R² from 0.99 to 0.01, while removing the same number of random directions
  leaves it at 0.99. Counted in *independent readouts* the three are equal (71 / 69 / 67 rounds);
  direction costs twice the *dimensions* only because its (sin, cos) readout is rank 2.
- Direction's centroids form a **Fourier ladder at m = 1, 2, 4**, where the m ≥ 2 components are
  180°-invariant: an *axis of motion* code, distinct from direction, which the network builds
  with depth (3.1 % at layer 0 → 23.2 % at layer 24).
- Multi-probe subspace steering works, but **a single probe barely moves direction** (28 % of
  the gap closed, against 74 % for either scalar), and enlarging the subspace to compensate
  destroys specificity — by dimension 16 every cross-variable cell leaks.

#### Part 2 — geometry, and why linear steering fails

- The intervention solves `min ‖Δx‖₂ s.t. W(x+Δx)+b = y` — the **Moore-Penrose minimum-norm
  edit**, verified against a direct pseudo-inverse to 3.1e−14. It is the smallest *Euclidean*
  step, and Euclidean smallness is blind to where the data lies.
- **Off-manifold does not imply off-behaviour.** Linear paths run 80–117× off manifold for
  all three variables, but only direction breaks. What separates them is **topology, not
  curvature**: direction's ring is closed, so a chord must cross an interior that corresponds
  to no direction at all.
- At that midpoint the model holds a **structured, physically impossible state** — axis intact
  (5.9° error), speed degraded but retained (per-clip *error* rises to 2.2× its baseline — 0.14 m/s
  of error on a 0.25–4.0 m/s label range, so the speed is still clearly encoded), direction annihilated (certainty 0.18, i.e. zero information).
- The edit is **not erased downstream but rotated**: 69 % of its magnitude survives four
  blocks while only 32 % of its readout-aligned component does. **Attention accounts for 77 %**
  of the decay; LayerNorm is ruled out and the MLP contributes little.
- **A single on-manifold spline edit retains 76 % of its intended readout after four blocks,
  against 25 % for the linear edit**, at comparable perturbation cost — reversing the ordering
  seen at the layer of the intervention.
- **Continuous feature clamping** across blocks also works (+0.36 → +0.79 at four blocks),
  but needs four interventions to achieve less than one geometry-respecting edit.
- **And it changes what the model forecasts.** Driving the shipped predictor after a layer-16
  edit, the on-manifold edit moves the forecast onto the target (81.4° → **14.4°**, 89 % of the
  floor-to-ceiling gap) while the Euclidean edit barely moves it at all (76.4°, 6.6 %) — 38 % even
  when rescaled to the same perturbation norm. Satisfying a probe is not the same as steering a
  world model.

---

## Reproducing

Tested on an Apple M2 Max (32 GB) using the PyTorch **MPS** backend; no CUDA GPU required.

**The clips are not in this repository.** They belong to World Mechanics and are
distributed by them at **[https://anonymous.4open.science/r/vjepa-physics-takehome-4E00/README.md](https://anonymous.4open.science/r/vjepa-physics-takehome-4E00/README.md)**. Download `data/` from there into the
repository root — the layout is `data/<variable>/{manifest.jsonl, videos/}`, exactly as
supplied — then:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m vjp.extract          # one-time, ~26 min, writes a 5.4 GB cache
```

Extraction runs the frozen encoder over all 4,572 clips and stores, for every clip and all
26 residual-stream states, three spatial poolings of the 2,048 tokens. Everything after that
reads the cache:

```bash
.venv/bin/python run_part1.py            # layer sweep, deciles, nullspace, steering   (~2 s)
.venv/bin/python run_part2.py            # manifolds, path geometry, endpoint accuracy (~14 s)
.venv/bin/python run_part2.py --causal   # + the causal path test                     (~4 min)
.venv/bin/python run_causal.py           # does a mid-layer edit survive propagation?
.venv/bin/python run_mechanism.py        # half-block diagnosis: where does it decay?
.venv/bin/python run_open2.py            # multi-probe vs spline, propagated
.venv/bin/python run_clamp.py            # clamping vs one-shot vs on-manifold
.venv/bin/python run_behaviour.py        # does the PREDICTOR's forecast change?      (~12 min)
.venv/bin/python run_open.py             # manifold vs linear path, propagated
```

`run_part1.py` and `run_part2.py` cover the three task steps; the rest are the follow-up
experiments. `run_behaviour.py` is the one that drives the predictor rather than reading an
encoder probe.

Experiment results are memoised by a hash of their full config, so re-running an analysis or
redrawing a figure costs nothing. The cache is treated as immutable once written — MPS
reductions are not run-to-run deterministic, so re-extracting mid-project would make probe
numbers drift between experiments.

---

## Layout

```
vjp/            library
  config.py       paths, model constants, HF token loading
  data.py         manifests, decoding, disk tracking, split construction
  features.py     the three pooling operators and the cache API
  extract.py      the single extraction pass
  experiments.py  split logic, layer sweeps, nullspace curves, memoisation
  probes.py       ridge probes, circular targets, INLP, the steering solve
  steering.py     probe banks, held-out-probe evaluation, leakage harness
  splines.py      periodic / open manifold fitting and steering
  part2.py        endpoint accuracy, path geometry, causal path test
  causal.py       forward-pass intervention through the live residual stream
  behaviour.py    drives the predictor; reads its forecast rather than the encoder
  mechanism.py    half-block instrumentation and LayerNorm statistics
  clamp.py        continuous feature clamping
  leakage.py      cross-variable specificity
  harmonics.py    Fourier analysis of the direction ring
  bootstrap.py    confidence tube around the fitted manifold
  figures.py      all plots
run_*.py        entry points, one per experiment
artifacts/      figures (28) and the presentation deck
data/           the supplied clips — not committed; see Reproducing
```

Derived caches, memoised results and the virtual environment are excluded from version
control; all are regenerable from the entry points above.

---

## Method notes

Three choices do most of the work, and each is justified by a measurement rather than a
convention:

**Pooling is a confound, not a detail.** The disk occupies ~1.3 of 256 patches per frame, so a
uniform mean dilutes it below 1 % of the pooled vector. Three poolings are cached. Layer
selection uses `sal` — a **leak-free** top-8-salient-token pooling whose ranking recovers the
true object tokens with recall 1.00 through layer 12. A tracker-selected pooling is also
stored but is **excluded from interpretation**: its mask derives from the trajectory, which is
the label, and the tracker-only baseline beats every encoder probe.

**Three independent held-out axes.** The data is 64 discrete values × 24 repetitions, so a
random *clip* split would place a given label in both train and test and let a probe memorise
it. The headline protocol holds out **values** — interleaved and strictly interior, so
reaching them is interpolation, which is what the spline work requires. Held-out *clips* and
held-out *probes* are separate axes, used where each is the right test.

**Controls before conclusions.** The nullspace curve is reported against random-subspace
removal; the layer-1 direction result against a position-only probe; the `obj` pooling against
its mask-only floor; the clamping arm against a subspace-size-matched baseline.

---

## Corrections

**[CORRECTIONS.md](CORRECTIONS.md)** logs ten errors found during the work, kept deliberately
rather than tidied away. Four changed a published result:

- **INLP was silently stalling** (C1) — the projection became a no-op while the dimension
  counter kept incrementing, so the curve never bent and there was no dimensionality number.
- **Speed steering appeared to fail** (C6) — the solve dropped a per-probe mean term, so the
  basis probes missed their own target and steering looked like it got *worse* with more probes.
- **The 2× dimension gap was readout rank, not redundancy** (C8) — INLP removes 2 dimensions
  per round for a (sin, cos) target and 1 for a scalar, which built the factor of two into the
  comparison before anything was measured.
- **The forecast ceiling came from the wrong split** (C10) — leaving a stated ceiling that the
  control arm beat.

Two more, **C2** and **C7**, would have broken Part 2's headline comparison before it was run.

---

## Limitations

- One high-contrast disk on an empty field. The layer-1 direction result and the clean ring
  geometry are both likely easier here than on natural video.
- Speed and acceleration are **mutually exclusive in every supplied dataset** — no clip has
  both non-zero — so the kinematic coupling `v = v₀ + at` cannot be probed, and two cells of
  the cross-variable leakage matrix are not estimable.
- The bottom of the speed range moves less than 0.05 px/frame, below the encoder's
  spatiotemporal resolution; metrics are reported stratified by label decile rather than
  aggregated.
- The checkpoint was trained at 64 frames per clip and is fed 16. Both run; the 64-frame
  variant is affordable as an ablation on a subset and has not been run.
- Retention is measured to eight blocks past the intervention, not to the final layer.
- "Behaviour" means the predictor's **latent** forecast read by a probe, not decoded pixels. The
  predictor forecasts encoder-space latents, so F18 closes the gap between representation and
  prediction, but not between prediction and rendered output.
