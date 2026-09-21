# Corrections

Errors found during the work and what they changed. Kept deliberately rather than tidied away:
several reversed a headline result, and the path to a number is part of the evidence for it.

Findings are in [FINDINGS.md](FINDINGS.md); every correction names the findings it touched.

Four changed a published result: **C1** (the dimensionality curve never bent), **C6** (speed
steering appeared to fail), **C8** (the 2× dimension gap was readout rank, not redundancy) and
**C10** (the forecast ceiling came from the wrong split). Two more, **C2** and **C7**, would have
broken Part 2's headline comparison before it was ever run.

---

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

**C6. Steering dropped the per-probe mean term (fixed; F8 did not exist before this).**
A probe standardises before reading, so `predict(x) = (x - mu) @ readout + ym` — the
intercept is `ym - mu @ readout`. `steer()` used `ym` alone, biasing every solve.
*Symptom:* the basis probes missed their own target by MAE 1.06 on speed, so "steering"
did nothing and speed appeared to get **worse** with more probes (0.757 → 1.572). The
error was near-invisible for direction because (sin, cos) targets are near zero-mean,
which is exactly how it survived the synthetic test. Fixed via `probe_intercept()`;
residuals are now ~1e-15 and an assertion enforces it.
**C7. Circular MAE is degenerate when the readout collapses (fixed; strengthened F14).**
Circular MAE takes `arctan2` of the predicted (sin, cos) pair. When an intervention drives
the readout toward zero norm, that angle is noise and the metric is arbitrary — the same
linear-steered state scored **13°** under one probe and **91°** under another, both
legitimately fitted. Detected when the half-block diagnosis reported 93° at a position
where F14 had reported 23°. All steering results that can drive the readout off-manifold
now report **certainty** (‖pred‖) and **alignment** (⟨pred, unit(intended)⟩), which are
well behaved at zero. F14's corrected numbers are considerably stronger than the originals;
F8's are unaffected, as certainty never collapses there.

**C8. "Direction occupies 2× the dimensions" conflated readout rank with redundancy (fixed; reframed F3).**
INLP removes `k` dimensions per round, where `k` is the rank of the probe's readout — 2 for a
(sin, cos) target, 1 for a scalar. Reporting *dimensions removed* therefore built a factor of two
into the comparison before any property of the representation was measured. Counting **rounds**
instead gives 71 / 69 / 67 — essentially equal. Confirmed decisively by re-running direction with a
1-d target (sin θ alone: 76 dims; cos θ alone: 54), which lands in the scalars' range. The random
control and the redundancy claim are unaffected; only the cross-variable *ratio* was wrong. A real
~1.2× asymmetry survives at the 10 % threshold.

**C9. "Speed untouched" overstated the result, and two documents quoted the wrong endpoint (fixed).**
The claim rested on the population *mean*, which is flat along the whole chord (2.10 → 2.17 m/s).
Per-clip error is not: it runs 0.071 → 0.136 → 0.258, i.e. **1.1× → 2.2× → 4.1×** the speed
probe's own baseline of 0.062. Speed is *degraded but retained*, not untouched — still a sharp
contrast with direction reaching zero information, but a different claim. Separately, README and
the deck quoted **2.10 → 2.17**, the t=0 → t=**1** pair, alongside the midpoint's direction
collapse — but at t=1 direction is *restored* (certainty 0.984), so two different points on the
path were being presented as one state. FINDINGS had the correct pair throughout. The figure now
plots per-clip error rather than the flat mean, and the speed badge reads DEGRADED past the
midpoint; calling it NULL would have been the opposite error, since 0.26 m/s on a 0.25–4.0 range
is still informative.



**C10. The forecast ceiling was quoted from the wrong split, and two arms were unreproducible (fixed).**
F18 originally quoted a forecast-probe ceiling of 11.1°, measured on the 150-clip
alpha-selection fold — which sits on *train* values and is not comparable to the steered
numbers. Worse, the control arm read its own label at 6.5°, i.e. *better than the stated
ceiling*, so a dashed ceiling line sat above a bar that beat it. The ceiling is now the control
arm itself: the same probe, same held-out-value test clips, reading each clip's own label. The
spline's gap-closed falls from 95 % to **89 %** and the matched-norm comparison from 41 % to
38 % — the finding is unchanged, the arithmetic is now consistent. Separately, the ×2.00 and
×2.61 rescaled arms were computed in a scratch script that was not kept, so the published table
could not be regenerated; `run_behaviour.py` now runs all five arms and saves floor, ceiling and
norms with the results.
