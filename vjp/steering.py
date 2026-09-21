"""Multi-probe subspace steering -- Part 1, step 3.

Reproduces the paper's intervention: stack K probe readouts into an orthonormal
basis V = QR([W_1 ... W_K]), decompose each activation into its component in that
subspace and the orthogonal remainder, solve for the coordinates that make the
probes read a TARGET value, and recompose.

Three things are held out, and they are independent:
  * VALUES  -- the clips being steered carry target values never seen in fitting;
  * PROBES  -- the evaluation probe is never one of the K that built the basis;
  * CLIPS   -- the evaluation probe is fitted on clips disjoint from all K.

Without the probe split, "steering worked" would only mean the intervention
satisfied the very equations it solved.
"""
from __future__ import annotations
import numpy as np

from . import probes as P


def _fold_probes(X, Y, folds: list[np.ndarray], val: np.ndarray):
    return [P.fit_ridge_cv(X[f], Y[f], X[val], Y[val]) for f in folds]


def make_probe_bank(X, Y, train: np.ndarray, val: np.ndarray, n_probes: int,
                    seed: int = 0):
    """Fit `n_probes` basis probes plus one evaluation probe on disjoint clips.

    Returns (basis_probes, eval_probe). The evaluation probe gets its own fold, so
    it shares no fitting data with any probe that shaped the steering subspace.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(train)
    folds = np.array_split(idx, n_probes + 1)
    eval_fold, basis_folds = folds[0], folds[1:]
    return (_fold_probes(X, Y, basis_folds, val),
            P.fit_ridge_cv(X[eval_fold], Y[eval_fold], X[val], Y[val]))


def steer_to(X, basis, target_values, dataset: str):
    """Steer activations so every basis probe reads `target_values`."""
    V = P.probe_subspace(basis)
    T = P.to_target(dataset, np.asarray(target_values, float))   # [N, k]
    target = np.tile(T, (1, len(basis)))                         # one block per probe
    return P.steer(X, V, basis, target), V


def evaluate_steering(dataset, X, Y, values, train, val, test, n_probes,
                      seed=0, n_targets=4):
    """Steer held-out clips to held-out target values; read with a held-out probe.

    Each test clip is steered to `n_targets` target values drawn from the held-out
    set, so the evaluation never asks the probe about a value it was fitted on.
    """
    basis, evp = make_probe_bank(X, Y, train, val, n_probes, seed)
    rng = np.random.default_rng(seed + 1)
    held_vals = np.unique(values[test])

    src, tgt = [], []
    for _ in range(n_targets):
        src.append(test)
        tgt.append(rng.choice(held_vals, size=len(test)))
    src = np.concatenate(src)
    tgt = np.concatenate(tgt)

    Xs, V = steer_to(X[src], basis, tgt, dataset)

    # The solve must be exact for the probes that defined the subspace. If this
    # is not ~0 the intervention is broken, and every downstream number is noise
    # rather than a statement about the representation.
    T = P.to_target(dataset, tgt)
    resid = float(np.mean([np.abs(b.predict(Xs) - T).mean() for b in basis]))
    assert resid < 1e-6, f"steering solve not satisfied (residual {resid:.2e})"

    pred = evp.predict(Xs)
    out = P.evaluate(dataset, pred, P.to_target(dataset, tgt), tgt)
    # Does the intervention ROTATE the representation, or just inject noise that
    # happens to satisfy the probe? If it genuinely rotates, error to the target
    # falls while error to each clip's OWN label rises by the same amount, and the
    # readout stays confident. Noise injection would collapse certainty instead.
    away = P.evaluate(dataset, pred, P.to_target(dataset, values[src]), values[src])
    # How well did the probe read the ORIGINAL clips? Steering cannot beat this.
    base = P.evaluate(dataset, evp.predict(X[src]), P.to_target(dataset, values[src]),
                      values[src])
    # Control: does the unsteered activation already look like the target?
    null = P.evaluate(dataset, evp.predict(X[src]), P.to_target(dataset, tgt), tgt)
    return {
        "n_probes": n_probes,
        "subspace_dim": int(V.shape[1]),
        "steered": out,
        "readout_ceiling": base,       # probe accuracy on unmodified clips
        "unsteered_vs_target": null,   # chance-level floor
        "n_eval": int(len(src)),
        "basis_residual": resid,
        "steered_vs_own_label": away,
        "certainty": float(np.linalg.norm(pred, axis=1).mean())
                     if dataset == "direction" else float(np.std(pred[:, 0])),
    }


def specificity(dataset_other: str, X_other, values_other, X, basis, target_values,
                dataset: str, probe_other):
    """Does steering variable A leave variable B alone?

    Only meaningful where both variables are labelled for the same clips.
    """
    Xs, _ = steer_to(X, basis, target_values, dataset)
    before = probe_other.predict(X)
    after = probe_other.predict(Xs)
    return float(np.abs(after - before).mean())
