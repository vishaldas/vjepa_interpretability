"""Cross-variable steering leakage: does steering A disturb the readout of B?

A full 3x3 matrix is NOT estimable on this data. Speed and acceleration are
mutually exclusive in every supplied dataset -- no clip has both non-zero
(`speed`: accel == 0; `acceleration`: speed == 0; `direction`: velocity clips have
accel == 0 and acceleration clips have speed == 0). So the kinematic coupling
v = v0 + a*t has been designed out, and the speed <-> acceleration cells have no
clips that could reveal them.

What IS measurable, because theta spans all 64 values in every dataset:
  * within `speed`:        direction <-> speed
  * within `acceleration`: direction <-> acceleration

Each cell reports drift in the OTHER variable's readout caused by the steer,
against that probe's own MAE on unmodified clips -- the natural yardstick, since
a drift below the probe's own error is not distinguishable from its noise.
"""
from __future__ import annotations
import numpy as np

from . import experiments as E
from . import probes as P
from . import steering as S
from . import features as F


def _labels(dataset: str, which: str) -> np.ndarray:
    idx = F.load_index(dataset)
    if which == "direction":
        return np.array(idx["theta_degrees"], float)
    if which == "speed":
        return np.array(idx["speed_mps"], float)
    return np.array(idx["acceleration_mps2"], float)


def _target(which: str, vals: np.ndarray) -> np.ndarray:
    return P.to_target("direction" if which == "direction" else "speed", vals)


def _err(which: str, pred: np.ndarray, vals: np.ndarray) -> float:
    if which == "direction":
        return P.circular_mae(pred, vals)
    return float(np.abs(pred[:, 0] - vals).mean())


def _drift(which: str, a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute change in the readout between steered and unsteered."""
    if which == "direction":
        pa = np.degrees(np.arctan2(a[:, 0], a[:, 1]))
        pb = np.degrees(np.arctan2(b[:, 0], b[:, 1]))
        return float(np.abs((pa - pb + 180) % 360 - 180).mean())
    return float(np.abs(a[:, 0] - b[:, 0]).mean())


def cell(dataset: str, layer: int, steer_var: str, read_var: str, K: int,
         pooling: str = "sal", seed: int = 0, n_targets: int = 2,
         n_read: int = 250) -> dict:
    """One matrix entry: steer `steer_var`, watch `read_var`."""
    X = F.load_pooled(dataset, pooling, layer, "mean")
    v_s, v_r = _labels(dataset, steer_var), _labels(dataset, read_var)
    Ys, Yr = _target(steer_var, v_s), _target(read_var, v_r)

    # hold out INTERLEAVED, INTERIOR values of the steered variable
    uniq = np.unique(v_s)
    held = {float(u) for i, u in enumerate(uniq) if 0 < i < len(uniq) - 1 and i % 8 == 7}
    test = np.flatnonzero([float(x) in held for x in v_s])
    train = np.flatnonzero([float(x) not in held for x in v_s])
    rng = np.random.default_rng(seed)
    train = rng.permutation(train)
    val, train = np.sort(train[:len(train) // 5]), np.sort(train[len(train) // 5:])

    # Reserve a FIXED-SIZE slice for the readout probes before the basis probes are
    # fitted. If the readout fold shrank as K grew (train/(K+1)), its baseline error
    # would degrade with K and the drift/baseline ratio would improve for free --
    # the comparison across K must hold the yardstick constant.
    perm = rng.permutation(train)
    read_fold, basis_pool = np.sort(perm[:n_read]), np.sort(perm[n_read:])
    basis, _ = S.make_probe_bank(X, Ys, basis_pool, val, K, seed)
    rp = P.fit_ridge_cv(X[read_fold], Yr[read_fold], X[val], Yr[val])

    held_vals = np.unique(v_s[test])
    src = np.concatenate([test] * n_targets)
    tgt = np.concatenate([rng.choice(held_vals, size=len(test)) for _ in range(n_targets)])
    Xs, _ = S.steer_to(X[src], basis, tgt, "direction" if steer_var == "direction" else "speed")

    before, after = rp.predict(X[src]), rp.predict(Xs)
    return {
        "dataset": dataset, "steer": steer_var, "read": read_var, "K": K,
        "baseline_mae": _err(read_var, before, v_r[src]),   # probe's own error, unsteered
        "drift": _drift(read_var, after, before),
        "on_target": _err(steer_var, _read_self(X, Ys, read_fold, val, Xs), tgt),
    }


def _read_self(X, Ys, read_fold, val, Xs):
    """On-target accuracy, read by a probe on the same fixed held-out fold."""
    return P.fit_ridge_cv(X[read_fold], Ys[read_fold], X[val], Ys[val]).predict(Xs)
