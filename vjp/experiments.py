"""Experiment API over the cached features, with on-disk result memoisation.

Every expensive call is keyed by its full config and cached under
artifacts/results/, so re-running an analysis or redrawing a figure is instant.
"""
from __future__ import annotations
import hashlib, json, time
import numpy as np

from .config import ART, N_STATES, N_T
from . import features as F
from . import probes as P

RESULTS = ART / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def _key(name: str, cfg: dict) -> "Path":
    h = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return RESULTS / f"{name}__{h}.json"


def cached(name: str, cfg: dict, fn, force: bool = False):
    """Memoise fn() to disk under a hash of cfg."""
    path = _key(name, cfg)
    if path.exists() and not force:
        return json.loads(path.read_text())["result"]
    t0 = time.time()
    result = fn()
    path.write_text(json.dumps(
        {"name": name, "config": cfg, "seconds": time.time() - t0, "result": result},
        default=float))
    return result


# ---------------------------------------------------------------- splits

def get_splits(dataset: str, axis: str = "value") -> dict[str, np.ndarray]:
    """Index arrays for one of the three held-out axes.

    axis='value'   held-out VALUES, interleaved and interior -> interpolation test.
                   The headline split; required for Part 2 (splines are fitted
                   through train-value centroids and must interpolate to reach the
                   held-out ones).
    axis='clip'    held-out CLIPS within each value -> nuisance generalisation
                   (start position, and for `direction` the motion regime).
    axis='extrap'  held-out TOP values -> extrapolation, reported separately.
    """
    idx = F.load_index(dataset)
    col = {"value": "split_value", "clip": "split_clip", "extrap": "split_extrap"}[axis]
    lab = np.array(idx[col])
    out = {s: np.flatnonzero(lab == s) for s in np.unique(lab)}
    if "val" not in out:                      # clip/extrap axes are train/test only
        rng = np.random.default_rng(1)
        tr = out["train"].copy()
        rng.shuffle(tr)
        cut = int(0.2 * len(tr))
        out["val"], out["train"] = np.sort(tr[:cut]), np.sort(tr[cut:])
    return out


def get_values(dataset: str) -> np.ndarray:
    return np.array(F.load_index(dataset)["values"], np.float64)


def get_xy(dataset: str, layer: int, pooling: str = "tmean", time: str = "mean"):
    """Design matrix, probe target, and raw label values for one layer."""
    X = F.load_pooled(dataset, pooling, layer, time)
    if X.ndim == 3:
        X = X.reshape(len(X), -1)
    v = get_values(dataset)
    return X, P.to_target(dataset, v), v


# ---------------------------------------------------------------- part 1.1

def layer_sweep(dataset: str, pooling: str = "tmean", time: str = "mean",
                axis: str = "value", force: bool = False) -> dict:
    """Probe every stored layer. Part 1, step 1."""
    cfg = dict(dataset=dataset, pooling=pooling, time=time, axis=axis, v=2)

    def run():
        sp = get_splits(dataset, axis)
        v = get_values(dataset)
        Y = P.to_target(dataset, v)
        rows = []
        for L in range(N_STATES):
            X = F.load_pooled(dataset, pooling, L, time)
            if X.ndim == 3:
                X = X.reshape(len(X), -1)
            m = P.fit_ridge_cv(X[sp["train"]], Y[sp["train"]], X[sp["val"]], Y[sp["val"]])
            r = P.evaluate(dataset, m.predict(X[sp["test"]]), Y[sp["test"]], v[sp["test"]])
            r["layer"] = L
            r["alpha"] = m.alpha
            rows.append(r)
        return rows

    return cached("layer_sweep", cfg, run, force)


def mask_baseline(dataset: str, axis: str = "value", force: bool = False) -> dict:
    """Label-leak floor for `obj` pooling: probe the tracker output ALONE.

    Features are built from the disk trajectory only -- no encoder activations at
    all. Any `obj`-pooled probe score at or below this is explained by the mask,
    not by the representation.

    The feature set deliberately gives the tracker its BEST shot, including the
    nonlinear quantities a linear probe cannot form for itself: per-tubelet
    displacement magnitudes (which are speed), their differences (acceleration),
    and unit-normalised displacement components (which are sin/cos of direction).
    A naive baseline on raw (x, y) coordinates scores R^2 ~ 0 on speed purely
    because a linear map cannot take a norm -- that would make the floor
    flatteringly low and the comparison meaningless.
    """
    cfg = dict(dataset=dataset, axis=axis, v=3)

    def run():
        cents, _ = F.load_tracks(dataset)                   # [N,16,2]
        c = np.nan_to_num(cents)
        tub = c.reshape(len(c), N_T, 2, 2).mean(2)          # per tubelet [N,8,2]
        d = np.diff(tub, axis=1)                            # [N,7,2] displacement
        nrm = np.linalg.norm(d, axis=-1)                    # [N,7]  speed proxy
        dn = d / np.clip(nrm, 1e-6, None)[..., None]        # unit direction
        acc = np.diff(nrm, axis=1)                          # [N,6]  accel proxy
        total = np.linalg.norm(tub[:, -1] - tub[:, 0], axis=-1)[:, None]
        X = np.concatenate([tub.reshape(len(c), -1), d.reshape(len(c), -1),
                            nrm, dn.reshape(len(c), -1), acc, total,
                            nrm ** 2, total ** 2], 1)
        sp = get_splits(dataset, axis)
        v = get_values(dataset)
        Y = P.to_target(dataset, v)
        m = P.fit_ridge_cv(X[sp["train"]], Y[sp["train"]], X[sp["val"]], Y[sp["val"]])
        return P.evaluate(dataset, m.predict(X[sp["test"]]), Y[sp["test"]], v[sp["test"]])

    return cached("mask_baseline", cfg, run, force)


# ---------------------------------------------------------------- part 1.2

def pca_fit(X: np.ndarray, train: np.ndarray, n_pca: int):
    """PCA basis fitted on the TRAIN rows only, then applied to everything.

    Fitting the SVD on all rows would leak held-out values into the basis, which
    is exactly what the value-split protocol exists to prevent.
    """
    mu = X[train].mean(0)
    _, _, Vt = np.linalg.svd(X[train] - mu, full_matrices=False)
    P = Vt[:n_pca]
    return (X - mu) @ P.T


def nullspace_curve(dataset: str, layer: int, pooling: str = "tmean",
                    time: str = "mean", axis: str = "value", n_iter: int = 30,
                    n_pca: int | None = None, force: bool = False) -> dict:
    """INLP curve plus its random-subspace control. Part 1, step 2.

    `n_pca` runs the whole procedure inside a PCA-reduced space. In the full
    1024-d space the curve does not bend within any reasonable budget (R^2 0.992
    -> 0.886 after removing 298 dims), because the code is extremely redundant --
    so there is no dimensionality number to report. Reducing first lets the curve
    actually terminate, at the cost of the claim being about the top-`n_pca`
    subspace rather than the raw residual stream. Report both.
    """
    cfg = dict(dataset=dataset, layer=layer, pooling=pooling, time=time,
               axis=axis, n_iter=n_iter, n_pca=n_pca, v=2)

    def run():
        X, Y, v = get_xy(dataset, layer, pooling, time)
        sp = get_splits(dataset, axis)
        if n_pca:
            X = pca_fit(X, sp["train"], n_pca)
        a = (X[sp["train"]], Y[sp["train"]], X[sp["val"]], Y[sp["val"]],
             X[sp["test"]], Y[sp["test"]], v[sp["test"]], dataset)
        hist, basis = P.inlp(*a, n_iter=n_iter)
        k = int(Y.shape[1])          # probe readout rank: 2 for (sin,cos), else 1
        ctrl = P.random_inlp(*a, n_iter=n_iter, k=k)
        return {"inlp": hist, "random": ctrl, "k_per_iter": k,
                "dims_removed_total": 0 if basis is None else int(basis.shape[1]),
                "n_dims": int(X.shape[1])}

    return cached("nullspace", cfg, run, force)


# ---------------------------------------------------------------- reporting

def summarise(rows: list[dict], dataset: str) -> str:
    metric = "circ_mae_deg" if dataset == "direction" else "mae"
    best = min(rows, key=lambda r: r[metric])
    return (f"{dataset}: best layer {best['layer']} "
            f"{metric}={best[metric]:.3f} r2={best['r2']:.3f}")
