"""Fourier analysis of the direction ring.

The 64 directions are equally spaced, so a DFT of the per-angle centroids in theta
reads off the harmonic content of the manifold directly -- no fitting required.
"""
from __future__ import annotations
import numpy as np

from . import experiments as E
from . import probes as P
from . import features as F


def centroids(dataset: str, layer: int, pooling: str = "sal"):
    X, _, v = E.get_xy(dataset, layer, pooling)
    th = np.unique(v)
    return np.stack([X[v == t].mean(0) for t in th]), th, X, v


def spectrum(C: np.ndarray) -> np.ndarray:
    """Share of centroid variance at each harmonic m (index 0 = DC, dropped)."""
    Fh = np.fft.rfft(C - C.mean(0), axis=0)
    p = (np.abs(Fh) ** 2).sum(1)
    return p / p.sum()


def noise_floor(X, v, th, seed=0):
    """Half-split the repetitions per angle: the difference is centroid noise."""
    rng = np.random.default_rng(seed)
    A, B = [], []
    for t in th:
        idx = rng.permutation(np.flatnonzero(v == t))
        h = len(idx) // 2
        A.append(X[idx[:h]].mean(0))
        B.append(X[idx[h:2 * h]].mean(0))
    return (np.stack(A) - np.stack(B)) / 2


def harmonic_of_pc(C: np.ndarray, th: np.ndarray, n_pc: int = 10, m_max: int = 8):
    """For each centroid PC, the harmonic it best matches and that fit's R^2."""
    r = np.radians(th)
    Cc = C - C.mean(0)
    _, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    Z = Cc @ Vt.T
    evr = S ** 2 / (S ** 2).sum()
    rows = []
    for j in range(n_pc):
        y = Z[:, j] - Z[:, j].mean()
        r2 = []
        for m in range(1, m_max + 1):
            A = np.stack([np.cos(m * r), np.sin(m * r)], 1)
            b, *_ = np.linalg.lstsq(A, y, rcond=None)
            r2.append(1 - ((y - A @ b) ** 2).sum() / (y ** 2).sum())
        half = len(th) // 2
        flip = float(np.corrcoef(Z[:half, j], Z[half:, j])[0, 1])
        rows.append({"pc": j + 1, "evr": float(evr[j]), "m": int(np.argmax(r2)) + 1,
                     "r2": float(max(r2)), "r2_by_m": [float(x) for x in r2],
                     "flip_corr": flip})
    return rows, Vt
