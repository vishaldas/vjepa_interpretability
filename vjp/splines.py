"""Manifold (spline) fitting and steering -- Part 2, after Wurgaft et al.

Pipeline mirrors the paper: reduce activations with PCA, average within each
label value to get concept centroids, fit a spline through the centroids in PCA
space, then steer by moving ALONG the spline rather than along a straight line.

The direction variable needs a PERIODIC spline: its centroids close a loop, and
an open spline would tear the circle at theta=0.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline


class Manifold:
    """A 1-D curve fitted through per-value centroids in PCA space."""

    def __init__(self, n_pca: int = 64, periodic: bool = False, smooth: float = 0.0):
        self.n_pca = n_pca
        self.periodic = periodic
        self.smooth = smooth

    # ---------------------------------------------------------------- fit
    def fit(self, X: np.ndarray, values: np.ndarray) -> "Manifold":
        """X [N,D] activations, `values` their labels. TRAIN values only.

        PCA is fitted here, on the rows passed in, so the caller must pass only
        training clips -- otherwise held-out values leak into the basis.
        """
        self.mu_ = X.mean(0)
        Xc = X - self.mu_
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(self.n_pca, Vt.shape[0])
        self.P_ = Vt[:k]                                  # [k, D]
        self.evr_ = (S[:k] ** 2) / (S ** 2).sum()
        Z = Xc @ self.P_.T                                # [N, k]

        self.values_ = np.unique(values)
        self.centroids_ = np.stack([Z[values == v].mean(0) for v in self.values_])

        u = self._param()
        C = self.centroids_
        if self.periodic:
            # CubicSpline's periodic boundary condition requires the curve to
            # close explicitly: repeat the first centroid one full period on.
            # Without this the period is inferred as u[-1] - u[0] (0.984 for
            # theta sampled at 5.625 deg steps), which tears the loop at 0/360.
            u = np.append(u, u[0] + 1.0)
            C = np.vstack([C, C[:1]])
            self.spline_ = CubicSpline(u, C, axis=0, bc_type="periodic")
        else:
            self.spline_ = CubicSpline(u, C, axis=0, bc_type="natural")
        self.u_ = u
        return self

    def _param(self) -> np.ndarray:
        """Intrinsic coordinate for each centroid, normalised so a period is 1."""
        v = self.values_
        if self.periodic:
            return v / 360.0                             # theta in degrees
        return (v - v.min()) / (v.max() - v.min())

    # ---------------------------------------------------------------- map
    def to_pca(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu_) @ self.P_.T

    def from_pca(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.P_ + self.mu_

    def curve(self, u: np.ndarray) -> np.ndarray:
        """Point(s) on the manifold at intrinsic coordinate u -> [M, k] in PCA space."""
        u = np.asarray(u, float)
        if self.periodic:
            u = np.mod(u, 1.0)
        else:
            u = np.clip(u, 0.0, 1.0)
        return np.atleast_2d(self.spline_(u))

    def u_of_value(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, float)
        if self.periodic:
            return np.mod(v / 360.0, 1.0)
        lo, hi = self.values_.min(), self.values_.max()
        return (v - lo) / (hi - lo)

    def project(self, Z: np.ndarray, n_grid: int = 2000) -> np.ndarray:
        """Nearest intrinsic coordinate on the curve for each point (the s^-1 of Eq. 2)."""
        g = np.linspace(0, 1, n_grid, endpoint=not self.periodic)
        C = self.curve(g)                                  # [n_grid, k]
        d = ((Z[:, None, :] - C[None]) ** 2).sum(-1)
        return g[d.argmin(1)]

    # ---------------------------------------------------------------- steer
    def steer(self, X: np.ndarray, target_values: np.ndarray) -> np.ndarray:
        """Manifold steering: move each activation along the curve to the target value.

        Follows Eq. 2 -- map to intrinsic coordinates, move there, map back --
        and preserves each clip's off-manifold residual so that only the
        manifold-aligned component changes.
        """
        return self.steer_at(X, target_values, t=1.0, mode="manifold")

    def steer_at(self, X: np.ndarray, target_values: np.ndarray, t: float = 1.0,
                 mode: str = "manifold") -> np.ndarray:
        """Partially-applied steering: move a fraction `t` of the way to the target.

        mode='manifold' travels ALONG the curve (Eq. 2); mode='linear' travels
        along the straight chord between the two centroids in activation space
        (Eq. 1), which for a curved manifold leaves the manifold in between.

        At t=1 the two modes coincide by construction -- the endpoint is the same
        point either way. The whole Goodfire comparison lives at INTERMEDIATE t,
        so an endpoint-only evaluation would find no difference between the
        methods. Sweep t and measure off-manifold energy / readout naturalness.
        """
        Z = self.to_pca(X)
        u0 = self.project(Z)
        resid = Z - self.curve(u0)
        u1 = self.u_of_value(target_values)
        if mode == "manifold":
            if self.periodic:                              # take the short way round
                d = (u1 - u0 + 0.5) % 1.0 - 0.5
                Zt = self.curve(u0 + t * d)
            else:
                Zt = self.curve(u0 + t * (u1 - u0))
        elif mode == "linear":
            Zt = (1 - t) * self.curve(u0) + t * self.curve(u1)
        else:
            raise ValueError(f"unknown mode={mode!r}")
        return self.from_pca(Zt + resid)

    def path(self, u0: float, u1: float, n: int = 32, mode: str = "manifold") -> np.ndarray:
        """The interpolating PATH between two values, in PCA space [n, k].

        mode='manifold' follows the curve; mode='linear' cuts straight through.
        Comparing these two paths is the core Goodfire experiment.
        """
        t = np.linspace(0, 1, n)
        if mode == "manifold":
            if self.periodic:                              # take the short way round
                d = (u1 - u0 + 0.5) % 1.0 - 0.5
                return self.curve(u0 + t * d)
            return self.curve(u0 + t * (u1 - u0))
        a, b = self.curve(np.array([u0]))[0], self.curve(np.array([u1]))[0]
        return (1 - t)[:, None] * a + t[:, None] * b


def off_manifold_energy(path: np.ndarray, man: "Manifold", n_grid: int = 2000) -> float:
    """Mean distance from a path to the manifold -- the paper's E_BC, discretised."""
    g = np.linspace(0, 1, n_grid, endpoint=not man.periodic)
    C = man.curve(g)
    d = np.sqrt(((path[:, None, :] - C[None]) ** 2).sum(-1)).min(1)
    return float(d.mean())
