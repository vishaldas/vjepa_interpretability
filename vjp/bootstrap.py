"""Bootstrap confidence tube around the fitted direction manifold.

Each bootstrap resample refits its OWN PCA, so its spline lives in its own basis
with arbitrary component signs and possible axis swaps. Comparing the curves
directly in their native coordinates would measure basis churn, not fit
uncertainty. Every bootstrap curve is therefore mapped back to the 1024-d
activation space and re-projected into a single reference basis before anything
is compared.
"""
from __future__ import annotations
import numpy as np

from .splines import Manifold


def reference(X, values, train, n_pca=64) -> Manifold:
    return Manifold(n_pca=n_pca, periodic=True).fit(X[train], values[train])


def tube(X, values, train, ref: Manifold, n_boot=100, n_grid=180, n_pca=64,
         seed=0):
    """Resample clips with replacement, refit the whole pipeline, collect curves.

    Returns
      G     [n_grid]            intrinsic coordinates the curves are sampled at
      REF   [n_grid, k]         the reference curve, in reference PCA coordinates
      B     [n_boot, n_grid, k] bootstrap curves, same coordinates
    """
    rng = np.random.default_rng(seed)
    G = np.linspace(0, 1, n_grid, endpoint=False)
    REF = ref.curve(G)
    out = []
    for b in range(n_boot):
        idx = train[rng.integers(0, len(train), len(train))]
        if len(np.unique(values[idx])) < 8:          # degenerate resample
            continue
        m = Manifold(n_pca=n_pca, periodic=True).fit(X[idx], values[idx])
        amb = m.from_pca(m.curve(G))                  # back to activation space
        out.append(ref.to_pca(amb))                   # into the reference basis
    return G, REF, np.stack(out)


def radius(REF, B, q=95.0):
    """Tube radius at each grid point: the q-th percentile bootstrap deviation."""
    d = np.linalg.norm(B - REF[None], axis=-1)        # [n_boot, n_grid]
    return np.percentile(d, q, axis=0)


def chord_profile(ref: Manifold, REF, rad, theta_a, theta_b, n_steps=41,
                  n_grid_search=2000):
    """Walk the straight chord between two angles; at each step report how far it
    sits from the manifold, and how that compares with the tube at the nearest point.

    Returns t, distance-to-manifold, tube radius there, and their ratio.
    """
    u = ref.u_of_value(np.array([theta_a, theta_b], float))
    a, b = ref.curve(u[:1])[0], ref.curve(u[1:])[0]
    g = np.linspace(0, 1, n_grid_search, endpoint=False)
    C = ref.curve(g)
    gi = np.linspace(0, 1, len(REF), endpoint=False)

    ts = np.linspace(0, 1, n_steps)
    d_man, r_at, = [], []
    for t in ts:
        p = (1 - t) * a + t * b
        dd = np.linalg.norm(C - p, axis=1)
        j = int(dd.argmin())
        d_man.append(float(dd[j]))
        r_at.append(float(np.interp(g[j], gi, rad, period=1.0)))
    d_man, r_at = np.array(d_man), np.array(r_at)
    return ts, d_man, r_at, d_man / np.maximum(r_at, 1e-9)


def data_cloud_test(X, values, train, ref: Manifold, theta_a, theta_b, seed=0):
    """The stronger question: is the chord midpoint inside the region real clips occupy?

    The manifold is a curve through centroids, but individual clips scatter widely
    around it. A point can be far from the centroid curve and still sit comfortably
    inside the data cloud -- so distance to the curve alone does not establish that
    a state is unreachable. This compares the midpoint's nearest-neighbour distance
    against the nearest-neighbour distances among real clips.
    """
    Z = ref.to_pca(X[train])
    u = ref.u_of_value(np.array([theta_a, theta_b], float))
    a, b = ref.curve(u[:1])[0], ref.curve(u[1:])[0]
    mid = 0.5 * (a + b)

    d_mid = np.linalg.norm(Z - mid, axis=1)
    # nearest-neighbour distance among real clips, as the yardstick
    rng = np.random.default_rng(seed)
    sub = Z[rng.choice(len(Z), size=min(400, len(Z)), replace=False)]
    dd = np.linalg.norm(sub[:, None] - Z[None], axis=-1)
    np.fill_diagonal(dd[:, :len(sub)], np.inf)
    nn_real = dd.min(1)
    return {
        "mid_nn": float(d_mid.min()),
        "real_nn_median": float(np.median(nn_real)),
        "real_nn_p95": float(np.percentile(nn_real, 95)),
        "real_nn_max": float(nn_real.max()),
        "pct_clips_closer_to_mid_than_their_own_nn":
            float((d_mid.min() < nn_real).mean() * 100),
    }
