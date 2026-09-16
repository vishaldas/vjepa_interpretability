"""Linear probes, circular targets, metrics, INLP, and subspace steering."""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------- targets

def to_target(dataset: str, values: np.ndarray) -> np.ndarray:
    """Probe target. Direction is circular -> (sin, cos); others are scalar."""
    if dataset == "direction":
        th = np.radians(values)
        return np.stack([np.sin(th), np.cos(th)], 1)
    return values[:, None].astype(np.float64)


def circular_mae(pred_sc: np.ndarray, true_deg: np.ndarray) -> float:
    """Mean absolute angular error in degrees, from predicted (sin, cos) pairs."""
    pred = np.degrees(np.arctan2(pred_sc[:, 0], pred_sc[:, 1]))
    d = np.abs((pred - true_deg + 180.0) % 360.0 - 180.0)
    return float(d.mean())


# ---------------------------------------------------------------- ridge

class Ridge:
    """Ridge regression with feature standardisation, solved in closed form.

    Fits on the covariance, so repeated alphas over the same X are cheap --
    which is what makes the layer sweeps and INLP iterations fast.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "Ridge":
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0) + 1e-6
        Z = (X - self.mu_) / self.sd_
        self.ym_ = Y.mean(0)
        Yc = Y - self.ym_
        G = Z.T @ Z
        G.flat[:: G.shape[0] + 1] += self.alpha * len(Z)
        self.W_ = np.linalg.solve(G, Z.T @ Yc)          # [D, K]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu_) / self.sd_) @ self.W_ + self.ym_

    @property
    def readout(self) -> np.ndarray:
        """Probe directions in the ORIGINAL feature space, [D, K].

        The probe acts on standardised features, so the direction that actually
        matters in raw activation space is W / sd -- using W directly would make
        the nullspace projection remove the wrong subspace.

        Features whose variance has already been projected away are zeroed. Their
        sd sits on the 1e-6 floor, so W/sd would explode along directions that are
        already dead, and INLP would then "remove" the same exhausted subspace
        over and over while the rank never drops.
        """
        live = self.sd_ > 1e-3 * self.sd_.max()
        R = np.zeros_like(self.W_)
        R[live] = self.W_[live] / self.sd_[live, None]
        return R


def fit_ridge_cv(Xtr, Ytr, Xva, Yva, alphas=(1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4)):
    """Pick alpha on the validation split; refit is implicit (closed form)."""
    best, best_mse = None, np.inf
    for a in alphas:
        m = Ridge(a).fit(Xtr, Ytr)
        mse = float(((m.predict(Xva) - Yva) ** 2).mean())
        if mse < best_mse:
            best, best_mse = m, mse
    return best


# ---------------------------------------------------------------- metrics

def r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def evaluate(dataset: str, pred: np.ndarray, Y: np.ndarray,
             values: np.ndarray) -> dict:
    """Headline metrics, plus the same metric stratified by label decile.

    The stratification is not decoration: the bottom of the speed range moves
    <0.05 px/frame, which is below the encoder's spatiotemporal resolution, so
    aggregate numbers hide where the representation actually runs out.
    """
    out = {"r2": r2(pred, Y)}
    if dataset == "direction":
        out["circ_mae_deg"] = circular_mae(pred, values)
        err = np.abs((np.degrees(np.arctan2(pred[:, 0], pred[:, 1])) - values + 180) % 360 - 180)
    else:
        out["mae"] = float(np.abs(pred[:, 0] - values).mean())
        err = np.abs(pred[:, 0] - values)
    edges = np.quantile(values, np.linspace(0, 1, 11))
    bins = np.clip(np.digitize(values, edges[1:-1]), 0, 9)
    out["decile_err"] = [float(err[bins == b].mean()) if (bins == b).any() else None
                         for b in range(10)]
    out["decile_centre"] = [float(values[bins == b].mean()) if (bins == b).any() else None
                            for b in range(10)]
    return out


# ---------------------------------------------------------------- INLP

def _deflate(R: np.ndarray, B: np.ndarray | None, tol: float = 1e-8):
    """Orthonormalise R against the already-removed basis B; drop spent columns.

    Returns None once the probe direction lies entirely inside B, which is the
    honest signal that the subspace is exhausted.
    """
    if B is not None and B.shape[1]:
        R = R - B @ (B.T @ R)
    keep = np.linalg.norm(R, axis=0) > tol * max(1.0, np.linalg.norm(R))
    if not keep.any():
        return None
    Q, _ = np.linalg.qr(R[:, keep])
    if B is not None and B.shape[1]:
        Q = Q - B @ (B.T @ Q)
        n = np.linalg.norm(Q, axis=0)
        if (n < tol).all():
            return None
        Q = Q[:, n > tol] / n[n > tol]
    return Q


def inlp(Xtr, Ytr, Xva, Yva, Xte, Yte, values_te, dataset, n_iter=40,
         alphas=(1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4)):
    """Iterative nullspace projection.

    Each round: fit a probe, orthonormalise its readout AGAINST EVERYTHING ALREADY
    REMOVED, project that subspace out of all splits, refit. The curve of test
    metric vs dimensions removed measures how many independent directions carry
    the variable -- i.e. its redundancy.

    Stops early when the probe direction lies inside the already-removed span, so
    `dims_removed` is always a true count of independent directions.
    """
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    hist, B = [], None
    for it in range(n_iter):
        m = fit_ridge_cv(Xtr, Ytr, Xva, Yva, alphas)
        rec = evaluate(dataset, m.predict(Xte), Yte, values_te)
        rec["iter"] = it
        rec["dims_removed"] = 0 if B is None else int(B.shape[1])
        hist.append(rec)
        Q = _deflate(m.readout, B)
        if Q is None:
            break
        B = Q if B is None else np.concatenate([B, Q], 1)
        for X in (Xtr, Xva, Xte):
            X -= (X @ Q) @ Q.T
    return hist, B


def random_inlp(Xtr, Ytr, Xva, Yva, Xte, Yte, values_te, dataset, n_iter=40,
                k=2, seed=12345, alphas=(1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4)):
    """Control for `inlp`: remove RANDOM k-dim subspaces instead of probe readouts.

    Without this the INLP curve is uninterpretable -- removing any directions
    eventually degrades a probe, so the claim has to be that removing *probe*
    directions degrades it markedly faster. Uses the same deflation bookkeeping so
    the two curves are plotted against an identical x-axis.
    """
    rng = np.random.default_rng(seed)
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    hist, B = [], None
    for it in range(n_iter):
        m = fit_ridge_cv(Xtr, Ytr, Xva, Yva, alphas)
        rec = evaluate(dataset, m.predict(Xte), Yte, values_te)
        rec["iter"] = it
        rec["dims_removed"] = 0 if B is None else int(B.shape[1])
        hist.append(rec)
        Q = _deflate(rng.standard_normal((Xtr.shape[1], k)), B)
        if Q is None:
            break
        B = Q if B is None else np.concatenate([B, Q], 1)
        for X in (Xtr, Xva, Xte):
            X -= (X @ Q) @ Q.T
    return hist


# ---------------------------------------------------------------- steering

def probe_subspace(probes: list[Ridge]) -> np.ndarray:
    """Orthonormal basis spanning K probe readouts: QR([W_1 ... W_K]) -> [D, <=2K]."""
    W = np.concatenate([p.readout for p in probes], 1)
    Q, _ = np.linalg.qr(W)
    return Q


def probe_intercept(p: Ridge) -> np.ndarray:
    """Effective intercept b such that p.predict(x) == x @ p.readout + b.

    A probe standardises before it reads, so its prediction is
    (x - mu) @ readout + ym, i.e. the intercept is ym - mu @ readout. Using ym
    alone drops the per-probe mean term and biases every steering solve -- the
    basis probes then miss their own target, badly for variables whose values are
    far from zero (speed 0.25-4.0) and less visibly for near-zero-mean targets
    like (sin, cos).
    """
    return p.ym_ - p.mu_ @ p.readout


def steer(X: np.ndarray, V: np.ndarray, probes: list[Ridge],
          target: np.ndarray) -> np.ndarray:
    """Multi-probe subspace steering (Part 1.3).

    Solves   min ||dx||_2^2   s.t.   W (x + dx) + b = target,
    where W stacks the K probe readouts (M = 2K rows for direction, K for a
    scalar). Equivalently dx = W^+ (target - (W x + b)), the Moore-Penrose
    minimum-norm solution -- verified to ~3e-14 against a direct pinv at every K
    (see FINDINGS.md F8).

    Implemented the paper's way rather than via an explicit pseudo-inverse:
    decompose x = V V^T x + x_perp, least-squares-solve for the coordinates c*
    that make every probe read `target`, and recompose x* = V c* + x_perp. The two
    agree because the stacked readouts are full column rank, so the constraint has
    a unique solution inside span(V), and the minimum-norm solution of W dx = r
    lies in span(W^T) = span(V) as well.

    Note the norm being minimised is Euclidean in the ambient activation space,
    which is blind to where the data actually lies -- this is precisely why the
    intervention takes an off-manifold shortcut (F11, F12, F15).
    """
    coef = np.concatenate([p.readout for p in probes], 1)        # [D, M]
    b = np.concatenate([probe_intercept(p) for p in probes])     # [M]
    A = V.T @ coef                                               # [r, M]
    Xperp = X - (X @ V) @ V.T
    rhs = (target - b) - Xperp @ coef                            # [N, M]
    # solve A^T c = rhs^T for each row
    c, *_ = np.linalg.lstsq(A.T, rhs.T, rcond=None)              # [r, N]
    return Xperp + c.T @ V.T
