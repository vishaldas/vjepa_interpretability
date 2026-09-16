"""Part 2: spline / manifold steering, after Wurgaft et al.

Three questions, in increasing order of how much they can tell us:

  E1  Does steering along the fitted manifold reach held-out target values as
      accurately as the multi-probe subspace method of Part 1?
  E2  Do the two methods differ in the PATH they take -- the paper's actual claim?
      (They coincide at the endpoint by construction; see splines.steer_at.)
  E3  Does an on-manifold edit survive propagation through the network better than
      a Euclidean one? F9 gives the linear baseline: 77 % retained at one block,
      15 % at four.

The manifold is fitted on TRAIN values only, so reaching a held-out value is
genuine interpolation along the curve.
"""
from __future__ import annotations
import numpy as np

from . import probes as P
from . import steering as S
from .splines import Manifold, off_manifold_energy


def fit_manifold(dataset: str, X, values, train, n_pca: int = 64) -> Manifold:
    return Manifold(n_pca=n_pca, periodic=(dataset == "direction")).fit(
        X[train], values[train])


def evaluate_spline_steering(dataset, X, Y, values, train, val, test,
                             n_pca=64, n_probes=16, seed=0, n_targets=4):
    """E1: spline steering to held-out values, read with a held-out probe.

    Uses the SAME evaluation probe and the SAME targets as the multi-probe
    baseline, so the two numbers are directly comparable.
    """
    basis, evp = S.make_probe_bank(X, Y, train, val, n_probes, seed)
    man = fit_manifold(dataset, X, values, train, n_pca)

    rng = np.random.default_rng(seed + 1)
    held = np.unique(values[test])
    src = np.concatenate([test] * n_targets)
    tgt = np.concatenate([rng.choice(held, size=len(test)) for _ in range(n_targets)])

    Xs = man.steer(X[src], tgt)
    Xm = P.steer(X[src], P.probe_subspace(basis), basis,
                 np.tile(P.to_target(dataset, tgt), (1, len(basis))))

    T = P.to_target(dataset, tgt)
    return {
        "n_pca": n_pca, "n_probes": n_probes,
        "spline": P.evaluate(dataset, evp.predict(Xs), T, tgt),
        "multiprobe": P.evaluate(dataset, evp.predict(Xm), T, tgt),
        "unsteered": P.evaluate(dataset, evp.predict(X[src]), T, tgt),
        "ceiling": P.evaluate(dataset, evp.predict(X[src]),
                              P.to_target(dataset, values[src]), values[src]),
        "explained_var": float(np.sum(man.evr_)),
    }


def path_geometry(dataset, X, values, train, n_pca=64, n_pairs=200, n_steps=32,
                  seed=0):
    """E2: how far off the manifold does each interpolation path stray?

    Endpoints are identical by construction, so any difference is purely the
    route taken -- which is the whole content of Eq. 1 vs Eq. 2.
    """
    man = fit_manifold(dataset, X, values, train, n_pca)
    rng = np.random.default_rng(seed)
    vals = man.values_
    em, el, chord = [], [], []
    for _ in range(n_pairs):
        a, b = rng.choice(vals, size=2, replace=False)
        u0, u1 = man.u_of_value(np.array([a]))[0], man.u_of_value(np.array([b]))[0]
        pm = man.path(u0, u1, n_steps, "manifold")
        pl = man.path(u0, u1, n_steps, "linear")
        em.append(off_manifold_energy(pm, man))
        el.append(off_manifold_energy(pl, man))
        chord.append(float(np.linalg.norm(pm[-1] - pm[0])))
    em, el, chord = np.array(em), np.array(el), np.array(chord)
    return {
        "n_pca": n_pca,
        "energy_manifold": float(em.mean()),
        "energy_linear": float(el.mean()),
        "ratio": float(el.mean() / max(em.mean(), 1e-12)),
        # normalised by how far the path travels, so long and short moves compare
        "energy_manifold_rel": float((em / chord).mean()),
        "energy_linear_rel": float((el / chord).mean()),
        "frac_linear_worse": float((el > em).mean()),
    }


def causal_path(dataset, layer, readout_layer, n_pca=64, n_clips=80,
                ts=(0.0, 0.25, 0.5, 0.75, 1.0), pooling="sal", seed=0,
                n_probes=16, model=None, device="mps", batch=4):
    """E3: walk the intervention along each path and see what the MODEL does.

    At t=1 the manifold and linear routes coincide by construction, so the whole
    comparison lives at intermediate t. For direction the chord between two angles
    cuts through the interior of the ring -- a region no real clip occupies -- so
    the prediction is that linear steering degrades there while manifold steering
    passes through genuine intermediate directions.

    Two things are measured downstream, after the edit has propagated:
      * accuracy  -- does the readout match the value the path is supposed to be at?
      * certainty -- for direction, ||(sin, cos)|| of the prediction. A confident
                     readout has norm ~1; a collapse toward 0 means the model has
                     no clear direction at all, which is what "off-manifold" is
                     supposed to feel like.
    """
    import torch
    from . import experiments as E
    from . import causal as C
    from . import data as D
    from . import features as F

    X, Y, v = E.get_xy(dataset, layer, pooling)
    sp = E.get_splits(dataset, "value")
    _, evp_L = S.make_probe_bank(X, Y, sp["train"], sp["val"], n_probes, seed)
    XR, YR, _ = E.get_xy(dataset, readout_layer, pooling)
    rng = np.random.default_rng(seed)
    eval_fold = np.array_split(rng.permutation(sp["train"]), n_probes + 1)[0]
    evp_R = P.fit_ridge_cv(XR[eval_fold], YR[eval_fold], XR[sp["val"]], YR[sp["val"]])

    man = fit_manifold(dataset, X, v, sp["train"], n_pca)
    test = sp["test"][:n_clips]
    # target: the held-out value furthest from each clip's own, i.e. the longest
    # chord -- where an off-manifold shortcut is most severe.
    held = np.unique(v[test])
    if dataset == "direction":
        d = np.abs((held[None] - v[test][:, None] + 180) % 360 - 180)
    else:
        d = np.abs(held[None] - v[test][:, None])
    tgt = held[d.argmax(1)]

    if model is None:
        model = C.load_model(device)
    recs = D.load_manifest(dataset)
    cents, _ = F.load_tracks(dataset)

    out = {}
    for mode in ("manifold", "linear"):
        for t in ts:
            Xs = man.steer_at(X[test], tgt, t=t, mode=mode)
            delta = (Xs - X[test]).astype(np.float32)
            reads = []
            for s in range(0, len(test), batch):
                sl = test[s:s + batch]
                vids = [D.decode(recs[i]["video"]) for i in sl]
                px = torch.from_numpy(np.stack([D.preprocess(x) for x in vids])).to(device)
                mk = torch.from_numpy(np.stack([D.object_mask(cents[i]) for i in sl])).to(device)
                dd = torch.from_numpy(delta[s:s + batch]).to(device)
                pooled = C.forward_steered(model, px, layer, dd, [readout_layer],
                                           pooling, mk, inject="all")
                reads.append(pooled[readout_layer].mean(axis=1))
            R = evp_R.predict(np.concatenate(reads))

            # the value the path is nominally at, along the manifold
            u0 = man.project(man.to_pca(X[test]))
            u1 = man.u_of_value(tgt)
            if man.periodic:
                step = (u1 - u0 + 0.5) % 1.0 - 0.5
                want = np.mod((u0 + t * step) * 360.0, 360.0)
            else:
                lo, hi = man.values_.min(), man.values_.max()
                want = (u0 + t * (u1 - u0)) * (hi - lo) + lo
            rec = {"t": t, "mode": mode,
                   "err_vs_path": P.evaluate(dataset, R, P.to_target(dataset, want), want),
                   "err_vs_target": P.evaluate(dataset, R, P.to_target(dataset, tgt), tgt)}
            if dataset == "direction":
                rec["certainty"] = float(np.linalg.norm(R, axis=1).mean())
            else:
                rec["certainty"] = float(np.std(R[:, 0]))
            out[(mode, t)] = rec
    return out
