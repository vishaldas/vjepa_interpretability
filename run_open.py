#!/usr/bin/env python
"""Open question: does an ON-MANIFOLD edit resist the attention rotation of F11?

F14 showed manifold steering stays accurate mid-path where linear steering collapses.
F11 showed the linear edit fails because attention rotates it out of the readout
subspace. This runs F11's instrumentation on both routes at the SAME t, so the only
difference is the geometry of the edit.
"""
import numpy as np
import torch

from vjp import (experiments as E, steering as S, probes as P, causal as C,
                 mechanism as M, part2 as T)

LAYER, BLOCKS, T_MID = 16, [16, 17, 18, 19, 20], 0.5


def main():
    X, Y, v = E.get_xy("direction", LAYER, "sal")
    sp = E.get_splits("direction", "value")
    basis, _ = S.make_probe_bank(X, Y, sp["train"], sp["val"], 16, 0)
    rng = np.random.default_rng(0)
    fit = np.sort(rng.permutation(sp["train"])[:600])

    model = C.load_model()

    # position probes, fitted once on unmodified activations
    print("fitting position probes ...", flush=True)
    _, ctrl_acts, ctrl_stats = M.diagnose(
        model, LAYER, BLOCKS, fit, np.zeros((len(fit), 1024), np.float32),
        {k: None for k in M.positions(BLOCKS)}, v[fit]) if False else (None, None, None)

    from vjp import data as D, features as F
    recs = D.load_manifest("direction")
    cents, _ = F.load_tracks("direction")

    def sweep(idx, deltas):
        acc = {}
        for s in range(0, len(idx), 4):
            sl = idx[s:s + 4]
            vids = [D.decode(recs[i]["video"]) for i in sl]
            px = torch.from_numpy(np.stack([D.preprocess(x) for x in vids])).to("mps")
            mk = torch.from_numpy(np.stack([D.object_mask(cents[i]) for i in sl])).to("mps")
            dd = None if deltas is None else torch.from_numpy(deltas[s:s + 4]).to("mps")
            with M.Recorder(model, BLOCKS, "sal") as rec:
                C.forward_steered(model, px, LAYER, dd, [LAYER], "sal", mk, inject="all")
                for k, val in rec.pooled(mk).items():
                    acc.setdefault(k, []).append(val)
        return {k: np.concatenate(w) for k, w in acc.items()}

    tr = sweep(fit, None)
    Ytr = P.to_target("direction", v[fit])
    nv = len(fit) // 4
    probes = {k: P.fit_ridge_cv(A[nv:], Ytr[nv:], A[:nv], Ytr[:nv]) for k, A in tr.items()}

    # the two routes, at the same t, to the same target
    test = sp["test"]
    man = T.fit_manifold("direction", X, v, sp["train"], 64)
    held = np.unique(v[test])
    dist = np.abs((held[None] - v[test][:, None] + 180) % 360 - 180)
    tgt = held[dist.argmax(1)]

    u0 = man.project(man.to_pca(X[test]))
    step = (man.u_of_value(tgt) - u0 + 0.5) % 1.0 - 0.5
    want = np.mod((u0 + T_MID * step) * 360.0, 360.0)   # value the path is nominally at

    deltas = {
        "manifold t=0.5": (man.steer_at(X[test], tgt, T_MID, "manifold") - X[test]),
        "linear   t=0.5": (man.steer_at(X[test], tgt, T_MID, "linear") - X[test]),
    }

    print("\ncontrol forward ...", flush=True)
    ctrl = sweep(test, None)

    out = {}
    for name, d in deltas.items():
        print(f"forward: {name} ...", flush=True)
        st = sweep(test, d.astype(np.float32))
        rows, base = [], None
        for k in M.positions(BLOCKS):
            dd = st[k] - ctrl[k]
            dn = float(np.linalg.norm(dd, axis=1).mean())
            base = dn if base is None else base
            Q, _ = np.linalg.qr(probes[k].readout)
            al = float(np.mean(np.linalg.norm(dd @ Q, axis=1)
                               / np.linalg.norm(dd, axis=1).clip(1e-9)))
            pred = probes[k].predict(st[k])
            W = P.to_target("direction", want)
            rows.append((k,
                         float(np.linalg.norm(pred, axis=1).mean()),   # certainty
                         float((pred * W).sum(1).mean()),              # alignment
                         dn / base, al, dn / base * al))
        out[name] = rows

    print("\n  certainty = mean ||pred||   alignment = mean <pred, unit(intended)>")
    print("  (circular MAE is omitted: when certainty collapses the angle is noise)\n")
    print(f"{'position':11s} | " + " | ".join(f"{n:^30s}" for n in out))
    print(f"{'':11s} | " + " | ".join(f"{'cert':>6s} {'align':>7s} {'|d|':>6s} {'sub':>6s}"
                                      for _ in out))
    print("-" * 84)
    for i, k in enumerate(M.positions(BLOCKS)):
        cells = []
        for n in out:
            _, c, a, dn, al, am = out[n][i]
            cells.append(f"{c:6.3f} {a:+7.3f} {dn:6.3f} {al:6.3f}")
        print(f"{k:11s} | " + " | ".join(cells))

    print("\nat L20.out, relative to the injection point:")
    for n, rows in out.items():
        print(f"  {n}:  perturbation magnitude {100*rows[-1][3]:5.1f}%   "
              f"readout-subspace component {100*rows[-1][5]/rows[0][5]:5.1f}%")
        print(f"  {'':16s}  alignment {rows[0][2]:+.3f} -> {rows[-1][2]:+.3f}   "
              f"certainty {rows[0][1]:.3f} -> {rows[-1][1]:.3f}")


if __name__ == "__main__":
    main()
