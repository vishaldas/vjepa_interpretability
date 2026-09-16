#!/usr/bin/env python
"""Continuous clamping vs one-shot editing vs on-manifold editing.

Four arms, same clips, same targets, same held-out readout probes:
  control     no edit
  once        one Euclidean edit at L16          (F14 baseline)
  reinject    the same edit re-added at L16..L19 (accumulates)
  clamp       per-block minimum-norm correction  (the LLM feature-clamping analogue)
  spline      one on-manifold edit at L16        (F15)
"""
import numpy as np, torch
from vjp import (experiments as E, steering as S, probes as P, causal as C,
                 clamp as CL, part2 as T, data as D, features as F)

LAYER, BLOCKS, READ = 16, 4, [17, 18, 20, 24]


def main():
    X, Y, v = E.get_xy("direction", LAYER, "sal")
    sp = E.get_splits("direction", "value")
    basis, _ = S.make_probe_bank(X, Y, sp["train"], sp["val"], 16, 0)
    rng = np.random.default_rng(0)
    fit = np.sort(rng.permutation(sp["train"])[:600])

    model = C.load_model()
    recs = D.load_manifest("direction")
    cents, _ = F.load_tracks("direction")

    # probes at every layer we clamp at or read from, on unmodified activations
    need = sorted(set(list(range(LAYER, LAYER + BLOCKS)) + READ))
    probes = {}
    for L in need:
        XL, YL, _ = E.get_xy("direction", L, "sal")
        probes[L] = P.fit_ridge_cv(XL[fit][150:], YL[fit][150:],
                                   XL[fit][:150], YL[fit][:150])

    test = sp["test"]
    rng2 = np.random.default_rng(1)
    tgt = rng2.choice(np.unique(v[test]), size=len(test))
    W = P.to_target("direction", tgt)

    # a K=16 basis at EVERY layer we clamp at, so the clamp arm and the one-shot
    # arm assert the same subspace rather than K=1 against K=16
    banks = {}
    for L in range(LAYER, LAYER + BLOCKS):
        XL, YL, _ = E.get_xy("direction", L, "sal")
        banks[L], _ = S.make_probe_bank(XL, YL, sp["train"], sp["val"], 16, 0)

    man = T.fit_manifold("direction", X, v, sp["train"], 64)
    deltas = {
        "once":     P.steer(X[test], P.probe_subspace(basis), basis,
                            np.tile(W, (1, len(basis)))) - X[test],
        "reinject": None,   # same as once
        "spline":   man.steer(X[test], tgt) - X[test],
    }
    deltas["reinject"] = deltas["once"]

    arms = [("control", "none", None), ("once", "once", "once"),
            ("reinject", "reinject", "reinject"),
            ("clamp K=16", "clamp", None), ("clamp K=1", "clamp1", None),
            ("spline", "once", "spline")]

    out = {}
    for name, mode, dkey in arms:
        print(f"forward: {name} ...", flush=True)
        acc, cost = {}, 0.0
        for s in range(0, len(test), 4):
            sl = test[s:s + 4]
            px = torch.from_numpy(np.stack([D.preprocess(D.decode(recs[i]["video"]))
                                            for i in sl])).to("mps")
            mk = torch.from_numpy(np.stack([D.object_mask(cents[i]) for i in sl])).to("mps")
            d0 = None
            if dkey:
                d0 = torch.from_numpy(deltas[dkey][s:s + 4].astype(np.float32)).to("mps")
            pooled, c = CL.clamp_forward(model, px, LAYER, W[s:s + 4], probes, READ,
                                         mode=mode, n_blocks=BLOCKS, delta0=d0,
                                         masks=mk, banks=banks)
            cost += c
            for L, val in pooled.items():
                acc.setdefault(L, []).append(val)
        out[name] = ({L: np.concatenate(w) for L, w in acc.items()}, cost / len(test))

    print(f"\n{'arm':10s} " + " ".join(f"{'L'+str(L):>16s}" for L in READ) + "   inject cost")
    print("  " + "-" * (11 + 17 * len(READ) + 14))
    for name, (acc, cost) in out.items():
        cells = []
        for L in READ:
            pr = probes[L].predict(acc[L])
            cert = np.linalg.norm(pr, axis=1).mean()
            align = (pr * W).sum(1).mean()
            cells.append(f"{cert:5.2f}/{align:+5.2f}")
        print(f"{name:10s} " + " ".join(f"{c:>16s}" for c in cells) + f"   {cost:9.1f}")
    print("\n  each cell is  certainty / alignment   (1.0 / +1.0 = perfect, 0 / 0 = no information)")


if __name__ == "__main__":
    main()
