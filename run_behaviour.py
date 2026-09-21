#!/usr/bin/env python
"""Does an on-manifold edit change what the model FORECASTS, not just what it encodes?

Every other Part 2 evaluation reads a probe on a downstream encoder layer. This one
injects the edit at layer 16, propagates it through the whole encoder, and asks the
predictor to forecast the second half of the clip from the first half. The probe then
reads that forecast -- so a change means the model's own forward prediction moved.
"""
import numpy as np, torch, json
from vjp import (experiments as E, steering as S, probes as P, causal as C,
                 part2 as T, data as D, features as F, behaviour as B)

LAYER = 16


def sweep(model, idx, deltas, recs):
    out = []
    for s in range(0, len(idx), 4):
        sl = idx[s:s + 4]
        px = torch.from_numpy(np.stack([D.preprocess(D.decode(recs[i]["video"]))
                                        for i in sl])).to("mps")
        dd = None if deltas is None else torch.from_numpy(deltas[s:s + 4].astype(np.float32)).to("mps")
        out.append(B.forecast(model, px, LAYER, dd))
    return np.concatenate(out)


def main():
    X, Y, v = E.get_xy("direction", LAYER, "sal")
    sp = E.get_splits("direction", "value")
    recs = D.load_manifest("direction")
    basis, _ = S.make_probe_bank(X, Y, sp["train"], sp["val"], 16, 0)
    rng = np.random.default_rng(0)
    fit = np.sort(rng.permutation(sp["train"])[:600])
    model = C.load_model()

    # a probe on UNSTEERED forecasts: can the model's forecast be read for direction at all?
    print("fitting the forecast probe on unsteered clips ...", flush=True)
    Ftr = sweep(model, fit, None, recs)
    Ytr = P.to_target("direction", v[fit])
    fp = P.fit_ridge_cv(Ftr[150:], Ytr[150:], Ftr[:150], Ytr[:150])
    print("  forecast probe, held-in: circ MAE %.2f°" %
          P.circular_mae(fp.predict(Ftr[:150]), v[fit][:150]), flush=True)

    test = sp["test"]
    rng2 = np.random.default_rng(1)
    tgt = rng2.choice(np.unique(v[test]), size=len(test))
    W = P.to_target("direction", tgt)
    Wown = P.to_target("direction", v[test])
    man = T.fit_manifold("direction", X, v, sp["train"], 64)

    arms = {
        "control (no edit)": None,
        "multi-probe K=16": P.steer(X[test], P.probe_subspace(basis), basis,
                                    np.tile(W, (1, len(basis)))) - X[test],
        "spline (on-manifold)": man.steer(X[test], tgt) - X[test],
    }
    res = {}
    for name, d in arms.items():
        print(f"forecast: {name} ...", flush=True)
        Fh = sweep(model, test, d, recs)
        pr = fp.predict(Fh)
        res[name] = {
            "mae_to_target": P.circular_mae(pr, tgt),
            "mae_to_own": P.circular_mae(pr, v[test]),
            "align_target": float((pr * W).sum(1).mean()),
            "align_own": float((pr * Wown).sum(1).mean()),
            "certainty": float(np.linalg.norm(pr, axis=1).mean()),
        }

    print(f"\n{'arm':24s} {'MAE→target':>11s} {'MAE→own':>9s} "
          f"{'align→target':>13s} {'align→own':>10s} {'certainty':>10s}")
    print("  " + "-" * 78)
    for n, r in res.items():
        print(f"{n:24s} {r['mae_to_target']:10.2f}° {r['mae_to_own']:8.2f}° "
              f"{r['align_target']:+13.3f} {r['align_own']:+10.3f} {r['certainty']:10.3f}")
    json.dump(res, open("artifacts/results/behaviour_predictor.json", "w"), indent=1)
    print("\nsaved artifacts/results/behaviour_predictor.json")


if __name__ == "__main__":
    main()
