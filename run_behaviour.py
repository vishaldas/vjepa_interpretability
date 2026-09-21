#!/usr/bin/env python
"""Does an on-manifold edit change what the model FORECASTS, not just what it encodes?

Every other Part 2 evaluation reads a probe on a downstream encoder layer. This one
injects the edit at layer 16, propagates it through the whole encoder, and asks the
predictor to forecast the second half of the clip from the first half. The probe then
reads that forecast -- so a change means the model's own forward prediction moved.

The multi-probe edit is also run RESCALED, so that "the spline wins" cannot be a
magnitude effect: its natural norm is 2.6x the multi-probe edit's.
"""
import numpy as np, torch, json
from vjp import (experiments as E, steering as S, probes as P, causal as C,
                 part2 as T, data as D, behaviour as B)

LAYER = 16
SCALES = (1.0, 2.0, 2.61)          # 2.61 matches the spline's perturbation norm


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

    print("fitting the forecast probe on unsteered clips ...", flush=True)
    Ftr = sweep(model, fit, None, recs)
    Ytr = P.to_target("direction", v[fit])
    fp = P.fit_ridge_cv(Ftr[150:], Ytr[150:], Ftr[:150], Ytr[:150])

    test = sp["test"]
    rng2 = np.random.default_rng(1)
    tgt = rng2.choice(np.unique(v[test]), size=len(test))
    W = P.to_target("direction", tgt)
    Wown = P.to_target("direction", v[test])
    man = T.fit_manifold("direction", X, v, sp["train"], 64)

    d_mp = P.steer(X[test], P.probe_subspace(basis), basis,
                   np.tile(W, (1, len(basis)))) - X[test]
    d_sp = man.steer(X[test], tgt) - X[test]
    n_mp = float(np.linalg.norm(d_mp, axis=1).mean())
    n_sp = float(np.linalg.norm(d_sp, axis=1).mean())

    arms = [("control (no edit)", None)] + \
           [(f"multi-probe ×{s:.2f}", d_mp * s) for s in SCALES] + \
           [("spline (on-manifold)", d_sp)]

    res = {}
    for name, d in arms:
        print(f"forecast: {name} ...", flush=True)
        pr = fp.predict(sweep(model, test, d, recs))
        res[name] = {
            "norm": 0.0 if d is None else float(np.linalg.norm(d, axis=1).mean()),
            "mae_to_target": P.circular_mae(pr, tgt),
            "mae_to_own": P.circular_mae(pr, v[test]),
            "align_target": float((pr * W).sum(1).mean()),
            "align_own": float((pr * Wown).sum(1).mean()),
            "certainty": float(np.linalg.norm(pr, axis=1).mean()),
        }

    # CEILING: the same probe, on the same held-out-value test clips, reading each
    # clip's OWN label. Measured on exactly the split the steering is evaluated on --
    # unlike the alpha-selection fold, which sits on train values and is not comparable.
    floor = res["control (no edit)"]["mae_to_target"]
    ceil = res["control (no edit)"]["mae_to_own"]
    for r in res.values():
        r["gap_closed_pct"] = 100.0 * (floor - r["mae_to_target"]) / (floor - ceil)
    res["_meta"] = {"floor_deg": floor, "ceiling_deg": ceil,
                    "ceiling_source": "control arm reading its own label, same test clips",
                    "norm_multiprobe": n_mp, "norm_spline": n_sp}

    print(f"\n{'arm':24s} {'‖Δ‖':>6s} {'MAE→target':>11s} {'MAE→own':>9s} "
          f"{'align→tgt':>10s} {'gap closed':>11s}")
    print("  " + "-" * 76)
    for n, r in res.items():
        if n == "_meta": continue
        print(f"{n:24s} {r['norm']:6.1f} {r['mae_to_target']:10.2f}° "
              f"{r['mae_to_own']:8.2f}° {r['align_target']:+10.3f} {r['gap_closed_pct']:10.1f}%")
    print(f"\n  floor {floor:.2f}°  ceiling {ceil:.2f}° (control reading its own label)")
    json.dump(res, open("artifacts/results/behaviour_predictor.json", "w"), indent=1)
    print("saved artifacts/results/behaviour_predictor.json")


if __name__ == "__main__":
    main()
