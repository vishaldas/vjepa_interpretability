#!/usr/bin/env python
"""Part 1 end to end, off the cache. Every step memoises, so re-runs are seconds.

    python run_part1.py                 # all three variables
    python run_part1.py --datasets speed
    python run_part1.py --force         # recompute, ignoring cached results
"""
import argparse
import numpy as np

from vjp import experiments as E, figures as G, steering as S
from vjp.config import DATASETS, POOLINGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--axis", default="value",
                    help="held-out axis: value (headline) | clip | extrap")
    ap.add_argument("--n-iter", dest="n_iter", type=int, default=200,
                    help="INLP rounds; full-1024 space needs ~200 to exhaust")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for ds in args.datasets:
        print(f"\n{'='*66}\n{ds}\n{'='*66}")

        # --- 1.1 layerwise probing, under every pooling -------------------
        sweeps = {p: E.layer_sweep(ds, pooling=p, axis=args.axis, force=args.force)
                  for p in POOLINGS}
        base = E.mask_baseline(ds, axis=args.axis, force=args.force)
        key = G.METRIC[ds][0]
        # Doubles as an ORACLE CEILING: what a perfect tracker recovers. The gap
        # between it and the best encoder probe is what the representation loses.
        print(f"  tracker-only oracle (no encoder features): "
              f"{key}={base[key]:.3f} r2={base['r2']:.3f}")
        for p, rows in sweeps.items():
            print("  " + E.summarise(rows, ds) + f"   [pooling={p}]")
        G.plot_layer_sweep(sweeps, ds, baseline=base)

        # pick the layer from the leak-free pooling, never from the oracle
        rows = sweeps["sal"]
        best = min(rows, key=lambda r: r[key])["layer"]
        print(f"  -> selected layer {best} (from leak-free 'sal' pooling)")
        # Deciles need every label value present in the test set, which the value
        # axis deliberately prevents (only ~7 held-out values). Use the clip axis
        # for the resolution-floor analysis and say so on the slide.
        rows_clip = E.layer_sweep(ds, pooling="sal", axis="clip", force=args.force)
        G.plot_decile(rows_clip, ds, best)

        # --- 1.2 iterative nullspace probing ------------------------------
        ns = E.nullspace_curve(ds, best, pooling="sal", axis=args.axis,
                               n_iter=args.n_iter, force=args.force)
        G.plot_nullspace(ns, ds, best)
        first = ns["inlp"][0]["r2"]
        floor = [r["dims_removed"] for r in ns["inlp"] if r["r2"] < 0.1 * first]
        ctrl_end = ns["random"][-1]["r2"]
        print(f"  nullspace: R2 {first:.3f} -> below 10% after "
              f"{floor[0] if floor else '>'+str(ns['inlp'][-1]['dims_removed'])} dims "
              f"(random control ends at R2={ctrl_end:.3f})")

        # --- 1.3 multi-probe subspace steering ----------------------------
        X, Y, vals = E.get_xy(ds, best, "sal")
        sp = E.get_splits(ds, args.axis)
        KS = [1, 2, 4, 8, 16, 32]
        st = E.cached("steering",
                      dict(dataset=ds, layer=best, pooling="sal", Ks=KS, v=1),
                      lambda: [S.evaluate_steering(ds, X, Y, vals, sp["train"],
                                                   sp["val"], sp["test"], K, n_targets=4)
                               for K in KS], force=args.force)
        G.plot_steering(st, ds, best)
        lo, hi = st[0], min(st, key=lambda r: r["steered"][key])
        print(f"  steering: K=1 {lo['steered'][key]:.3f} -> K={hi['n_probes']} "
              f"{hi['steered'][key]:.3f}  (floor {lo['unsteered_vs_target'][key]:.3f}, "
              f"ceiling {hi['readout_ceiling'][key]:.3f})")


if __name__ == "__main__":
    main()
