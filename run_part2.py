#!/usr/bin/env python
"""Part 2 end to end: manifold fitting, path geometry, and the causal path test.

    python run_part2.py                # E1 + E2 (cached, seconds)
    python run_part2.py --causal       # also E3 (forward passes, ~4 min/variable)
"""
import argparse

from vjp import experiments as E, part2 as T, figures as G, causal as C
from vjp.config import DATASETS

LAYER = {"direction": 16, "speed": 14, "acceleration": 14}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--n-pca", dest="n_pca", type=int, default=64)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    model = [None]

    for ds in a.datasets:
        L = LAYER[ds]
        key = G.METRIC[ds][0]
        X, Y, v = E.get_xy(ds, L, "sal")
        sp = E.get_splits(ds, "value")
        print(f"\n{'='*66}\n{ds}  (layer {L})\n{'='*66}")

        man = T.fit_manifold(ds, X, v, sp["train"], a.n_pca)
        G.plot_manifold(man, v, ds)

        # E1: endpoint accuracy, spline vs multi-probe
        cfg = dict(dataset=ds, layer=L, n_pca=a.n_pca, v=1)
        e1 = E.cached("spline_steer", cfg, lambda: T.evaluate_spline_steering(
            ds, X, Y, v, sp["train"], sp["val"], sp["test"], n_pca=a.n_pca),
            force=a.force)
        print(f"  E1 endpoint : spline {e1['spline'][key]:.3f} | "
              f"multi-probe {e1['multiprobe'][key]:.3f} | "
              f"floor {e1['unsteered'][key]:.3f} | ceiling {e1['ceiling'][key]:.3f}")

        # E2: path geometry
        e2 = E.cached("path_geom", cfg, lambda: T.path_geometry(
            ds, X, v, sp["train"], n_pca=a.n_pca, n_pairs=150), force=a.force)
        print(f"  E2 path     : manifold {e2['energy_manifold']:.3f} | "
              f"linear {e2['energy_linear']:.3f} | ratio {e2['ratio']:.0f}x | "
              f"linear worse in {100*e2['frac_linear_worse']:.0f}% of pairs")

        if a.causal:
            if model[0] is None:
                model[0] = C.load_model()
            r = T.causal_path(ds, L, L + 1, n_pca=a.n_pca, n_clips=80, model=model[0])
            G.plot_causal_path(r, ds, L, L + 1)
            mid = (r[("manifold", 0.5)]["err_vs_path"][key],
                   r[("linear", 0.5)]["err_vs_path"][key])
            print(f"  E3 mid-path : manifold {mid[0]:.3f} | linear {mid[1]:.3f} "
                  f"| penalty {mid[1]/mid[0]:.2f}x")


if __name__ == "__main__":
    main()
