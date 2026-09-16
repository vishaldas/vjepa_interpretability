#!/usr/bin/env python
"""Causal steering sweep: intervene at layer L, propagate, read downstream.

    python run_causal.py [--dataset direction] [--layer 16]
"""
import argparse
import numpy as np

from vjp import causal, experiments as E, figures as G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="direction")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    RL = [a.layer + 1, a.layer + 2, a.layer + 4, a.layer + 8]
    RL = [r for r in RL if r <= 25]
    KS = [1, 4, 16]
    model = [None]

    def run(K, mode):
        if model[0] is None:
            model[0] = causal.load_model()
        return causal.run_causal(a.dataset, a.layer, RL, K, model=model[0], inject=mode)

    res = {}
    for mode in ("all", "salient"):
        for K in KS:
            cfg = dict(dataset=a.dataset, layer=a.layer, K=K, inject=mode,
                       readout=RL, v=1)
            res[(mode, K)] = E.cached("causal", cfg, lambda K=K, m=mode: run(K, m),
                                      force=a.force)
    G.plot_causal(res, a.dataset, a.layer, RL, KS)

    key = G.METRIC[a.dataset][0]
    for (mode, K), r in res.items():
        vals = "  ".join(f"L{R}:{G.get_layer(r, R)['steered_vs_target'][key]:6.1f}" for R in RL)
        print(f"  {mode:8s} K={K:2d}  {vals}")


if __name__ == "__main__":
    main()
