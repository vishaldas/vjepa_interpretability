#!/usr/bin/env python
"""Where does the F9 decay happen? Half-block readouts + LayerNorm statistics."""
import argparse
import numpy as np
import torch

from vjp import (experiments as E, steering as S, probes as P, causal as C,
                 mechanism as M, data as D, features as F)

BLOCKS = [16, 17, 18, 19, 20]


def collect(model, idx, layer, deltas, blocks, pooling="sal", batch=4, device="mps"):
    """Forward `idx` clips with (or without) an injected delta; return pooled acts."""
    recs = D.load_manifest("direction")
    cents, _ = F.load_tracks("direction")
    acc, stats = {}, []
    for s in range(0, len(idx), batch):
        sl = idx[s:s + batch]
        vids = [D.decode(recs[i]["video"]) for i in sl]
        px = torch.from_numpy(np.stack([D.preprocess(x) for x in vids])).to(device)
        mk = torch.from_numpy(np.stack([D.object_mask(cents[i]) for i in sl])).to(device)
        dd = None if deltas is None else torch.from_numpy(deltas[s:s + batch]).to(device)
        with M.Recorder(model, blocks, pooling) as rec:
            C.forward_steered(model, px, layer, dd, [layer], pooling, mk, inject="all")
            pooled = rec.pooled(mk)
            raw = rec.raw()
            stats.append({k: M.token_stats(v) for k, v in raw.items()})
        for k, val in pooled.items():
            acc.setdefault(k, []).append(val)
    return {k: np.concatenate(v) for k, v in acc.items()}, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=500)
    a = ap.parse_args()

    X, Y, v = E.get_xy("direction", a.layer, "sal")
    sp = E.get_splits("direction", "value")
    basis, _ = S.make_probe_bank(X, Y, sp["train"], sp["val"], 16, 0)
    # Probes at half-block positions need their own fitting clips. They must be
    # disjoint from the K basis probes that built the steering subspace, but there
    # is no reason to restrict them to one 1/17 fold -- use the whole eval side.
    rng = np.random.default_rng(0)
    perm = rng.permutation(sp["train"])
    eval_fold = np.sort(perm[:a.n_train])

    model = C.load_model()
    pos = M.positions(BLOCKS)

    # 1. probes at every half-block position, fitted on UNMODIFIED activations
    print(f"fitting position probes on {len(eval_fold)} clips ...", flush=True)
    tr_acts, _ = collect(model, eval_fold, a.layer, None, BLOCKS)
    Ytr = P.to_target("direction", v[eval_fold])
    n_val = len(eval_fold) // 4
    probes = {k: P.fit_ridge_cv(A[n_val:], Ytr[n_val:], A[:n_val], Ytr[:n_val])
              for k, A in tr_acts.items()}

    # 2. steered vs control on held-out clips
    test = sp["test"]
    rng2 = np.random.default_rng(1)
    tgt = rng2.choice(np.unique(v[test]), size=len(test))
    Xs, _ = S.steer_to(X[test], basis, tgt, "direction")
    deltas = (Xs - X[test]).astype(np.float32)

    print("forward: steered ...", flush=True)
    st_acts, st_stats = collect(model, test, a.layer, deltas, BLOCKS)
    print("forward: control ...", flush=True)
    ct_acts, ct_stats = collect(model, test, a.layer, None, BLOCKS)

    T = P.to_target("direction", tgt)
    print(f"\n{'position':12s} {'steered':>8s} {'ctrl':>7s} | {'|delta|':>8s} "
          f"{'aligned':>8s} | {'LN l2':>7s} {'LN std':>7s}")
    print("  " + "-" * 68)
    base_d = None
    rows = []
    for k in pos:
        st, ct = st_acts[k], ct_acts[k]
        e_st = P.circular_mae(probes[k].predict(st), tgt)
        e_ct = P.circular_mae(probes[k].predict(ct), tgt)
        d = st - ct
        dnorm = float(np.linalg.norm(d, axis=1).mean())
        if base_d is None:
            base_d = dnorm
        # Is the surviving perturbation still pointing along the direction-readout
        # subspace at this position? This separates "the edit was removed" from
        # "the edit survives but no longer means direction".
        R = probes[k].readout                                # [D, 2]
        Q, _ = np.linalg.qr(R)
        proj = np.linalg.norm(d @ Q, axis=1) / np.linalg.norm(d, axis=1).clip(1e-9)
        l2r = np.mean([s[k]["l2"] for s in st_stats]) / np.mean([s[k]["l2"] for s in ct_stats])
        sdr = (np.mean([s[k]["chan_std"] for s in st_stats])
               / np.mean([s[k]["chan_std"] for s in ct_stats]))
        rows.append((k, e_st, dnorm / base_d, float(proj.mean())))
        print(f"{k:12s} {e_st:7.1f}° {e_ct:6.1f}° | {dnorm/base_d:8.3f} "
              f"{proj.mean():8.3f} | {l2r:7.4f} {sdr:7.4f}")

    print("\n  attribution of the readout decay (deg added by each half-block):")
    att = mlp = 0.0
    for i in BLOCKS:
        a_in = dict((r[0], r[1]) for r in rows)
        da = a_in[f"L{i}.attn"] - a_in[f"L{i}.in"]
        dm = a_in[f"L{i}.out"] - a_in[f"L{i}.attn"]
        att += da; mlp += dm
        print(f"    block {i}:  attention {da:+6.1f}°   MLP {dm:+6.1f}°")
    print(f"    TOTAL:     attention {att:+6.1f}°   MLP {mlp:+6.1f}°   "
          f"-> attention accounts for {100*att/(att+mlp):.0f}%")


if __name__ == "__main__":
    main()
