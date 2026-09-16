#!/usr/bin/env python
"""Apples-to-apples: multi-probe (Part 1) vs spline (Part 2) edits to the SAME target,
propagated through the same blocks, scored with the same non-degenerate metric."""
import numpy as np, torch
from vjp import (experiments as E, steering as S, probes as P, causal as C,
                 mechanism as M, part2 as T, data as D, features as F)

LAYER, BLOCKS = 16, [16, 17, 18, 19, 20]

X, Y, v = E.get_xy("direction", LAYER, "sal")
sp = E.get_splits("direction", "value")
basis, _ = S.make_probe_bank(X, Y, sp["train"], sp["val"], 16, 0)
rng = np.random.default_rng(0)
fit = np.sort(rng.permutation(sp["train"])[:600])
model = C.load_model()
recs = D.load_manifest("direction"); cents, _ = F.load_tracks("direction")

def sweep(idx, deltas):
    acc = {}
    for s in range(0, len(idx), 4):
        sl = idx[s:s+4]
        px = torch.from_numpy(np.stack([D.preprocess(D.decode(recs[i]["video"])) for i in sl])).to("mps")
        mk = torch.from_numpy(np.stack([D.object_mask(cents[i]) for i in sl])).to("mps")
        dd = None if deltas is None else torch.from_numpy(deltas[s:s+4]).to("mps")
        with M.Recorder(model, BLOCKS, "sal") as rec:
            C.forward_steered(model, px, LAYER, dd, [LAYER], "sal", mk, inject="all")
            for k, val in rec.pooled(mk).items():
                acc.setdefault(k, []).append(val)
    return {k: np.concatenate(w) for k, w in acc.items()}

print("fitting position probes ...", flush=True)
tr = sweep(fit, None); Ytr = P.to_target("direction", v[fit])
probes = {k: P.fit_ridge_cv(A[150:], Ytr[150:], A[:150], Ytr[:150]) for k, A in tr.items()}

test = sp["test"]
rng2 = np.random.default_rng(1)
tgt = rng2.choice(np.unique(v[test]), size=len(test))
W = P.to_target("direction", tgt)
man = T.fit_manifold("direction", X, v, sp["train"], 64)

deltas = {
    "multi-probe (K=16)": P.steer(X[test], P.probe_subspace(basis), basis,
                                  np.tile(W, (1, len(basis)))) - X[test],
    "spline (manifold)": man.steer(X[test], tgt) - X[test],
}
print("control ...", flush=True); ctrl = sweep(test, None)
out = {}
for n, d in deltas.items():
    print(f"forward: {n} ...", flush=True)
    st = sweep(test, d.astype(np.float32)); rows, base = [], None
    for k in M.positions(BLOCKS):
        dd = st[k] - ctrl[k]; dn = float(np.linalg.norm(dd, axis=1).mean())
        base = dn if base is None else base
        Q, _ = np.linalg.qr(probes[k].readout)
        sub = float(np.mean(np.linalg.norm(dd @ Q, axis=1) / np.linalg.norm(dd, axis=1).clip(1e-9)))
        pred = probes[k].predict(st[k])
        rows.append((k, float(np.linalg.norm(pred, axis=1).mean()),
                     float((pred*W).sum(1).mean()), dn/base, sub))
    out[n] = rows

print(f"\n{'position':11s} | " + " | ".join(f"{n:^30s}" for n in out))
print(f"{'':11s} | " + " | ".join(f"{'cert':>6s} {'align':>7s} {'|d|':>6s} {'sub':>6s}" for _ in out))
print("-"*84)
for i, k in enumerate(M.positions(BLOCKS)):
    print(f"{k:11s} | " + " | ".join(
        f"{out[n][i][1]:6.3f} {out[n][i][2]:+7.3f} {out[n][i][3]:6.3f} {out[n][i][4]:6.3f}" for n in out))
print("\nretention of the intended readout over 4 blocks (L16.in -> L20.out):")
for n, r in out.items():
    print(f"  {n:22s} alignment {r[0][2]:+.3f} -> {r[-1][2]:+.3f}  ({100*r[-1][2]/r[0][2]:5.1f}% retained)"
          f"   |delta| {100*r[-1][3]:.0f}%")
