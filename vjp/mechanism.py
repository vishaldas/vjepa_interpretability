"""Mechanistic diagnosis of the F9 decay: why does the network undo a Euclidean edit?

VJEPA2Layer is a standard pre-norm block:

    residual = h ;  h = norm1(h) ; h = attention(h) + residual      # post-attn
    residual = h ;  h = norm2(h) ; h = mlp(h)       + residual      # post-mlp (= block out)

so we can read the residual stream at half-block resolution:
  `L{i}.in`   block input          (pre-hook on layer i)
  `L{i}.n1`   after norm1          (forward hook on norm1)
  `L{i}.attn` post-attention residual (pre-hook on norm2)
  `L{i}.n2`   after norm2          (forward hook on norm2)
  `L{i}.out`  post-MLP residual    (forward hook on layer i)

Two hypotheses this separates:
  H1  LayerNorm rescaling. LN subtracts each token's mean ACROSS CHANNELS, so any
      component of the injected delta along the all-ones direction is deleted
      outright; it then divides by the token's std, which the delta itself has
      inflated, shrinking the surviving signal.
  H2  Attention dilution vs MLP overwrite. If the readout collapses across the
      attention half, unmodified context tokens are averaging the edit away; if it
      collapses across the MLP half, the feed-forward map is acting as an
      error-correcting projection back onto the learned distribution.
"""
from __future__ import annotations
import numpy as np
import torch

from .config import N_T, N_SPATIAL, HIDDEN
from . import features as F


def positions(blocks: list[int]) -> list[str]:
    out = []
    for i in blocks:
        out += [f"L{i}.in", f"L{i}.attn", f"L{i}.out"]
    return out


class Recorder:
    """Captures pooled activations and token statistics at half-block resolution."""

    def __init__(self, model, blocks: list[int], pooling: str = "sal"):
        self.model, self.blocks, self.pooling = model, blocks, pooling
        self.handles, self.buf = [], {}

    def _save(self, key):
        def fn(t):
            self.buf[key] = t.detach()
        return fn

    def __enter__(self):
        m = self.model
        for i in self.blocks:
            L = m.encoder.layer[i]
            self.handles += [
                L.register_forward_pre_hook(
                    lambda mod, inp, k=f"L{i}.in": self._save(k)(inp[0])),
                L.norm1.register_forward_hook(
                    lambda mod, inp, out, k=f"L{i}.n1": self._save(k)(out)),
                L.norm2.register_forward_pre_hook(
                    lambda mod, inp, k=f"L{i}.attn": self._save(k)(inp[0])),
                L.norm2.register_forward_hook(
                    lambda mod, inp, out, k=f"L{i}.n2": self._save(k)(out)),
                L.register_forward_hook(
                    lambda mod, inp, out, k=f"L{i}.out": self._save(k)(
                        out[0] if isinstance(out, tuple) else out)),
            ]
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []

    def pooled(self, masks=None) -> dict[str, np.ndarray]:
        """Pooled [B, 8, 1024] per position, averaged over time -> [B, 1024]."""
        return {k: F.POOL_FNS[self.pooling](v, masks).float().mean(1).cpu().numpy()
                for k, v in self.buf.items()}

    def raw(self) -> dict[str, torch.Tensor]:
        return dict(self.buf)


def token_stats(t: torch.Tensor) -> dict:
    """Per-token statistics that LayerNorm acts on."""
    return {
        "l2": float(t.norm(dim=-1).mean()),
        "chan_std": float(t.std(dim=-1).mean()),       # what LN divides by
        "chan_mean": float(t.mean(dim=-1).abs().mean()),  # what LN subtracts
    }


def delta_decomposition(delta: torch.Tensor) -> dict:
    """How much of the injected edit is LayerNorm-visible?

    LN removes the per-token channel mean, so the component of delta along the
    all-ones direction is annihilated by the very first norm it meets.
    """
    d = delta.float()
    ones = torch.ones(d.shape[-1], device=d.device) / d.shape[-1] ** 0.5
    along = (d @ ones).abs()
    total = d.norm(dim=-1)
    return {
        "frac_along_ones": float((along / total.clamp_min(1e-9)).mean()),
        "frac_energy_removed_by_mean_subtraction":
            float(((along ** 2) / (total ** 2).clamp_min(1e-9)).mean()),
    }


def diagnose(model, layer, blocks, clips, deltas, probes, target_values,
             control_acts=None, control_stats=None, pooling="sal", batch=4,
             device="mps"):
    """Per-position decay of one intervention, given a delta per clip.

    Returns rows of (position, readout error vs `target_values`, surviving ‖delta‖
    relative to the injection point, and the fraction of that delta lying in the
    position's own readout subspace).

    `control_acts` is the unmodified forward pass; pass it in to reuse across
    several interventions rather than recomputing it each time.
    """
    from . import probes as P
    from . import causal as C
    from . import data as D

    recs = D.load_manifest("direction")
    cents, _ = F.load_tracks("direction")

    def sweep(dd_all):
        acc, stats = {}, []
        for s in range(0, len(clips), batch):
            sl = clips[s:s + batch]
            vids = [D.decode(recs[i]["video"]) for i in sl]
            px = torch.from_numpy(np.stack([D.preprocess(x) for x in vids])).to(device)
            mk = torch.from_numpy(np.stack([D.object_mask(cents[i]) for i in sl])).to(device)
            dd = None if dd_all is None else torch.from_numpy(dd_all[s:s + batch]).to(device)
            with Recorder(model, blocks, pooling) as rec:
                C.forward_steered(model, px, layer, dd, [layer], pooling, mk, inject="all")
                for k, val in rec.pooled(mk).items():
                    acc.setdefault(k, []).append(val)
                stats.append({k: token_stats(v) for k, v in rec.raw().items()})
        return {k: np.concatenate(v) for k, v in acc.items()}, stats

    if control_acts is None:
        control_acts, control_stats = sweep(None)
    st_acts, st_stats = sweep(deltas)

    rows, base = [], None
    for k in positions(blocks):
        d = st_acts[k] - control_acts[k]
        dn = float(np.linalg.norm(d, axis=1).mean())
        if base is None:
            base = dn
        Q, _ = np.linalg.qr(probes[k].readout)
        al = float(np.mean(np.linalg.norm(d @ Q, axis=1)
                           / np.linalg.norm(d, axis=1).clip(1e-9)))
        rows.append({
            "pos": k,
            "err": P.circular_mae(probes[k].predict(st_acts[k]), target_values),
            "dnorm": dn / base,
            "aligned": al,
            "aligned_mag": (dn / base) * al,
        })
    return rows, control_acts, control_stats
