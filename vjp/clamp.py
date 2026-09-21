"""Continuous feature clamping across blocks — the world-model analogue of
per-layer feature clamping in LLM interpretability.

F11 showed a one-shot Euclidean edit at layer 16 is not erased but *rotated* out
of the readout subspace by attention, keeping only 32% of its aligned component
four blocks later. The obvious counter is to stop asking the edit to survive and
instead re-assert it at every block.

Two things that get conflated, and only the second is clamping:

  re-injection   add the SAME delta again at each layer. The perturbation
                 accumulates (4x by block 19), so any recovered readout is partly
                 bought with a much larger intervention.
  clamping       at each layer, read the probe, and apply the minimum-norm
                 correction that makes it read the target. Self-limiting: if the
                 representation has drifted little, the correction is small.

`clamp_forward` implements both, plus the un-edited control.
"""
from __future__ import annotations
import numpy as np
import torch

from .config import N_T, N_SPATIAL, HIDDEN
from . import features as F


def _pool_mean(h: torch.Tensor, pooling: str, masks) -> torch.Tensor:
    """Pooled, time-averaged representation the probe sees: [B, 1024]."""
    return F.POOL_FNS[pooling](h, masks).mean(dim=1)


def _probe_parts(p, device):
    R = torch.from_numpy(np.ascontiguousarray(p.readout)).float().to(device)   # [D,k]
    mu = torch.from_numpy(p.mu_.astype(np.float32)).to(device)
    ym = torch.from_numpy(p.ym_.astype(np.float32)).to(device)
    Rp = torch.from_numpy(np.linalg.pinv(p.readout).astype(np.float32)).to(device)  # [k,D]
    return R, mu, ym, Rp


@torch.inference_mode()
def clamp_forward(model, pixels, layer, target, probes, readout_layers,
                  mode="clamp", n_blocks=4, delta0=None, pooling="sal",
                  masks=None, device="mps", banks=None, dataset="direction"):
    """Run the encoder while asserting the target readout across several blocks.

    mode='none'    untouched control
    mode='once'    inject delta0 at `layer` only  (the F11 baseline)
    mode='reinject' add delta0 again at each of the next n_blocks
    mode='clamp'   at each block, apply the minimum-norm correction that makes
                   that block's probe read `target`

    Returns (pooled readouts at each readout layer, total L2 of everything injected).
    """
    handles, spent = [], {"n": 0.0}
    tgt = torch.from_numpy(target.astype(np.float32)).to(device)               # [B,k]

    def make(Lp, kind):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if kind == "fixed":
                d = delta0
            elif kind == "clamp1":
                R, mu, ym, Rp = _probe_parts(probes[Lp], device)
                cur = (_pool_mean(h, pooling, masks) - mu) @ R + ym            # [B,k]
                d = (tgt - cur) @ Rp                                           # min-norm
            else:
                # Clamp the SAME multi-probe subspace used for the one-shot edit,
                # refitted at this layer. Without this the clamp arm is a K=1
                # intervention competing against a K=16 one, and F8 already showed
                # K=1 barely moves direction -- the comparison would be rigged.
                from . import probes as _P
                bank = banks[Lp]
                cur = _pool_mean(h, pooling, masks).float().cpu().numpy()
                tg = np.tile(target, (1, len(bank)))
                want = _P.steer(cur, _P.probe_subspace(bank), bank, tg)
                d = torch.from_numpy((want - cur).astype(np.float32)).to(device)
            spent["n"] += float(d.norm(dim=-1).sum())
            return ((h + d[:, None, :]),) + tuple(out[1:])
        return hook

    if mode != "none":
        # inject at `layer` == modify the output of block layer-1
        handles.append(model.encoder.layer[layer - 1].register_forward_hook(
            make(layer, "fixed" if mode in ("once", "reinject") else
                 ("clamp1" if mode == "clamp1" else "clampK"))))
        if mode in ("reinject", "clamp", "clamp1"):
            kind = "fixed" if mode == "reinject" else ("clamp1" if mode == "clamp1" else "clampK")
            for j in range(1, n_blocks):
                Lp = layer + j
                handles.append(model.encoder.layer[Lp - 1].register_forward_hook(
                    make(Lp, kind)))
    try:
        res = model(pixel_values_videos=pixels, output_hidden_states=True,
                    skip_predictor=True)
        states = list(res.hidden_states) + [res.last_hidden_state]
        return ({L: _pool_mean(states[L], pooling, masks).float().cpu().numpy()
                 for L in readout_layers}, spent["n"])
    finally:
        for h in handles:
            h.remove()
