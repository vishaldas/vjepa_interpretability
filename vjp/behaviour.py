"""Behaviour channel: read the PREDICTOR's forecast, not an encoder probe.

Every other Part 2 evaluation reads a ridge probe on a downstream encoder layer,
so its claims are about the representation. Wurgaft et al.'s thesis links the
activation manifold to a *behaviour* manifold, and for a JEPA the nearest thing
to behaviour is what the predictor forecasts.

Here the edit is injected at `layer`, propagated through the whole encoder, and
the predictor is then asked to forecast the LATER half of the clip from the
earlier half. The probe reads that forecast. A change in its readout means the
model's own forward prediction moved, not merely its internal state.

Token layout is t*256 + h*16 + w (verified in setup), so temporal blocks 0-3 are
indices 0..1023 and blocks 4-7 are 1024..2047.
"""
from __future__ import annotations
import numpy as np
import torch

from .config import N_T, N_SPATIAL, SAL_TOPK

CTX_BLOCKS = 4                      # forecast the second half from the first


def masks_for(n_tokens: int, device: str, batch: int, ctx_blocks: int = CTX_BLOCKS):
    cut = ctx_blocks * N_SPATIAL
    idx = torch.arange(n_tokens, device=device)
    ctx = [idx[:cut].unsqueeze(0).repeat(batch, 1)]
    tgt = [idx[cut:].unsqueeze(0).repeat(batch, 1)]
    return ctx, tgt


def pool_forecast(h: torch.Tensor, topk: int = SAL_TOPK) -> torch.Tensor:
    """Leak-free top-K salience pooling over the forecast's spatial patches.

    h is [B, n_target_tokens, D]; reshaped to [B, blocks, 256, D], the top-K most
    deviant patches per block are averaged, then averaged over blocks -> [B, D].
    Deliberately NOT the tracker mask: that mask comes from the trajectory, i.e.
    the label. It would be defensible here because the mask is identical across
    conditions, but salience avoids the argument entirely.
    """
    B, N, D = h.shape
    blocks = N // N_SPATIAL
    g = h.view(B, blocks, N_SPATIAL, D)
    dev = (g - g.mean(dim=2, keepdim=True)).norm(dim=-1)
    idx = dev.topk(min(topk, N_SPATIAL), dim=2).indices
    sel = torch.gather(g, 2, idx.unsqueeze(-1).expand(-1, -1, -1, D))
    return sel.mean(dim=2).mean(dim=1)


@torch.inference_mode()
def forecast(model, pixels, layer: int, delta: torch.Tensor | None,
             ctx_blocks: int = CTX_BLOCKS, device: str = "mps"):
    """Inject `delta` at `layer`, propagate, and return the pooled forecast [B, D]."""
    handle = None
    if delta is not None:
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            return ((h + delta[:, None, :]),) + tuple(out[1:])
        handle = model.encoder.layer[layer - 1].register_forward_hook(hook)
    try:
        B = pixels.shape[0]
        n_tok = N_T * N_SPATIAL
        ctx, tgt = masks_for(n_tok, device, B, ctx_blocks)
        out = model(pixel_values_videos=pixels, context_mask=ctx, target_mask=tgt)
        pred = out.predictor_output.last_hidden_state          # the FORECAST
        return pool_forecast(pred).float().cpu().numpy()
    finally:
        if handle is not None:
            handle.remove()
