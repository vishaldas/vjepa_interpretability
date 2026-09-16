"""Causal steering: intervene at layer L, run the rest of the network, read later.

F8 steers a *pooled feature* and reads it back with a probe -- it never propagates
the edit. That shows the subspace is decodable and internally consistent, not that
it drives what the model computes next. Here the edit is injected into the live
residual stream and the remaining blocks actually run.

The intervention adds a single vector `delta` to EVERY token at layer L. That is
exact for both cached poolings:
  * `tmean` is a mean over tokens, so it shifts by exactly delta;
  * `sal` ranks tokens by deviation from the token mean, which a uniform shift
    leaves invariant, so the selection is unchanged and it also shifts by delta.
Both verified numerically in setup.
"""
from __future__ import annotations
import numpy as np
import torch

from .config import MODEL_ID, N_STATES, load_hf_token
from . import data as D
from . import features as F
from . import probes as P
from . import steering as S


def load_model(device: str = "mps"):
    from transformers import VJEPA2Model
    load_hf_token()
    return VJEPA2Model.from_pretrained(MODEL_ID, dtype=torch.float32).eval().to(device)


@torch.inference_mode()
def forward_steered(model, pixels: torch.Tensor, layer: int,
                    delta: torch.Tensor | None, readout_layers: list[int],
                    pooling: str = "sal", masks: torch.Tensor | None = None,
                    inject: str = "all"):
    """Run the encoder, optionally adding `delta` to every token at `layer`.

    `delta` is [B, 1024] or None for the unmodified control. Returns
    {readout_layer: pooled [B, 8, 1024]}.

    NOTE on indexing: hidden_states[i] is the output of block i-1, so injecting at
    the output of block `layer-1` shifts the INPUT to block `layer`. The recorded
    hidden_states[layer] does not show the edit (it is captured before the forward
    hook returns), which is why readouts must be taken strictly downstream.
    """
    from .config import N_T, N_SPATIAL, HIDDEN, SAL_TOPK
    handle = None
    if delta is not None:
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            if inject == "all":
                # uniform shift: `tmean` and `sal` both move by exactly delta
                o = o + delta[:, None, :]
            elif inject == "salient":
                # add to ONLY the top-K salient tokens, scaled so the sal-pooled
                # vector still moves by exactly delta. A far more localised edit:
                # it perturbs the tokens that carry the object rather than
                # displacing the entire residual stream.
                g = o.view(o.shape[0], N_T, N_SPATIAL, HIDDEN)
                dev = (g - g.mean(dim=2, keepdim=True)).norm(dim=-1)
                idx = dev.topk(SAL_TOPK, dim=2).indices               # [B,8,K]
                add = torch.zeros_like(g)
                src = delta[:, None, None, :].expand(-1, N_T, SAL_TOPK, -1)
                add.scatter_(2, idx.unsqueeze(-1).expand(-1, -1, -1, HIDDEN), src)
                o = (g + add).view_as(o)
            else:
                raise ValueError(inject)
            return (o,) + tuple(out[1:])
        handle = model.encoder.layer[layer - 1].register_forward_hook(hook)
    try:
        res = model(pixel_values_videos=pixels, output_hidden_states=True,
                    skip_predictor=True)
        states = list(res.hidden_states) + [res.last_hidden_state]
        return {L: F.POOL_FNS[pooling](states[L], masks).float().cpu().numpy()
                for L in readout_layers}
    finally:
        if handle is not None:
            handle.remove()


def run_causal(dataset: str, layer: int, readout_layers: list[int], K: int,
               pooling: str = "sal", batch: int = 4, device: str = "mps",
               seed: int = 0, n_test: int | None = None, model=None,
               inject: str = "all"):
    """Steer at `layer`, read at each of `readout_layers` with held-out probes.

    The basis probes live at `layer` (where the intervention happens); each readout
    probe is fitted at its own layer on clips disjoint from every basis probe.
    """
    from . import experiments as E
    sp = E.get_splits(dataset, "value")
    v = E.get_values(dataset)
    X, Y, _ = E.get_xy(dataset, layer, pooling)

    basis, _ = S.make_probe_bank(X, Y, sp["train"], sp["val"], K, seed)
    rng = np.random.default_rng(seed)
    eval_fold = np.array_split(rng.permutation(sp["train"]), K + 1)[0]

    # readout probes, fitted on unmodified cached features at each readout layer
    read_probes = {}
    for R in readout_layers:
        XR, YR, _ = E.get_xy(dataset, R, pooling)
        read_probes[R] = P.fit_ridge_cv(XR[eval_fold], YR[eval_fold],
                                        XR[sp["val"]], YR[sp["val"]])

    test = sp["test"] if n_test is None else sp["test"][:n_test]
    targets = rng.choice(np.unique(v[test]), size=len(test))

    Xs, _ = S.steer_to(X[test], basis, targets, dataset)
    deltas = (Xs - X[test]).astype(np.float32)          # [N, 1024] pooled-space edit

    if model is None:
        model = load_model(device)
    recs = D.load_manifest(dataset)
    cents, _ = F.load_tracks(dataset)

    out = {R: {"steered": [], "control": []} for R in readout_layers}
    for s in range(0, len(test), batch):
        sl = test[s:s + batch]
        vids = [D.decode(recs[i]["video"]) for i in sl]
        px = torch.from_numpy(np.stack([D.preprocess(x) for x in vids])).to(device)
        mk = torch.from_numpy(np.stack([D.object_mask(cents[i]) for i in sl])).to(device)
        d = torch.from_numpy(deltas[s:s + batch]).to(device)
        for name, dd in (("steered", d), ("control", None)):
            pooled = forward_steered(model, px, layer, dd, readout_layers, pooling,
                                     mk, inject=inject)
            for R in readout_layers:
                out[R][name].append(pooled[R].mean(axis=1))   # mean over 8 temporal

    res = {}
    for R in readout_layers:
        st = np.concatenate(out[R]["steered"])
        ct = np.concatenate(out[R]["control"])
        Ttgt = P.to_target(dataset, targets)
        Tsrc = P.to_target(dataset, v[test])
        pr = read_probes[R]
        res[R] = {
            "steered_vs_target": P.evaluate(dataset, pr.predict(st), Ttgt, targets),
            "control_vs_target": P.evaluate(dataset, pr.predict(ct), Ttgt, targets),
            "control_vs_source": P.evaluate(dataset, pr.predict(ct), Tsrc, v[test]),
            "steered_vs_source": P.evaluate(dataset, pr.predict(st), Tsrc, v[test]),
        }
    return res
