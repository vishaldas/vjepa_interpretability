"""Token pooling operators and the feature-cache read/write API."""
from __future__ import annotations
import json
import numpy as np
import torch

from .config import (FEATS, CACHE, N_T, N_SPATIAL, HIDDEN, N_STATES,
                     POST_LN_INDEX, POOLINGS, FEAT_VERSION, SAL_TOPK)

# ---------------------------------------------------------------- poolings
# Every pooling maps [B, 2048, 1024] -> [B, 8, 1024]: it collapses the 256
# spatial patches within each temporal token, leaving the time axis intact.
# Global mean pooling is then derivable downstream as .mean(axis=-2).

def _grid(h: torch.Tensor) -> torch.Tensor:
    return h.view(h.shape[0], N_T, N_SPATIAL, HIDDEN)


def pool_tmean(h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Uniform spatial mean. The paper's 'mean-pooled space-time patches'.

    Dilutes the disk (~1.3 patches of 256) to <1% of the pooled vector.
    """
    return _grid(h).mean(dim=2)


def pool_sal(h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Label-free salience pooling: uniform mean over the top-K most salient patches.

    Salience is how far a patch deviates from its own timestep's mean token. The
    encoder's ranking recovers the true object tokens with recall 1.00 through
    layer 12 and >=0.89 at layer 24 (measured, K=8), so this is object-centric
    *without* ever consulting the trajectory -- unlike `pool_obj` it carries no
    label leak, and it is the pooling we can defend as "what the encoder itself
    makes salient".

    Soft deviation-proportional weights were tried first and rejected: the weight
    distribution flattens with depth (object weight share 0.61 at L1 -> 0.03 at
    L20), so the weighted mean collapses onto the uniform mean, cos(tmean,sal)=1.000.
    Top-K keeps the pooling sharp at every depth.
    """
    g = _grid(h)
    dev = (g - g.mean(dim=2, keepdim=True)).norm(dim=-1)        # [B,8,256]
    idx = dev.topk(SAL_TOPK, dim=2).indices                      # [B,8,K]
    sel = torch.gather(g, 2, idx.unsqueeze(-1).expand(-1, -1, -1, g.shape[-1]))
    return sel.mean(dim=2)


def pool_obj(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Oracle object-token pooling: uniform mean over tracker-selected patches.

    LABEL LEAK: the mask is built from the disk trajectory, i.e. from the label.
    Positional (RoPE) information alone makes these features predictive, so probe
    scores here are only meaningful against the mask-only baseline.
    """
    g = _grid(h)
    m = mask.to(g.dtype)                                        # [B,8,256]
    w = m / m.sum(dim=2, keepdim=True).clamp_min(1e-6)
    return (w.unsqueeze(-1) * g).sum(dim=2)


POOL_FNS = {"tmean": pool_tmean, "sal": pool_sal, "obj": pool_obj}

# ---------------------------------------------------------------- cache io

def feat_path(dataset: str, pooling: str) -> "Path":
    return FEATS / f"{dataset}_{pooling}.npy"


def index_path(dataset: str) -> "Path":
    return CACHE / f"index_{dataset}.json"


def tracks_path(dataset: str) -> "Path":
    return CACHE / f"tracks_{dataset}.npz"


def is_cached(dataset: str) -> bool:
    return (index_path(dataset).exists() and tracks_path(dataset).exists()
            and all(feat_path(dataset, p).exists() for p in POOLINGS))


def load_index(dataset: str) -> dict:
    return json.loads(index_path(dataset).read_text())


def load_tracks(dataset: str):
    z = np.load(tracks_path(dataset))
    return z["centroids"], z["areas"]


def load_feats(dataset: str, pooling: str = "tmean", layer: int | None = None,
               mmap: bool = True) -> np.ndarray:
    """Load cached features.

    Layout is LAYER-MAJOR [26, N, 8, 1024] fp16, so `layer=L` is one contiguous
    ~25 MB read rather than a strided scan of the whole file.

    Returns float32. Index 0..24 are hidden_states (residual stream at each block
    boundary, PRE final LayerNorm); index 25 is the post-LayerNorm output.
    """
    arr = np.load(feat_path(dataset, pooling), mmap_mode="r" if mmap else None)
    out = arr[layer] if layer is not None else arr
    return np.asarray(out, dtype=np.float32)


def load_pooled(dataset: str, pooling: str = "tmean", layer: int = 12,
                time: str = "mean") -> np.ndarray:
    """Convenience: features for one layer as a flat [N, D] design matrix.

    time='mean'    -> average the 8 temporal tokens        -> [N, 1024]
    time='concat'  -> concatenate them                     -> [N, 8192]
    time='keep'    -> leave the time axis alone            -> [N, 8, 1024]
    """
    f = load_feats(dataset, pooling, layer)
    if time == "mean":
        return f.mean(axis=1)
    if time == "concat":
        return f.reshape(len(f), -1)
    if time == "keep":
        return f
    raise ValueError(f"unknown time={time!r}")
