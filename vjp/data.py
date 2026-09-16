"""Manifest loading, video decoding, disk tracking, and split assignment."""
from __future__ import annotations
import json
import numpy as np
import av

from .config import (DATA, N_FRAMES, IMG, GRID, PATCH, N_T, TUBELET,
                     DISK_RADIUS_PX, MEAN, STD)

# ---------------------------------------------------------------- manifests

def load_manifest(dataset: str) -> list[dict]:
    """Return one record per clip: id, absolute video path, and all metadata fields."""
    root = DATA / dataset
    recs = []
    with open(root / "manifest.jsonl") as f:
        for line in f:
            row = json.loads(line)
            md = json.loads((root / row["metadata"]).read_text())
            recs.append({"id": row["id"], "video": str(root / row["video"]), **md})
    recs.sort(key=lambda r: r["id"])
    return recs


def target_value(dataset: str, rec: dict) -> float:
    """The primary target: theta for `direction`, magnitude otherwise."""
    return rec["theta_degrees"] if dataset == "direction" else rec["magnitude"]


# ---------------------------------------------------------------- decoding

def decode(path: str) -> np.ndarray:
    """Decode an mp4 to uint8 [T, H, W, 3]."""
    container = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    arr = np.stack(frames)
    assert arr.shape == (N_FRAMES, IMG, IMG, 3), f"unexpected shape {arr.shape} for {path}"
    return arr


_MEAN = np.array(MEAN, np.float32)
_STD = np.array(STD, np.float32)


def preprocess(video: np.ndarray) -> np.ndarray:
    """uint8 [T,H,W,3] -> float32 [T,3,H,W], ImageNet-normalised.

    The clips are already exactly 256x256, so we deliberately bypass the HF
    processor's resize-to-292-then-centre-crop, which would have thrown away
    ~12% of each frame's border. Documented as an explicit choice.
    """
    x = (video.astype(np.float32) / 255.0 - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


# ---------------------------------------------------------------- tracking

def track_disk(video: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Locate the (orange) disk in each frame by intensity-weighted centroid.

    Returns (centroids [T,2] as (x,y) in pixels, areas [T]).
    Frames with no detection get NaN centroids and 0 area.
    """
    v = video.astype(np.float32)
    orangeness = v[..., 0] - 0.5 * (v[..., 1] + v[..., 2])
    cents = np.full((len(v), 2), np.nan, np.float32)
    areas = np.zeros(len(v), np.float32)
    for i, o in enumerate(orangeness):
        thresh = o.max() * 0.5
        if o.max() < 10:            # nothing bright enough -> no disk
            continue
        mask = o > thresh
        ys, xs = np.nonzero(mask)
        w = o[mask]
        cents[i] = ((xs * w).sum() / w.sum(), (ys * w).sum() / w.sum())
        areas[i] = mask.sum()
    return cents, areas


# patch-centre coordinates, shape [GRID*GRID, 2] as (x, y)
_PC = np.stack(np.meshgrid(
    np.arange(GRID) * PATCH + PATCH / 2,   # x
    np.arange(GRID) * PATCH + PATCH / 2,   # y
    indexing="xy",
), -1).reshape(-1, 2).astype(np.float32)
# NOTE: indexing="xy" gives [row=y, col=x]; flattening matches token order h*16+w.


def object_mask(cents: np.ndarray) -> np.ndarray:
    """Per-temporal-token boolean mask over the 256 spatial patches.

    Temporal token t covers frames 2t and 2t+1; a patch is selected if its centre
    lies within (disk radius + half a patch) of the disk in EITHER frame.

    WARNING: this mask is derived from the disk trajectory, which is the label.
    Any probe on object-pooled features therefore has a label-leak floor that must
    be measured with the mask-only baseline (see experiments/mask_baseline).
    """
    radius = DISK_RADIUS_PX + PATCH / 2
    mask = np.zeros((N_T, GRID * GRID), bool)
    for t in range(N_T):
        for f in (TUBELET * t, TUBELET * t + 1):
            c = cents[f]
            if np.isnan(c).any():
                continue
            mask[t] |= np.hypot(*(_PC - c).T) <= radius
        if not mask[t].any():        # never leave a token with an empty mask
            mask[t] = True
    return mask


# ---------------------------------------------------------------- splits

def value_splits(values: np.ndarray, seed: int = 0) -> dict[float, str]:
    """Assign each unique target value to train / val / test.

    Held-out values are INTERLEAVED and strictly INTERIOR (never the min or max),
    so that evaluating on them is an interpolation test. This matters for Part 2:
    splines are fitted through train-value centroids and must interpolate to reach
    the held-out ones. An extrapolation split is provided separately.
    """
    uniq = np.unique(values)
    assign = {}
    for i, v in enumerate(uniq):
        interior = 0 < i < len(uniq) - 1
        if interior and i % 8 == 3:
            assign[float(v)] = "val"
        elif interior and i % 8 == 7:
            assign[float(v)] = "test"
        else:
            assign[float(v)] = "train"
    return assign


def extrapolation_split(values: np.ndarray, frac: float = 0.15) -> dict[float, str]:
    """Separate, separately-labelled split holding out the TOP values only."""
    uniq = np.unique(values)
    k = max(1, int(round(frac * len(uniq))))
    return {float(v): ("test" if i >= len(uniq) - k else "train")
            for i, v in enumerate(uniq)}
