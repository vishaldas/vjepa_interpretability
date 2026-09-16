"""One-pass feature extraction with on-disk caching.

Run:  python -m vjp.extract  [--datasets speed ...] [--batch 4] [--force]

Extracts, for every clip and every one of the 26 stored residual-stream states,
three spatial poolings of the 2048 encoder tokens, keeping the 8 temporal tokens.
Output per dataset is layer-major [26, N, 8, 1024] fp16, ~654 MB per pooling.

The cache is treated as IMMUTABLE once written: MPS reductions are not run-to-run
deterministic (~1e-3 relative at depth), so re-extracting mid-project would make
probe numbers drift between experiments.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import torch
from transformers import VJEPA2Model

from .config import (MODEL_ID, DATASETS, N_T, HIDDEN, N_STATES, POST_LN_INDEX,
                     POOLINGS, CACHE, FEAT_VERSION, load_hf_token)
from . import data as D
from . import features as F


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_index(dataset: str, recs: list[dict]) -> dict:
    """Clip metadata plus the three split axes, resolved once and cached."""
    vals = np.array([D.target_value(dataset, r) for r in recs], np.float64)
    vsplit = D.value_splits(vals)
    xsplit = D.extrapolation_split(vals)

    # Clip-level split WITHIN each value: tests nuisance generalisation
    # (start position, and for `direction` the speed/accel regime) with the
    # target value itself seen in training.
    rng = np.random.default_rng(0)
    clip_split = np.empty(len(recs), object)
    for v in np.unique(vals):
        idx = np.flatnonzero(vals == v)
        perm = rng.permutation(idx)
        n_test = max(1, int(round(0.25 * len(perm))))
        clip_split[perm[:n_test]] = "test"
        clip_split[perm[n_test:]] = "train"

    return {
        "dataset": dataset,
        "n": len(recs),
        "ids": [r["id"] for r in recs],
        "target": "theta_degrees" if dataset == "direction" else "magnitude",
        "values": vals.tolist(),
        "theta_degrees": [r["theta_degrees"] for r in recs],
        "speed_mps": [r["speed_mps"] for r in recs],
        "acceleration_mps2": [r["acceleration_mps2"] for r in recs],
        "motion": [r["motion"] for r in recs],
        "start_xy": [r["start_position_xy_m"] for r in recs],
        "split_value": [vsplit[float(v)] for v in vals],
        "split_clip": clip_split.tolist(),
        "split_extrap": [xsplit[float(v)] for v in vals],
    }


@torch.inference_mode()
def extract_dataset(model, dataset: str, batch: int = 4, device: str = "mps") -> None:
    recs = D.load_manifest(dataset)
    n = len(recs)
    print(f"[{dataset}] {n} clips", flush=True)

    out = {p: np.zeros((N_STATES, n, N_T, HIDDEN), np.float16) for p in POOLINGS}
    centroids = np.zeros((n, 16, 2), np.float32)
    areas = np.zeros((n, 16), np.float32)

    t0 = time.time()
    for s in range(0, n, batch):
        chunk = recs[s:s + batch]
        vids = [D.decode(r["video"]) for r in chunk]
        x = torch.from_numpy(np.stack([D.preprocess(v) for v in vids])).to(device)

        masks = []
        for j, v in enumerate(vids):
            c, a = D.track_disk(v)
            centroids[s + j], areas[s + j] = c, a
            masks.append(D.object_mask(c))
        mask = torch.from_numpy(np.stack(masks)).to(device)

        res = model(pixel_values_videos=x, output_hidden_states=True, skip_predictor=True)
        states = list(res.hidden_states) + [res.last_hidden_state]
        assert len(states) == N_STATES, f"{len(states)} != {N_STATES}"

        for li, h in enumerate(states):
            for p in POOLINGS:
                v = F.POOL_FNS[p](h, mask)
                assert torch.isfinite(v).all(), f"non-finite at {dataset} layer {li} pool {p}"
                out[p][li, s:s + len(chunk)] = v.float().cpu().numpy().astype(np.float16)

        if s % (batch * 40) == 0 or s + batch >= n:
            done = min(s + batch, n)
            el = time.time() - t0
            print(f"  {done}/{n}  {el:.0f}s elapsed  eta {el/done*(n-done):.0f}s", flush=True)

    for p in POOLINGS:
        np.save(F.feat_path(dataset, p), out[p])
    np.savez_compressed(F.tracks_path(dataset), centroids=centroids, areas=areas)
    F.index_path(dataset).write_text(json.dumps(build_index(dataset, recs)))
    print(f"[{dataset}] saved in {time.time()-t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    todo = [d for d in args.datasets if args.force or not F.is_cached(d)]
    for d in args.datasets:
        if d not in todo:
            print(f"[{d}] cached, skipping (use --force to redo)")
    if not todo:
        return

    auth = load_hf_token()
    dev = _device()
    print(f"device={dev} model={MODEL_ID} hf_auth={'yes' if auth else 'no (anonymous)'}", flush=True)
    model = VJEPA2Model.from_pretrained(MODEL_ID, dtype=torch.float32).eval().to(dev)

    for d in todo:
        extract_dataset(model, d, batch=args.batch, device=dev)

    (CACHE / "meta.json").write_text(json.dumps({
        "version": FEAT_VERSION, "model": MODEL_ID, "device": dev,
        "dtype": "float32 compute / float16 storage", "batch": args.batch,
        "n_states": N_STATES, "post_ln_index": POST_LN_INDEX,
        "layout": "[n_states, n_clips, 8_temporal, 1024] layer-major",
        "poolings": list(POOLINGS),
        "frames": 16, "note": "16f native (checkpoint is fpc64); processor resize bypassed",
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    print("meta written")


if __name__ == "__main__":
    main()
