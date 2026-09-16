# The feature cache — what is stored and how to use it

Everything expensive is computed once and cached, so experiments are interactive.

## One-time cost

```bash
.venv/bin/python -m vjp.extract            # ~23 min on MPS, all 4,572 clips
```

Idempotent: it skips datasets already cached (`--force` to redo). Checkpointing is
**per dataset**, so an interruption costs at most ~8 minutes.

> Treat the cache as **immutable** once written. MPS reductions are not run-to-run
> deterministic (~1e-3 relative at depth), so re-extracting mid-project would make probe
> numbers drift between experiments.

## Layout

```
artifacts/
  cache/
    meta.json                  extraction config: model, device, dtype, layout
    index_<ds>.json            per-clip labels + the three split assignments
    tracks_<ds>.npz            disk centroid [N,16,2] and area [N,16]
    feats/<ds>_<pool>.npy      [26, N, 8, 1024] float16   (~650 MB each)
  results/<name>__<hash>.json  memoised experiment outputs
  figures/*.png
```

Total ≈ 5.8 GB for 3 datasets × 3 poolings.

**Layout is layer-major** — `arr[L]` is one contiguous ~25 MB read rather than a strided
scan of the whole file, which is what every downstream experiment wants.

**26 states, not 24.** Indices 0–24 are the residual stream at each block boundary
(**pre** final LayerNorm); index 25 is the **post**-LayerNorm output (`last_hidden_state`).
These genuinely differ — see SETUP_FINDINGS.md.

## Three poolings

Each collapses the 256 spatial patches within a temporal token, leaving the 8 temporal
tokens. Global mean pooling is derivable downstream (`time="mean"`), so nothing is lost.

| name | what | use |
|---|---|---|
| `tmean` | uniform spatial mean | the paper's pooling; comparable to published curves |
| `sal` | mean over top-8 most salient patches | **leak-free object-centric — select layers with this** |
| `obj` | mean over tracker-selected patches | oracle; only meaningful *above* the mask-only floor |

`obj` carries a **label leak** (the mask is built from the trajectory, which is the
label). Always read it against `experiments.mask_baseline()`.

## Reading it

```python
from vjp import experiments as E, features as F

X, Y, values = E.get_xy("speed", layer=12, pooling="sal")   # [N,1024], [N,1], [N]
sp = E.get_splits("speed", "value")                          # train / val / test indices

F.load_pooled("speed", "sal", layer=12, time="mean")    # [N, 1024]  average the 8 steps
F.load_pooled("speed", "sal", layer=12, time="concat")  # [N, 8192]  keep them separately
F.load_pooled("speed", "sal", layer=12, time="keep")    # [N, 8, 1024]
```

## Three held-out axes

`E.get_splits(ds, axis)` — these are **not** interchangeable, and a random clip split
would silently invalidate both parts of the take-home.

| axis | holds out | why |
|---|---|---|
| `value` | 15 of 64 target values, **interleaved and strictly interior** | headline interpolation test; required for Part 2 |
| `clip` | 25 % of clips within each value | nuisance generalisation (start position, motion regime) |
| `extrap` | the top 15 % of values | extrapolation, reported separately |

The data is 64 discrete values × 24 reps. A random *clip* split puts a given magnitude in
both train and test, so a probe can memorise value→activation and R² tests nothing about
the continuous structure that Parts 1 and 2 are about.

Held-out values are **interior** (never the min or max) so that reaching them is genuine
interpolation — which is what Part 2 needs, since splines are fitted through train-value
centroids.

## Results memoisation

`E.cached(name, cfg, fn)` keys every result by a hash of its full config, so re-running an
analysis or redrawing a figure costs nothing. Pass `force=True` to recompute.

## Running things

```bash
.venv/bin/python run_part1.py                  # layer sweep + deciles + nullspace, all 3 vars
.venv/bin/python run_part1.py --axis clip      # same, on the nuisance split
```

## Not cached (deliberately)

Full 2,048-token grids and the 64-frame variant are **not** stored — 57 GB and 188 min
respectively, against a 23-minute re-extraction. Add them on demand when Stage 4/5 needs
token-level steering or the clip-length ablation.

## Hugging Face authentication (optional)

The encoder `facebook/vjepa2-vitl-fpc64-256` is **public and not gated**, and its weights
are already in `~/.cache/huggingface` (1.2 GB). A token is therefore a convenience —
higher rate limits and faster downloads — never a requirement for this project.

`vjp.config.load_hf_token()` reads one into the environment and is called automatically
by `vjp.extract`. It is a no-op if `HF_TOKEN` is already set or the file is missing, and
it never logs the value. It accepts a bare token, `HF_TOKEN=hf_...`, or
`export HF_TOKEN=hf_...`.

Default location is `hf_token.env` one level above the project root; override with the
`VJP_HF_TOKEN_FILE` environment variable.

```python
from vjp.config import load_hf_token
load_hf_token()          # -> True if a token was set
```

Keep the file **outside** the repo. `.gitignore` now also covers `*token*.env`, `.env`
and `hf_token*` in case one is ever copied in. Keep it mode 600.
