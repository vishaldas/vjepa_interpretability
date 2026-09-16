"""Central configuration. Single source of truth for paths and extraction constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
CACHE = ART / "cache"
FEATS = CACHE / "feats"
FIGS = ART / "figures"
for _p in (ART, CACHE, FEATS, FIGS):
    _p.mkdir(parents=True, exist_ok=True)

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
DATASETS = ("direction", "speed", "acceleration")

# Verified empirically in setup (see notes/SETUP_FINDINGS.md):
N_FRAMES = 16
IMG = 256
PATCH = 16
TUBELET = 2
GRID = IMG // PATCH          # 16 x 16 spatial patches
N_T = N_FRAMES // TUBELET    # 8 temporal tokens
N_SPATIAL = GRID * GRID      # 256
N_TOKENS = N_T * N_SPATIAL   # 2048, ordered as t*256 + h*16 + w
HIDDEN = 1024
N_LAYERS = 24
# We store hidden_states[0..24] (residual stream at each block boundary, PRE final
# LayerNorm) plus one extra slot for the post-final-LayerNorm output = last_hidden_state.
N_STATES = N_LAYERS + 2      # 26; index 25 == post-LN
POST_LN_INDEX = N_STATES - 1

# ImageNet normalisation, from video_preprocessor_config.json.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

POOLINGS = ("tmean", "sal", "obj")
DISK_RADIUS_PX = 10.5        # measured: disk area ~335 px
SAL_TOPK = 8                 # top-K salient patches for label-free pooling

FEAT_VERSION = "v1"


# ---------------------------------------------------------------- HF auth
# Optional. Unauthenticated Hub requests are rate-limited and slower; the weights
# are cached locally after the first download, so this is a convenience, never a
# requirement. Supports either a bare token or a KEY=value line, and never logs
# the value.
HF_TOKEN_FILE = Path(
    os.environ.get("VJP_HF_TOKEN_FILE", ROOT.parent / "hf_token.env")
).expanduser()


def load_hf_token(path: "Path | None" = None) -> bool:
    """Read a Hugging Face token into the environment. Returns whether one was set.

    Does nothing if HF_TOKEN is already set, or the file is absent/empty.
    """
    if os.environ.get("HF_TOKEN"):
        return True
    path = Path(path or HF_TOKEN_FILE)
    try:
        raw = path.read_text().strip()
    except OSError:
        return False
    if not raw:
        return False
    # accept "hf_xxx", "HF_TOKEN=hf_xxx", or "export HF_TOKEN=hf_xxx"
    line = raw.splitlines()[0].strip()
    line = line.removeprefix("export ").strip()
    token = line.split("=", 1)[1].strip() if "=" in line else line
    token = token.strip("'\"")
    if not token:
        return False
    os.environ["HF_TOKEN"] = token
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    return True
