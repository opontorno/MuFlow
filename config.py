import os

# ---- Data root ----
# Set via MUFLOW_DATA_DIR, or hardcode your own path here instead.
DATA_DIR = os.environ.get("MUFLOW_DATA_DIR")
if DATA_DIR is None:
    raise RuntimeError(
        "MUFLOW_DATA_DIR is not set. Either `export MUFLOW_DATA_DIR=/path/to/your/data`, "
        "or edit DATA_DIR directly in config.py."
    )
DATA_DIR = DATA_DIR.rstrip("/")  # avoid double slashes if the value has a trailing "/"

# ---- Data sources (globs; edit these to point at your data) ----
PATH_REAL     = [f"{DATA_DIR}/datasets/ffhq/**/*.*g"]
PATH_REAL_OOD = [f"{DATA_DIR}/datasets/celeba_hq/**/*.*g"]   # None → reuse PATH_REAL at test time
PATH_FAKE     = [
    f"{DATA_DIR}/datasets/WILD/**/*.*g",
    f"{DATA_DIR}/datasets/other_sources/**/*.*g",
]

# ---- Generator families for the per-family report (names must match the fake folders) ----
GANS       = {'StyleGAN', 'StyleGAN2', 'StyleGAN3', 'STARGAN', 'AttGAN', 'GDWCT'}
DM_OPEN    = {'Flux.1', 'Stable DIffusion 3.5', 'Stable Diffusion XL',
              'Stable Cascade', 'Stable Diffusion Attend and Excite'}
DM_CLOSED  = {'Dall-E 3', 'Midjourney', 'Starry AI', 'Deep AI', 'Hotpot AI',
              'Nvidia Sana PAG', 'Tencent Hunyuan', 'Flux.1.1 Pro'}
MIX_2CLASS = {'STARGAN', 'StyleGAN2', 'Stable DIffusion 3.5', 'Flux.1.1 Pro'}

# ---- Console reporting ----
SHOW_PER_CLASS = False
SHOW_FAMILIES  = False

# ---- Dataset preparation (scripts/) ----
MEAN_SIZE      = 500                 # images averaged into each mean image
NUM_MEANS      = 1000                # mean images produced per source
FAKE_PER_CLASS = 1000                # cap per fake generator in the split CSV
SPLIT          = [0.70, 0.15, 0.15]  # train / val / test ratios


def real_tag(path_globs):
    """Derive a short real-source tag from a list of glob patterns."""
    head = path_globs[0].split("*")[0]
    return os.path.basename(head.rstrip("/"))
