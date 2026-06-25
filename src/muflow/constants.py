import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# All paths can be overridden via environment variables so the project is
# portable across machines without editing the source.
#
#   MUFLOW_DATA_DIR     root of the datasets (see README for the expected layout)
#   MUFLOW_WORKING_DIR  repo root (auto-detected from this file by default)
#   MUFLOW_CHECKPOINT_DIR  where training runs are written (default: "logs/")

# Repo root = two levels up from this file (src/muflow/constants.py → repo root)
_DEFAULT_WORKING_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORKING_DIR    = os.environ.get("MUFLOW_WORKING_DIR", _DEFAULT_WORKING_DIR)
DATA_DIR       = os.environ.get("MUFLOW_DATA_DIR", "/media/orazio_mattia_group/ad4dd")
CHECKPOINT_DIR = os.environ.get("MUFLOW_CHECKPOINT_DIR", "logs/")

# ── Patch-based representation ────────────────────────────────────────────────
# An image is represented through native patches of side = config["input_size"]
# (NO resize). The flow is trained on single real patches; at inference / GMM
# fitting an image is summarised by aggregating its patches.
PATCH_NUM_TRAIN = 4    # random patches sampled per training image (per step)
PATCH_NUM_REPR  = 16   # patches per image for inference & mean-image GMM centroid
PATCH_SEED      = 42   # fixed seed → reproducible representation patches

# ── Classic backbones ─────────────────────────────────────────────────────────
BACKBONE_DEIT         = "deit_base_distilled_patch16_384"
BACKBONE_CAIT         = "cait_m48_448"
BACKBONE_RESNET18     = "resnet18"
BACKBONE_RESNET50     = "resnet50"
BACKBONE_RESNET101    = "resnet101"
BACKBONE_WIDE_RESNET50= "wide_resnet50_2"
BACKBONE_DENSENET121  = "densenet121"

# ── DINOv2 backbones (timm) ───────────────────────────────────────────────────
BACKBONE_DINOV2_VITS14 = "dinov2_vits14"
BACKBONE_DINOV2_VITB14 = "dinov2_vitb14"
BACKBONE_DINOV2_VITL14 = "dinov2_vitl14"

# Timm model identifiers for DINOv2
DINO_TIMM_NAMES = {
    BACKBONE_DINOV2_VITS14: "vit_small_patch14_dinov2.lvd142m",
    BACKBONE_DINOV2_VITB14: "vit_base_patch14_dinov2.lvd142m",
    BACKBONE_DINOV2_VITL14: "vit_large_patch14_dinov2.lvd142m",
}
# Channel widths
DINO_CHANNELS = {
    BACKBONE_DINOV2_VITS14: 384,
    BACKBONE_DINOV2_VITB14: 768,
    BACKBONE_DINOV2_VITL14: 1024,
}
DINO_PATCH_SIZE = {
    BACKBONE_DINOV2_VITS14: 14,
    BACKBONE_DINOV2_VITB14: 14,
    BACKBONE_DINOV2_VITL14: 14,
}
# DINOv2 uses the same normalization as ImageNet
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD  = [0.229, 0.224, 0.225]

# Sets
DINO_BACKBONES = [BACKBONE_DINOV2_VITS14, BACKBONE_DINOV2_VITB14, BACKBONE_DINOV2_VITL14]

# ── CLIP backbones (official OpenAI CLIP) ─────────────────────────────────────
BACKBONE_CLIP_VITB32 = "clip_vitb32"
BACKBONE_CLIP_VITB16 = "clip_vitb16"
BACKBONE_CLIP_VITL14 = "clip_vitl14"

# Official OpenAI CLIP model strings (used with clip.load())
CLIP_OPENAI_NAMES = {
    BACKBONE_CLIP_VITB32: "ViT-B/32",
    BACKBONE_CLIP_VITB16: "ViT-B/16",
    BACKBONE_CLIP_VITL14: "ViT-L/14",
}
CLIP_CHANNELS = {
    BACKBONE_CLIP_VITB32: 768,
    BACKBONE_CLIP_VITB16: 768,
    BACKBONE_CLIP_VITL14: 1024,
}
CLIP_PATCH_SIZE = {
    BACKBONE_CLIP_VITB32: 32,
    BACKBONE_CLIP_VITB16: 16,
    BACKBONE_CLIP_VITL14: 14,
}
CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLIP_BACKBONES = [BACKBONE_CLIP_VITB32, BACKBONE_CLIP_VITB16, BACKBONE_CLIP_VITL14]


def get_norm_stats(backbone_name: str):
    """Return (mean, std) normalization stats appropriate for backbone_name."""
    if backbone_name in CLIP_BACKBONES:
        return CLIP_MEAN, CLIP_STD
    return IMAGENET_MEAN, IMAGENET_STD

# ── All supported backbones ───────────────────────────────────────────────────
SUPPORTED_BACKBONES = [
    BACKBONE_DEIT,
    BACKBONE_CAIT,
    BACKBONE_RESNET18,
    BACKBONE_RESNET50,
    BACKBONE_RESNET101,
    BACKBONE_WIDE_RESNET50,
    BACKBONE_DENSENET121,
    *DINO_BACKBONES,
    *CLIP_BACKBONES,
]
