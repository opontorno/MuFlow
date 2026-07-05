import os

_DEFAULT_WORKING_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORKING_DIR    = os.environ.get("MUFLOW_WORKING_DIR", _DEFAULT_WORKING_DIR)
DATA_DIR       = os.environ.get("MUFLOW_DATA_DIR", "/media/orazio_mattia_group/ad4dd")
CHECKPOINT_DIR = os.environ.get("MUFLOW_CHECKPOINT_DIR", "logs/")

PATH_REAL = [f"{DATA_DIR}/datasets/ffhq/**/*.*g"]
PATH_REAL_OOD = [f"{DATA_DIR}/datasets/celeba_hq/**/*.*g"]
PATH_FAKE = [
    f"{DATA_DIR}/datasets/WILD/**/*.*g",
    f"{DATA_DIR}/datasets/other_sources/**/*.*g",
]

SHOW_PER_CLASS = False
SHOW_FAMILIES  = False


def real_tag(path_globs):
    """Derive a short real-source tag from a list of glob patterns."""
    head = path_globs[0].split("*")[0]
    return os.path.basename(head.rstrip("/"))


PATCH_NUM_TRAIN = 4
PATCH_NUM_REPR  = 16
PATCH_SEED      = 42

BACKBONE_RESNET50     = "resnet50"


DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD  = [0.229, 0.224, 0.225]

BACKBONE_CLIP_VITL14 = "clip_vitl14"

CLIP_OPENAI_NAMES = {
    BACKBONE_CLIP_VITL14: "ViT-L/14",
}
CLIP_CHANNELS = {
    BACKBONE_CLIP_VITL14: 1024,
}
CLIP_PATCH_SIZE = {
    BACKBONE_CLIP_VITL14: 14,
}
CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLIP_BACKBONES = [BACKBONE_CLIP_VITL14]


def get_norm_stats(backbone_name: str):
    """Return the (mean, std) normalization stats for a backbone."""
    if backbone_name in CLIP_BACKBONES:
        return CLIP_MEAN, CLIP_STD
    return IMAGENET_MEAN, IMAGENET_STD

SUPPORTED_BACKBONES = [
    BACKBONE_RESNET50,
    *CLIP_BACKBONES,
]
