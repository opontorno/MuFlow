CHECKPOINT_DIR = "logs/"

DATA_DIR = "/media/orazio_mattia_group/ad4dd"
WORKING_DIR = "/home/opontorno/projects/MuFlow"

MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

BACKBONE_DEIT = "deit_base_distilled_patch16_384"
BACKBONE_CAIT = "cait_m48_448"
BACKBONE_RESNET18 = "resnet18"
BACKBONE_RESNET50 = "resnet50"
BACKBONE_RESNET101 = "resnet101"
BACKBONE_WIDE_RESNET50 = "wide_resnet50_2"
BACKBONE_DENSENET121 = "densenet121"

SUPPORTED_BACKBONES = [
    BACKBONE_DEIT,
    BACKBONE_CAIT,
    BACKBONE_RESNET18,
    BACKBONE_RESNET50,
    BACKBONE_RESNET101,
    BACKBONE_WIDE_RESNET50,
    BACKBONE_DENSENET121,
]
