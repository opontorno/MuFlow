"""
MuFlow — Evaluation script.

Normalizing-flow evaluation pipeline for one-class deepfake detection.
Shared logic (eval_once, build_model, compute_threshold, …) is imported
directly from main.py to avoid duplication.
"""
import argparse
import os
import glob as glob_module
import torch
import yaml
import joblib
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from PIL import Image

from muflow import constants as const
from muflow import dataset
from muflow.dataset import make_patch_transform
from muflow.patches import repr_patches

# ── Shared evaluation logic ──────────────────────────────────────────────────
from main import (
    eval_once,
    score_per_image,
    _compute_class_metrics,
    _aggregate_by_family,
    _print_family_summary,
    build_model,
    compute_threshold,
    build_val_data_loader,
    build_test_data_loader,
    GANS, DM_OPEN, DM_CLOSED, MIX_2CLASS,
)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a MuFlow run. All settings are loaded automatically "
                    "from run_dir/run_config.yaml. Only eval-specific options are needed.")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Directory of a training run (must contain best.pt, "
                             "thresholds.npz and run_config.yaml).")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--custom_dirs", type=str, nargs='+', default=None,
                        help="One or more folder paths or glob patterns containing images to evaluate "
                             "(png/jpg/jpeg). Each dir needs a corresponding entry in --custom_labels.")
    parser.add_argument("--custom_labels", type=int, nargs='+', default=None,
                        help="Class label for each --custom_dirs entry: 0=real, 1=fake. "
                             "Must have the same length as --custom_dirs.")

    args = parser.parse_args()

    # Load run_config.yaml and attach every key directly onto args
    run_cfg_path = os.path.join(args.run_dir, 'run_config.yaml')
    if not os.path.exists(run_cfg_path):
        parser.error(f"run_config.yaml not found in {args.run_dir}. "
                     "Make sure the run has saved at least one best checkpoint.")
    saved = yaml.safe_load(open(run_cfg_path))
    for k, v in saved.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # Resolve file paths relative to run_dir
    args.checkpoint     = os.path.join(args.run_dir, 'best.pt')
    args.threshold_path = os.path.join(args.run_dir, 'thresholds.npz')
    args.lof_checkpoint = os.path.join(args.run_dir, 'lof_model.pkl')

    print(f"[eval] Run dir  : {args.run_dir}")
    print(f"[eval] Config   : {args.config}")
    print(f"[eval] alpha    : {args.alpha}")
    return args


# ═══════════════════════════════════════════════════════════════════════════
# Eval-specific data loaders
# ═══════════════════════════════════════════════════════════════════════════

def create_dataloader(args, config, opt):
    attack_type   = getattr(opt, 'attack_type', 'none')
    attack_params = getattr(opt, 'attack_params', {})

    norm_mean, norm_std = const.get_norm_stats(config["backbone_name"])
    test_dataset = dataset.Dataset(
        reals_name=args.reals,
        input_size=config["input_size"],
        is_train=False,
        is_val=False,
        num_repr_patches=getattr(args, 'num_repr_patches', const.PATCH_NUM_REPR),
        attack_type=attack_type,
        attack_params=attack_params,
        norm_mean=norm_mean,
        norm_std=norm_std,
    ).create_dataset()

    num_workers = getattr(args, 'num_workers', 4)
    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return data_loader, {v: k for k, v in test_dataset.class_to_idx.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Custom-dir helpers
# ═══════════════════════════════════════════════════════════════════════════

def _dir_name_from_path(path_or_glob):
    parts = path_or_glob.replace('\\', '/').rstrip('/').split('/')
    for part in reversed(parts):
        if part and '*' not in part and '?' not in part:
            return part
    return f"dir_{abs(hash(path_or_glob)) % 10000}"


def collect_custom_images(path_or_glob):
    valid_ext = {'.png', '.jpg', '.jpeg'}
    img_exts  = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']

    if os.path.isdir(path_or_glob):
        paths = []
        for ext in img_exts:
            paths.extend(glob_module.glob(os.path.join(path_or_glob, ext)))
        return list(np.unique(paths))

    matched = glob_module.glob(path_or_glob, recursive=True)
    paths = []
    for p in matched:
        if os.path.isdir(p):
            for ext in img_exts:
                paths.extend(glob_module.glob(os.path.join(p, ext)))
        elif os.path.isfile(p) and os.path.splitext(p)[1].lower() in valid_ext:
            paths.append(p)
    return list(np.unique(paths))


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, input_size, norm_mean=None, norm_std=None,
                 num_repr_patches=const.PATCH_NUM_REPR):
        self.image_paths = image_paths
        self.labels      = labels
        self.P           = input_size if isinstance(input_size, int) else input_size[0]
        self.num_repr_patches = num_repr_patches
        self.transform   = make_patch_transform(norm_mean, norm_std)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        plist = repr_patches(img, self.P, self.num_repr_patches, const.PATCH_SEED)
        patches = torch.stack([self.transform(p).float() for p in plist])  # (N, 3, P, P)
        return patches, self.labels[idx]


def create_custom_dataloader(args, config):
    if args.custom_labels is None or len(args.custom_labels) != len(args.custom_dirs):
        raise ValueError(
            "--custom_labels must be provided and have the same length as --custom_dirs.")

    all_paths, all_group_ids, groups = [], [], []
    for gid, (d, lbl) in enumerate(zip(args.custom_dirs, args.custom_labels)):
        if lbl not in (0, 1):
            raise ValueError(f"--custom_labels must be 0 or 1, got {lbl} for dir '{d}'.")
        name  = _dir_name_from_path(d)
        paths = collect_custom_images(d)
        if not paths:
            print(f"[warn] No images found: {d}")
            continue
        groups.append({'name': name, 'expected_label': lbl, 'group_id': gid, 'n': len(paths)})
        all_paths.extend(paths)
        all_group_ids.extend([gid] * len(paths))
        print(f"  [{gid}] {name}  → {'real(0)' if lbl == 0 else 'fake(1)'}: {len(paths)} images")

    if not all_paths:
        raise ValueError("No images found in any of the provided custom dirs.")

    norm_mean, norm_std = const.get_norm_stats(config["backbone_name"])
    ds = CustomImageDataset(all_paths, all_group_ids, config["input_size"],
                            norm_mean=norm_mean, norm_std=norm_std)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=64, shuffle=False,
        num_workers=getattr(args, 'num_workers', 4), drop_last=False,
    )
    return loader, groups


def eval_custom(dataloader, model, groups, threshold_info):
    model.eval()
    preds_list, group_ids_list = [], []
    device = next(model.nf_flows[0].parameters()).device

    for data, gids in tqdm(dataloader, desc="Evaluating custom dirs"):
        data = data.to(device)                 # (B, N, 3, P, P)
        with torch.no_grad():
            ret = score_per_image(model, data)
        preds_list.append(ret["loss"].cpu())
        group_ids_list.append(gids)

    scores   = torch.cat(preds_list).numpy()
    group_ids = torch.cat(group_ids_list).numpy()

    if 'lof' in threshold_info:
        binary_preds = (threshold_info['lof'].predict(scores.reshape(-1, 1)) < 0).astype(int)
    elif 'l_threshold' in threshold_info and 'u_threshold' in threshold_info:
        binary_preds = ((scores < threshold_info['l_threshold']) |
                        (scores > threshold_info['u_threshold'])).astype(int)
    elif 'threshold' in threshold_info and threshold_info['threshold'] is not None:
        binary_preds = (scores > threshold_info['threshold']).astype(int)
    else:
        raise ValueError("threshold_info must contain thresholds or 'lof'.")

    print("\nCustom-dirs evaluation results")
    print("=" * 50)
    for g in groups:
        gid      = g['group_id']
        expected = g['expected_label']
        mask     = group_ids == gid
        if mask.sum() == 0:
            continue
        g_scores = scores[mask]
        g_preds  = binary_preds[mask]
        acc = (g_preds == expected).mean()
        print(f"  [{g['name']}]  expected={'real(0)' if expected == 0 else 'fake(1)'}  "
              f"N={mask.sum()}  Accuracy={acc:.4f}  "
              f"Score={g_scores.mean():.4f}±{g_scores.std():.4f}")
    print("=" * 50)


# ═══════════════════════════════════════════════════════════════════════════
# Main evaluation entry
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(args):
    config     = yaml.safe_load(open(args.config, "r"))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    model = build_model(config, args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.cuda()

    # ── Threshold ────────────────────────────────────────────────────────────
    threshold_info = None
    if os.path.exists(args.threshold_path):
        if args.threshold_path.endswith('.npz'):
            loaded = np.load(args.threshold_path, allow_pickle=True)
            threshold_info = {k: loaded[k] for k in loaded.files}
            for sk in ('threshold', 'l_threshold', 'u_threshold', 'mean', 'std'):
                if sk in threshold_info and isinstance(threshold_info[sk], np.ndarray):
                    if threshold_info[sk].size == 1:
                        threshold_info[sk] = float(threshold_info[sk])
        else:
            td = np.load(args.threshold_path, allow_pickle=True)
            threshold_info = td.item() if (isinstance(td, np.ndarray) and td.dtype == object) else {
                'mean': float(td[0]), 'std': float(td[1]),
                'l_threshold': float(td[0]) - 3 * float(td[1]),
                'u_threshold': float(td[0]) + 3 * float(td[1]),
            }
        print(f"Threshold loaded from {args.threshold_path}")
    else:
        print("Computing threshold on validation set...")
        val_dataloader = build_val_data_loader(args, config)
        threshold_info = compute_threshold(
            val_dataloader, model,
            use_lof=bool(getattr(args, 'use_lof', False)),
            contamination=args.contamination,
            alpha=args.alpha,
        )

    if getattr(args, 'use_lof', False) and os.path.exists(args.lof_checkpoint):
        threshold_info['lof'] = joblib.load(args.lof_checkpoint)
        print(f"Loaded LOF model from {args.lof_checkpoint}")

    # ── Custom-dirs mode ─────────────────────────────────────────────────────
    if args.custom_dirs is not None:
        print(f"\n{'='*60}\nCustom dirs evaluation\n{'='*60}")
        custom_loader, groups = create_custom_dataloader(args, config)
        eval_custom(custom_loader, model, groups, threshold_info)
        return

    # ── Standard eval with optional attacks ──────────────────────────────────
    attacks_configs = [
        ('none', {}),
        # ('jpeg', {'quality': 90}),
        # ('jpeg', {'quality': 80}),
        # ('jpeg', {'quality': 70}),
        # ('jpeg', {'quality': 60}),
        # ('jpeg', {'quality': 50}),
        # ('jpeg', {'quality': 40}),
        # ('jpeg', {'quality': 30}),
        # ('jpeg', {'quality': 20}),
        # ('gaussian_blur', {'kernel_size': 3, 'sigma': 1.0}),
        # ('gaussian_blur', {'kernel_size': 5, 'sigma': 2.0}),
        # ('rotation', {'angle': 30}),
        # ('rotation', {'angle': 180}),
        # ('gaussian_noise', {'mean': 0, 'std': 0.05}),
        # ('salt_pepper', {'amount': 0.02}),
        # ('resize', {'scale_factor': 0.5}),
        # ('horizontal_flip', {}),
        # ('random_crop', {'crop_ratio': 0.95}),
    ]

    for attack_type, attack_params in attacks_configs:
        print(f"\n{'='*60}")
        print(f"Attack: {attack_type} {attack_params}")
        print(f"{'='*60}\n")

        class Options:
            isTrain = False
            isVal   = False
            batch_size = 64
        opt = Options()
        opt.attack_type   = attack_type
        opt.attack_params = attack_params
        test_dataloader, class2idx = create_dataloader(args, config, opt)

        eval_once(test_dataloader, model,
                  class2idx=class2idx, threshold_info=threshold_info)


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
