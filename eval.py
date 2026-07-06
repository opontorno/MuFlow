import argparse
import json
import os
import glob as glob_module

import joblib
import numpy as np
import torch
import yaml
from PIL import Image
from scipy.stats import norm
from tqdm import tqdm

from muflow import constants as const
from muflow import dataset
from muflow import calibration
from muflow.gpu_utils import resolve_device
from muflow.patch_utils import make_patch_transform, repr_patches

import config as cfg

from main import (
    eval_once,
    score_per_image,
    _collect_val_scores,
    build_model,
    compute_threshold,
    build_val_data_loader,
    build_test_data_loader,
)


ROBUSTNESS_ATTACKS = [
    ('jpeg', {'quality': 90}),
    ('jpeg', {'quality': 80}),
    ('jpeg', {'quality': 70}),
    ('jpeg', {'quality': 60}),
    ('jpeg', {'quality': 50}),
    ('jpeg', {'quality': 40}),
    ('jpeg', {'quality': 30}),
    ('jpeg', {'quality': 20}),
    ('gaussian_blur', {'kernel_size': 3, 'sigma': 1.0}),
    ('gaussian_blur', {'kernel_size': 5, 'sigma': 2.0}),
    ('rotation', {'angle': 30}),
    ('rotation', {'angle': 180}),
    ('gaussian_noise', {'mean': 0, 'std': 0.05}),
    ('salt_pepper', {'amount': 0.02}),
    ('resize', {'scale_factor': 0.5}),
    ('horizontal_flip', {}),
    ('random_crop', {'crop_ratio': 0.95}),
]


def parse_args():
    """Parse CLI args and merge them with the run's saved config."""
    parser = argparse.ArgumentParser(
        description="Evaluate a MuFlow run; settings are loaded from run_dir/run_config.yaml.")
    parser.add_argument("--run_dir", type=str, required=True, help="training run directory")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--custom_dirs", type=str, nargs='+', default=None,
                        help="folders or globs of images to evaluate; one --custom_labels each")
    parser.add_argument("--custom_labels", type=int, nargs='+', default=None,
                        help="label per --custom_dirs entry: 0=real, 1=fake")
    parser.add_argument("--gpu_id", type=int, default=None, help="GPU id; None auto-selects the freest")
    parser.add_argument("--top_k", type=int, default=argparse.SUPPRESS,
                        help="override the run's top_k aggregation and recompute the threshold")
    parser.add_argument("--alpha", type=float, default=argparse.SUPPRESS,
                        help="override the threshold with a gaussian band at this false-positive rate")
    parser.add_argument("--recalibrate", action="store_true", default=False,
                        help="sweep calibration candidates and persist the best by real accuracy")
    parser.add_argument("--robustness", action="store_true", default=False,
                        help="also evaluate under the robustness degradations")

    args = parser.parse_args()
    args._top_k_override = hasattr(args, 'top_k')
    args._alpha_override = hasattr(args, 'alpha')

    run_cfg_path = os.path.join(args.run_dir, 'run_config.yaml')
    if not os.path.exists(run_cfg_path):
        parser.error(f"run_config.yaml not found in {args.run_dir}. "
                     "Make sure the run has saved at least one best checkpoint.")
    saved = yaml.safe_load(open(run_cfg_path))
    for k, v in saved.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    args.checkpoint     = os.path.join(args.run_dir, 'best.pt')
    args.threshold_path = os.path.join(args.run_dir, 'thresholds.npz')
    args.lof_checkpoint = os.path.join(args.run_dir, 'lof_model.pkl')

    print(f"[eval] Run dir     : {args.run_dir}")
    print(f"[eval] Config      : {args.config}")
    print(f"[eval] recalibrate : {args.recalibrate}")
    if not args.recalibrate:
        print(f"[eval] alpha       : {getattr(args, 'alpha', 'N/A')}")
        print(f"[eval] use_lof     : {getattr(args, 'use_lof', False)}")
    return args


def create_dataloader(args, config, opt):
    """Build a test DataLoader with an optional robustness attack."""
    attack_type   = getattr(opt, 'attack_type', 'none')
    attack_params = getattr(opt, 'attack_params', {})

    norm_mean, norm_std = const.get_norm_stats(config["backbone_name"])
    test_dataset = dataset.Dataset(
        real_paths=cfg.PATH_REAL_OOD if cfg.PATH_REAL_OOD else cfg.PATH_REAL,
        fake_paths=cfg.PATH_FAKE,
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


def _dir_name_from_path(path_or_glob):
    """Derive a readable group name from a path or glob."""
    parts = path_or_glob.replace('\\', '/').rstrip('/').split('/')
    for part in reversed(parts):
        if part and '*' not in part and '?' not in part:
            return part
    return f"dir_{abs(hash(path_or_glob)) % 10000}"


def collect_custom_images(path_or_glob):
    """Collect image files from a folder or glob."""
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
        """Return one item."""
        img = Image.open(self.image_paths[idx]).convert('RGB')
        plist = repr_patches(img, self.P, self.num_repr_patches, const.PATCH_SEED)
        patches = torch.stack([self.transform(p).float() for p in plist])
        return patches, self.labels[idx]


def create_custom_dataloader(args, config):
    """Build a DataLoader over user-provided custom dirs."""
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
    """Score custom dirs and print per-group accuracy."""
    model.eval()
    preds_list, group_ids_list = [], []
    device = next(model.nf_flows[0].parameters()).device

    ti = threshold_info if isinstance(threshold_info, dict) else {}
    mu_patch = ti.get('mu_patch')
    if isinstance(mu_patch, np.ndarray):
        mu_patch = float(mu_patch)
    top_k = ti.get('top_k')
    if isinstance(top_k, np.ndarray):
        top_k = int(top_k)

    for data, gids in tqdm(dataloader, desc="Evaluating custom dirs"):
        data = data.to(device)
        with torch.no_grad():
            ret = score_per_image(model, data, top_k=top_k, mu_patch=mu_patch)
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


def _maybe_promote_calibration(threshold_info: dict, metrics_summary: dict, args,
                               preds=None, labels=None) -> None:
    """Compare an eval-time calibration override against the saved champion, promote if better."""
    if not (getattr(args, '_alpha_override', False) or getattr(args, '_top_k_override', False)):
        return

    new_acc = calibration.real_acc(metrics_summary)
    canonical_acc = calibration.load_canonical_accuracy(args.run_dir)
    print(f"\n[champion] Real Acc: this={new_acc:.4f}  canonical={canonical_acc:.4f}")

    if new_acc > canonical_acc:
        cand = {
            'use_lof':       'lof' in threshold_info,
            'alpha':         float(getattr(args, 'alpha', 0.1)),
            'contamination': getattr(args, 'contamination', 'auto'),
        }
        calibration.persist_calibration(
            threshold_info, cand, metrics_summary,
            run_dir=args.run_dir, threshold_path=args.threshold_path, lof_checkpoint=args.lof_checkpoint,
            preds=preds, labels=labels)
        print(f"[champion] New champion calibration!  {new_acc:.4f} > {canonical_acc:.4f}")
    else:
        print(f"[champion] No promotion. Canonical remains at {canonical_acc:.4f}")


def _recalibrate_sweep(model, args, config, class2idx, test_dataloader):
    """Try all calibration candidates, keep the best by real-baseline accuracy, persist it."""
    print("\n" + "=" * 64)
    print("Recalibration sweep")
    print("=" * 64)
    print("Collecting val NLL scores...", flush=True)
    val_loader = build_val_data_loader(args, config)
    top_k = getattr(args, 'top_k', None)
    val_losses, mu_patch = _collect_val_scores(model, val_loader, top_k)
    print(f"  {len(val_losses)} real images — "
          f"mean={val_losses.mean():.4f}  std={val_losses.std():.4f}")

    print("\nSweeping calibration candidates (clean test set)...")
    best_cand, best_thr, best_metrics, best_real, best_preds, best_labels, rows = calibration.run_sweep(
        model, val_losses, mu_patch, top_k, test_dataloader, class2idx, eval_once)
    calibration.print_sweep_table(rows)
    print(f"\n  Winner: {best_cand['label']}  (Real Acc = {best_real:.4f})\n")

    print("=" * 64)
    print("Full evaluation — winner calibration")
    print("=" * 64)
    eval_once(test_dataloader, model, class2idx=class2idx,
              threshold_info=best_thr, wandb_log=False)

    calibration.persist_calibration(
        best_thr, best_cand, best_metrics,
        run_dir=args.run_dir, threshold_path=args.threshold_path, lof_checkpoint=args.lof_checkpoint,
        preds=best_preds, labels=best_labels)

    return best_thr


def evaluate(args):
    """Run evaluation for a training run (sweep, custom dirs, or standard)."""
    config     = yaml.safe_load(open(args.config, "r"))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    model = build_model(config, args)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = resolve_device(getattr(args, 'gpu_id', None))
    model.to(device)

    if args.recalibrate:
        test_dataloader, class2idx = build_test_data_loader(args, config)
        _recalibrate_sweep(model, args, config, class2idx, test_dataloader)
        return

    if getattr(args, '_alpha_override', False):
        args.use_lof = False

    threshold_info = None
    if os.path.exists(args.threshold_path):
        if args.threshold_path.endswith('.npz'):
            loaded = np.load(args.threshold_path, allow_pickle=True)
            threshold_info = {k: loaded[k] for k in loaded.files}
            for sk in ('threshold', 'l_threshold', 'u_threshold', 'mean', 'std', 'mu_patch'):
                if sk in threshold_info and isinstance(threshold_info[sk], np.ndarray):
                    if threshold_info[sk].size == 1:
                        threshold_info[sk] = float(threshold_info[sk])
            if 'top_k' in threshold_info and isinstance(threshold_info['top_k'], np.ndarray):
                threshold_info['top_k'] = int(threshold_info['top_k'])
        else:
            td = np.load(args.threshold_path, allow_pickle=True)
            threshold_info = td.item() if (isinstance(td, np.ndarray) and td.dtype == object) else {
                'mean': float(td[0]), 'std': float(td[1]),
                'l_threshold': float(td[0]) - 3 * float(td[1]),
                'u_threshold': float(td[0]) + 3 * float(td[1]),
            }
        print(f"Threshold loaded from {args.threshold_path}")

        if getattr(args, '_top_k_override', False) and args.top_k != threshold_info.get('top_k'):
            print(f"[eval] --top_k override ({threshold_info.get('top_k')} → {args.top_k}); "
                  "recomputing threshold on validation set...")
            val_dataloader = build_val_data_loader(args, config)
            threshold_info = compute_threshold(
                val_dataloader, model,
                use_lof=bool(getattr(args, 'use_lof', False)),
                contamination=getattr(args, 'contamination', 'auto'),
                alpha=getattr(args, 'alpha', 0.1),
                top_k=args.top_k,
            )
    else:
        print("Computing threshold on validation set...")
        val_dataloader = build_val_data_loader(args, config)
        threshold_info = compute_threshold(
            val_dataloader, model,
            use_lof=bool(getattr(args, 'use_lof', False)),
            contamination=getattr(args, 'contamination', 'auto'),
            alpha=getattr(args, 'alpha', 0.1),
            top_k=getattr(args, 'top_k', None),
        )

    if getattr(args, 'use_lof', False) and os.path.exists(args.lof_checkpoint):
        threshold_info['lof'] = joblib.load(args.lof_checkpoint)
        print(f"Loaded LOF model from {args.lof_checkpoint}")

    if getattr(args, '_alpha_override', False):
        z = norm.ppf(1 - args.alpha)
        threshold_info.pop('lof', None)
        threshold_info['l_threshold'] = threshold_info['mean'] - z * threshold_info['std']
        threshold_info['u_threshold'] = threshold_info['mean'] + z * threshold_info['std']
        print(f"[eval] --alpha override → gaussian band recomputed with alpha={args.alpha} "
              f"(l={threshold_info['l_threshold']:.4f}, u={threshold_info['u_threshold']:.4f})")

    if args.custom_dirs is not None:
        print(f"\n{'='*60}\nCustom dirs evaluation\n{'='*60}")
        custom_loader, groups = create_custom_dataloader(args, config)
        eval_custom(custom_loader, model, groups, threshold_info)
        return

    attacks_configs = [('none', {})]
    if getattr(args, 'robustness', False):
        attacks_configs += ROBUSTNESS_ATTACKS

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

        _, preds, labels, metrics_summary = eval_once(
            test_dataloader, model, class2idx=class2idx, threshold_info=threshold_info)

        if attack_type == 'none':
            _maybe_promote_calibration(threshold_info, metrics_summary, args, preds, labels)


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
