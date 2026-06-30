import argparse
import contextlib
import io
import json
import os
import glob as glob_module

import joblib
import numpy as np
import torch
import yaml
from PIL import Image
from scipy.stats import norm
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm

from muflow import constants as const
from muflow import dataset
from muflow.gpu_utils import resolve_device
from muflow.patch_utils import make_patch_transform, repr_patches

from main import (
    eval_once,
    score_per_image,
    _collect_val_scores,
    _compute_class_metrics,
    _aggregate_by_family,
    _print_family_summary,
    build_model,
    compute_threshold,
    build_val_data_loader,
    build_test_data_loader,
    GANS, DM_OPEN, DM_CLOSED, MIX_2CLASS,
)


def parse_args():
    """Parse CLI args and merge them with the run's saved config.
    Returns: argparse.Namespace with run_config.yaml values and resolved paths.
    """
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
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="GPU to use (default: auto-select the one with most free memory).")
    parser.add_argument("--top_k", type=int, default=argparse.SUPPRESS,
                        help="override the run's top_k patch aggregation (inference-only); "
                             "recomputes the validation threshold to stay consistent. None/0 = all patches.")
    parser.add_argument("--recalibrate", action="store_true", default=False,
                        help="Run a full calibration sweep (threshold α∈{0.01,0.05,0.10} + LOF) "
                             "on the val set, pick the best by OOD accuracy, and overwrite the "
                             "saved calibration in run_dir. Always uses the clean test set.")

    args = parser.parse_args()
    args._top_k_override = hasattr(args, 'top_k')

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
    """Build a test DataLoader with an optional robustness attack.
    args: parsed CLI args.
    config: backbone config dict.
    opt: object carrying attack_type, attack_params and batch_size.
    Returns: (DataLoader, idx2class).
    """
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


def _dir_name_from_path(path_or_glob):
    """Derive a readable group name from a path or glob.
    path_or_glob: folder path or glob pattern.
    Returns: the last literal path component, or a hash-based fallback.
    """
    parts = path_or_glob.replace('\\', '/').rstrip('/').split('/')
    for part in reversed(parts):
        if part and '*' not in part and '?' not in part:
            return part
    return f"dir_{abs(hash(path_or_glob)) % 10000}"


def collect_custom_images(path_or_glob):
    """Collect image files from a folder or glob.
    path_or_glob: folder path or glob pattern.
    Returns: unique list of image file paths.
    """
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
        """
        image_paths: list of image file paths.
        labels: per-image group id / label.
        input_size: patch side.
        norm_mean: normalization mean.
        norm_std: normalization std.
        num_repr_patches: deterministic patches per image.
        """
        self.image_paths = image_paths
        self.labels      = labels
        self.P           = input_size if isinstance(input_size, int) else input_size[0]
        self.num_repr_patches = num_repr_patches
        self.transform   = make_patch_transform(norm_mean, norm_std)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Return one item.
        idx: sample index.
        Returns: ((N, 3, P, P) patches, label).
        """
        img = Image.open(self.image_paths[idx]).convert('RGB')
        plist = repr_patches(img, self.P, self.num_repr_patches, const.PATCH_SEED)
        patches = torch.stack([self.transform(p).float() for p in plist])
        return patches, self.labels[idx]


def create_custom_dataloader(args, config):
    """Build a DataLoader over user-provided custom dirs.
    args: parsed CLI args (custom_dirs, custom_labels).
    config: backbone config dict.
    Returns: (DataLoader, groups metadata list).
    """
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
    """Score custom dirs and print per-group accuracy.
    dataloader: DataLoader yielding (patches, group_id).
    model: FastFlow model.
    groups: group metadata list from create_custom_dataloader.
    threshold_info: calibration dict.
    """
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


SWEEP_CANDIDATES = [
    {'label': 'threshold  α=0.01', 'use_lof': False, 'alpha': 0.01, 'contamination': None},
    {'label': 'threshold  α=0.05', 'use_lof': False, 'alpha': 0.05, 'contamination': None},
    {'label': 'threshold  α=0.10', 'use_lof': False, 'alpha': 0.10, 'contamination': None},
    {'label': "LOF  (contamination='auto')", 'use_lof': True,  'alpha': None, 'contamination': 'auto'},
]


@contextlib.contextmanager
def _mute():
    """Context manager that suppresses stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _fit_calibrator(val_losses: np.ndarray, cand: dict, mu_patch=None, top_k=None) -> dict:
    """Fit one calibration candidate on validation scores.
    val_losses: per-image validation NLL scores.
    cand: candidate spec (use_lof, alpha, contamination).
    mu_patch: validation patch-mean to store.
    top_k: aggregation top_k to store.
    Returns: threshold_info dict.
    """
    mean, std = float(val_losses.mean()), float(val_losses.std())
    info = {'mean': mean, 'std': std, 'losses': val_losses, 'mu_patch': mu_patch, 'top_k': top_k}
    if cand['use_lof']:
        lof = LocalOutlierFactor(novelty=True, contamination=cand['contamination'], n_jobs=-1)
        lof.fit(val_losses.reshape(-1, 1))
        info['lof'] = lof
    else:
        z = norm.ppf(1 - cand['alpha'])
        info['l_threshold'] = mean - z * std
        info['u_threshold'] = mean + z * std
    return info


def _ood_acc(metrics_summary: dict) -> float:
    """Extract the selection metric from a metrics summary.
    metrics_summary: output of eval_once.
    Returns: mean OOD accuracy, or in_domain_real accuracy as fallback.
    """
    if 'vs_ood_real' in metrics_summary:
        return metrics_summary['vs_ood_real'].get('mean_acc', 0.0)
    return metrics_summary.get('in_domain_real', {}).get('mean_acc', 0.0)


def _recalibrate_sweep(model, args, config, class2idx, test_dataloader):
    """Try all calibration candidates, keep the best by OOD accuracy, persist it.
    model: FastFlow model.
    args: parsed CLI args.
    config: backbone config dict.
    class2idx: mapping class id -> name.
    test_dataloader: clean test DataLoader shared across candidates.
    Returns: winning threshold_info dict.
    """
    device = next(model.nf_flows[0].parameters()).device

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
    rows = []
    for cand in SWEEP_CANDIDATES:
        thr = _fit_calibrator(val_losses, cand, mu_patch=mu_patch, top_k=top_k)
        with _mute():
            _, _, _, metrics = eval_once(
                test_dataloader, model,
                class2idx=class2idx,
                threshold_info=thr,
                wandb_log=False,
            )
        rows.append((cand, thr, metrics, _ood_acc(metrics)))

    print(f"\n{'─'*64}")
    print(f"  {'Candidate':<36} {'OOD Acc':>9} {'Real Acc':>9}")
    print(f"{'─'*64}")
    for cand, _, metrics, ood_acc_val in rows:
        real_acc = metrics.get('in_domain_real', {}).get('mean_acc', 0.0)
        print(f"  {cand['label']:<36} {ood_acc_val:>9.4f} {real_acc:>9.4f}")
    print(f"{'─'*64}")

    best_cand, best_thr, best_metrics, best_ood = max(
        rows, key=lambda x: (x[3], 0 if not x[0]['use_lof'] else -1)
    )
    print(f"\n  Winner: {best_cand['label']}  (OOD Acc = {best_ood:.4f})\n")

    print("=" * 64)
    print("Full evaluation — winner calibration")
    print("=" * 64)
    eval_once(test_dataloader, model, class2idx=class2idx,
              threshold_info=best_thr, wandb_log=False)

    _persist_calibration(best_thr, best_cand, best_metrics, args)

    return best_thr


def _persist_calibration(threshold_info: dict, cand: dict,
                         metrics_summary: dict, args) -> None:
    """Persist the winning calibration to the run folder.
    threshold_info: winning calibration dict.
    cand: winning candidate spec.
    metrics_summary: metrics of the winner.
    args: parsed CLI args (provides run_dir and file paths).
    """
    run_dir = args.run_dir

    save_keys = ('l_threshold', 'u_threshold', 'threshold', 'losses', 'mean', 'std', 'mu_patch', 'top_k')
    np.savez(args.threshold_path,
             **{k: threshold_info[k] for k in save_keys
                if k in threshold_info and threshold_info[k] is not None})
    print(f"\n  Saved thresholds     → {args.threshold_path}")

    if 'lof' in threshold_info:
        joblib.dump(threshold_info['lof'], args.lof_checkpoint)
        print(f"  Saved LOF model      → {args.lof_checkpoint}")
    elif os.path.exists(args.lof_checkpoint):
        os.remove(args.lof_checkpoint)

    metrics_path = os.path.join(run_dir, 'best_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            payload = json.load(f)
        payload['metrics']     = metrics_summary
        payload['calibration'] = {
            'method':        'lof' if cand['use_lof'] else 'gaussian',
            'alpha':         cand['alpha'],
            'contamination': cand['contamination'],
        }
        with open(metrics_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"  Updated best_metrics.json")

    cfg_path = os.path.join(run_dir, 'run_config.yaml')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg['use_lof']       = cand['use_lof']
        cfg['alpha']         = cand['alpha']
        cfg['contamination'] = cand['contamination']
        with open(cfg_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"  Updated run_config.yaml")


def evaluate(args):
    """Run evaluation for a training run (sweep, custom dirs, or standard).
    args: parsed CLI args.
    """
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

    if args.custom_dirs is not None:
        print(f"\n{'='*60}\nCustom dirs evaluation\n{'='*60}")
        custom_loader, groups = create_custom_dataloader(args, config)
        eval_custom(custom_loader, model, groups, threshold_info)
        return

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
