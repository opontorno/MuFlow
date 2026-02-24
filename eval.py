import argparse
import os
import torch
import yaml
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

from muflow import constants as const
from muflow import dataset
from muflow import model as fastflow

from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from joblib import Parallel, delayed
import time

GANS = ['StyleGAN', 'StyleGAN2', 'StyleGAN3', 'STARGAN', 'AttGAN', 'GDWCT']
DM_OPEN = ['Flux.1', 'Stable DIffusion 3.5', 'Stable Diffusion XL', 'Stable Cascade', 'Stable Diffusion Attend and Excite']
DM_CLOSED = ['Dall-E 3', 'Midjourney', 'Starry AI', 'Deep AI', 'Hotpot AI', 'Nvidia Sana PAG', 'Tencent Hunyuan', 'Flux.1.1 Pro']

MIX_2CLASS = ['StyleGAN', 'StyleGAN2', 'Stable DIffusion 3.5', 'Flux.1.1 Pro']




def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a MuFlow run. All settings are loaded automatically "
                    "from run_dir/run_config.yaml. Only eval-specific options are needed.")
    parser.add_argument("run_dir", type=str,
                        help="Directory of a training run (must contain best.pt, "
                             "thresholds.npz and run_config.yaml).")
    parser.add_argument("--test", type=str, default=None,
                        help="Optional test-set name override (passed to Dataset).")
    parser.add_argument("--on_celeba", action='store_true',
                        help="Use CelebA-HQ test dataloader (adds class 99 OOD Real).")
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    # Load run_config.yaml and attach every key directly onto args
    run_cfg_path = os.path.join(args.run_dir, 'run_config.yaml')
    if not os.path.exists(run_cfg_path):
        parser.error(f"run_config.yaml not found in {args.run_dir}. "
                     "Make sure the run has saved at least one best checkpoint.")
    saved = yaml.safe_load(open(run_cfg_path))
    for k, v in saved.items():
        if not hasattr(args, k):   # don't overwrite CLI args (run_dir, test, on_celeba, num_workers)
            setattr(args, k, v)

    # Resolve file paths relative to run_dir
    args.checkpoint    = os.path.join(args.run_dir, 'best.pt')
    args.threshold_path = os.path.join(args.run_dir, 'thresholds.npz')
    args.lof_checkpoint = os.path.join(args.run_dir, 'lof_model.pkl')

    print(f"[eval] Run dir  : {args.run_dir}")
    print(f"[eval] Config   : {args.config}")
    print(f"[eval] std_weight={args.std_weight}  alpha={args.alpha}")
    return args


def build_val_data_loader(args, config):
    """Build validation dataloader (real images only, same split used during training)."""
    val_dataset = dataset.Dataset(
        dataset_name=args.data,
        reals_name=args.reals,
        input_size=config["input_size"],
        is_train=True,
        is_val=True,
        use_fourier=True if args.use_fourier == 1 else False,
        use_augs=False,
    ).create_dataset()

    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )


def compute_threshold(val_dataloader, model, model_type="FastFlow",
                      use_lof=False, contamination='auto',
                      use_percentile=False, percentile=95,
                      alpha=0.1, std_weight=0.0):
    """
    Compute anomaly-detection threshold from the validation set (real images).
    Mirrors compute_threshold() in main.py exactly.
    score = loss + std_weight * z_std
    """
    model.eval()
    loss_values = []
    z_std_values = []
    device = next(model.nf_flows[0].parameters()).device

    for batch in val_dataloader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            data, _ = batch
        else:
            data = batch
        data = data.to(device)
        with torch.no_grad():
            ret = model(data) if model_type == "FastFlow" else model(data, eval_mode=True)
            if model_type == "FastFlow":
                loss_values.append(ret["loss"].cpu())
                z_std_values.append(ret["z_std"].cpu())

    if not loss_values:
        raise ValueError("No validation samples found. Check val dataloader.")

    losses = torch.cat(loss_values).numpy()
    z_stds = torch.cat(z_std_values).numpy()
    scores = losses + std_weight * z_stds
    mean, std = float(scores.mean()), float(scores.std())

    result = {
        'threshold': None,
        'mean': mean,
        'std': std,
        'losses': losses,
        'scores': scores,
        'std_weight': std_weight,
    }

    if use_lof:
        lof = LocalOutlierFactor(novelty=True, contamination=contamination, n_jobs=-1)
        lof.fit(scores.reshape(-1, 1))
        result['lof'] = lof
    elif use_percentile:
        result['threshold'] = float(np.percentile(scores, percentile))
    else:
        z = norm.ppf(1 - alpha)
        result['l_threshold'] = float(mean - z * std)
        result['u_threshold'] = float(mean + z * std)

    l = result.get('l_threshold')
    u = result.get('u_threshold')
    bounds = f"[{l:.4f}, {u:.4f}]" if l is not None else "(LOF / percentile)"
    print(f"Threshold computation done.  mean={mean:.4f}  std={std:.4f}  {bounds}")
    return result


def create_dataloader(args, config, opt):

    attack_type = getattr(opt, 'attack_type', 'none')
    attack_params = getattr(opt, 'attack_params', {})

    # The data loading code remains the same
    test_dataset = dataset.Dataset(
        dataset_name=args.data,
        reals_name=args.reals,
        test_name=args.test,
        input_size=config["input_size"],
        is_train=False,
        is_val=False,
        use_fourier=True if args.use_fourier == 1 else False,
        use_augs=True if args.use_augs == 1 else False,
        attack_type=attack_type,
        attack_params=attack_params
    ).create_dataset()

    class_to_idx_ = test_dataset.class_to_idx

    num_workers = getattr(args, 'num_workers', 4)
    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return data_loader, {v: k for k, v in class_to_idx_.items()}


def create_dataloader_w_celeba(args, config, opt):

    attack_type = getattr(opt, 'attack_type', 'none')
    attack_params = getattr(opt, 'attack_params', {})

    test_dataset = dataset.Dataset_celeba(
        dataset_name=args.data,
        reals_name=args.reals,
        test_name=args.test,
        input_size=config["input_size"],
        is_train=False,
        is_val=False,
        use_fourier=True if args.use_fourier == 1 else False,
        use_augs=True if args.use_augs == 1 else False,
        attack_type=attack_type,
        attack_params=attack_params
    ).create_dataset()

    class_to_idx_ = test_dataset.class_to_idx

    num_workers = getattr(args, 'num_workers', 4)
    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return data_loader, {v: k for k, v in class_to_idx_.items()}


def build_model(config, model_type, args):
    out_indices = config.get("out_indices", [1, 2, 3])
    out_indices_str = str(out_indices)
    pooling_type = getattr(args, 'pooling_type', config.get('pooling_type', 'mean'))
    n_components = getattr(args, 'n_components', config.get('gmm_n_components', 1))

    if args.use_fourier == 1:
        gmm_parameters = f"{const.WORKING_DIR}/parameters/{n_components}-gmm_parameters_{config['backbone_name']}_indices_{out_indices_str}_fourier_{args.reals}_{config['input_size']}_{pooling_type}.npy"
    else:
        gmm_parameters = f"{const.WORKING_DIR}/parameters/{n_components}-gmm_parameters_{config['backbone_name']}_indices_{out_indices_str}_{args.reals}_{config['input_size']}_{pooling_type}.npy"

    gmm_values = np.load(gmm_parameters, allow_pickle=True).item()
    print(f"Loading GMM parameters from {gmm_parameters}")

    if model_type == "FastFlow":
        model = fastflow.FastFlow(
            backbone_name=config["backbone_name"],
            flow_steps=config["flow_step"],
            input_size=config["input_size"],
            conv3x3_only=config["conv3x3_only"],
            hidden_ratio=config["hidden_ratio"],
            gmm_values=gmm_values,
            in_channels=1 if args.use_fourier == 1 else 3,
            backbone_weights=args.backbone_weights if hasattr(args, 'backbone_weights') and args.backbone_weights else None,
            out_indices=out_indices,
            pooling_type=pooling_type,
            use_adversarial=args.use_adversarial == 1 if hasattr(args, 'use_adversarial') else False,
            noise_differentiable=args.noise_differentiable == 1 if hasattr(args, 'noise_differentiable') else False,
            use_proj_layer=args.use_proj_layer == 1 if hasattr(args, 'use_proj_layer') else False,
            proj_hidden_ratio=args.proj_hidden_ratio if hasattr(args, 'proj_hidden_ratio') else 0.5,
            use_autoencoder=args.use_autoencoder == 1 if hasattr(args, 'use_autoencoder') else False,
            ae_hidden_ratio=args.ae_hidden_ratio if hasattr(args, 'ae_hidden_ratio') else 0.5,
        )
        print(f"Model A.D. Param#: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


def _compute_class_metrics(c, class_masks, labels, preds, preds_, class2idx, y_true_0_base, y_pred_0_base):
    """Compute balanced binary metrics for one fake class vs the real baseline."""
    mask_c = class_masks[c]
    y_true_c = labels[mask_c]
    y_pred_c = preds[mask_c]
    preds_c = preds_[mask_c]

    min_len = min(len(y_true_c), len(y_true_0_base))
    if min_len == 0:
        return None

    class_name = class2idx[c] if class2idx else str(c)
    rng = np.random.RandomState(42 + c)
    idx = rng.permutation(len(y_true_0_base))
    y_pred_0_shuffled = y_pred_0_base[idx]

    y_pred_balanced = np.concatenate([y_pred_c[:min_len], y_pred_0_shuffled[:min_len]])
    y_true_binary = np.concatenate([np.ones(min_len, dtype=np.int8), np.zeros(min_len, dtype=np.int8)])

    return {
        'class_id': c,
        'class_name': class_name,
        'min_len': min_len,
        'loss_mean': float(preds_c.mean()),
        'loss_std': float(preds_c.std()),
        'accuracy': accuracy_score(y_true_binary, y_pred_balanced),
        'ap': average_precision_score(y_true_binary, y_pred_balanced),
        'roc': roc_auc_score(y_true_binary, y_pred_balanced),
    }

def eval_once(dataloader, model, model_type="FastFlow", class2idx=None, threshold_info=None, std_weight=0.0):
    model.eval()
    labels_list = []
    preds_list = []
    device = next(model.nf_flows[0].parameters()).device

    for data, targets in dataloader:
        data, targets = data.to(device), targets.to(device)
        with torch.no_grad():
            ret = model(data) if model_type == "FastFlow" else model(data, eval_mode=True)
        # Combined anomaly score: loss + std_weight * z_std
        score = ret["loss"].cpu() + std_weight * ret["z_std"].cpu()
        preds_list.append(score)
        labels_list.append(targets.cpu())

    preds_ = torch.cat(preds_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    print("Testing done")

    # --- Apply threshold ---
    if threshold_info is None:
        raise ValueError("threshold_info must be provided.")

    if 'lof' in threshold_info:
        preds = (threshold_info['lof'].predict(preds_.reshape(-1, 1)) < 0).astype(int)
    elif 'l_threshold' in threshold_info and 'u_threshold' in threshold_info:
        preds = ((preds_ < threshold_info['l_threshold']) | (preds_ > threshold_info['u_threshold'])).astype(int)
    elif 'threshold' in threshold_info and threshold_info['threshold'] is not None:
        # Legacy single-threshold (only upper bound)
        preds = (preds_ > threshold_info['threshold']).astype(int)
    else:
        raise ValueError("threshold_info must contain 'l_threshold'/'u_threshold', 'threshold', or 'lof'.")

    classes = np.unique(labels)
    class_masks = {c: labels == c for c in classes}

    # --- Real class (0) ---
    mask_0 = class_masks[0]
    preds_0_scores = preds_[mask_0]
    acc_real = accuracy_score(np.zeros(mask_0.sum(), dtype=np.int8), preds[mask_0])
    print(f"Accuracy Real (class 0): {acc_real:.4f}  "
          f"Score = {preds_0_scores.mean():.4f}\u00b1{preds_0_scores.std():.4f}")

    # --- OOD Real (class 99, e.g. CelebA-HQ) ---
    if 99 in class_masks:
        mask_99 = class_masks[99]
        preds_99 = preds_[mask_99]
        acc_ood = accuracy_score(np.zeros(mask_99.sum(), dtype=np.int8), preds[mask_99])
        ood_name = class2idx.get(99, "OOD_Real") if class2idx else "OOD_Real"
        print("-" * 30)
        print(f"Accuracy OOD Real ({ood_name}): {acc_ood:.4f}  "
              f"Score = {preds_99.mean():.4f}\u00b1{preds_99.std():.4f}")
        print("-" * 30)

    # --- Per-class metrics (balanced vs Real baseline) ---
    classes_to_process = [c for c in classes if c != 0 and c != 99]
    y_true_0 = labels[mask_0]
    y_pred_0 = preds[mask_0]

    print("\nPer-class metrics (vs Real baseline):")
    print("=" * 30)

    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(_compute_class_metrics)(
            c, class_masks, labels, preds, preds_, class2idx, y_true_0, y_pred_0
        ) for c in classes_to_process
    )

    aps, accs, rocs = [], [], []
    aps_gan, accs_gan = [], []
    aps_dmo, accs_dmo = [], []
    aps_dmc, accs_dmc = [], []
    aps_mix, accs_mix = [], []

    for result in results:
        if result is None:
            continue
        cname = result['class_name']
        print(f"  > {cname} (N={result['min_len']*2}):  "
              f"Acc={result['accuracy']:.4f}  AP={result['ap']:.4f}  "
              f"ROC={result['roc']:.4f}  Score={result['loss_mean']:.4f}\u00b1{result['loss_std']:.4f}")
        print("-" * 30)

        aps.append(result['ap'])
        accs.append(result['accuracy'])
        rocs.append(result['roc'])

        if cname in GANS:
            aps_gan.append(result['ap']); accs_gan.append(result['accuracy'])
        if cname in DM_OPEN:
            aps_dmo.append(result['ap']); accs_dmo.append(result['accuracy'])
        if cname in DM_CLOSED:
            aps_dmc.append(result['ap']); accs_dmc.append(result['accuracy'])
        if cname in MIX_2CLASS:
            aps_mix.append(result['ap']); accs_mix.append(result['accuracy'])

    print("=" * 30)
    print(f"Mean Accuracy : {np.mean(accs):.4f}")
    print(f"Mean AP       : {np.mean(aps):.4f}")
    print(f"Mean ROC AUC  : {np.mean(rocs):.4f}")
    print("\n--- Results by family ---")
    if aps_gan:
        print(f"  GANs     :  Acc={np.mean(accs_gan):.4f}  AP={np.mean(aps_gan):.4f}")
    if aps_dmo:
        print(f"  DM-Open  :  Acc={np.mean(accs_dmo):.4f}  AP={np.mean(aps_dmo):.4f}")
    if aps_dmc:
        print(f"  DM-Closed:  Acc={np.mean(accs_dmc):.4f}  AP={np.mean(aps_dmc):.4f}")
    if aps_mix:
        print(f"  Mix      :  Acc={np.mean(accs_mix):.4f}  AP={np.mean(aps_mix):.4f}")
    print("=" * 30 + "\n")


def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    model = build_model(config, args.model_type, args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.cuda()

    # ----------------------------------------------------------------
    # Threshold: compute live on val set (same as main.py), unless an
    # explicit --threshold_path override is given.
    # ----------------------------------------------------------------
    threshold_info = None
    std_weight = args.std_weight  # default; overridden below if loaded from file

    if os.path.exists(args.threshold_path):
        # --- Load pre-computed thresholds from file ---
        threshold_path = args.threshold_path
        if threshold_path.endswith('.npz'):
            loaded = np.load(threshold_path, allow_pickle=True)
            threshold_info = {key: loaded[key] for key in loaded.files}
            for scalar_key in ('threshold', 'l_threshold', 'u_threshold', 'mean', 'std', 'std_weight'):
                if scalar_key in threshold_info and isinstance(threshold_info[scalar_key], np.ndarray):
                    if threshold_info[scalar_key].size == 1:
                        threshold_info[scalar_key] = float(threshold_info[scalar_key])
        else:
            threshold_data = np.load(threshold_path, allow_pickle=True)
            if isinstance(threshold_data, np.ndarray) and threshold_data.dtype == object:
                threshold_info = threshold_data.item()
            else:
                threshold_info = {
                    'mean': float(threshold_data[0]),
                    'std': float(threshold_data[1]),
                    'l_threshold': float(threshold_data[0]) - 3 * float(threshold_data[1]),
                    'u_threshold': float(threshold_data[0]) + 3 * float(threshold_data[1]),
                }
        # Prefer std_weight stored in the file over the CLI value
        std_weight = float(threshold_info.get('std_weight', args.std_weight))
        print(f"Threshold loaded from {threshold_path}  (std_weight={std_weight})")
    else:
        # --- Primary path: compute on validation set ---
        print("Computing threshold on validation set...")
        val_dataloader = build_val_data_loader(args, config)
        threshold_info = compute_threshold(
            val_dataloader, model,
            model_type=args.model_type,
            use_lof=args.use_lof == 1,
            contamination=args.contamination,
            use_percentile=args.use_percentile == 1,
            percentile=args.percentile,
            alpha=args.alpha,
            std_weight=args.std_weight,
        )
        std_weight = args.std_weight

    # --- LOF: load from file if available ---
    lof_checkpoint = args.lof_checkpoint
    if args.use_lof == 1 and os.path.exists(lof_checkpoint):
        threshold_info['lof'] = joblib.load(lof_checkpoint)
        print(f"Loaded LOF model from {lof_checkpoint}")

    attacks_configs = [
        ('none', {}),
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
    ]

    for attack_type, attack_params in attacks_configs:
        class Options:
            isTrain = False
            isVal = False
            batch_size = 64

        opt = Options()
        opt.attack_type = attack_type
        opt.attack_params = attack_params

        print(f"\n{'='*60}")
        print(f"Attack: {attack_type} {attack_params}")
        print(f"{'='*60}\n")

        if not args.on_celeba:
            test_dataloader, class2idx = create_dataloader(args, config, opt)
        else:
            test_dataloader, class2idx = create_dataloader_w_celeba(args, config, opt)

        eval_once(test_dataloader, model, model_type=args.model_type,
                  class2idx=class2idx, threshold_info=threshold_info, std_weight=std_weight)


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
