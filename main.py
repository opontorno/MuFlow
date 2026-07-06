import argparse
from pprint import pprint
import os
import json
import shutil
import subprocess
import sys
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import wandb
import joblib

from muflow import constants as const
from muflow import dataset
from muflow import model as fastflow
from muflow import utils
from muflow import calibration
from muflow.gpu_utils import resolve_device

import config as cfg

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, roc_curve
from sklearn.neighbors import LocalOutlierFactor
from joblib import Parallel, delayed


def _aggregate_by_family(results):
    """Group per-class result dicts by generator family."""
    fam = {'all': [], 'gan': [], 'dm_open': [], 'dm_closed': [], 'mix': []}
    for r in results:
        if r is None:
            continue
        n = r['class_name']
        fam['all'].append(r)
        if n in cfg.GANS:       fam['gan'].append(r)
        if n in cfg.DM_OPEN:    fam['dm_open'].append(r)
        if n in cfg.DM_CLOSED:  fam['dm_closed'].append(r)
        if n in cfg.MIX_2CLASS: fam['mix'].append(r)
    return fam


def _print_family_summary(fam):
    """Print mean metrics overall and per family."""
    keys = ['accuracy', 'acc_oracle', 'ap', 'roc']

    def _m(rs, k): return np.mean([r[k] for r in rs])

    if not fam['all']:
        return
    print(f"Mean Accuracy : {_m(fam['all'], 'accuracy'):.4f}"
          f" (oracle={_m(fam['all'], 'acc_oracle'):.4f})")
    print(f"Mean AP       : {_m(fam['all'], 'ap'):.4f}")
    print(f"Mean ROC AUC  : {_m(fam['all'], 'roc'):.4f}")
    print("--- by family ---")
    for label, key in [('GANs', 'gan'), ('DM-Open', 'dm_open'),
                       ('DM-Closed', 'dm_closed'), ('Mix', 'mix')]:
        if fam[key]:
            print(f"  {label:<10}: Acc={_m(fam[key], 'accuracy'):.4f}"
                  f" (oracle={_m(fam[key], 'acc_oracle'):.4f})"
                  f"  AP={_m(fam[key], 'ap'):.4f}"
                  f"  ROC={_m(fam[key], 'roc'):.4f}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to the model config YAML")
    parser.add_argument("--checkpoint", type=str, help="checkpoint to resume from")

    parser.add_argument('--wandb', default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_entity', type=str, default='orazio-mattia')
    parser.add_argument('--wandb_project', type=str, default='MuFlow')

    parser.add_argument('--num_train_patches', type=int, default=const.PATCH_NUM_TRAIN,
                        help="random patches per training image")
    parser.add_argument('--num_repr_patches', type=int, default=const.PATCH_NUM_REPR,
                        help="patches per image for the inference representation")
    parser.add_argument('--top_k', type=int, default=None,
                        help="signed-mean over the top_k most deviant patches; None = all")
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--backbone_weights', type=str, help="backbone weights to load")
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--debug', action='store_true', help="reduced dataset for quick runs")
    parser.add_argument('--gpu_id', type=int, default=None, help="GPU id; None auto-selects the freest")

    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'sgd'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--scheduler', action='store_true', default=True)
    parser.add_argument('--lr_decay', type=float, default=0.7)
    parser.add_argument('--lr_patience', type=int, default=35)

    parser.add_argument('--alpha', type=float, default=0.1, help="target false-positive rate for the threshold band")
    parser.add_argument('--use_lof', action='store_true', default=False, help="calibrate with LOF instead of a band")
    parser.add_argument('--contamination', default='auto')
    parser.add_argument('--calibrate', action='store_true', default=True,
                        help="recalibrate the champion threshold after promotion")

    parser.add_argument('-patience', '--early_stopping_patience', type=float, default=50)

    args = parser.parse_args()
    return args


def _load_canonical_metric(canonical_dir):
    """Read the reference metric of the canonical champion folder."""
    path = os.path.join(canonical_dir, 'best_metrics.json')
    if not os.path.exists(path):
        return 0.0
    try:
        with open(path) as f:
            return float(json.load(f).get('reference_metric', 0.0))
    except Exception:
        return 0.0


def _promote_to_canonical(temp_dir, canonical_dir):
    """Copy the temporary run files into the canonical champion folder."""
    os.makedirs(canonical_dir, exist_ok=True)
    for fname in os.listdir(temp_dir):
        src = os.path.join(temp_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(canonical_dir, fname))
    print(f"[champion] Promoted → {canonical_dir}")


def _calibrate_champion(model, val_dataloader, test_dataloader, class2idx,
                               canonical_checkpoint_dir, top_k, device):
    """Sweep calibration candidates on the promoted champion, persist the best by real accuracy."""
    print("\n" + "=" * 64)
    print("[auto-recalibrate] Sweeping calibration on the champion...")
    print("=" * 64)

    champion_ckpt = torch.load(os.path.join(canonical_checkpoint_dir, "best.pt"), map_location=device)
    model.load_state_dict(champion_ckpt["model_state_dict"])

    print("Collecting val NLL scores...", flush=True)
    val_losses, mu_patch = _collect_val_scores(model, val_dataloader, top_k)
    print(f"  {len(val_losses)} real images — "
          f"mean={val_losses.mean():.4f}  std={val_losses.std():.4f}")

    print("\nSweeping calibration candidates (clean test set)...")
    best_cand, best_thr, best_metrics, best_real, best_preds, best_labels, rows = calibration.run_sweep(
        model, val_losses, mu_patch, top_k, test_dataloader, class2idx, eval_once)
    calibration.print_sweep_table(rows)
    print(f"\n  Winner: {best_cand['label']}  (Real Acc = {best_real:.4f})\n")

    calibration.persist_calibration(
        best_thr, best_cand, best_metrics,
        run_dir=canonical_checkpoint_dir,
        threshold_path=os.path.join(canonical_checkpoint_dir, "thresholds.npz"),
        lof_checkpoint=os.path.join(canonical_checkpoint_dir, "lof_model.pkl"),
        preds=best_preds, labels=best_labels)


def _cleanup_temp(temp_dir):
    """Delete the temporary run folder."""
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"[champion] Removed temp folder: {temp_dir}")


def _build_metrics_json(run_name, canonical_dir, epoch, reference_metric, metrics):
    """Assemble the best_metrics.json payload."""
    def _r(x, d=6):
        return round(float(x), d)

    return {
        'run_name':         run_name,
        'canonical_name':   os.path.basename(canonical_dir),
        'epoch':            int(epoch),
        'reference_metric': _r(reference_metric),
        'metrics':          metrics,
    }


def _build_data_loader_common(args, config, is_train, is_val, shuffle, drop_last, return_class2idx=False):
    """Build a DataLoader over the configured dataset."""
    norm_mean, norm_std = const.get_norm_stats(config["backbone_name"])
    if is_train:
        real_paths = cfg.PATH_REAL
        fake_paths = []
    else:
        real_paths = cfg.PATH_REAL_OOD if cfg.PATH_REAL_OOD else cfg.PATH_REAL
        fake_paths = cfg.PATH_FAKE
    dataset_instance = dataset.Dataset(
        real_paths=real_paths,
        fake_paths=fake_paths,
        input_size=config["input_size"],
        is_train=is_train,
        is_val=is_val,
        num_train_patches=args.num_train_patches,
        num_repr_patches=args.num_repr_patches,
        debug=args.debug,
        norm_mean=norm_mean,
        norm_std=norm_std,
    ).create_dataset()

    num_workers = getattr(args, 'num_workers', 4)
    dataloader = torch.utils.data.DataLoader(
        dataset_instance,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    if return_class2idx:
        return dataloader, {v: k for k, v in dataset_instance.class_to_idx.items()}
    return dataloader


def build_train_data_loader(args, config):
    """Build the training DataLoader."""
    return _build_data_loader_common(args, config, is_train=True, is_val=False, shuffle=True, drop_last=True)


def build_val_data_loader(args, config):
    """Build the validation DataLoader."""
    return _build_data_loader_common(args, config, is_train=True, is_val=True, shuffle=False, drop_last=False)


def build_test_data_loader(args, config):
    """Build the test DataLoader."""
    return _build_data_loader_common(args, config, is_train=False, is_val=False, shuffle=False, drop_last=False, return_class2idx=True)


def build_model(config, args):
    """Build the FastFlow model, generating GMM parameters if missing."""
    out_indices = config.get("out_indices", [1, 2, 3])
    out_indices_str = str(out_indices)
    pooling_type = config.get("pooling_type", "mean")
    n_components = config.get("gmm_n_components", 1)

    reals_tag = cfg.real_tag(cfg.PATH_REAL)
    gmm_parameters = f"{const.WORKING_DIR}/parameters/{n_components}-gmm_parameters_{config['backbone_name']}_indices_{out_indices_str}_{reals_tag}_{config['input_size']}_{pooling_type}.npy"

    if not os.path.exists(gmm_parameters):
        print(f"GMM parameters not found at {gmm_parameters}.")
        print(f"Running generate_parameters.py...")
        gen_script = os.path.join(const.WORKING_DIR, "scripts", "generate_parameters.py")
        if not os.path.exists(gen_script):
            raise FileNotFoundError(
                f"Cannot generate GMM parameters: script not found at {gen_script}")
        cmd = [
            sys.executable,
            gen_script,
            "--model_name", config["backbone_name"],
            "--reals", reals_tag,
        ]
        subprocess.run(cmd, check=True)
        if not os.path.exists(gmm_parameters):
            raise FileNotFoundError(
                f"generate_parameters.py ran but expected output is missing: {gmm_parameters}")
        print("GMM parameters generated successfully.")

    gmm_values = np.load(gmm_parameters, allow_pickle=True).item()
    print(f"Loading gmm parameters from {gmm_parameters}")

    model = fastflow.FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
        gmm_values=gmm_values,
        in_channels=3,
        backbone_weights=args.backbone_weights,
        out_indices=config.get("out_indices", [1, 2, 3]),
        pooling_type=pooling_type,

    )
    print("Model A.D. Param#: {}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))
    return model


def build_optimizer(args, model, config):
    """Build the optimizer."""
    if args.optimizer == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def _per_patch_scores(model, patches):
    """Score every patch of every image."""
    B, N = patches.shape[0], patches.shape[1]
    flat = patches.reshape(B * N, *patches.shape[2:])
    ret = model(flat)
    return ret["loss"].reshape(B, N), ret["mahalanobis"].reshape(B, N)


def aggregate_scores(loss_pp, maha_pp, top_k=None, mu_patch=None):
    """Aggregate per-patch scores into one score per image."""
    N = loss_pp.shape[1]
    if not top_k or top_k >= N or mu_patch is None:
        return loss_pp.mean(dim=1), maha_pp.mean(dim=1)
    idx = (loss_pp - mu_patch).abs().topk(top_k, dim=1).indices
    return torch.gather(loss_pp, 1, idx).mean(dim=1), torch.gather(maha_pp, 1, idx).mean(dim=1)


def score_per_image(model, patches, top_k=None, mu_patch=None):
    """Compute one score per image from its patches."""
    loss_pp, maha_pp = _per_patch_scores(model, patches)
    loss, maha = aggregate_scores(loss_pp, maha_pp, top_k, mu_patch)
    return {"loss": loss, "mahalanobis": maha}


def _collect_val_scores(model, val_dataloader, top_k=None):
    """Collect per-image validation scores and the patch-mean."""
    model.eval()
    device = model.nf_flows[0].parameters().__next__().device
    loss_pp_all, maha_pp_all = [], []
    for batch in val_dataloader:
        data = batch[0] if isinstance(batch, (list, tuple)) else batch
        data = data.to(device)
        with torch.no_grad():
            lpp, mpp = _per_patch_scores(model, data)
        loss_pp_all.append(lpp.cpu())
        maha_pp_all.append(mpp.cpu())
    if len(loss_pp_all) == 0:
        return np.array([]), None
    mu_patch = float(torch.cat([x.reshape(-1) for x in loss_pp_all]).mean())
    losses = torch.cat([aggregate_scores(l, m, top_k, mu_patch)[0]
                        for l, m in zip(loss_pp_all, maha_pp_all)]).numpy()
    return losses, mu_patch


def train_one_epoch(dataloader, model, optimizer, epoch, args, scheduler=None):
    """Train the flow for one epoch (per-patch loss)."""
    start_time = time.time()
    model.train()
    loss_meter = utils.AverageMeter()
    loss_values = []

    device = args.device if hasattr(args, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for step, data in enumerate(dataloader):
        data = data.to(device)
        data = data.reshape(-1, *data.shape[2:])
        ret = model(data)
        loss = ret["loss"].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item())
        loss_values.append(ret["loss"].detach().cpu())

        if (step + 1) % args.log_interval == 0 or (step + 1) == len(dataloader):
            print("Epoch {} - Iteration {}/{} - Loss: {:.3f}".format(
                epoch + 1, step + 1, len(dataloader), loss_meter.avg))
            wandb.log({"Train Loss": loss_meter.val}, step=epoch + 1)

    if scheduler:
        scheduler.step(loss_meter.val)

    preds_train = torch.cat(loss_values, dim=0).numpy().reshape(-1, 1)
    train_mean = preds_train.mean()
    train_std = preds_train.std()

    wandb.log({"Train loss mean": train_mean, "Train loss std": train_std}, step=epoch + 1)

    training_time = time.time() - start_time
    print(f"⏱️  Training epoch time: {int(training_time // 60)}m {int(training_time % 60)}s")
    return train_mean, train_std, preds_train


def compute_threshold(val_dataloader, model, use_lof=False, contamination='auto', alpha=0.1, top_k=None):
    """Calibrate the anomaly threshold on the validation reals."""
    losses, mu_patch = _collect_val_scores(model, val_dataloader, top_k)

    if len(losses) == 0:
        raise ValueError("No validation loss values collected. Check validation dataloader.")

    mean = losses.mean()
    std = losses.std()

    result = {'threshold': None, 'mean': mean, 'std': std, 'losses': losses,
              'mu_patch': mu_patch, 'top_k': top_k}

    if use_lof:
        lof = LocalOutlierFactor(novelty=True, contamination=contamination, n_jobs=-1)
        lof.fit(losses.reshape(-1, 1))
        result['lof'] = lof
    else:
        z = norm.ppf(1 - alpha)
        result['l_threshold'] = mean - z * std
        result['u_threshold'] = mean + z * std

    print(f"Threshold computation done.")
    return result


def _compute_class_metrics(c, class_masks, labels, preds, preds_, class2idx,
                           y_true_0_base, y_pred_0_base, preds_0_base, mu_val):
    """Compute metrics for one fake class against a real baseline."""
    mask_c  = class_masks[c]
    y_pred_c = preds[mask_c]
    preds_c  = preds_[mask_c]

    min_len = min(len(preds_c), len(y_true_0_base))
    if min_len == 0:
        return None

    class_name     = class2idx[c] if class2idx else str(c)
    loss_mean_fake = float(preds_c.mean())
    loss_std_fake  = float(preds_c.std())

    rng = np.random.RandomState(42 + c)
    idx = rng.permutation(len(y_true_0_base))

    y_pred_balanced = np.concatenate([y_pred_c[:min_len], y_pred_0_base[idx][:min_len]])
    scores_balanced = np.concatenate([preds_c[:min_len],  preds_0_base[idx][:min_len]])
    y_true_binary   = np.concatenate([np.ones(min_len, dtype=np.int8),
                                      np.zeros(min_len, dtype=np.int8)])

    _mu_real          = float(preds_0_base.mean())
    _distances        = np.abs(scores_balanced - _mu_real)
    _fpr_o, _tpr_o, _thr_o = roc_curve(y_true_binary, _distances)
    _best_o           = np.argmax(_tpr_o - _fpr_o)
    _preds_oracle     = (_distances >= _thr_o[_best_o]).astype(np.int8)

    scores_bilateral = np.abs(scores_balanced - mu_val)

    return {
        'class_id':          c, 'class_name': class_name, 'min_len': min_len,
        'loss_mean':         loss_mean_fake, 'loss_std': loss_std_fake,
        'accuracy':          accuracy_score(y_true_binary, y_pred_balanced),
        'acc_oracle':        accuracy_score(y_true_binary, _preds_oracle),
        'thr_oracle':        float(_thr_o[_best_o]),
        'mu_oracle':         _mu_real,
        'ap':                average_precision_score(y_true_binary, scores_bilateral),
        'roc':               roc_auc_score(y_true_binary, scores_bilateral),
    }


def eval_once(dataloader, model, epoch=None, class2idx=None, threshold_info=None,
              wandb_log=False, wandb_prefix: str = ""):
    """Score the test set and compute all metrics."""
    inference_start_time = time.time()
    model.eval()
    labels_list = []
    preds_list = []
    device = model.nf_flows[0].parameters().__next__().device

    ti = threshold_info if isinstance(threshold_info, dict) else {}
    mu_patch = ti.get('mu_patch')
    if isinstance(mu_patch, np.ndarray):
        mu_patch = float(mu_patch)
    top_k = ti.get('top_k')
    if isinstance(top_k, np.ndarray):
        top_k = int(top_k)

    for data, targets in dataloader:
        data, targets = data.to(device), targets.to(device)
        with torch.no_grad():
            ret = score_per_image(model, data, top_k=top_k, mu_patch=mu_patch)
        preds_list.append(ret["loss"].cpu())
        labels_list.append(targets.cpu())

    inference_time = time.time() - inference_start_time
    print(f"Testing done")
    if epoch is not None:
        print(f"⏱️  Test inference time: {int(inference_time // 60)}m {int(inference_time % 60)}s")

    preds_ = torch.cat(preds_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()

    if threshold_info is None:
        raise ValueError("threshold_info must be provided. Use compute_threshold() first.")

    if 'lof' in threshold_info:
        lof = threshold_info['lof']
        preds = (lof.predict(preds_.reshape(-1, 1)) < 0).astype(int)
    else:
        l_threshold = threshold_info['l_threshold']
        u_threshold = threshold_info['u_threshold']
        preds = ((preds_ < l_threshold) | (preds_ > u_threshold)).astype(int)

    classes = np.unique(labels)
    class_masks = {c: labels == c for c in classes}

    mask_fake = np.zeros(len(labels), dtype=bool)
    for c in classes:
        if c != 0:
            mask_fake |= class_masks[c]

    if epoch is not None and epoch % 50 == 0 and not wandb_prefix:
        likelihood_real = preds_[class_masks[0]]
        likelihood_fake = preds_[mask_fake]

        plt.figure(figsize=(10, 6))
        if len(likelihood_real) > 0:
            sns.kdeplot(likelihood_real, label='Real', color='blue')
        if len(likelihood_fake) > 0:
            sns.kdeplot(likelihood_fake, label='Fake', color='orange')
        plt.title('Real vs Fake')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(const.CHECKPOINT_DIR + "/real_fake_likelihoods_{}.png".format(epoch))
        plt.close()

    aps, accs, rocs, cls = [], [], [], []
    per_class_metrics = {}

    mask_0   = class_masks[0]
    y_true_0 = labels[mask_0]
    y_pred_0 = preds[mask_0]
    preds_0  = preds_[mask_0]

    mu_val = threshold_info.get('mean') if isinstance(threshold_info, dict) else None
    if isinstance(mu_val, np.ndarray):
        mu_val = float(mu_val)
    if mu_val is None:
        mu_val = float(preds_0.mean()) if len(preds_0) > 0 else 0.0
        print("[warn] threshold_info has no 'mean'; folding AP/ROC around the test real mean.")

    loss_mean_real = loss_std_real = 0.0
    if len(preds_0) > 0:
        loss_mean_real = preds_0.mean()
        loss_std_real  = preds_0.std()
        per_class_metrics["loss_per_class/Real"]     = loss_mean_real
        per_class_metrics["loss_std_per_class/Real"] = loss_std_real

    acc_real = accuracy_score(np.zeros(len(y_true_0), dtype=np.int8), y_pred_0) \
               if len(y_true_0) > 0 else 0.0

    if 'lof' in threshold_info:
        deployed_thr_str = "thr=LOF"
    else:
        deployed_thr_str = f"thr=[{threshold_info['l_threshold']:.4f}, {threshold_info['u_threshold']:.4f}]"

    classes_to_process = list(classes[1:])

    if cfg.SHOW_PER_CLASS:
        print("\nComputing metrics using Real as baseline...")
        print("-" * 30)
        print(f"  > Class Real      : \t Accuracy = {acc_real:.4f} ({deployed_thr_str}), \t Loss = {preds_0.mean() if len(preds_0) > 0 else 0:.4f}±{preds_0.std() if len(preds_0) > 0 else 0:.4f}")
        print("=" * 30)
    metrics_real_start_time = time.time()
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(_compute_class_metrics)(
            c, class_masks, labels, preds, preds_, class2idx, y_true_0, y_pred_0, preds_0, mu_val
        ) for c in classes_to_process
    )

    accs_oracle = []
    for result in results:
        if result is None:
            continue
        if cfg.SHOW_PER_CLASS:
            print(f"  > Class {result['class_name']} : \t Accuracy = {result['accuracy']:.4f} "
                  f"(oracle={result['acc_oracle']:.4f} @[{result['mu_oracle']-result['thr_oracle']:.4f}, "
                  f"{result['mu_oracle']+result['thr_oracle']:.4f}]), "
                  f"\t AP = {result['ap']:.4f}, \t ROC AUC = {result['roc']:.4f}, \t Loss = {result['loss_mean']:.4f}±{result['loss_std']:.4f}")
            print("-" * 30)
        per_class_metrics[f"loss_per_class/{result['class_name']}"]          = result['loss_mean']
        per_class_metrics[f"loss_std_per_class/{result['class_name']}"]      = result['loss_std']
        per_class_metrics[f"acc_per_class/{result['class_name']}"]           = result['accuracy']
        per_class_metrics[f"acc_oracle_per_class/{result['class_name']}"]    = result['acc_oracle']
        per_class_metrics[f"ap_per_class/{result['class_name']}"]            = result['ap']
        per_class_metrics[f"roc_per_class/{result['class_name']}"]           = result['roc']
        aps.append(result['ap'])
        accs.append(result['accuracy'])
        accs_oracle.append(result['acc_oracle'])
        rocs.append(result['roc'])
        cls.append(result['class_id'])

    mean_acc           = np.mean(accs)
    mean_acc_oracle    = np.mean(accs_oracle)
    mean_ap            = np.mean(aps)
    mean_roc           = np.mean(rocs)

    metrics_real_time = time.time() - metrics_real_start_time
    if cfg.SHOW_FAMILIES:
        print("-" * 30)
        _print_family_summary(_aggregate_by_family(results))
    if cfg.SHOW_PER_CLASS and epoch is not None:
        print(f"⏱️  Metrics computation time (Real baseline): {int(metrics_real_time // 60)}m {int(metrics_real_time % 60)}s")
    if cfg.SHOW_PER_CLASS or cfg.SHOW_FAMILIES:
        print("=" * 30 + "\n")

    per_class_metrics["Val acc"] = mean_acc
    per_class_metrics["Val AP"]  = mean_ap
    per_class_metrics["Val ROC"] = mean_roc

    test_loss_real_mean = loss_mean_real
    test_loss_real_std  = loss_std_real

    preds_fake        = preds_[mask_fake]
    preds_fake_binary = preds[mask_fake]
    test_loss_fake_mean = preds_fake.mean() if len(preds_fake) > 0 else 0
    test_loss_fake_std  = preds_fake.std()  if len(preds_fake) > 0 else 0

    global_metrics = {}
    if len(preds_fake) > 0 and len(preds_0) > 0:
        min_n  = min(len(preds_0), len(preds_fake))
        rng    = np.random.RandomState(42)
        idx_r  = rng.permutation(len(preds_0))[:min_n]
        idx_f  = rng.permutation(len(preds_fake))[:min_n]

        sc_g   = np.concatenate([preds_fake[idx_f],        preds_0[idx_r]])
        yp_g   = np.concatenate([preds_fake_binary[idx_f], y_pred_0[idx_r]])
        yt_g   = np.concatenate([np.ones(min_n, dtype=np.int8),
                                  np.zeros(min_n, dtype=np.int8)])

        sc_g_bi = np.abs(sc_g - mu_val)
        acc_g  = accuracy_score(yt_g, yp_g)
        ap_g   = average_precision_score(yt_g, sc_g_bi)
        roc_g  = roc_auc_score(yt_g, sc_g_bi)

        _mu_g  = float(preds_0.mean())
        _dist_g = np.abs(sc_g - _mu_g)
        _fgo, _tgo, _thr_go = roc_curve(yt_g, _dist_g)
        _ogo = np.argmax(_tgo - _fgo)
        acc_oracle_g = accuracy_score(yt_g, (_dist_g >= _thr_go[_ogo]).astype(np.int8))

        print(f"\n{'=' * 30}")
        print(f"Global — Real vs All Fake")
        print(f"  Acc={acc_g:.4f}  (oracle={acc_oracle_g:.4f})")
        print(f"  AP={ap_g:.4f}   ROC AUC={roc_g:.4f}")
        print(f"{'=' * 30}\n")

        per_class_metrics["global/acc"]        = acc_g
        per_class_metrics["global/acc_oracle"] = acc_oracle_g
        per_class_metrics["global/ap"]         = ap_g
        per_class_metrics["global/roc"]        = roc_g

        global_metrics = {
            'n_balanced':    int(min_n),
            'n_fake_total':  int(len(preds_fake)),
            'acc':           float(round(acc_g, 6)),
            'acc_oracle':    float(round(acc_oracle_g, 6)),
            'ap':            float(round(ap_g, 6)),
            'roc':           float(round(roc_g, 6)),
        }

    if wandb_log:
        p = wandb_prefix
        wandb_log_dict = {
            f"{p}Val acc":             mean_acc,
            f"{p}Test loss real mean": test_loss_real_mean,
            f"{p}Test loss fake mean": test_loss_fake_mean,
            f"{p}Test loss real std":  test_loss_real_std,
            f"{p}Test loss fake std":  test_loss_fake_std,
        }
        wandb_log_dict.update({f"{p}{k}": v for k, v in per_class_metrics.items()})
        wandb.log(wandb_log_dict, step=(epoch + 1) if epoch is not None else 0)

    def _r(x, d=6): return round(float(x), d)

    def _per_class_dict(result_list):
        return {
            r['class_name']: {
                'acc':           _r(r['accuracy']),
                'acc_oracle':    _r(r['acc_oracle']),
                'ap':            _r(r['ap']),
                'roc':           _r(r['roc']),
                'loss_mean':     _r(r['loss_mean'], 4),
                'loss_std':      _r(r['loss_std'],  4),
            }
            for r in result_list if r is not None
        }

    metrics_summary = {
        'real': {
            'mean_acc':           _r(mean_acc),
            'mean_acc_oracle':    _r(mean_acc_oracle),
            'mean_ap':            _r(mean_ap),
            'mean_roc':           _r(mean_roc),
            'per_class':          _per_class_dict(results),
        },
    }
    if global_metrics:
        metrics_summary['global'] = global_metrics

    return mean_roc, preds_, labels, metrics_summary


def train(args, config, canonical_checkpoint_dir):
    """Run the full training loop with champion promotion."""
    checkpoint_dir = const.CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)

    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config=config,
        name=args.run_name,
        mode="disabled" if args.debug else args.wandb)
    wandb.config.update(args)

    model = build_model(config, args)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        print('Model loaded!')

    device = resolve_device(args.gpu_id)

    model.to(device)
    print(f"✓ Model moved to {device}")
    args.device = device

    optimizer = build_optimizer(args, model, config)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay,
                                  patience=args.lr_patience, verbose=True) if args.scheduler else None

    train_dataloader = build_train_data_loader(args, config)
    val_dataloader   = build_val_data_loader(args, config)

    test_dataloader, class2idx = build_test_data_loader(args, config)

    best_metric = 0.0
    patience_counter = 0
    eval_metrics = {}

    for epoch in range(args.num_epochs):
        train_mean, train_std, _ = train_one_epoch(
            train_dataloader, model, optimizer, epoch, args, scheduler=scheduler)

        current_metric = -1.0

        if (epoch + 1) % args.eval_interval == 0:
            threshold_info = compute_threshold(
                val_dataloader, model,
                use_lof=args.use_lof,
                contamination=args.contamination,
                alpha=args.alpha,
                top_k=args.top_k)

            l_threshold_val = threshold_info.get('l_threshold')
            u_threshold_val = threshold_info.get('u_threshold')
            val_mean = threshold_info.get('mean', None)
            val_std  = threshold_info.get('std', None)

            print(f"Training loss stats: mean: {train_mean:.4f}, std: {train_std:.4f}")

            log_dict = {
                "Train Loss Mean": train_mean, "Train Loss Std": train_std,
            }
            if l_threshold_val is not None:
                log_dict["Lower Threshold"] = l_threshold_val
            if u_threshold_val is not None:
                log_dict["Upper Threshold"] = u_threshold_val
            if val_mean is not None:
                log_dict["Val Loss Mean"] = val_mean
            if val_std is not None:
                log_dict["Val Loss Std"] = val_std
            wandb.log(log_dict, step=epoch + 1)

            current_metric, preds, labels, eval_metrics = eval_once(
                test_dataloader, model, epoch=epoch, class2idx=class2idx,
                threshold_info=threshold_info, wandb_log=True)

            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0

                print(f"Epoch {epoch+1}: New best ROC: {best_metric:.4f}. Saving model.")
                wandb.run.summary["best_roc"] = best_metric

                checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, checkpoint_path)

                args_dict = vars(args).copy()
                args_dict.pop('device', None)
                with open(os.path.join(checkpoint_dir, "run_config.yaml"), 'w') as f:
                    yaml.dump(args_dict, f, default_flow_style=False)

                save_dict = {}
                for key in ('l_threshold', 'u_threshold', 'threshold', 'losses', 'mean', 'std', 'lof_scores', 'mu_patch', 'top_k'):
                    if key in threshold_info and threshold_info[key] is not None:
                        save_dict[key] = threshold_info[key]
                np.savez(os.path.join(checkpoint_dir, "thresholds.npz"), **save_dict)

                np.save(os.path.join(checkpoint_dir, "preds_best.npy"), preds)
                np.save(os.path.join(checkpoint_dir, "labels_best.npy"), labels)
                with open(os.path.join(checkpoint_dir, "class2idx_best.json"), 'w') as _f:
                    json.dump({int(k): v for k, v in class2idx.items()}, _f)

                if 'lof' in threshold_info:
                    joblib.dump(threshold_info['lof'], os.path.join(checkpoint_dir, "lof_model.pkl"))

                metrics_payload = _build_metrics_json(
                    run_name=args.run_name,
                    canonical_dir=canonical_checkpoint_dir,
                    epoch=epoch,
                    reference_metric=current_metric,
                    metrics=eval_metrics,
                )
                with open(os.path.join(checkpoint_dir, 'best_metrics.json'), 'w') as _f:
                    json.dump(metrics_payload, _f, indent=2)

            else:
                patience_counter += 1
                print(f"Epoch {epoch+1}: ROC ({current_metric:.4f}) not improved compared to {best_metric:.4f}. Patience: {patience_counter}/{args.early_stopping_patience}")

        if patience_counter >= args.early_stopping_patience:
            print(f"Stopping early at epoch {epoch + 1} after {args.early_stopping_patience} epochs without improvement of ROC.")
            break

    print(f"Training finished. Best ROC: {best_metric:.4f}")

    canonical_best_metric = _load_canonical_metric(canonical_checkpoint_dir)
    print(f"[champion] This run: {best_metric:.4f}  |  Canonical: {canonical_best_metric:.4f}")
    if best_metric > canonical_best_metric:
        _promote_to_canonical(checkpoint_dir, canonical_checkpoint_dir)
        print(f"[champion] New champion!  {best_metric:.4f} > {canonical_best_metric:.4f}")
        if args.calibrate:
            _calibrate_champion(
                model, val_dataloader, test_dataloader, class2idx,
                canonical_checkpoint_dir, args.top_k, device)
    else:
        print(f"[champion] No promotion. Canonical remains at {canonical_best_metric:.4f}")

    _cleanup_temp(checkpoint_dir)


if __name__ == "__main__":
    args = parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    pprint(vars(args))
    pprint(config)

    canonical_run_name = f"{config['backbone_name']}_{cfg.real_tag(cfg.PATH_REAL)}_{config['pooling_type']}"

    if args.run_name is None:
        calib_tag = "_lof" if args.use_lof else f"_t-alpha{args.alpha}"
        args.run_name = (canonical_run_name + calib_tag +
                         f"_np{args.num_train_patches}_lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}"
                         f"_ld{args.lr_decay}_lp{args.lr_patience}")
        if args.top_k:
            args.run_name += f"_topk{args.top_k}"

    canonical_checkpoint_dir = os.path.join(const.CHECKPOINT_DIR, canonical_run_name)
    const.CHECKPOINT_DIR     = os.path.join(const.CHECKPOINT_DIR, args.run_name)

    print(f"[champion] Canonical dir : {canonical_checkpoint_dir}")
    print(f"[champion] Temporary dir : {const.CHECKPOINT_DIR}")

    train(args, config, canonical_checkpoint_dir=canonical_checkpoint_dir)
