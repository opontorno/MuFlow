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
from muflow.gpu_utils import resolve_device

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, roc_curve
from sklearn.neighbors import LocalOutlierFactor
from joblib import Parallel, delayed

GANS      = {'StyleGAN', 'StyleGAN2', 'StyleGAN3', 'STARGAN', 'AttGAN', 'GDWCT'}
DM_OPEN   = {'Flux.1', 'Stable DIffusion 3.5', 'Stable Diffusion XL',
             'Stable Cascade', 'Stable Diffusion Attend and Excite'}
DM_CLOSED = {'Dall-E 3', 'Midjourney', 'Starry AI', 'Deep AI', 'Hotpot AI',
             'Nvidia Sana PAG', 'Tencent Hunyuan', 'Flux.1.1 Pro'}
MIX_2CLASS = {'STARGAN', 'StyleGAN2', 'Stable DIffusion 3.5', 'Flux.1.1 Pro'}


def _aggregate_by_family(results):
    """Group per-class result dicts by generator family.
    results: list of per-class metric dicts (None entries ignored).
    Returns: dict {'all','gan','dm_open','dm_closed','mix'} -> list of dicts.
    """
    fam = {'all': [], 'gan': [], 'dm_open': [], 'dm_closed': [], 'mix': []}
    for r in results:
        if r is None:
            continue
        n = r['class_name']
        fam['all'].append(r)
        if n in GANS:       fam['gan'].append(r)
        if n in DM_OPEN:    fam['dm_open'].append(r)
        if n in DM_CLOSED:  fam['dm_closed'].append(r)
        if n in MIX_2CLASS: fam['mix'].append(r)
    return fam


def _print_family_summary(fam):
    """Print mean metrics overall and per family.
    fam: output of _aggregate_by_family.
    """
    keys = ['accuracy', 'acc_oracle', 'acc_oracle_bi', 'ap', 'roc']

    def _m(rs, k): return np.mean([r[k] for r in rs])

    if not fam['all']:
        return
    print(f"Mean Accuracy : {_m(fam['all'], 'accuracy'):.4f}"
          f" (oracle={_m(fam['all'], 'acc_oracle'):.4f},"
          f" bi={_m(fam['all'], 'acc_oracle_bi'):.4f})")
    print(f"Mean AP       : {_m(fam['all'], 'ap'):.4f}")
    print(f"Mean ROC AUC  : {_m(fam['all'], 'roc'):.4f}")
    print("--- by family ---")
    for label, key in [('GANs', 'gan'), ('DM-Open', 'dm_open'),
                       ('DM-Closed', 'dm_closed'), ('Mix', 'mix')]:
        if fam[key]:
            print(f"  {label:<10}: Acc={_m(fam[key], 'accuracy'):.4f}"
                  f" (oracle={_m(fam[key], 'acc_oracle'):.4f},"
                  f" bi={_m(fam[key], 'acc_oracle_bi'):.4f})"
                  f"  AP={_m(fam[key], 'ap'):.4f}"
                  f"  ROC={_m(fam[key], 'roc'):.4f}")


def parse_args():
    """Parse command-line arguments.
    Returns: argparse.Namespace.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/resnet50.yaml', help="path to config file")

    parser.add_argument("--reals", type=str, default='ffhq', help="reals dataset", choices=['ffhq', 'celeba_hq', 'ffhq+celeba_hq'])
    parser.add_argument("--checkpoint", type=str, help="path to load checkpoint")

    parser.add_argument('--wandb', default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_entity', type=str, default='orazio-mattia', help="W&B entity (team/user). Default: your default W&B entity.")
    parser.add_argument('--wandb_project', type=str, default='MuFlow', help="W&B project name.")

    parser.add_argument('--num_train_patches', type=int, default=const.PATCH_NUM_TRAIN,
                        help="random native patches sampled per training image (per step)")
    parser.add_argument('--num_repr_patches', type=int, default=const.PATCH_NUM_REPR,
                        help="patches per image to build the inference/threshold representation")
    parser.add_argument('--top_k', type=int, default=None,
                        help="signed-mean over the top_k patches most deviant from the val patch mean; None/0 = all patches (plain mean)")
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--backbone_weights', type=str, help="path to load backbone weights")
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4, help="number of data loading workers")
    parser.add_argument('--debug', action='store_true', help="Debug mode with reduced dataset size")
    parser.add_argument('--gpu_id', type=int, default=None, help="Manually specify GPU ID to use (default: auto-select GPU with most free memory)")

    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'sgd'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--scheduler', action='store_true', default=True, help="Enable LR scheduler (default: on)")
    parser.add_argument('--lr_decay', type=float, default=0.7)
    parser.add_argument('--lr_patience', type=int, default=35)

    parser.add_argument('--alpha', type=float, default=0.1, help="Target false positive rate under the normality assumption")
    parser.add_argument('--use_lof', action='store_true', default=False, help="Enable Local Outlier Factor threshold")
    parser.add_argument('--contamination', default='auto')

    parser.add_argument('-patience', '--early_stopping_patience', type=float, default=50,
                        help="Patience epochs for early stopping based on Val Acc")

    args = parser.parse_args()
    return args


def _load_canonical_metric(canonical_dir):
    """Read the reference metric of the canonical champion folder.
    canonical_dir: champion folder path.
    Returns: float reference metric (0.0 if missing/unreadable).
    """
    path = os.path.join(canonical_dir, 'best_metrics.json')
    if not os.path.exists(path):
        return 0.0
    try:
        with open(path) as f:
            return float(json.load(f).get('reference_metric', 0.0))
    except Exception:
        return 0.0


def _promote_to_canonical(temp_dir, canonical_dir):
    """Copy the temporary run files into the canonical champion folder.
    temp_dir: source run folder.
    canonical_dir: destination champion folder.
    """
    os.makedirs(canonical_dir, exist_ok=True)
    for fname in os.listdir(temp_dir):
        src = os.path.join(temp_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(canonical_dir, fname))
    print(f"[champion] Promoted → {canonical_dir}")


def _cleanup_temp(temp_dir):
    """Delete the temporary run folder.
    temp_dir: folder to remove.
    """
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"[champion] Removed temp folder: {temp_dir}")


def _build_metrics_json(run_name, canonical_dir, epoch, reference_metric, metrics):
    """Assemble the best_metrics.json payload.
    run_name: temporary run name.
    canonical_dir: champion folder path.
    epoch: best epoch.
    reference_metric: selection metric value.
    metrics: metrics summary dict.
    Returns: dict payload.
    """
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
    """Build a DataLoader over the configured dataset.
    args: parsed CLI args.
    config: backbone config dict.
    is_train: train/val dataset if True, else test.
    is_val: select the validation split.
    shuffle: shuffle samples.
    drop_last: drop the last incomplete batch.
    return_class2idx: also return the idx->class mapping.
    Returns: DataLoader, or (DataLoader, idx2class) if return_class2idx.
    """
    norm_mean, norm_std = const.get_norm_stats(config["backbone_name"])
    dataset_instance = dataset.Dataset(
        reals_name=args.reals,
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
    """Build the training DataLoader.
    args: parsed CLI args.
    config: backbone config dict.
    Returns: DataLoader.
    """
    return _build_data_loader_common(args, config, is_train=True, is_val=False, shuffle=True, drop_last=True)


def build_val_data_loader(args, config):
    """Build the validation DataLoader.
    args: parsed CLI args.
    config: backbone config dict.
    Returns: DataLoader.
    """
    return _build_data_loader_common(args, config, is_train=True, is_val=True, shuffle=False, drop_last=False)


def build_test_data_loader(args, config):
    """Build the test DataLoader.
    args: parsed CLI args.
    config: backbone config dict.
    Returns: (DataLoader, idx2class).
    """
    return _build_data_loader_common(args, config, is_train=False, is_val=False, shuffle=False, drop_last=False, return_class2idx=True)


def build_model(config, args):
    """Build the FastFlow model, generating GMM parameters if missing.
    config: backbone config dict.
    args: parsed CLI args.
    Returns: FastFlow model.
    """
    out_indices = config.get("out_indices", [1, 2, 3])
    out_indices_str = str(out_indices)
    pooling_type = config.get("pooling_type", "mean")
    n_components = config.get("gmm_n_components", 1)

    gmm_parameters = f"{const.WORKING_DIR}/parameters/{n_components}-gmm_parameters_{config['backbone_name']}_indices_{out_indices_str}_{args.reals}_{config['input_size']}_{pooling_type}.npy"

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
            "--reals", args.reals,
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
    """Build the optimizer.
    args: parsed CLI args.
    model: model whose parameters are optimized.
    config: backbone config dict.
    Returns: torch optimizer.
    """
    if args.optimizer == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def _per_patch_scores(model, patches):
    """Score every patch of every image.
    model: FastFlow model.
    patches: (B, N, 3, P, P) tensor.
    Returns: (loss, mahalanobis) tensors of shape (B, N).
    """
    B, N = patches.shape[0], patches.shape[1]
    flat = patches.reshape(B * N, *patches.shape[2:])
    ret = model(flat)
    return ret["loss"].reshape(B, N), ret["mahalanobis"].reshape(B, N)


def aggregate_scores(loss_pp, maha_pp, top_k=None, mu_patch=None):
    """Aggregate per-patch scores into one score per image.
    loss_pp: (B, N) per-patch NLL.
    maha_pp: (B, N) per-patch Mahalanobis distance.
    top_k: keep the top_k patches most deviant from mu_patch; None/0/>=N = all.
    mu_patch: validation patch-mean used as the deviation center.
    Returns: (loss, mahalanobis) tensors of shape (B,).
    """
    N = loss_pp.shape[1]
    if not top_k or top_k >= N or mu_patch is None:
        return loss_pp.mean(dim=1), maha_pp.mean(dim=1)
    idx = (loss_pp - mu_patch).abs().topk(top_k, dim=1).indices
    return torch.gather(loss_pp, 1, idx).mean(dim=1), torch.gather(maha_pp, 1, idx).mean(dim=1)


def score_per_image(model, patches, top_k=None, mu_patch=None):
    """Compute one score per image from its patches.
    model: FastFlow model.
    patches: (B, N, 3, P, P) tensor.
    top_k: patches kept for the signed top-k aggregation (None = all).
    mu_patch: validation patch-mean for the deviation center.
    Returns: dict with 'loss' and 'mahalanobis' tensors of shape (B,).
    """
    loss_pp, maha_pp = _per_patch_scores(model, patches)
    loss, maha = aggregate_scores(loss_pp, maha_pp, top_k, mu_patch)
    return {"loss": loss, "mahalanobis": maha}


def _collect_val_scores(model, val_dataloader, top_k=None):
    """Collect per-image validation scores and the patch-mean.
    model: FastFlow model.
    val_dataloader: validation DataLoader (real images).
    top_k: patches kept for the signed top-k aggregation (None = all).
    Returns: (per-image losses numpy array, mu_patch float or None).
    """
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
    """Train the flow for one epoch (per-patch loss).
    dataloader: training DataLoader yielding (B, K, 3, P, P) patches.
    model: FastFlow model.
    optimizer: optimizer.
    epoch: zero-based epoch index.
    args: parsed CLI args.
    scheduler: optional LR scheduler stepped on the loss.
    Returns: (train_mean, train_std, per-patch loss array).
    """
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
    """Calibrate the anomaly threshold on the validation reals.
    val_dataloader: validation DataLoader (real images).
    model: FastFlow model.
    use_lof: fit a Local Outlier Factor instead of a Gaussian band.
    contamination: LOF contamination parameter.
    alpha: target false-positive rate for the Gaussian band.
    top_k: patches kept for the signed top-k aggregation (None = all).
    Returns: dict with mean/std, band or lof, losses, mu_patch and top_k.
    """
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
    """Compute metrics for one fake class against a real baseline.
    c: fake class id.
    class_masks: dict {class_id: boolean mask}.
    labels: integer labels for all samples.
    preds: binary predictions for all samples.
    preds_: continuous scores for all samples.
    class2idx: mapping class id -> name.
    y_true_0_base: labels of the baseline real class.
    y_pred_0_base: binary predictions of the baseline real class.
    preds_0_base: continuous scores of the baseline real class.
    mu_val: validation real mean, fold center for the bilateral AP/ROC.
    Returns: dict of per-class metrics, or None if the class is empty.
    """
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

    _fpr, _tpr, _thr = roc_curve(y_true_binary, scores_balanced)
    _best            = np.argmax(_tpr - _fpr)
    _preds_oracle    = (scores_balanced >= _thr[_best]).astype(np.int8)

    _mu_real          = float(preds_0_base.mean())
    _distances        = np.abs(scores_balanced - _mu_real)
    _fpr_b, _tpr_b, _thr_b = roc_curve(y_true_binary, _distances)
    _best_b           = np.argmax(_tpr_b - _fpr_b)
    _preds_oracle_b   = (_distances >= _thr_b[_best_b]).astype(np.int8)

    scores_bilateral = np.abs(scores_balanced - mu_val)

    return {
        'class_id':          c, 'class_name': class_name, 'min_len': min_len,
        'loss_mean':         loss_mean_fake, 'loss_std': loss_std_fake,
        'accuracy':          accuracy_score(y_true_binary, y_pred_balanced),
        'acc_oracle':        accuracy_score(y_true_binary, _preds_oracle),
        'acc_oracle_bi':     accuracy_score(y_true_binary, _preds_oracle_b),
        'thr_oracle':        float(_thr[_best]),
        'thr_oracle_bi':     float(_thr_b[_best_b]),
        'mu_oracle_bi':      _mu_real,
        'ap':                average_precision_score(y_true_binary, scores_bilateral),
        'roc':               roc_auc_score(y_true_binary, scores_bilateral),
    }


def eval_once(dataloader, model, epoch=None, class2idx=None, threshold_info=None,
              wandb_log=False, wandb_prefix: str = ""):
    """Score the test set and compute all metrics.
    dataloader: test DataLoader yielding (patches, label).
    model: FastFlow model.
    epoch: zero-based epoch index, or None outside training.
    class2idx: mapping class id -> name.
    threshold_info: calibration dict from compute_threshold.
    wandb_log: log metrics to Weights & Biases.
    wandb_prefix: prefix prepended to W&B keys.
    Returns: (mean_roc_ood, per-image scores, labels, metrics_summary).
    """
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
        if c != 0 and c != 99:
            mask_fake |= class_masks[c]

    if epoch is not None and epoch % 50 == 0 and not wandb_prefix:
        likelihood_real = preds_[class_masks[0]]
        likelihood_ood_real = preds_[class_masks.get(99, np.zeros(len(labels), dtype=bool))]
        likelihood_fake = preds_[mask_fake]

        plt.figure(figsize=(10, 6))
        if len(likelihood_real) > 0:
            sns.kdeplot(likelihood_real, label='Real', color='blue')
        if len(likelihood_ood_real) > 0:
            sns.kdeplot(likelihood_ood_real, label='OOD Real', color='green')
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

    preds_99         = np.array([])
    y_pred_ood_real  = np.array([], dtype=int)
    y_true_ood_real  = np.array([])
    mean_roc_ood     = 0.0
    acc_ood_real     = 0.0
    class_ood_real_name = "OOD_Real"

    if 99 in class_masks:
        class_ood_real_name = class2idx[99] if class2idx else "OOD_Real"
        mask_99         = class_masks[99]
        y_true_ood_real = labels[mask_99]
        y_pred_ood_real = preds[mask_99]
        preds_99        = preds_[mask_99]

        if len(preds_99) > 0:
            per_class_metrics[f"loss_per_class/{class_ood_real_name}"]     = float(preds_99.mean())
            per_class_metrics[f"loss_std_per_class/{class_ood_real_name}"] = float(preds_99.std())
            acc_ood_real = accuracy_score(
                np.zeros(len(y_true_ood_real), dtype=np.int8), y_pred_ood_real)
            per_class_metrics[f"acc_per_class/{class_ood_real_name}"] = acc_ood_real

    acc_real = accuracy_score(np.zeros(len(y_true_0), dtype=np.int8), y_pred_0) \
               if len(y_true_0) > 0 else 0.0

    if 'lof' in threshold_info:
        deployed_thr_str = "thr=LOF"
    else:
        deployed_thr_str = f"thr=[{threshold_info['l_threshold']:.4f}, {threshold_info['u_threshold']:.4f}]"

    classes_to_process = [c for c in classes[1:] if c != 99]

    print("\nComputing metrics using in-domain Real as baseline...")
    print("-" * 30)
    print(f"  > Class In-domain Real      : \t Accuracy = {acc_real:.4f} ({deployed_thr_str}), \t Loss = {preds_0.mean() if len(preds_0) > 0 else 0:.4f}±{preds_0.std() if len(preds_0) > 0 else 0:.4f}")
    print("=" * 30)
    metrics_real_start_time = time.time()
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(_compute_class_metrics)(
            c, class_masks, labels, preds, preds_, class2idx, y_true_0, y_pred_0, preds_0, mu_val
        ) for c in classes_to_process
    )

    accs_oracle, accs_oracle_bi = [], []
    for result in results:
        if result is None:
            continue
        print(f"  > Class {result['class_name']} : \t Accuracy = {result['accuracy']:.4f} "
              f"(oracle={result['acc_oracle']:.4f} @thr={result['thr_oracle']:.4f}, "
              f"bi={result['acc_oracle_bi']:.4f} @[{result['mu_oracle_bi']-result['thr_oracle_bi']:.4f}, "
              f"{result['mu_oracle_bi']+result['thr_oracle_bi']:.4f}]), "
              f"\t AP = {result['ap']:.4f}, \t ROC AUC = {result['roc']:.4f}, \t Loss = {result['loss_mean']:.4f}±{result['loss_std']:.4f}")
        print("-" * 30)
        per_class_metrics[f"loss_per_class/{result['class_name']}"]          = result['loss_mean']
        per_class_metrics[f"loss_std_per_class/{result['class_name']}"]      = result['loss_std']
        per_class_metrics[f"acc_per_class/{result['class_name']}"]           = result['accuracy']
        per_class_metrics[f"acc_oracle_per_class/{result['class_name']}"]    = result['acc_oracle']
        per_class_metrics[f"acc_oracle_bi_per_class/{result['class_name']}"] = result['acc_oracle_bi']
        per_class_metrics[f"ap_per_class/{result['class_name']}"]            = result['ap']
        per_class_metrics[f"roc_per_class/{result['class_name']}"]           = result['roc']
        aps.append(result['ap'])
        accs.append(result['accuracy'])
        accs_oracle.append(result['acc_oracle'])
        accs_oracle_bi.append(result['acc_oracle_bi'])
        rocs.append(result['roc'])
        cls.append(result['class_id'])

    mean_acc           = np.mean(accs)
    mean_acc_oracle    = np.mean(accs_oracle)
    mean_acc_oracle_bi = np.mean(accs_oracle_bi)
    mean_ap            = np.mean(aps)
    mean_roc           = np.mean(rocs)

    metrics_real_time = time.time() - metrics_real_start_time
    print("-" * 30)
    _print_family_summary(_aggregate_by_family(results))
    if epoch is not None:
        print(f"⏱️  Metrics computation time (Real baseline): {int(metrics_real_time // 60)}m {int(metrics_real_time % 60)}s")
    print("=" * 30 + "\n")

    aps_ood, accs_ood, rocs_ood = [], [], []
    mean_acc_ood = mean_ap_ood = mean_roc_ood = 0.0
    mean_acc_oracle_bi_ood = 0.0
    accs_oracle_ood = []

    if 99 in class_masks and len(preds_99) > 0:
        print("Computing metrics using OOD Real as baseline...")
        print("-" * 30)
        print(f"  > Class {class_ood_real_name} : \t Accuracy = {acc_ood_real:.4f} ({deployed_thr_str}), \t Loss = {preds_99.mean():.4f}±{preds_99.std():.4f}")
        print("=" * 30)
        metrics_ood_start_time = time.time()
        results_ood = Parallel(n_jobs=-1, backend='threading')(
            delayed(_compute_class_metrics)(
                c, class_masks, labels, preds, preds_, class2idx, y_true_ood_real, y_pred_ood_real, preds_99, mu_val
            ) for c in classes_to_process
        )
        accs_oracle_ood, accs_oracle_bi_ood = [], []
        for result in results_ood:
            if result is None:
                continue
            print(f"  > Class {result['class_name']} : \t Accuracy = {result['accuracy']:.4f} "
                  f"(oracle={result['acc_oracle']:.4f} @thr={result['thr_oracle']:.4f}, "
                  f"bi={result['acc_oracle_bi']:.4f} @[{result['mu_oracle_bi']-result['thr_oracle_bi']:.4f}, "
                  f"{result['mu_oracle_bi']+result['thr_oracle_bi']:.4f}]), "
                  f"\t AP = {result['ap']:.4f}, \t ROC AUC = {result['roc']:.4f}, \t Loss = {result['loss_mean']:.4f}±{result['loss_std']:.4f}")
            print("-" * 30)
            per_class_metrics[f"acc_per_class_ood/{result['class_name']}"]           = result['accuracy']
            per_class_metrics[f"acc_oracle_per_class_ood/{result['class_name']}"]    = result['acc_oracle']
            per_class_metrics[f"acc_oracle_bi_per_class_ood/{result['class_name']}"] = result['acc_oracle_bi']
            per_class_metrics[f"ap_per_class_ood/{result['class_name']}"]            = result['ap']
            per_class_metrics[f"roc_per_class_ood/{result['class_name']}"]           = result['roc']
            aps_ood.append(result['ap'])
            accs_ood.append(result['accuracy'])
            accs_oracle_ood.append(result['acc_oracle'])
            accs_oracle_bi_ood.append(result['acc_oracle_bi'])
            rocs_ood.append(result['roc'])

        mean_acc_ood = np.mean(accs_ood) if len(accs_ood) > 0 else 0.0
        mean_ap_ood  = np.mean(aps_ood)  if len(aps_ood)  > 0 else 0.0
        mean_roc_ood = np.mean(rocs_ood) if len(rocs_ood) > 0 else 0.0

        metrics_ood_time = time.time() - metrics_ood_start_time
        print("-" * 30)
        _print_family_summary(_aggregate_by_family(results_ood))
        if epoch is not None:
            print(f"⏱️  Metrics computation time (OOD Real baseline): {int(metrics_ood_time // 60)}m {int(metrics_ood_time % 60)}s")
        print("=" * 30 + "\n")

        per_class_metrics["Val acc OOD"] = mean_acc_ood
        per_class_metrics["Val AP OOD"] = mean_ap_ood
        per_class_metrics["Val ROC OOD"] = mean_roc_ood

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

        _fg, _tg, _thr_g = roc_curve(yt_g, sc_g)
        _og = np.argmax(_tg - _fg)
        acc_oracle_g = accuracy_score(yt_g, (sc_g >= _thr_g[_og]).astype(np.int8))

        _mu_g  = float(preds_0.mean())
        _dist_g = np.abs(sc_g - _mu_g)
        _fgb, _tgb, _thr_gb = roc_curve(yt_g, _dist_g)
        _ogb = np.argmax(_tgb - _fgb)
        acc_oracle_bi_g = accuracy_score(yt_g, (_dist_g >= _thr_gb[_ogb]).astype(np.int8))

        print(f"\n{'=' * 30}")
        print(f"Global — Real vs All Fake")
        print(f"  Acc={acc_g:.4f}  (oracle={acc_oracle_g:.4f}, bi={acc_oracle_bi_g:.4f})")
        print(f"  AP={ap_g:.4f}   ROC AUC={roc_g:.4f}")
        print(f"{'=' * 30}\n")

        per_class_metrics["global/acc"]           = acc_g
        per_class_metrics["global/acc_oracle"]    = acc_oracle_g
        per_class_metrics["global/acc_oracle_bi"] = acc_oracle_bi_g
        per_class_metrics["global/ap"]            = ap_g
        per_class_metrics["global/roc"]           = roc_g

        global_metrics = {
            'n_balanced':    int(min_n),
            'n_fake_total':  int(len(preds_fake)),
            'acc':           float(round(acc_g, 6)),
            'acc_oracle':    float(round(acc_oracle_g, 6)),
            'acc_oracle_bi': float(round(acc_oracle_bi_g, 6)),
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
                'acc_oracle_bi': _r(r['acc_oracle_bi']),
                'ap':            _r(r['ap']),
                'roc':           _r(r['roc']),
                'loss_mean':     _r(r['loss_mean'], 4),
                'loss_std':      _r(r['loss_std'],  4),
            }
            for r in result_list if r is not None
        }

    metrics_summary = {
        'in_domain_real': {
            'mean_acc':           _r(mean_acc),
            'mean_acc_oracle':    _r(mean_acc_oracle),
            'mean_acc_oracle_bi': _r(mean_acc_oracle_bi),
            'mean_ap':            _r(mean_ap),
            'mean_roc':           _r(mean_roc),
            'per_class':          _per_class_dict(results),
        },
    }
    if 99 in class_masks and len(preds_99) > 0:
        metrics_summary['vs_ood_real'] = {
            'mean_acc':           _r(mean_acc_ood),
            'mean_acc_oracle':    _r(np.mean(accs_oracle_ood) if accs_oracle_ood else 0),
            'mean_acc_oracle_bi': _r(mean_acc_oracle_bi_ood),
            'mean_ap':            _r(mean_ap_ood),
            'mean_roc':           _r(mean_roc_ood),
            'per_class':          _per_class_dict(results_ood),
        }
    if global_metrics:
        metrics_summary['global'] = global_metrics

    return mean_roc_ood, preds_, labels, metrics_summary


def train(args, config, canonical_checkpoint_dir):
    """Run the full training loop with champion promotion.
    args: parsed CLI args.
    config: backbone config dict.
    canonical_checkpoint_dir: champion folder for the best run.
    """
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

                print(f"Epoch {epoch+1}: New best ROC OOD: {best_metric:.4f}. Saving model.")
                wandb.run.summary["best_roc_ood"] = best_metric

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
                print(f"Epoch {epoch+1}: ROC OOD ({current_metric:.4f}) not improved compared to {best_metric:.4f}. Patience: {patience_counter}/{args.early_stopping_patience}")

        if patience_counter >= args.early_stopping_patience:
            print(f"Stopping early at epoch {epoch + 1} after {args.early_stopping_patience} epochs without improvement of ROC OOD.")
            break

    print(f"Training finished. Best ROC OOD: {best_metric:.4f}")

    canonical_best_metric = _load_canonical_metric(canonical_checkpoint_dir)
    print(f"[champion] This run: {best_metric:.4f}  |  Canonical: {canonical_best_metric:.4f}")
    if best_metric > canonical_best_metric:
        _promote_to_canonical(checkpoint_dir, canonical_checkpoint_dir)
        print(f"[champion] New champion!  {best_metric:.4f} > {canonical_best_metric:.4f}")
    else:
        print(f"[champion] No promotion. Canonical remains at {canonical_best_metric:.4f}")

    _cleanup_temp(checkpoint_dir)


if __name__ == "__main__":
    args = parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    pprint(vars(args))
    pprint(config)

    canonical_run_name = f"{config['backbone_name']}_{args.reals}_{config['pooling_type']}"
    if args.use_lof:
        canonical_run_name += "_lof"
    else:
        canonical_run_name += f"_t-alpha{args.alpha}"

    if args.run_name is None:
        args.run_name = (canonical_run_name +
                         f"_np{args.num_train_patches}_lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}"
                         f"_ld{args.lr_decay}_lp{args.lr_patience}")
        if args.top_k:
            args.run_name += f"_topk{args.top_k}"

    canonical_checkpoint_dir = os.path.join(const.CHECKPOINT_DIR, canonical_run_name)
    const.CHECKPOINT_DIR     = os.path.join(const.CHECKPOINT_DIR, args.run_name)

    print(f"[champion] Canonical dir : {canonical_checkpoint_dir}")
    print(f"[champion] Temporary dir : {const.CHECKPOINT_DIR}")

    train(args, config, canonical_checkpoint_dir=canonical_checkpoint_dir)
