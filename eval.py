import argparse
import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import wandb
from ignite.contrib import metrics
import joblib

from muflow import constants as const
from muflow import dataset
from muflow import model as fastflow
#from muflow import vae
from muflow import utils

import cv2, pdb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.neighbors import LocalOutlierFactor

GANS = ['StyleGAN', 'StyleGAN2', 'StyleGAN3', 'STARGAN', 'AttGAN', 'GDWCT']
DM_OPEN = ['Flux.1', 'Stable DIffusion 3.5', 'Stable Diffusion XL', 'Stable Cascade', 'Stable Diffusion Attend and Excite']
DM_CLOSED = ['Dall-E 3', 'Midjourney', 'Starry AI', 'Deep AI', 'Hotpot AI', 'Nvidia Sana PAG', 'Tencent Hunyuan', 'Flux.1.1 Pro']

MIX_2CLASS = ['StyleGAN', 'StyleGAN2', 'Stable DIffusion 3.5', 'Flux.1.1 Pro']




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/resnet18.yaml', help="path to config file")

    parser.add_argument("--data", type=str, default='WILD', help="path to mvtec folder", choices=['FF++', 'WILD', 'progan'])
    parser.add_argument("--reals", type=str, default='ffhq', help="reals dataset", choices=['ffhq', 'celeba_hq', 'ffhq+celeba_hq'])
    parser.add_argument("--test", type=str, help="test name")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to load checkpoint")
    parser.add_argument('--gmm_checkpoint', type=str, help="path to GMM checkpoint (for GMM-based threshold)")
    parser.add_argument('--threshold_path', type=str, help="path to threshold file (.npz or .npy)")
    parser.add_argument('--lof_checkpoint', type=str, help="path to LOF model checkpoint (.pkl)")

    parser.add_argument('--use_augs', type=int, default=0, choices=[0, 1], help="Whether to use data augmentation")
    parser.add_argument('--use_fourier', type=int, default=0, choices=[0, 1], help="Whether to use Fourier transform")
    parser.add_argument('--backbone_weights', type=str, help="path to load backbone weights")
    parser.add_argument('--model_type', type=str, choices=['FastFlow', 'VAE'], default='FastFlow', help="Choose the model to train")
    parser.add_argument('--on_celeba', action='store_true', help="Use celeba dataset")
    parser.add_argument('--num_workers', type=int, default=4, help="number of data loading workers")
    parser.add_argument('--pooling_type', type=str, default='mean', choices=['mean', 'flatten', 'max', 'mean_std'], help="Spatial pooling type: mean, flatten, max, or mean_std")
    parser.add_argument('--n_components', type=int, default=1, help="Number of GMM components")

    # Threshold method arguments
    parser.add_argument('--use_lof', type=int, default=0, choices=[0, 1], help="Whether to use LOF (requires lof_checkpoint)")
    parser.add_argument('--contamination', default='auto', help="Contamination rate for LOF")
    parser.add_argument('--use_percentile', type=int, default=0, choices=[0, 1], help="Whether to use percentile")
    parser.add_argument('--percentile', type=int, default=95, help="Percentile to use if use_percentile=True")

    args = parser.parse_args()
    
    return args

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
    """
    Build model - unified function shared with main.py
    """
    # Get out_indices for filename
    out_indices = config.get("out_indices", [1, 2, 3])
    out_indices_str = str(out_indices)
    pooling_type = getattr(args, 'pooling_type', 'mean')
    
    # Try new naming convention (with out_indices and pooling_type)
    if args.use_fourier == 1:
        gmm_parameters = f"{const.WORKING_DIR}/parameters/{args.n_components}-gmm_parameters_{config['backbone_name']}_indices_{out_indices_str}_fourier_{args.reals}_{config['input_size']}_{pooling_type}.npy"
    else:
        gmm_parameters = f"{const.WORKING_DIR}/parameters/{args.n_components}-gmm_parameters_{config['backbone_name']}_indices_{out_indices_str}_{args.reals}_{config['input_size']}_{pooling_type}.npy"
    
    gmm_values = np.load(gmm_parameters, allow_pickle=True).item() 
    print(f"Loading gmm parameters from {gmm_parameters}")

    if model_type == "FastFlow":
        model = fastflow.FastFlow(
            backbone_name=config["backbone_name"],
            flow_steps=config["flow_step"],
            input_size=config["input_size"],
            conv3x3_only=config["conv3x3_only"],
            hidden_ratio=config["hidden_ratio"],
            gmm_values=gmm_values,
            in_channels=3,  # Always 3 channels (RGB or replicated Fourier magnitude)
            backbone_weights=args.backbone_weights if hasattr(args, 'backbone_weights') and args.backbone_weights else None,
            out_indices=config.get("out_indices", [1, 2, 3]),  # Default [1,2,3] if not specified
            pooling_type=pooling_type
        )
        print(
            "Model A.D. Param#: {}".format(
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            )
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model

def eval_once(dataloader, model, epoch=None, model_type="FastFlow", gmm=None, threshold_info=None, class2idx=None): 
    model.eval()
    labels_list = []
    preds_list = []
    for data, targets in dataloader:
        data, targets = data.cuda(), targets.cuda()
        with torch.no_grad():
            ret = model(data) if model_type == "FastFlow" else model(data, eval_mode=True)
        
        if model_type == "FastFlow":
            # Move to CPU immediately to free GPU memory
            outputs = ret["loss"].cpu()
        
        preds_list.append(outputs)
        labels_list.append(targets.cpu())

    print("Testing done")

    # Concatenate tensors directly for better memory efficiency
    preds_ = torch.cat(preds_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()

    # Visualization: plot real vs fake separation
    likelihood_real = preds_[labels == 0]
    likelihood_fake = preds_[labels > 0]

    # Apply threshold or GMM/LOF to get predictions
    if threshold_info and 'lof' in threshold_info:
        # Use LOF prediction
        lof = threshold_info['lof']
        preds = (lof.predict(np.array(preds_).reshape(-1,1)) < 0).astype(int)
    elif threshold_info:
        # Use standard threshold
        threshold = threshold_info['threshold']
        preds = (preds_ > threshold).astype(int)
    else:
        raise ValueError("Either lof or threshold_info must be provided")
    
    # --- INIZIO NUOVA VALUTAZIONE BILANCIATA (One-vs-All) ---
    # Print the accuracy only for class 0 (real class)
    idx_real = (labels == 0)
    acc_real = accuracy_score(labels[idx_real], preds[idx_real])
    num_real = np.sum(idx_real)
    num_real_correct = np.sum((labels[idx_real] == preds[idx_real]))
    print(f"Accuracy for class 0 (real): {acc_real:.4f} ({num_real_correct}/{num_real} caught)")
    
    aps, aps_gan, aps_dmo, aps_dmc, aps_mix = [], [], [], [], []
    accs, accs_gan, accs_dmo, accs_dmc, accs_mix = [], [], [], [], []
    cls = []  # Lista delle classi valutate

    classes = np.unique(labels)
    y_true_0 = labels[labels == 0]
    y_pred_0 = preds[labels == 0]

    np.random.seed(42)
    
    for c in classes[1:]:
        
        idx = (labels == c)
        y_true_c = labels[idx]
        y_pred_c = preds[idx]
        
        min_len = min(len(y_true_c), len(y_true_0))
        if min_len == 0:
            print(f"  > Class {class2idx[c] if class2idx else c}: SKIPPED (0 samples)")
            continue
        
        # Get class name for logging
        class_name = class2idx[c] if class2idx else str(c)
        
        # Compute loss statistics for this class
        loss_mean_fake = np.mean(preds_[labels == c])
        loss_std_fake = np.std(preds_[labels == c])
        
        #Shuffling real to get different subset each time
        idx = np.random.permutation(len(y_true_0))
        y_true_0 = y_true_0[idx]
        y_pred_0 = y_pred_0[idx]

        y_true_balanced = np.array(y_true_c[:min_len].tolist() + y_true_0[:min_len].tolist())
        y_pred_balanced = np.array(y_pred_c[:min_len].tolist() + y_pred_0[:min_len].tolist())
        
        y_true_binary = (y_true_balanced > 0).astype(np.int8)
        
        ap = average_precision_score(y_true_binary, y_pred_balanced)
        acc0 = accuracy_score(y_true_binary, y_pred_balanced)
        
        print(f"  > Class {class_name} (N={min_len*2}): \t AP = {ap:.4f}, \t Accuracy = {acc0:.4f}, \t Loss = {loss_mean_fake:.4f}±{loss_std_fake:.4f}")
        
        aps.append(ap)
        accs.append(acc0)
        cls.append(c)

        if class2idx and class2idx[c] in GANS:
            aps_gan.append(ap)
            accs_gan.append(acc0)

        elif class2idx and class2idx[c] in DM_OPEN:
            aps_dmo.append(ap)
            accs_dmo.append(acc0)

        elif class2idx and class2idx[c] in DM_CLOSED:
            aps_dmc.append(ap)
            accs_dmc.append(acc0)

        elif class2idx and class2idx[c] in MIX_2CLASS:
            aps_mix.append(ap)
            accs_mix.append(acc0)
    
    if aps_gan:
        print(f'Result in GANS: acc {np.mean(np.array(accs_gan)):.4f} ap {np.mean(np.array(aps_gan)):.4f}')
    if aps_dmo:
        print(f'Result in DMO: acc {np.mean(np.array(accs_dmo)):.4f} ap {np.mean(np.array(aps_dmo)):.4f}')
    if aps_dmc:
        print(f'Result in DMC: acc {np.mean(np.array(accs_dmc)):.4f} ap {np.mean(np.array(aps_dmc)):.4f}')
    if aps_mix:
        print(f'Result in MIX: acc {np.mean(np.array(accs_mix)):.4f} ap {np.mean(np.array(aps_mix)):.4f}')

    mean_ap = np.mean(aps)
    mean_acc = np.mean(accs)
    
    print("-" * 30)
    print(f"Average Accuracy: {mean_acc:.4f}")
    print(f"Average Precision: {mean_ap:.4f}")
    print("="*30 + "\n")


def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))

    checkpoint = torch.load(args.checkpoint)

    # Load GMM if provided
    gmm = joblib.load(args.gmm_checkpoint) if args.gmm_checkpoint else None

    model = build_model(config, args.model_type, args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.cuda()

    # Load threshold_info: can be a dict (from .npz) or numpy array (legacy .npy)
    threshold_info = None
    threshold_path = args.threshold_path
    
    # If threshold_path not provided, try to find it in the same directory as checkpoint
    if not threshold_path:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        potential_threshold_path = os.path.join(checkpoint_dir, "thresholds.npz")
        if os.path.exists(potential_threshold_path):
            threshold_path = potential_threshold_path
            print(f"Found threshold file: {threshold_path}")
    pdb.set_trace()
    if threshold_path:
        if threshold_path.endswith('.npz'):
            # New format: dictionary with threshold, mean, std, losses
            loaded = np.load(threshold_path, allow_pickle=True)
            threshold_info = {key: loaded[key] for key in loaded.files}
            # If 'threshold' is an array, extract scalar
            if 'threshold' in threshold_info and isinstance(threshold_info['threshold'], np.ndarray):
                if threshold_info['threshold'].size == 1:
                    threshold_info['threshold'] = float(threshold_info['threshold'])
        else:
            # Legacy format: just a dict with mean and std
            threshold_data = np.load(threshold_path, allow_pickle=True)
            if isinstance(threshold_data, np.ndarray) and threshold_data.dtype == object:
                # It's a dict stored as object array
                threshold_info = threshold_data.item()
            else:
                # Assume it's mean/std values
                threshold_info = {'mean': threshold_data[0], 'std': threshold_data[1], 'threshold': threshold_data[0] + 3*threshold_data[1]}
    
    # Load LOF model if provided (or auto-detect in checkpoint dir)
    lof_checkpoint = args.lof_checkpoint if hasattr(args, 'lof_checkpoint') else None
    if not lof_checkpoint and args.use_lof == 1:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        potential_lof_path = os.path.join(checkpoint_dir, "lof_model.pkl")
        if os.path.exists(potential_lof_path):
            lof_checkpoint = potential_lof_path
            print(f"Found LOF model: {lof_checkpoint}")
    
    if args.use_lof == 1 and lof_checkpoint:
        lof = joblib.load(lof_checkpoint)
        if threshold_info is None:
            threshold_info = {}
        threshold_info['lof'] = lof
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
        # ('crop', {'crop_ratio': 0.8}),
        ('horizontal_flip', {})
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
        print(f"Evaluating with attack: {attack_type} {attack_params}")
        print(f"{'='*60}\n")

        if not args.on_celeba:
            test_dataloader, class2idx = create_dataloader(args, config, opt)
            eval_once(test_dataloader, model, model_type=args.model_type, gmm=gmm, threshold_info=threshold_info, class2idx=class2idx)
        else:
            test_dataloader, class2idx = create_dataloader_w_celeba(args, config, opt)
            eval_once(test_dataloader, model, model_type=args.model_type, gmm=gmm, threshold_info=threshold_info, class2idx=class2idx)


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
