import argparse
from pprint import pprint
import os, pdb
import subprocess
import sys
import time
import timm
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import wandb
import joblib
import GPUtil

from muflow import constants as const
from muflow import dataset
from muflow import model as fastflow
from muflow import utils

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from joblib import Parallel, delayed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/resnet50.yaml', help="path to config file")

    parser.add_argument("--data", type=str, default='WILD', help="path to mvtec folder", choices=['FF++', 'WILD', 'progan'])
    parser.add_argument("--reals", type=str, default='ffhq', help="reals dataset", choices=['ffhq', 'celeba_hq', 'ffhq+celeba_hq'])
    parser.add_argument("--checkpoint", type=str, help="path to load checkpoint")

    parser.add_argument('--wandb', default= 'online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--use_augs', type=int, default=0, choices=[0, 1], help="Whether to use data augmentation")
    parser.add_argument('--use_fourier', type=int, default=0, choices=[0, 1], help="Whether to use Fourier transform")
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--backbone_weights', type=str, help="path to load backbone weights")
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4, help="number of data loading workers")
    parser.add_argument('--debug', action='store_true', help="Debug mode with reduced dataset size")
    parser.add_argument('--gpu_id', type=int, default=None, help="Manually specify GPU ID to use (default: auto-select GPU with most free memory)")

    # Training Hyperparameters
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'sgd'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--scheduler', type=int, default=1, choices=[0, 1], help="Whether to use scheduler")
    parser.add_argument('--lr_decay', type=float, default=0.3)
    parser.add_argument('--lr_patience', type=int, default=35)

    # Model Hyperparameters
    parser.add_argument('--alpha', type=float, default=0.01, help="Target false positive rate under the normality assumption, i.e., the probability of flagging a normal sample as anomalous.")
    parser.add_argument('--use_lof', type=int, default=0, choices=[0, 1], help="Whether to use LOF")
    parser.add_argument('--contamination', default='auto')
    
    # Projection Layer Hyperparameters
    parser.add_argument('-use_proj', '--use_proj_layer', type=int, default=0, choices=[0, 1], help="Whether to use projection layer with contrastive learning")
    parser.add_argument('-proj_hidden', '--proj_hidden_ratio', type=float, default=0.5, help="Hidden layer ratio for projection layer")
    parser.add_argument('-l_contr', '--lambda_contrastive', type=float, default=1.0, help="Weight for contrastive loss")
    parser.add_argument('-l_rec', '--lambda_reconstruction', type=float, default=1.0, help="Weight for reconstruction loss")
    parser.add_argument('-proj_noise', '--proj_noise_std', type=float, default=0.1, help="Noise std for creating negative pairs in projection training")
    parser.add_argument('-margin', '--contrastive_margin', type=float, default=2.0, help="Margin for contrastive loss")

    # Adversarial Training Hyperparameters
    parser.add_argument('--use_adversarial', type=int, default=0, choices=[0, 1], help="Whether to use adversarial training")
    parser.add_argument('--adv_strategy', type=str, default='same_batch', choices=['same_batch', 'alternate_batch', 'cycle_2_1'], help="Strategy for adversarial training")
    parser.add_argument('--adv_lambda', type=float, default=-1, help="Weight for adversarial loss (negative to maximize)")
    parser.add_argument('--noise_std', type=float, default=0.1, help="Standard deviation of Gaussian noise")
    parser.add_argument('--noise_schedule', type=str, default='fixed', choices=['fixed', 'linear', 'exponential'], help="Noise schedule during training")
    parser.add_argument('--noise_std_max', type=float, default=1.0, help="Maximum std if using noise schedule")
    parser.add_argument('--noise_differentiable', type=int, default=0, choices=[0, 1], help="Whether noise should be differentiable")

    # Autoencoder Training Hyperparameters
    parser.add_argument('--use_autoencoder', type=int, default=0, choices=[0, 1], help="Whether to use convolutional autoencoder on backbone features")
    parser.add_argument('--ae_lambda', type=float, default=1.0, help="Weight for autoencoder reconstruction loss")
    parser.add_argument('--ae_hidden_ratio', type=float, default=0.5, help="Hidden channel ratio for autoencoder bottleneck")
    parser.add_argument('--ae_training_mode', type=str, default='two_phase', choices=['two_phase', 'end_to_end'],
                        help="AE training strategy: 'two_phase' alternates AE/NF phases per epoch; "
                             "'end_to_end' trains AE+NF jointly with a single optimizer")

    parser.add_argument('-patience', '--early_stopping_patience', type=float, default=float("inf"), help="Patience epochs for early stopping based on Val Acc")

    args = parser.parse_args()
    
    return args


def select_best_gpu():
    """
    Automatically select the GPU with the most free memory.
    
    Returns:
        int: GPU ID with most free memory, or 0 if no GPU available
    """
    if not torch.cuda.is_available():
        print("No CUDA GPUs available, using CPU")
        return None
    
    gpus = GPUtil.getGPUs()
    if not gpus:
        print("No GPUs found by GPUtil, using cuda:0")
        return 0
    
    best_gpu = max(gpus, key=lambda gpu: gpu.memoryFree)
    print(f"🎯 Auto-selected GPU {best_gpu.id}: {best_gpu.name} (Free: {best_gpu.memoryFree}MB / {best_gpu.memoryTotal}MB)")
    
    return best_gpu.id


def _build_data_loader_common(args, config, is_train, is_val, shuffle, drop_last, return_class2idx=False):
    """
    Helper function to create DataLoader with common logic.
    
    Args:
        args: Arguments object
        config: Config dictionary
        is_train: Whether this is training data
        is_val: Whether this is validation data
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop last incomplete batch
        return_class2idx: Whether to return class_to_idx mapping
    
    Returns:
        DataLoader and optionally class_to_idx mapping
    """
    dataset_instance = dataset.Dataset(
        dataset_name=args.data,
        reals_name=args.reals,
        input_size=config["input_size"],
        is_train=is_train,
        is_val=is_val,
        use_fourier=True if args.use_fourier == 1 else False,
        use_augs=True if args.use_augs == 1 else False,
        debug=args.debug
    ).create_dataset()
    
    num_workers = getattr(args, 'num_workers', 4)
    dataloader = torch.utils.data.DataLoader(
        dataset_instance,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,  # Faster CPU-GPU transfer
        persistent_workers=True if num_workers > 0 else False,  # Reuse workers
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches for efficiency
    )
    
    if return_class2idx:
        return dataloader, {v: k for k, v in dataset_instance.class_to_idx.items()}
    return dataloader


def build_train_data_loader(args, config):
    return _build_data_loader_common(args, config, is_train=True, is_val=False, shuffle=True, drop_last=True)


def build_val_data_loader(args, config):
    return _build_data_loader_common(args, config, is_train=True, is_val=True, shuffle=False, drop_last=False)


def build_test_data_loader(args, config):
    return _build_data_loader_common(args, config, is_train=False, is_val=False, shuffle=False, drop_last=False, return_class2idx=True)


def build_pair_data_loader(args, config):
    """
    Build dataloader for projection layer training with PairDataset.
    Returns pairs of real images for contrastive learning.
    """
    from muflow.dataset import PairDataset
    
    # Create PairDataset instance
    pair_dataset = PairDataset(
        root_dir=[f"{dataset.DATA_DIR}/ffhq/*"] if args.reals == 'ffhq' else [f"{dataset.DATA_DIR}/celeba_hq/train/*"],
        file_pattern="*.*g",
        input_size=config["input_size"],
        is_train=True,
        is_val=False,
        reals_name=args.reals,
        use_fourier=True if args.use_fourier == 1 else False,
        use_augs=True if args.use_augs == 1 else False,
        debug=args.debug
    )
    
    num_workers = getattr(args, 'num_workers', 4)
    dataloader = torch.utils.data.DataLoader(
        pair_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    return dataloader


def build_model(config, args):
    out_indices = config.get("out_indices", [1, 2, 3])
    out_indices_str = str(out_indices)
    pooling_type = config.get("pooling_type", "mean")
    n_components = config.get("gmm_n_components", 1)
    
    if args.use_fourier == 1:
        gmm_parameters = f"{const.WORKING_DIR}/parameters/{n_components}-gmm_parameters_{config['backbone_name']}_indices_{out_indices_str}_fourier_{args.reals}_{config['input_size']}_{pooling_type}.npy"
    else:
        gmm_parameters = f"{const.WORKING_DIR}/parameters/{n_components}-gmm_parameters_{config['backbone_name']}_indices_{out_indices_str}_{args.reals}_{config['input_size']}_{pooling_type}.npy"
    
    if not os.path.exists(gmm_parameters):
        print(f"GMM parameters not found at {gmm_parameters}.")
        print(f"Running generate_parameters.py...")
        cmd = [
            sys.executable,
            os.path.join(const.WORKING_DIR, "generate_parameters.py"),
            "--model_name", config["backbone_name"],
            "--reals", args.reals,
        ]
        if args.use_fourier == 1:
            cmd.append("--use_fourier")
        subprocess.run(cmd, check=True)
        print("GMM parameters generated successfully.")

    gmm_values = np.load(gmm_parameters, allow_pickle=True).item() 
    print(f"Loading gmm parameters from {gmm_parameters}")

    # Prepare adversarial training config
    adversarial_config = {
        'use_adversarial': args.use_adversarial == 1,
        'noise_differentiable': args.noise_differentiable == 1,
    } if hasattr(args, 'use_adversarial') else {'use_adversarial': False, 'noise_differentiable': False}

    model = fastflow.FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
        gmm_values=gmm_values,
        in_channels=1 if args.use_fourier == 1 else 3,
        backbone_weights=args.backbone_weights,
        out_indices=config.get("out_indices", [1, 2, 3]),
        pooling_type=pooling_type,
        use_adversarial=adversarial_config['use_adversarial'],
        noise_differentiable=adversarial_config['noise_differentiable'],
        use_proj_layer=args.use_proj_layer == 1 if hasattr(args, 'use_proj_layer') else False,
        proj_hidden_ratio=args.proj_hidden_ratio if hasattr(args, 'proj_hidden_ratio') else 0.5,
        use_autoencoder=args.use_autoencoder == 1 if hasattr(args, 'use_autoencoder') else False,
        ae_hidden_ratio=args.ae_hidden_ratio if hasattr(args, 'ae_hidden_ratio') else 0.5,
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model


def build_optimizer(args, model, config):
    """
    Build optimizer(s) for the model.

    Returns:
        Standard training : single optimizer for all parameters
        use_proj_layer=1  : tuple (optimizer_proj, optimizer_fastflow)
        use_autoencoder=1 : tuple (optimizer_ae, optimizer_fastflow)
    """
    use_proj = hasattr(args, 'use_proj_layer') and args.use_proj_layer == 1
    use_ae = hasattr(args, 'use_autoencoder') and args.use_autoencoder == 1

    # ---- Autoencoder branch ----
    if use_ae and model.autoencoders is not None:
        ae_mode = getattr(args, 'ae_training_mode', 'two_phase')

        if ae_mode == 'end_to_end':
            # Single optimizer: AE + NF flows trained jointly
            joint_params = list(model.autoencoders.parameters()) + list(model.nf_flows.parameters())
            if args.optimizer == "AdamW":
                return torch.optim.AdamW(joint_params, lr=args.lr, weight_decay=args.weight_decay)
            elif args.optimizer == "sgd":
                return torch.optim.SGD(joint_params, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
            else:
                raise ValueError(f"Unknown optimizer: {args.optimizer}")
        else:
            # two_phase: separate optimizers
            if args.optimizer == "AdamW":
                optimizer_ae = torch.optim.AdamW(
                    model.autoencoders.parameters(),
                    lr=args.lr, weight_decay=args.weight_decay
                )
                optimizer_fastflow = torch.optim.AdamW(
                    model.nf_flows.parameters(),
                    lr=args.lr, weight_decay=args.weight_decay
                )
            elif args.optimizer == "sgd":
                optimizer_ae = torch.optim.SGD(
                    model.autoencoders.parameters(),
                    lr=args.lr, weight_decay=args.weight_decay, momentum=0.9
                )
                optimizer_fastflow = torch.optim.SGD(
                    model.nf_flows.parameters(),
                    lr=args.lr, weight_decay=args.weight_decay, momentum=0.9
                )
            else:
                raise ValueError(f"Unknown optimizer: {args.optimizer}")
            return optimizer_ae, optimizer_fastflow

    # ---- Projection-layer branch ----
    if use_proj and model.projection_layers is not None:
        # Create two separate optimizers for end-to-end training
        if args.optimizer == "AdamW":
            optimizer_proj = torch.optim.AdamW(
                model.projection_layers.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )
            optimizer_fastflow = torch.optim.AdamW(
                model.nf_flows.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )
        elif args.optimizer == "sgd":
            optimizer_proj = torch.optim.SGD(
                model.projection_layers.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                momentum=0.9
            )
            optimizer_fastflow = torch.optim.SGD(
                model.nf_flows.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {args.optimizer}")
        return optimizer_proj, optimizer_fastflow

    # ---- Standard branch ----
    if args.optimizer == "AdamW":
        return torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def train_autoencoder_one_epoch(dataloader, model, optimizer_ae, epoch, args):
    """
    Train autoencoder layers for one epoch using only reconstruction loss.
    FastFlow (nf_flows) is frozen during this phase.

    Args:
        dataloader: Standard training DataLoader
        model: FastFlow model with autoencoders
        optimizer_ae: Optimizer for autoencoder parameters
        epoch: Current epoch number
        args: Arguments namespace

    Returns:
        Average reconstruction loss for the epoch
    """
    import torch.nn.functional as F

    start_time = time.time()
    model.feature_extractor.eval()
    model.nf_flows.eval()
    for ae in model.autoencoders:
        ae.train()

    recon_meter = utils.AverageMeter()
    device = args.device if hasattr(args, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for step, data in enumerate(dataloader):
        data = data.to(device)

        with torch.no_grad():
            # ---- Extract and normalise backbone features (frozen) ----
            bt = model.backbone_type
            if bt == 'cait_deit':
                if isinstance(model.feature_extractor, timm.models.vision_transformer.VisionTransformer):
                    x = model.feature_extractor.patch_embed(data)
                    cls_token = model.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
                    if model.feature_extractor.dist_token is None:
                        x = torch.cat((cls_token, x), dim=1)
                    else:
                        x = torch.cat((cls_token,
                                       model.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                                       x), dim=1)
                    x = model.feature_extractor.pos_drop(x + model.feature_extractor.pos_embed)
                    for i in range(8):
                        x = model.feature_extractor.blocks[i](x)
                    x = model.feature_extractor.norm(x)
                    x = x[:, 2:, :]
                    N, _, C = x.shape
                    x = x.permute(0, 2, 1)
                    x = x.reshape(N, C, model.input_size // 16, model.input_size // 16)
                    features = [x]
                else:  # CaiT
                    x = model.feature_extractor.patch_embed(data)
                    x = x + model.feature_extractor.pos_embed
                    x = model.feature_extractor.pos_drop(x)
                    for i in range(41):
                        x = model.feature_extractor.blocks[i](x)
                    N, _, C = x.shape
                    x = model.feature_extractor.norm(x)
                    x = x.permute(0, 2, 1)
                    x = x.reshape(N, C, model.input_size // 16, model.input_size // 16)
                    features = [x]
            elif bt == 'dino':
                features = model.feature_extractor.get_intermediate_layers(
                    data, n=model.dino_out_blocks, reshape=True
                )
                features = [model.norms[i](f) for i, f in enumerate(features)]
            elif bt == 'clip':
                features = model.feature_extractor(data)
                features = [model.norms[i](f) for i, f in enumerate(features)]
            else:
                features = model.feature_extractor(data)
                features = [model.norms[i](feat) for i, feat in enumerate(features)]

        # ---- Reconstruction loss over all scales ----
        recon_losses = []
        for i, feat in enumerate(features):
            x_hat = model.autoencoders[i](feat)
            recon_losses.append(F.mse_loss(x_hat, feat))
        recon_loss = torch.stack(recon_losses).mean()

        optimizer_ae.zero_grad()
        recon_loss.backward()
        optimizer_ae.step()

        recon_meter.update(recon_loss.item())

        if (step + 1) % args.log_interval == 0 or (step + 1) == len(dataloader):
            print(
                f"Epoch {epoch+1} [AE] Step [{step+1}/{len(dataloader)}] "
                f"Recon Loss: {recon_meter.avg:.4f}"
            )

    training_time = time.time() - start_time
    print(f"\u23f1\ufe0f  AE training epoch time: {int(training_time // 60)}m {int(training_time % 60)}s")

    wandb.log({"AE Recon Loss": recon_meter.avg}, step=epoch + 1)
    return recon_meter.avg


def get_noise_std(epoch, args):
    """Calculate noise std based on schedule"""
    if args.noise_schedule == 'fixed':
        return args.noise_std
    elif args.noise_schedule == 'linear':
        # Linear increase from noise_std to noise_std_max
        progress = min(epoch / args.num_epochs, 1.0)
        return args.noise_std + (args.noise_std_max - args.noise_std) * progress
    elif args.noise_schedule == 'exponential':
        # Exponential increase
        progress = min(epoch / args.num_epochs, 1.0)
        return args.noise_std * (args.noise_std_max / args.noise_std) ** progress
    return args.noise_std


def should_use_noise_batch(step, strategy):
    """Determine if current batch should use noise based on strategy"""
    if strategy == 'same_batch':
        # Always use both pure and noisy (handled in forward)
        return True
    elif strategy == 'alternate_batch':
        # Alternate every batch: 0=pure, 1=noisy, 2=pure, 3=noisy...
        return step % 2 == 1
    elif strategy == 'cycle_2_1':
        # Pattern: 2 pure, 1 noisy, 2 pure, 1 noisy...
        # step % 3: 0=pure, 1=pure, 2=noisy
        return step % 3 == 2
    return False


def train_projection_one_epoch(dataloader_pair, model, optimizer_proj, epoch, args):
    """
    Train projection layer for one epoch using contrastive and reconstruction loss.
    
    Args:
        dataloader_pair: DataLoader with PairDataset (returns image pairs)
        model: FastFlow model with projection layers
        optimizer_proj: Optimizer for projection layers
        epoch: Current epoch number
        args: Arguments with hyperparameters
    
    Returns:
        Tuple of (avg_contrastive_loss, avg_reconstruction_loss, avg_total_loss)
    """
    from muflow.proj_layer import contrastive_loss
    
    start_time = time.time()
    
    # Set training mode: projection trainable, FastFlow frozen
    if model.projection_layers is not None:
        for proj in model.projection_layers:
            proj.train()
    model.nf_flows.eval()
    model.feature_extractor.eval()
    
    loss_contrastive_meter = utils.AverageMeter()
    loss_recon_meter = utils.AverageMeter()
    loss_total_meter = utils.AverageMeter()
    
    lambda_c = args.lambda_contrastive if hasattr(args, 'lambda_contrastive') else 1.0
    lambda_r = args.lambda_reconstruction if hasattr(args, 'lambda_reconstruction') else 0.5
    proj_noise_std = args.proj_noise_std if hasattr(args, 'proj_noise_std') else 0.1
    margin = args.contrastive_margin if hasattr(args, 'contrastive_margin') else 1.0
    device = args.device if hasattr(args, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for step, (img1, img2) in enumerate(dataloader_pair):
        img1, img2 = img1.to(device), img2.to(device)
        
        # Extract features (frozen feature extractor)
        with torch.no_grad():
            bt = model.backbone_type
            if bt == 'dino':
                features1 = model.feature_extractor.get_intermediate_layers(
                    img1, n=model.dino_out_blocks, reshape=True)
                features1 = [model.norms[i](f) for i, f in enumerate(features1)]
                features2 = model.feature_extractor.get_intermediate_layers(
                    img2, n=model.dino_out_blocks, reshape=True)
                features2 = [model.norms[i](f) for i, f in enumerate(features2)]
            elif bt == 'clip':
                features1 = model.feature_extractor(img1)
                features1 = [model.norms[i](f) for i, f in enumerate(features1)]
                features2 = model.feature_extractor(img2)
                features2 = [model.norms[i](f) for i, f in enumerate(features2)]
            elif bt == 'cait_deit':
                # For legacy transformers just use standard forward (features already good)
                features1 = model.feature_extractor(img1)
                features1 = [features1]
                features2 = model.feature_extractor(img2)
                features2 = [features2]
            else:  # cnn
                features1 = model.feature_extractor(img1)
                features1 = [model.norms[i](feat) for i, feat in enumerate(features1)]
                features2 = model.feature_extractor(img2)
                features2 = [model.norms[i](feat) for i, feat in enumerate(features2)]
            
            # Add noise to features1 for negative pairs
            noisy_features1 = model.add_gaussian_noise(features1, proj_noise_std)
        
        # Project features
        proj_clean1 = [model.projection_layers[i](features1[i]) for i in range(len(features1))]
        proj_clean2 = [model.projection_layers[i](features2[i]) for i in range(len(features2))]
        proj_noisy1 = [model.projection_layers[i](noisy_features1[i]) for i in range(len(noisy_features1))]
        
        # Compute losses for each scale and average
        loss_contrastive_total = 0
        loss_recon_total = 0
        
        for i in range(len(proj_clean1)):
            # Contrastive loss with positive and negative pairs
            # Positive pairs: (proj_clean1, proj_clean2) - both clean, different images -> label=1
            # Negative pairs: (proj_clean1, proj_noisy1) - clean vs noisy -> label=0
            
            batch_size = proj_clean1[i].size(0)
            
            # Concatenate positive and negative pairs
            features_a = torch.cat([proj_clean1[i], proj_clean1[i]], dim=0)  # [2*B, C, H, W]
            features_b = torch.cat([proj_clean2[i], proj_noisy1[i]], dim=0)  # [2*B, C, H, W]
            
            # Create labels: 1 for positive pairs (similar), 0 for negative pairs (dissimilar)
            labels = torch.cat([
                torch.ones(batch_size, device=device),   # positive pairs
                torch.zeros(batch_size, device=device)   # negative pairs
            ], dim=0)  # [2*B]
            
            # Compute contrastive loss
            loss_c = contrastive_loss(
                features1=features_a,
                features2=features_b,
                labels=labels,
                margin=margin
            )
            loss_contrastive_total += loss_c
            
            # Reconstruction loss: only on clean features
            loss_r = F.mse_loss(proj_clean1[i], features1[i])
            loss_recon_total += loss_r
        
        # Average over scales
        loss_contrastive_avg = loss_contrastive_total / len(proj_clean1)
        loss_recon_avg = loss_recon_total / len(proj_clean1)
        
        # Total loss
        loss_proj = lambda_c * loss_contrastive_avg + lambda_r * loss_recon_avg
        
        # Backward and optimize
        optimizer_proj.zero_grad()
        loss_proj.backward()
        optimizer_proj.step()
        
        # Update meters
        loss_contrastive_meter.update(loss_contrastive_avg.item())
        loss_recon_meter.update(loss_recon_avg.item())
        loss_total_meter.update(loss_proj.item())
        
        if (step + 1) % args.log_interval == 0 or (step + 1) == len(dataloader_pair):
            print(
                f"Epoch {epoch+1} [PROJ] Step [{step+1}/{len(dataloader_pair)}] "
                f"Loss: {loss_total_meter.avg:.4f} "
                f"(Contrastive: {loss_contrastive_meter.avg:.4f}, Recon: {loss_recon_meter.avg:.4f})"
            )
    
    training_time = time.time() - start_time
    print(f"⏱️  Projection training epoch time: {int(training_time // 60)}m {int(training_time % 60)}s")
    
    # Log to wandb
    wandb.log({
        "Projection Loss Contrastive": loss_contrastive_meter.avg,
        "Projection Loss Reconstruction": loss_recon_meter.avg,
        "Projection Loss Total": loss_total_meter.avg,
    }, step=epoch + 1)
    
    return loss_contrastive_meter.avg, loss_recon_meter.avg, loss_total_meter.avg


def train_one_epoch(dataloader, model, optimizer, epoch, args, scheduler=None, ae_lambda=0.0):
    start_time = time.time()
    model.train()
    loss_meter = utils.AverageMeter()
    loss_values = []

    use_adversarial = args.use_adversarial == 1 if hasattr(args, 'use_adversarial') else False
    use_autoencoder = args.use_autoencoder == 1 if hasattr(args, 'use_autoencoder') else False
    current_noise_std = get_noise_std(epoch, args) if use_adversarial else 0.0
    
    # Separate meters for adversarial training
    if use_adversarial:
        loss_pure_meter = utils.AverageMeter()
        loss_adv_meter = utils.AverageMeter()
    
    device = args.device if hasattr(args, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for step, data in enumerate(dataloader):
        # Forward pass
        data = data.to(device)
        
        if use_adversarial:
            # Determine strategy
            if args.adv_strategy == 'same_batch':
                ret = model(data, noise_std=current_noise_std, adversarial_mode='same_batch')
                loss_pure = ret["loss_pure"].mean()
                loss_adv = ret["loss_adv"].mean()
                loss = loss_pure + args.adv_lambda * loss_adv
                loss_pure_meter.update(loss_pure.item())
                loss_adv_meter.update(loss_adv.item())
            else:
                use_noise = should_use_noise_batch(step, args.adv_strategy)
                if use_noise:
                    ret = model(data, noise_std=current_noise_std, adversarial_mode='noisy_only')
                    loss_adv = ret["loss"].mean()
                    loss = args.adv_lambda * loss_adv
                    loss_adv_meter.update(loss_adv.item())
                else:
                    ret = model(data, noise_std=0.0, adversarial_mode='pure_only')
                    loss_pure = ret["loss"].mean()
                    loss = loss_pure
                    loss_pure_meter.update(loss_pure.item())
        else:
            # Standard training (with or without autoencoder)
            ret = model(data, ae_lambda=ae_lambda)
            loss = ret["loss"].mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item())
        loss_values.append((ret["loss"] if not use_adversarial or args.adv_strategy != 'same_batch' else ret.get("loss_pure", ret["loss"])).detach().cpu())
        
        if (step + 1) % args.log_interval == 0 or (step + 1) == len(dataloader):
            if use_adversarial and args.adv_strategy == 'same_batch':
                print(
                    "Epoch {} - Iteration {}/{} - Loss: {:.3f} (Pure: {:.3f}, Adv: {:.3f}, NoiseStd: {:.4f})".format(
                        epoch + 1, step + 1, len(dataloader), loss_meter.avg, loss_pure.item(), loss_adv.item(), current_noise_std
                    )
                )
                wandb.log({
                    "Train Loss": loss_meter.val,
                    "Train Loss Pure": loss_pure_meter.val,
                    "Train Loss Adv": loss_adv_meter.val
                }, step=epoch + 1)
            else:
                mode_str = "[NOISY]" if (use_adversarial and should_use_noise_batch(step, args.adv_strategy)) else "[PURE]"
                print(
                    "Epoch {} - Iteration {}/{} {} - Loss: {:.3f}".format(
                        epoch + 1, step + 1, len(dataloader), mode_str if use_adversarial else "", loss_meter.avg
                    )
                )
                log_dict = {"Train Loss": loss_meter.val}
                if use_adversarial:
                    use_noise = should_use_noise_batch(step, args.adv_strategy)
                    if use_noise:
                        log_dict["Train Loss Adv"] = loss_adv_meter.val
                    else:
                        log_dict["Train Loss Pure"] = loss_pure_meter.val
                wandb.log(log_dict, step=epoch + 1)

    if scheduler:
      scheduler.step(loss_meter.val)

    preds_train = torch.cat(loss_values, dim=0).numpy().reshape(-1, 1)
    train_mean = preds_train.mean()
    train_std = preds_train.std()
    
    # Log additional metrics
    wandb_dict = {
        "Train loss mean": train_mean,
        "Train loss std": train_std,
    }
    if use_adversarial:
        wandb_dict["Noise std"] = current_noise_std
        wandb_dict["Train Loss Pure Mean"] = loss_pure_meter.avg
        wandb_dict["Train Loss Adv Mean"] = loss_adv_meter.avg
    if use_autoencoder:
        wandb_dict["ae_lambda"] = ae_lambda
    wandb.log(wandb_dict, step=epoch + 1)
    
    training_time = time.time() - start_time
    print(f"⏱️  Training epoch time: {int(training_time // 60)}m {int(training_time % 60)}s")
    
    return train_mean, train_std, preds_train


def compute_threshold(val_dataloader, model, use_lof=False, contamination='auto', alpha=0.1):
    """
    Compute threshold for anomaly detection using validation set.
    
    Args:
        val_dataloader: DataLoader for validation set
        model: Trained model
        use_lof: If True, use LOF instead of Gaussian threshold
        contamination: Contamination rate for LOF (default: 'auto')
        alpha: Target false positive rate under the normality assumption, i.e., the probability of flagging a normal sample as anomalous. (default: 0.1)
    Returns:
        dict with 'threshold' (scalar), 'mean', 'std', 'losses' (numpy array), 
        and optionally 'lof' (if use_lof=True)
    """
    model.eval()
    loss_values = []
    device = model.nf_flows[0].parameters().__next__().device  # Get device from model
    
    for batch in val_dataloader:
        # Validation set returns only images (no labels) since is_train=True and is_val=True
        # But check if it's a tuple/list in case the dataset returns (image, label)
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            data, _ = batch
        else:
            data = batch
        data = data.to(device)

        with torch.no_grad():
            ret = model(data)
            # Move to CPU immediately to free GPU memory
            loss_values.append(ret["loss"].cpu())
    
    if len(loss_values) == 0:
        raise ValueError("No validation loss values collected. Check validation dataloader.")
    
    losses = torch.cat(loss_values, dim=0).numpy()
    mean = losses.mean()
    std = losses.std()
    
    # Always compute mean and std for logging purposes
    result = {
        'threshold': None,
        'mean': mean,
        'std': std,
        'losses': losses,
    }
    
    if use_lof:
        lof = LocalOutlierFactor(novelty=True, contamination=contamination, n_jobs=-1)
        lof.fit(losses.reshape(-1, 1))
        result['lof'] = lof
    else:
        z = norm.ppf(1 - alpha)
        l_threshold = mean - z * std
        u_threshold = mean + z * std
        result['l_threshold'] = l_threshold
        result['u_threshold'] = u_threshold
    print(f"Threshold computation done.")
    return result


def _compute_class_metrics(c, class_masks, labels, preds, preds_, class2idx, y_true_0_base, y_pred_0_base):
    """
    Helper function to compute metrics for a single class in parallel.
    """
    mask_c = class_masks[c]
    y_true_c = labels[mask_c]
    y_pred_c = preds[mask_c]
    preds_c = preds_[mask_c]
    
    min_len = min(len(y_true_c), len(y_true_0_base))
    if min_len == 0:
        return None
    
    class_name = class2idx[c] if class2idx else str(c)
    
    # Compute loss statistics for this class
    loss_mean_fake = float(preds_c.mean())
    loss_std_fake = float(preds_c.std())
    
    # Shuffling real to get different subset each time
    # Note: We use a fixed seed per class for reproducibility
    rng = np.random.RandomState(42 + c)  # Different seed per class but reproducible
    idx = rng.permutation(len(y_true_0_base))
    y_pred_0_shuffled = y_pred_0_base[idx]
    
    # Use concatenate instead of tolist() for efficiency
    y_pred_balanced = np.concatenate([y_pred_c[:min_len], y_pred_0_shuffled[:min_len]])
    
    # Create binary labels directly (1s for fake, 0s for real)
    y_true_binary = np.concatenate([np.ones(min_len, dtype=np.int8), np.zeros(min_len, dtype=np.int8)])
    
    acc0 = accuracy_score(y_true_binary, y_pred_balanced)
    ap = average_precision_score(y_true_binary, y_pred_balanced)
    roc = roc_auc_score(y_true_binary, y_pred_balanced)
    
    return {
        'class_id': c,
        'class_name': class_name,
        'min_len': min_len,
        'loss_mean': loss_mean_fake,
        'loss_std': loss_std_fake,
        'accuracy': acc0,
        'ap': ap,
        'roc': roc
    }


def eval_once(dataloader, model, epoch, class2idx=None, threshold_info=None): 
    inference_start_time = time.time()
    model.eval()
    labels_list = []
    preds_list = []
    device = model.nf_flows[0].parameters().__next__().device  # Get device from model
    
    for data, targets in dataloader:
        data, targets = data.to(device), targets.to(device)
        with torch.no_grad():
            ret = model(data)
        score = ret["loss"].cpu()
        preds_list.append(score)
        labels_list.append(targets.cpu())

    inference_time = time.time() - inference_start_time
    print(f"Testing done")
    print(f"⏱️  Test inference time: {int(inference_time // 60)}m {int(inference_time % 60)}s")
    
    # Concatenate tensors directly for better memory efficiency
    preds_ = torch.cat(preds_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()

    # Use threshold from validation set instead of training set
    if threshold_info is None:
        raise ValueError("threshold_info must be provided. Use compute_threshold() first.")
    
    # If using LOF, apply LOF to test data and compare scores
    if 'lof' in threshold_info:
        lof = threshold_info['lof']
        preds = (lof.predict(preds_.reshape(-1,1)) < 0).astype(int)
    else:
        # Standard threshold comparison: loss > threshold means anomaly
        l_threshold = threshold_info['l_threshold']
        u_threshold = threshold_info['u_threshold']
        preds = ((preds_ < l_threshold) | (preds_ > u_threshold)).astype(int)
    
    # Get unique classes and pre-compute masks for efficiency
    classes = np.unique(labels)
    class_masks = {c: labels == c for c in classes}
    
    # Pre-compute fake mask once (all classes except 0 and 99)
    mask_fake = np.zeros(len(labels), dtype=bool)
    for c in classes:
        if c != 0 and c != 99:
            mask_fake |= class_masks[c]
    
    if epoch % 50 == 0:
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
        plt.savefig(const.CHECKPOINT_DIR+"/real_fake_likelihoods_{}.png".format(epoch))
        plt.close()
    
    aps, accs, rocs, cls = [], [], [], []
    
    # Dictionary to store per-class metrics for wandb logging
    per_class_metrics = {}

    # Pre-compute data for Real class (class 0)
    mask_0 = class_masks[0]
    y_true_0 = labels[mask_0]
    y_pred_0 = preds[mask_0]
    preds_0 = preds_[mask_0]
    
    if len(preds_0) > 0:
        loss_mean_real = preds_0.mean()
        loss_std_real = preds_0.std()
        per_class_metrics["loss_per_class/Real"] = loss_mean_real
        per_class_metrics["loss_std_per_class/Real"] = loss_std_real
    
    # Logs metrics for OOD Real class (class 99)
    if 99 in class_masks:
        class_ood_real_name = class2idx[99] if class2idx else "OOD_Real"
        mask_99 = class_masks[99]
        y_true_ood_real = labels[mask_99]
        y_pred_ood_real = preds[mask_99]
        preds_99 = preds_[mask_99]

        if len(preds_99) > 0:
            loss_mean_ood_real = preds_99.mean()
            loss_std_ood_real = preds_99.std()
            per_class_metrics[f"loss_per_class/{class_ood_real_name}"] = loss_mean_ood_real
            per_class_metrics[f"loss_std_per_class/{class_ood_real_name}"] = loss_std_ood_real
            
            # Log accuracy for OOD Real class
            y_true_ood_real_binary = np.zeros(len(y_true_ood_real), dtype=np.int8)
            acc_ood_real = accuracy_score(y_true_ood_real_binary, y_pred_ood_real)
            print("-" * 30)
            print(f"  > Class {class_ood_real_name} (N={len(y_true_ood_real)}): \t Accuracy = {acc_ood_real:.4f}, \t Loss = {loss_mean_ood_real:.4f}±{loss_std_ood_real:.4f}")
            print("-" * 30)
            per_class_metrics[f"acc_per_class/{class_ood_real_name}"] = acc_ood_real

    # Parallel computation of metrics for each class
    # Filter out class 0 (real) and 99 (OOD real)
    classes_to_process = [c for c in classes[1:] if c != 99]
    
    print("\n"+"Computing metrics using Real as baseline...")
    print("="*30)
    metrics_real_start_time = time.time()
    # Compute metrics in parallel using all available CPU cores
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(_compute_class_metrics)(
            c, class_masks, labels, preds, preds_, class2idx, y_true_0, y_pred_0
        ) for c in classes_to_process
    )
    
    # Filter out None results (classes with no samples) and process results
    for result in results:
        if result is None:
            continue
        
        print(f"  > Class {result['class_name']} (N={result['min_len']*2}): \t Accuracy = {result['accuracy']:.4f}, \t AP = {result['ap']:.4f}, \t ROC AUC = {result['roc']:.4f}, \t Loss = {result['loss_mean']:.4f}±{result['loss_std']:.4f}")
        print("-" * 30)
        
        # Store metrics for wandb logging
        per_class_metrics[f"loss_per_class/{result['class_name']}"] = result['loss_mean']
        per_class_metrics[f"loss_std_per_class/{result['class_name']}"] = result['loss_std']
        per_class_metrics[f"acc_per_class/{result['class_name']}"] = result['accuracy']
        per_class_metrics[f"ap_per_class/{result['class_name']}"] = result['ap']
        per_class_metrics[f"roc_per_class/{result['class_name']}"] = result['roc']
        aps.append(result['ap'])
        accs.append(result['accuracy'])
        rocs.append(result['roc'])
        cls.append(result['class_id'])

    mean_acc = np.mean(accs)
    mean_ap = np.mean(aps)
    mean_roc = np.mean(rocs)
    
    metrics_real_time = time.time() - metrics_real_start_time
    print("-" * 30)
    print(f"Average Accuracy (vs Real): {mean_acc:.4f}")
    print(f"Average Precision (vs Real): {mean_ap:.4f}")
    print(f"Average ROC AUC (vs Real): {mean_roc:.4f}")
    print(f"⏱️  Metrics computation time (Real baseline): {int(metrics_real_time // 60)}m {int(metrics_real_time % 60)}s")
    print("="*30 + "\n")

    # ============ Compute metrics using OOD Real as baseline ============
    aps_ood, accs_ood, rocs_ood = [], [], []
    
    if 99 in class_masks and len(preds_99) > 0:
        print("Computing metrics using OOD Real as baseline...")
        print("="*30)
        metrics_ood_start_time = time.time()
        
        # Compute metrics in parallel using OOD Real as baseline
        results_ood = Parallel(n_jobs=-1, backend='threading')(
            delayed(_compute_class_metrics)(
                c, class_masks, labels, preds, preds_, class2idx, y_true_ood_real, y_pred_ood_real
            ) for c in classes_to_process
        )
        
        # Process results with OOD Real baseline
        for result in results_ood:
            if result is None:
                continue
            
            print(f"  > Class {result['class_name']} (N={result['min_len']*2}): \t Accuracy = {result['accuracy']:.4f}, \t AP = {result['ap']:.4f}, \t ROC AUC = {result['roc']:.4f}")
            print("-" * 30)
            
            # Store metrics for wandb logging with _ood suffix
            per_class_metrics[f"acc_per_class_ood/{result['class_name']}"] = result['accuracy']
            per_class_metrics[f"ap_per_class_ood/{result['class_name']}"] = result['ap']
            per_class_metrics[f"roc_per_class_ood/{result['class_name']}"] = result['roc']
            aps_ood.append(result['ap'])
            accs_ood.append(result['accuracy'])
            rocs_ood.append(result['roc'])
        
        mean_acc_ood = np.mean(accs_ood) if len(accs_ood) > 0 else 0
        mean_ap_ood = np.mean(aps_ood) if len(aps_ood) > 0 else 0
        mean_roc_ood = np.mean(rocs_ood) if len(rocs_ood) > 0 else 0
        
        metrics_ood_time = time.time() - metrics_ood_start_time
        print("-" * 30)
        print(f"Average Accuracy (vs OOD Real): {mean_acc_ood:.4f}")
        print(f"Average Precision (vs OOD Real): {mean_ap_ood:.4f}")
        print(f"Average ROC AUC (vs OOD Real): {mean_roc_ood:.4f}")
        print(f"⏱️  Metrics computation time (OOD Real baseline): {int(metrics_ood_time // 60)}m {int(metrics_ood_time % 60)}s")
        print("="*30 + "\n")
        
        # Add to wandb metrics
        per_class_metrics["Val acc OOD"] = mean_acc_ood
        per_class_metrics["Val AP OOD"] = mean_ap_ood
        per_class_metrics["Val ROC OOD"] = mean_roc_ood
    # ====================================================================

    # Log test statistics (use pre-computed values to avoid redundant calculations)
    test_loss_real_mean = loss_mean_real if len(preds_0) > 0 else 0
    test_loss_real_std = loss_std_real if len(preds_0) > 0 else 0
    
    # Use pre-computed fake mask and extract statistics (mask_fake already computed above)
    preds_fake = preds_[mask_fake]
    test_loss_fake_mean = preds_fake.mean() if len(preds_fake) > 0 else 0
    test_loss_fake_std = preds_fake.std() if len(preds_fake) > 0 else 0
    
    wandb_log_dict = {
        "Val acc": mean_acc, 
        "Test loss real mean": test_loss_real_mean,
        "Test loss fake mean": test_loss_fake_mean,
        "Test loss real std": test_loss_real_std,
        "Test loss fake std": test_loss_fake_std
    }
    
    wandb_log_dict.update(per_class_metrics)
    wandb.log(wandb_log_dict, step=epoch + 1)

    return mean_acc, preds_, labels


def train(args, config):
    checkpoint_dir = const.CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Validate conflicting options
    use_proj = hasattr(args, 'use_proj_layer') and args.use_proj_layer == 1
    use_adv = hasattr(args, 'use_adversarial') and args.use_adversarial == 1
    use_ae = hasattr(args, 'use_autoencoder') and args.use_autoencoder == 1

    active_modes = [m for m, f in [("--use_proj_layer", use_proj), ("--use_adversarial", use_adv), ("--use_autoencoder", use_ae)] if f]
    if len(active_modes) > 1:
        raise ValueError(
            f"The following options are mutually exclusive: {', '.join(active_modes)}. "
            "Please choose only one."
        )

    wandb.init(
        entity="orazio-mattia",
        project="MuFlow", 
        config=config, 
        name=args.run_name, 
        mode="disabled" if args.debug else args.wandb)
    wandb.config.update(args)

    model = build_model(config, args)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        print('Model loaded!')
    
    # GPU Selection
    if args.gpu_id is not None:
        # Manual GPU selection
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"📌 Using manually specified GPU {args.gpu_id}")
    else:
        # Automatic GPU selection (default behavior)
        gpu_id = select_best_gpu()
        if gpu_id is not None:
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cpu')
            print("⚠️  No GPU available, using CPU")
    
    model.to(device)
    print(f"✓ Model moved to {device}")
    
    # Save device to args for use in training functions
    args.device = device

    # Build optimizer(s)
    optimizer_result = build_optimizer(args, model, config)

    # Handle cases: standard / proj / autoencoder (two_phase or end_to_end)
    ae_mode = getattr(args, 'ae_training_mode', 'two_phase')
    if use_proj:
        optimizer_proj, optimizer_fastflow = optimizer_result
        scheduler_proj = None
        scheduler_fastflow = ReduceLROnPlateau(optimizer_fastflow, mode='min', factor=args.lr_decay, patience=args.lr_patience, verbose=True) if args.scheduler == 1 else None
    elif use_ae and ae_mode == 'two_phase':
        optimizer_ae, optimizer_fastflow = optimizer_result
        scheduler_ae = None
        scheduler_fastflow = ReduceLROnPlateau(optimizer_fastflow, mode='min', factor=args.lr_decay, patience=args.lr_patience, verbose=True) if args.scheduler == 1 else None
    elif use_ae and ae_mode == 'end_to_end':
        optimizer = optimizer_result  # single joint optimizer
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay, patience=args.lr_patience, verbose=True) if args.scheduler == 1 else None
    else:
        optimizer = optimizer_result
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay, patience=args.lr_patience, verbose=True) if args.scheduler == 1 else None

    # Build dataloaders
    if use_proj:
        pair_dataloader = build_pair_data_loader(args, config)
        train_dataloader = build_train_data_loader(args, config)
    else:
        train_dataloader = build_train_data_loader(args, config)
    
    val_dataloader = build_val_data_loader(args, config)
    test_dataloader, class2idx = build_test_data_loader(args, config)
    
    best_acc = 0
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        # ================================================================
        # PHASE 1: Train auxiliary module — only in two_phase modes
        # ================================================================
        if use_proj:
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{args.num_epochs} - PHASE 1: Training Projection Layer")
            print(f"{'='*60}")
            model.freeze_fastflow()
            model.unfreeze_projection_layers()
            train_projection_one_epoch(pair_dataloader, model, optimizer_proj, epoch, args)

        elif use_ae and ae_mode == 'two_phase':
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{args.num_epochs} - PHASE 1: Training Autoencoder (NF frozen)")
            print(f"{'='*60}")
            model.freeze_fastflow()
            model.unfreeze_autoencoders()
            train_autoencoder_one_epoch(train_dataloader, model, optimizer_ae, epoch, args)

        # ================================================================
        # PHASE 2: Train FastFlow (NF flows)
        # ================================================================
        if use_proj:
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{args.num_epochs} - PHASE 2: Training FastFlow")
            print(f"{'='*60}")
            model.freeze_projection_layers()
            model.unfreeze_fastflow()
            train_mean, train_std, _ = train_one_epoch(
                train_dataloader, model, optimizer_fastflow, epoch, args, scheduler=scheduler_fastflow
            )
        elif use_ae and ae_mode == 'two_phase':
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{args.num_epochs} - PHASE 2: Training FastFlow (AE frozen)")
            print(f"{'='*60}")
            model.freeze_autoencoders()
            model.unfreeze_fastflow()
            train_mean, train_std, _ = train_one_epoch(
                train_dataloader, model, optimizer_fastflow, epoch, args,
                scheduler=scheduler_fastflow,
                ae_lambda=args.ae_lambda,
            )
        elif use_ae and ae_mode == 'end_to_end':
            # AE + NF trained jointly — no phase split, single optimizer
            model.unfreeze_autoencoders()
            model.unfreeze_fastflow()
            train_mean, train_std, _ = train_one_epoch(
                train_dataloader, model, optimizer, epoch, args,
                scheduler=scheduler,
                ae_lambda=args.ae_lambda,
            )
        else:
            # Standard training
            train_mean, train_std, _ = train_one_epoch(
                train_dataloader, model, optimizer, epoch, args, scheduler=scheduler
            )
        
        current_acc = -1
        
        if (epoch + 1) % args.eval_interval == 0:
            # Compute threshold on validation set
            threshold_info = compute_threshold(val_dataloader, model, use_lof=True if args.use_lof==1 else False, contamination=args.contamination, alpha=args.alpha)
            
            # Log validation threshold statistics
            l_threshold_val = threshold_info['l_threshold']
            u_threshold_val = threshold_info['u_threshold']
            val_mean = threshold_info.get('mean', None)
            val_std = threshold_info.get('std', None)
                
            print(f"Training loss stats: mean: {train_mean:.4f}, std: {train_std:.4f}")
            
            # Log to wandb
            log_dict = {
                "Train Loss Mean": train_mean,
                "Train Loss Std": train_std,
                "Lower Threshold": l_threshold_val,
                "Upper Threshold": u_threshold_val
            }
            if val_mean is not None:
                log_dict["Val Loss Mean"] = val_mean
            if val_std is not None:
                log_dict["Val Loss Std"] = val_std
            wandb.log(log_dict, step=epoch + 1)
            
            # Evaluate on test set using threshold from validation
            acc, preds, labels = eval_once(test_dataloader, model, epoch, class2idx, threshold_info)
            current_acc = acc
            
            # Log test loss statistics (preds_ contains loss values from eval_once)
            # These will be logged in eval_once function

            if current_acc > best_acc:
                best_acc = current_acc
                patience_counter = 0
                
                print(f"Epoch {epoch+1}: New best accuracy: {best_acc:.4f}. Saving model.")
                
                
                wandb.run.summary["best_accuracy"] = best_acc
                
                checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
                
                # Save checkpoint with appropriate optimizer state
                checkpoint_dict = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                }
                
                if use_proj:
                    checkpoint_dict["optimizer_proj_state_dict"] = optimizer_proj.state_dict()
                    checkpoint_dict["optimizer_fastflow_state_dict"] = optimizer_fastflow.state_dict()
                elif use_ae and ae_mode == 'two_phase':
                    checkpoint_dict["optimizer_ae_state_dict"] = optimizer_ae.state_dict()
                    checkpoint_dict["optimizer_fastflow_state_dict"] = optimizer_fastflow.state_dict()
                else:
                    # end_to_end or standard: single optimizer
                    checkpoint_dict["optimizer_state_dict"] = optimizer.state_dict()
                
                torch.save(checkpoint_dict, checkpoint_path)

                # Save all run hyperparameters so eval.py can reload them
                # without re-specifying every flag.
                args_dict = vars(args).copy()
                args_dict.pop('device', None)  # non-serializable torch.device
                with open(os.path.join(checkpoint_dir, "run_config.yaml"), 'w') as f:
                    yaml.dump(args_dict, f, default_flow_style=False)

                # Save threshold information (only save fields that exist)
                # Note: LOF object cannot be saved in npz, only with joblib
                save_dict = {}
                for key in ('l_threshold', 'u_threshold', 'threshold',
                            'losses', 'mean', 'std',
                            'lof_scores'):
                    if key in threshold_info and threshold_info[key] is not None:
                        save_dict[key] = threshold_info[key]
                
                np.savez(
                    os.path.join(checkpoint_dir, "thresholds.npz"),
                    **save_dict
                )
                np.save(
                    os.path.join(checkpoint_dir, "preds_best.npy"), preds
                )
                np.save(
                    os.path.join(checkpoint_dir, "labels_best.npy"), labels
                )
                # Save LOF model only if it exists
                if 'lof' in threshold_info:
                    joblib.dump(threshold_info['lof'], os.path.join(checkpoint_dir, "lof_model.pkl"))
            
            else:
                patience_counter += 1 
                print(f"Epoch {epoch+1}: Accuracy ({current_acc:.4f}) not improved compared to {best_acc:.4f}. Patience: {patience_counter}/{args.early_stopping_patience}")

        if patience_counter >= args.early_stopping_patience:
            print(f"Stopping early at epoch {epoch + 1} after {args.early_stopping_patience} epochs without improvement of Val Acc.")
            break
    
    print(f"Training finished. Best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    args = parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    pprint(vars(args))
    pprint(config)

    if args.run_name is None:
        args.run_name = f"{config['backbone_name']}_{args.data}_{args.reals}"
        if args.use_fourier == 1:
            args.run_name += "_fourier"
        args.run_name += f"_lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}_ld{args.lr_decay}_lp{args.lr_patience}_{config['pooling_type']}"

        if args.use_lof == 1:
            args.run_name += "_lof"
        else:
            args.run_name += f"_t-alpha{args.alpha}"
        
        if args.use_augs == 1:
            args.run_name += "_augs"
        
        if args.use_proj_layer == 1:
            args.run_name += f"_proj-lc{args.lambda_contrastive}_lr{args.lambda_reconstruction}_noise{args.proj_noise_std}"

        if args.use_adversarial == 1:
            args.run_name += f"_adv-{args.adv_strategy}_lambda{args.adv_lambda}_noise{args.noise_std}_{args.noise_schedule}"

        if args.use_autoencoder == 1:
            args.run_name += f"_ae-{args.ae_training_mode}_lambda{args.ae_lambda}"

    const.CHECKPOINT_DIR += "/" + args.run_name 

    train(args, config)
