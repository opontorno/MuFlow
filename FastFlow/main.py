import argparse
from pprint import pprint
import os, pdb
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import wandb
import joblib

import constants as const
import dataset
import fastflow
import utils

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
    parser.add_argument("--test", type=str, help="test name")
    parser.add_argument("--checkpoint", type=str, help="path to load checkpoint")

    parser.add_argument('--wandb', default= 'disabled', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--use_augs', type=int, default=0, choices=[0, 1], help="Whether to use data augmentation")
    parser.add_argument('--use_fourier', type=int, default=0, choices=[0, 1], help="Whether to use Fourier transform")
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--model_type', type=str, choices=['FastFlow', 'VAE'], default='FastFlow', help="Choose the model to train")
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--backbone_weights', type=str, help="path to load backbone weights")
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4, help="number of data loading workers")
    parser.add_argument('--debug', action='store_true', help="Debug mode with reduced dataset size")

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
    parser.add_argument('--use_proj', type=int, default=0, choices=[0, 1], help="Whether to use projection layer")
    parser.add_argument('--projection_type', type=str, default='conv', choices=['conv', 'mlp', 'autoencoder', 'identity'], help="Type of projection layer")
    parser.add_argument('--use_lof', type=int, default=0, choices=[0, 1], help="Whether to use LOF")
    parser.add_argument('--contamination', default='auto')
    parser.add_argument('--use_percentile', type=int, default=0, choices=[0, 1], help="Whether to use percentile")
    parser.add_argument('--percentile', type=int, default=95)

    parser.add_argument('--early_stopping_patience', type=float, default=float("inf"), help="Patience epochs for early stopping based on Val Acc")

    args = parser.parse_args()
    
    return args


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
        test_name=args.test,
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


def build_model(config, model_type, args):
    out_indices = config.get("out_indices", [1, 2, 3])
    out_indices_str = str(out_indices)
    pooling_type = config.get("pooling_type", "mean")
    n_components = config.get("gmm_n_components", 1)
    
    if args.use_fourier == 1:
        gmm_parameters = f"{const.WORKING_DIR}/parameters/{n_components}-gmm_parameters_{config['backbone_name']}_indices_{out_indices_str}_fourier_{args.reals}_{config['input_size']}_{pooling_type}.npy"
    else:
        gmm_parameters = f"{const.WORKING_DIR}/parameters/{n_components}-gmm_parameters_{config['backbone_name']}_indices_{out_indices_str}_{args.reals}_{config['input_size']}_{pooling_type}.npy"
    
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
            in_channels=3,
            backbone_weights=args.backbone_weights,
            out_indices=config.get("out_indices", [1, 2, 3]),
            use_proj=True if args.use_proj == 1 else False,
            projection_type=getattr(args, 'projection_type', 'conv'),
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


def build_optimizer(args, model, model_type, config):
    if model_type == "FastFlow":
        if args.optimizer == "AdamW":
            return torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        if args.optimizer == "sgd":
            return torch.optim.SGD(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9
            )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_one_epoch(dataloader, model, optimizer, epoch, args, scheduler=None):
    model.train()
    loss_meter = utils.AverageMeter()
    loss_values = []
    for step, data in enumerate(dataloader):
        data = data.cuda()
        ret = model(data)
        loss = ret["loss"].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        
        loss_values.append(ret["loss"].detach().cpu())
        if (step + 1) % args.log_interval == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )
            
            wandb.log({"Train Loss": loss_meter.val}, step=epoch + 1)

    if scheduler:
      scheduler.step(loss_meter.val)

    preds_train = torch.cat(loss_values, dim=0).numpy().reshape(-1, 1)
    train_mean = preds_train.mean()
    train_std = preds_train.std()
    
    return train_mean, train_std, preds_train


def compute_threshold(val_dataloader, model, model_type="FastFlow", use_lof=False, contamination='auto', use_percentile=False, percentile=95, alpha=0.1):
    """
    Compute threshold for anomaly detection using validation set.
    
    Args:
        val_dataloader: DataLoader for validation set
        model: Trained model
        model_type: Type of model (default: "FastFlow")
        use_lof: If True, use LOF instead of mean+3*std
        contamination: Contamination rate for LOF (default: 'auto')
        use_percentile: If True, use percentile instead of mean+3*std
        percentile: Percentile to use if use_percentile=True (default: 95)
        alpha: Target false positive rate under the normality assumption, i.e., the probability of flagging a normal sample as anomalous. (default: 0.1)
    Returns:
        dict with 'threshold' (scalar), 'mean', 'std', 'losses' (numpy array), 
        and optionally 'lof' (if use_lof=True)
    """
    model.eval()
    loss_values = []
    
    for batch in val_dataloader:
        # Validation set returns only images (no labels) since is_train=True and is_val=True
        # But check if it's a tuple/list in case the dataset returns (image, label)
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            data, _ = batch
        else:
            data = batch
        data = data.cuda()

        with torch.no_grad():
            ret = model(data) if model_type == "FastFlow" else model(data, eval_mode=True)
            if model_type == "FastFlow":
                # Move to CPU immediately to free GPU memory
                outputs = ret["loss"].cpu()
                loss_values.append(outputs)
    
    if len(loss_values) == 0:
        raise ValueError("No validation loss values collected. Check validation dataloader.")
    
    # Concatenate tensors directly instead of converting to numpy first
    losses = torch.cat(loss_values, dim=0).numpy()
    mean = losses.mean()
    std = losses.std()
    
    # Always compute mean and std for logging purposes
    result = {
        'threshold': None,
        'mean': mean,
        'std': std,
        'losses': losses
    }
    
    if use_lof:
        lof = LocalOutlierFactor(novelty=True, contamination=contamination, n_jobs=-1)
        lof.fit(losses.reshape(-1, 1))
        # For LOF, we use the threshold as the median of negative scores (outliers have negative scores)
        # Or we can use a percentile of the scores
        result['lof'] = lof
    elif use_percentile:
        threshold = np.percentile(losses, percentile)
        result['threshold'] = threshold
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


def eval_once(dataloader, model, epoch, model_type="FastFlow", class2idx=None, threshold_info=None): 
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
    
    print("-" * 30)
    print(f"Average Accuracy (vs Real): {mean_acc:.4f}")
    print(f"Average Precision (vs Real): {mean_ap:.4f}")
    print(f"Average ROC AUC (vs Real): {mean_roc:.4f}")
    print("="*30 + "\n")

    # ============ Compute metrics using OOD Real as baseline ============
    aps_ood, accs_ood, rocs_ood = [], [], []
    
    if 99 in class_masks and len(preds_99) > 0:
        print("Computing metrics using OOD Real as baseline...")
        print("="*30)
        
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
        
        print("-" * 30)
        print(f"Average Accuracy (vs OOD Real): {mean_acc_ood:.4f}")
        print(f"Average Precision (vs OOD Real): {mean_ap_ood:.4f}")
        print(f"Average ROC AUC (vs OOD Real): {mean_roc_ood:.4f}")
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

    wandb.init(
        entity="orazio-mattia",
        project="MuFlow", 
        config=config, 
        name=args.run_name, 
        mode=args.wandb)
    wandb.config.update(args)

    model = build_model(config, args.model_type, args)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        print('Model loaded!')
    
    model.cuda()

    optimizer = build_optimizer(args, model, args.model_type, config)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay, patience=args.lr_patience, verbose=True) if args.scheduler==1 else None

    train_dataloader = build_train_data_loader(args, config)
    val_dataloader = build_val_data_loader(args, config)
    test_dataloader, class2idx = build_test_data_loader(args, config)
    
    best_acc = 0
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        train_mean, train_std, _ = train_one_epoch(train_dataloader, model, optimizer, epoch, args, scheduler=scheduler)
        
        current_acc = -1
        
        if (epoch + 1) % args.eval_interval == 0:
            # Compute threshold on validation set
            threshold_info = compute_threshold(val_dataloader, model, args.model_type, use_lof=True if args.use_lof==1 else False, contamination=args.contamination, use_percentile=True if args.use_percentile==1 else False, percentile=args.percentile, alpha=args.alpha)
            
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
            acc, preds, labels = eval_once(test_dataloader, model, epoch, args.model_type, class2idx, threshold_info)
            current_acc = acc
            
            # Log test loss statistics (preds_ contains loss values from eval_once)
            # These will be logged in eval_once function

            if current_acc > best_acc:
                best_acc = current_acc
                patience_counter = 0
                
                print(f"Epoch {epoch+1}: New best accuracy: {best_acc:.4f}. Saving model.")
                
                
                wandb.run.summary["best_accuracy"] = best_acc
                
                checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()
                    },
                    checkpoint_path,
                )
                # Save threshold information (only save fields that exist)
                # Note: LOF object cannot be saved in npz, only with joblib
                save_dict = {
                    'l_threshold': threshold_info['l_threshold'],
                    'u_threshold': threshold_info['u_threshold'],
                    'losses': threshold_info['losses']
                }
                if 'mean' in threshold_info:
                    save_dict['mean'] = threshold_info['mean']
                if 'std' in threshold_info:
                    save_dict['std'] = threshold_info['std']
                if 'lof_scores' in threshold_info:
                    save_dict['lof_scores'] = threshold_info['lof_scores']
                
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
        if args.use_percentile == 1:
            args.run_name += f"_percentile{args.percentile}"
        if args.use_lof == 0 and args.use_percentile == 0:
            args.run_name += f"_t-alpha{args.alpha}"

        if args.use_augs == 1:
            args.run_name += "_augs"

    const.CHECKPOINT_DIR += "/" + args.run_name 

    train(args, config)
