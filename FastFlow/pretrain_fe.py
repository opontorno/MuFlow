import argparse
import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import wandb
import timm
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pdb

import constants as const
import dataset
import utils


def build_train_data_loader(args, config):
    train_dataset = dataset.Dataset_multi_class(
        dataset_name=args.data,
        reals_name=args.reals,
        test_name=args.test,
        input_size=config["input_size"],
        is_train=True,
        is_val=False,
        use_fourier=args.use_fourier
    ).create_dataset()
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

def build_test_data_loader(args, config):
    test_dataset = dataset.Dataset_multi_class(
        dataset_name=args.data,
        reals_name=args.reals,
        test_name=args.test,
        input_size=config["input_size"],
        is_train=True,
        is_val=True,
        use_fourier=args.use_fourier
    ).create_dataset()
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    ), {v: k for k, v in test_dataset.class_to_idx.items()}

def build_model(config, num_classes):
    """Build ResNet18 model with custom number of output classes"""
    model = timm.create_model(
        config["backbone_name"],
        pretrained=True,
        features_only=False,
        in_chans=3
    )
    return model

def build_optimizer(args, model):
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


def train_one_epoch(dataloader, model, optimizer, epoch, device, scheduler=None, log_interval=50):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    
    loss_meter = utils.AverageMeter()
    correct = 0
    total = 0

    for step, (images, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


        # Update metrics
        loss_meter.update(loss.item())

        # Logging
        if (step + 1) % log_interval == 0 or (step + 1) == len(dataloader):
            train_acc = 100. * correct / total
            print(f"Epoch {epoch+1} - Step {step+1}/{len(dataloader)}: "
                  f"loss = {loss_meter.val:.4f} (avg: {loss_meter.avg:.4f}), "
                  f"acc = {train_acc:.2f}%")

    train_acc = 100. * correct / total
    wandb.log({
        "train/loss": loss_meter.avg,
        "train/accuracy": train_acc,
        "epoch": epoch + 1
    })

    if scheduler:
        scheduler.step(loss_meter.avg)

    return loss_meter.avg, train_acc

def eval_once(dataloader, model, epoch, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    
    loss_meter = utils.AverageMeter()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, targets)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loss_meter.update(loss.item())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    accuracy = 100. * correct / total
    
    wandb.log({
        "val/loss": loss_meter.avg,
        "val/accuracy": accuracy,
        "epoch": epoch + 1
    })
    
    print(f"Validation - Epoch {epoch+1}: loss = {loss_meter.avg:.4f}, accuracy = {accuracy:.2f}%")
    
    return accuracy, np.array(all_preds), np.array(all_labels)

def train(args):
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = const.CHECKPOINT_DIR
    
    config = yaml.safe_load(open(args.config, "r"))
    
    wandb.init(project="AD4DD", config=config, name=args.run_name, mode=args.wandb_mode)
    wandb.config.update(args)

    train_dataloader = build_train_data_loader(args, config)
    test_dataloader, class2idx = build_test_data_loader(args, config)
    num_classes = len(class2idx)+1
    
    print(f"Number of classes: {num_classes}")
    
    # Build model with correct number of classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, num_classes)
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        print('Model loaded from checkpoint!')
    
    model = model.to(device)
    
    optimizer = build_optimizer(args, model)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_decay, 
        patience=args.lr_patience, verbose=True
    ) if args.scheduler == 1 else None
    
    best_acc = 0
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_one_epoch(
            train_dataloader, model, optimizer, epoch, device, 
            scheduler, args.log_interval
        )
        
        current_acc = -1
        
        if (epoch + 1) % args.eval_interval == 0:
            acc, preds, labels = eval_once(test_dataloader, model, epoch, device)
            current_acc = acc

            if current_acc > best_acc:
                best_acc = current_acc
                patience_counter = 0
                
                print(f"Epoch {epoch+1}: New best accuracy: {best_acc:.2f}%. Saving model.")
                
                wandb.run.summary["best_accuracy"] = best_acc
                
                checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "accuracy": best_acc,
                    },
                    checkpoint_path,
                )
            else:
                patience_counter += 1 
                print(f"Epoch {epoch+1}: Accuracy ({current_acc:.2f}%) not improved "
                      f"from {best_acc:.2f}%. Patience: {patience_counter}/{args.early_stopping_patience}")

        if patience_counter >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1} after {args.early_stopping_patience} "
                  f"epochs without improvement in validation accuracy.")
            break
    
    print(f"Training finished. Best accuracy: {best_acc:.2f}%")
    wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/resnet18.yaml', 
                        help="path to config file")
    parser.add_argument("--data", type=str, default='FF4ALL', 
                        help="dataset name", 
                        choices=['FF++', 'FF4ALL', 'progan'])
    parser.add_argument("--reals", type=str, default='ffhq', 
                        help="reals dataset", 
                        choices=['ffhq', 'celeba_hq', 'ffhq+celeba_hq'])
    parser.add_argument("--test", type=str, help="test name")
    parser.add_argument("--checkpoint", type=str, help="path to load checkpoint")

    parser.add_argument('--wandb_mode', default='disabled', 
                        choices=['online', 'offline', 'disabled'])
    parser.add_argument('--use_fourier', action='store_true')
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=50)

    # Hyperparameters
    parser.add_argument('--optimizer', type=str, default='AdamW', 
                        choices=['AdamW', 'sgd'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--scheduler', type=int, default=0)
    parser.add_argument('--lr_decay', type=float, default=0.7)
    parser.add_argument('--lr_patience', type=int, default=50)
    parser.add_argument('--early_stopping_patience', type=int, default=10, 
                        help="Patience epochs for early stopping based on Val Acc")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.run_name is None:
        args.run_name = (f"{args.data}_{args.reals}_lr{args.lr}_wd{args.weight_decay}_"
                        f"bs{args.batch_size}_ld{args.lr_decay}_lp{args.lr_patience}")
    const.CHECKPOINT_DIR += "/feature_extractor/" + args.run_name 

    train(args)