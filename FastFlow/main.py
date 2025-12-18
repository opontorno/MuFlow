import argparse
import os
import torch
from torch.optim import optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import wandb
from ignite.contrib import metrics
import joblib

import constants as const
import dataset
import fastflow
#import vanillaVAE as vae
import utils

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import pdb
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.neighbors import LocalOutlierFactor


def build_train_data_loader(args, config):
    # The data loading code remains the same
    train_dataset = dataset.Dataset(
        dataset_name=args.data,
        reals_name=args.reals,
        test_name=args.test,
        input_size=config["input_size"],
        is_train=True,
        use_fourier=False
    ).create_dataset()
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )


def build_test_data_loader(args, config):
    # The data loading code remains the same
    test_dataset = dataset.Dataset(
        dataset_name=args.data,
        reals_name=args.reals,
        test_name=args.test,
        input_size=config["input_size"],
        is_train=False,
        use_fourier=False
    ).create_dataset()
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    ), {v: k for k, v in test_dataset.class_to_idx.items()}


def build_model(config, model_type):
    
    gmm_parameters = f"{const.WORKING_DIR}/parameters/gmm_parameters_{config['backbone_name']}_fourier_{args.reals}_{config['input_size']}.npy" if args.use_fourier \
                else f"{const.WORKING_DIR}/parameters/gmm_parameters_{config['backbone_name']}_{args.reals}_{config['input_size']}.npy"
    
    gmm_values = np.load(gmm_parameters, allow_pickle=True).item() 
    print(f"Loading gmm parameters from {gmm_parameters}")

    if model_type == "FastFlow":
        model = fastflow.FastFlow(
            backbone_name=config["backbone_name"],
            flow_steps=config["flow_step"],
            input_size=384 if args.use_fourier else config["input_size"],
            conv3x3_only=config["conv3x3_only"],
            hidden_ratio=config["hidden_ratio"],
            gmm_values=gmm_values,
            in_channels=1 if args.use_fourier else 3,
            backbone_weights=args.backbone_weights
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


def train_one_epoch(dataloader, model, optimizer, epoch, contamination='auto', scheduler=None):
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
        loss_values.append(ret["loss"])
        if (step + 1) % args.log_interval == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )
            
            wandb.log({"Train Loss": loss_meter.val, "Epoch": epoch + 1}, step=epoch + 1)

    if scheduler:
      scheduler.step(loss_meter.val)

    gmm = LocalOutlierFactor(novelty=True, contamination=contamination)
    gmm.fit(np.array(torch.cat(loss_values).detach().cpu()).reshape(-1,1))
    return gmm, np.array(torch.cat(loss_values).detach().cpu()).mean(), np.array(torch.cat(loss_values).detach().cpu()).std(), np.array(torch.cat(loss_values).detach().cpu()).reshape(-1,1)


def eval_once(dataloader, model, epoch=None, model_type="FastFlow", gmm=None, class2idx=None, preds_train=None): 
    model.eval()
    labels = []
    preds = []
    for data, targets in dataloader:
        data, targets = data.cuda(), targets.cuda()
        with torch.no_grad():
            ret = model(data) if model_type == "FastFlow" else model(data, eval_mode=True)
        
        if model_type == "FastFlow":
            outputs = ret["loss"].cpu().detach()
        
        preds.append(outputs)
        labels.append(targets.detach().cpu())

    print("Testing done")
    
    preds_ = np.concatenate(preds)
    labels = np.concatenate(labels)
    
    # preds = (gmm.predict(np.array(preds_).reshape(-1,1)) < 0).astype(int)

    threshold = preds_train.mean() + 3 * preds_train.std()
    preds = (preds_ > threshold).astype(int)
    
    if epoch:
        likelihood_real = preds_[labels == 0]
        likelihood_fake = preds_[labels > 0]

        plt.figure(figsize=(10, 6))

        # plt.scatter(range(len(likelihood_real)), likelihood_real, label='Real', color='blue', alpha=0.5)
        # plt.scatter(range(len(likelihood_fake)), likelihood_fake, label='Fake', color='orange', alpha=0.5)
        
        sns.kdeplot(likelihood_real, label='Real', color='blue')
        sns.kdeplot(likelihood_fake, label='Fake', color='orange')


        plt.title('Real vs Fake')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(const.CHECKPOINT_DIR+"/real_fake_likelihoods_{}.png".format(epoch))
    
    aps, accs, cls = [], [], []

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
            print(f"  > Classe {class2idx[c] if class2idx else c}: SALTATA (0 campioni)")
            continue
        
        #Shuffling real to get different subset each time
        idx = np.random.permutation(len(y_true_0))
        y_true_0 = y_true_0[idx]
        y_pred_0 = y_pred_0[idx]

        y_true_balanced = np.array(y_true_c[:min_len].tolist() + y_true_0[:min_len].tolist())
        y_pred_balanced = np.array(y_pred_c[:min_len].tolist() + y_pred_0[:min_len].tolist())
        
        y_true_binary = (y_true_balanced > 0).astype(np.int8)
        
        ap = average_precision_score(y_true_binary, y_pred_balanced)
        acc0 = accuracy_score(y_true_binary, y_pred_balanced)
        
        print(f"  > Class {class2idx[c] if class2idx else c} (N={min_len*2}): \t AP = {ap:.4f}, \t Accuracy = {acc0:.4f}")
        
        aps.append(ap)
        accs.append(acc0)
        cls.append(c)

    mean_ap = np.mean(aps)
    mean_acc = np.mean(accs)
    
    print("-" * 30)
    print(f"Average Accuracy: {mean_acc:.4f}")
    print(f"Average Precision: {mean_ap:.4f}")
    print("="*30 + "\n")

    
    wandb.log({"Val acc": mean_acc, "Val loss real": np.mean(preds_[labels == 0]), "Val loss fake": np.mean(preds_[labels > 0])}, step=epoch + 1)

    return mean_acc, preds_, labels, {'threshold': threshold, 'mean': preds_train.mean(), 'std': preds_train.std()}

def train(args):
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = const.CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    config = yaml.safe_load(open(args.config, "r"))

    
    wandb.init(project="AD4DD", config=config, name=args.run_name, mode=args.wandb)
    wandb.config.update(args)

    model = build_model(config, args.model_type)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        print('Model loaded!')
        model.cuda()

    optimizer = build_optimizer(args, model, args.model_type, config)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay, patience=args.lr_patience, verbose=True) if args.scheduler==1 else None

    train_dataloader = build_train_data_loader(args, config)
    test_dataloader, class2idx = build_test_data_loader(args, config)
    model.cuda()
    
    best_acc = 0
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        gmm, means, stds, preds_train = train_one_epoch(train_dataloader, model, optimizer, epoch, contamination=args.contamination, scheduler=scheduler)
        
        current_acc = -1
        
        if (epoch + 1) % args.eval_interval == 0:
            # acc, preds, labels = eval_once(test_dataloader, model, epoch, args.model_type, gmm, means, stds)
            acc, preds, labels, threshold_ = eval_once(test_dataloader, model, epoch, args.model_type, gmm, class2idx, preds_train)
            current_acc = acc

            if current_acc > best_acc:
                best_acc = current_acc
                patience_counter = 0
                
                print(f"Epoch {epoch+1}: Nuova best accuracy: {best_acc:.4f}. Salvataggio modello.")
                
                
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
                np.savez(
                    os.path.join(checkpoint_dir, "thresholds.npz"),
                        threshold=threshold_['threshold'],
                        mean=threshold_['mean'],
                        std=threshold_['std']
                )
                np.save(
                    os.path.join(checkpoint_dir, "preds_best.npy"), preds
                )
                np.save(
                    os.path.join(checkpoint_dir, "labels_best.npy"), labels
                )
                joblib.dump(gmm, os.path.join(checkpoint_dir, "gmm_model.pkl"))
            
            else:
                patience_counter += 1 
                print(f"Epoch {epoch+1}: Accuracy ({current_acc:.4f}) non migliorata rispetto a {best_acc:.4f}. Pazienza: {patience_counter}/{args.early_stopping_patience}")

        if patience_counter >= args.early_stopping_patience:
            print(f"Stopping early all'epoca {epoch + 1} dopo {args.early_stopping_patience} epoche senza miglioramento della Val Acc.")
            #break
    
    print(f"Training terminato. Migliore accuracy: {best_acc:.4f}")

def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))

    checkpoint = torch.load(args.checkpoint)
    gmm = joblib.load(args.gmm_checkpoint)

    model = build_model(config, args.model_type)  # Pass the model type here
    model.load_state_dict(checkpoint["model_state_dict"])

    test_dataloader, class2idx = build_test_data_loader(args, config)
    model.cuda()
    
    eval_once(test_dataloader, model, model_type=args.model_type, gmm=gmm)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/resnet18.yaml', help="path to config file")

    parser.add_argument("--data", type=str, default='FF4ALL', help="path to mvtec folder", choices=['FF++', 'FF4ALL', 'progan'])
    parser.add_argument("--reals", type=str, default='ffhq', help="reals dataset", choices=['ffhq', 'celeba_hq', 'ffhq+celeba_hq'])
    parser.add_argument("--test", type=str, help="test name")
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument("--checkpoint", type=str, help="path to load checkpoint")
    parser.add_argument('--gmm_checkpoint', type=str)

    parser.add_argument('--wandb', default= 'disabled', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--use_fourier', action='store_true')
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--model_type', type=str, choices=['FastFlow', 'VAE'], default='FastFlow', help="Choose the model to train")
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--backbone_weights', type=str, help="path to load backbone weights")
    parser.add_argument('--log_interval', type=int, default=10)

    # Hyperparameters
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'sgd'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--scheduler', type=int, default=0)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--lr_patience', type=int, default=30)
    parser.add_argument('--contamination', default='auto')

    parser.add_argument('--early_stopping_patience', type=int, default=400, help="Patience epochs for early stopping based on Val Acc")

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.run_name is None:
        args.run_name = f"{args.data}_{args.reals}_lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}_ld{args.lr_decay}_lp{args.lr_patience}"
    const.CHECKPOINT_DIR += "/" + args.run_name 

    if args.eval:
        evaluate(args)
    else:
        train(args)
