import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, average_precision_score
import torch
import dataset
import fastflow
import constants as const
import yaml
import utils
import pdb
import matplotlib.pyplot as plt

run = 'ricercadellaricercadelbuono'

labels = np.load(f'/home/mlitrico/AD4DD/FastFlow/logs/{run}/labels_best.npy')
preds_ = np.load(f'/home/mlitrico/AD4DD/FastFlow/logs/{run}/preds_best.npy')

"""likelihood_real = preds_[labels == 0][:1000]
likelihood_fake = preds_[labels == 14]

#likelihood_fake = likelihood_fake[np.abs(likelihood_fake - likelihood_fake.mean()) < 10000000]

plt.figure(figsize=(10, 6))
plt.scatter(range(len(likelihood_real)), likelihood_real, label='Real', color='blue', alpha=0.5)
plt.scatter(range(len(likelihood_fake)), likelihood_fake, label='Fake', color='orange', alpha=0.5)
plt.savefig("/home/mlitrico/real_fake_separation.png")

pdb.set_trace()"""

def build_train_data_loader(config):
    # The data loading code remains the same
    train_dataset = dataset.Dataset(
        dataset_name='FF4ALL',
        reals_name='ffhq',
        input_size=config["input_size"],
        is_train=True,
        use_fourier=False
    ).create_dataset()
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

def build_test_data_loader(config):
    # The data loading code remains the same
    test_dataset = dataset.Dataset(
        dataset_name='FF4ALL',
        reals_name='ffhq',
        input_size=config["input_size"],
        is_train=False,
        use_fourier=False
    ).create_dataset()
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    ), {v: k for k, v in test_dataset.class_to_idx.items()}




def build_model(config, model_type):
    
    gmm_parameters = f"{const.WORKING_DIR}/parameters/gmm_parameters_{config['backbone_name']}_ffhq_{config['input_size']}.npy"
    
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
            in_channels= 3,
            backbone_weights=None
        )
        print(
            "Model A.D. Param#: {}".format(
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            )
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


config = yaml.safe_load(open('/home/mlitrico/AD4DD/FastFlow/configs/resnet18.yaml', "r"))

model = build_model(config, 'FastFlow')
model.cuda()

model.train()

checkpoint = torch.load(f'/home/mlitrico/AD4DD/FastFlow/logs/{run}/best.pt')

model.load_state_dict(checkpoint["model_state_dict"])

train_dataloader = build_train_data_loader(config)
test_dataloader, class2idx = build_test_data_loader(config)

loss_values = []
for step, data in enumerate(train_dataloader):
    data = data.cuda()
    with torch.no_grad():
        ret = model(data)
    # loss = ret["loss"].mean()
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # loss_meter.update(loss.item())
    loss_values.append(ret["loss"])

preds_train = np.array(torch.cat(loss_values).detach().cpu()).reshape(-1,1)

# contamination = 'auto'

# gmm = LocalOutlierFactor(novelty=True, contamination=contamination)
# gmm = OneClassSVM()

# preds_0 = preds_[labels==0]

# gmm.fit(np.array(torch.cat(loss_values).detach().cpu()).reshape(-1,1))
# gmm.fit(np.array(preds_).reshape(-1,1))


model.eval()

labels = []
preds = []
for data, targets in test_dataloader:
    data, targets = data.cuda(), targets.cuda()
    with torch.no_grad():
        ret = model(data)
    
        outputs = ret["loss"].cpu().detach()
    
    preds.append(outputs)
    labels.append(targets.detach().cpu())

print("Testing done")

# Ottieni un array 1D di punteggi e uno di etichette
preds_ = np.concatenate(preds)
labels = np.concatenate(labels)

# preds = (gmm.predict(np.array(preds_).reshape(-1,1)) < 0).astype(int)
# pdb.set_trace()

# threshold = -700000

threshold = preds_train.mean() + 3 * preds_train.std()
preds = (preds_ > threshold).astype(int)
# pdb.set_trace()
print(accuracy_score(labels>0, preds))
# pdb.set_trace()

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

np.savez(
    f'/home/mlitrico/AD4DD/FastFlow/logs/{run}/thresholds.npz',
    threshold=threshold,
    mean=preds_train.mean(),
    std=preds_train.std()
)
pdb.set_trace()


# print(np.unique(preds, return_counts=True))

# # Visualizza le predizioni per gli indici in cui label==0
# preds_label0 = preds[label == 0]
# print("Predizioni corrispondenti agli indici in cui label==0:")
# print(preds_label0.mean())
# # print("Distribuzione delle predizioni per label==0 (valore, conteggio):")
# # print(np.unique(preds_label0, return_counts=True))

# preds_labels1 = preds[label == 1]
# print("Predizioni corrispondenti agli indici in cui label==1:")
# print(preds_labels1.mean())
# # print("Distribuzione delle predizioni per label==1 (valore, conteggio):")
# # print(np.unique(preds_labels1, return_counts=True))

# preds_labels2 = preds[label == 2]
# print("Predizioni corrispondenti agli indici in cui label==2:")
# print(preds_labels2.mean())
# # print("Distribuzione delle predizioni per label==2 (valore, conteggio):")
# # print(np.unique(preds_labels2, return_counts=True))

# preds_labels3 = preds[label == 3]
# print("Predizioni corrispondenti agli indici in cui label==3:")
# print(preds_labels3.mean())
# # print("Distribuzione delle predizioni per label==3 (valore, conteggio):")
# # print(np.unique(preds_labels3, return_counts=True))

# preds_labels4 = preds[label == 4]
# print("Predizioni corrispondenti agli indici in cui label==4:")
# print(preds_labels4.mean())
# # print("Distribuzione delle predizioni per label==4 (valore, conteggio):")
# # print(np.unique(preds_labels4, return_counts=True))

# preds_labels5 = preds[label == 5]
# print("Predizioni corrispondenti agli indici in cui label==5:")
# print(preds_labels5.mean())
# # print("Distribuzione delle predizioni per label==5 (valore, conteggio):")
# # print(np.unique(preds_labels5, return_counts=True))

# preds_labels6 = preds[label == 6]
# print("Predizioni corrispondenti agli indici in cui label==6:")
# print(preds_labels6.mean())
# # print("Distribuzione delle predizioni per label==6 (valore, conteggio):")
# # print(np.unique(preds_labels6, return_counts=True))

# preds_labels7 = preds[label == 7]
# print("Predizioni corrispondenti agli indici in cui label==7:")
# print(preds_labels7.mean())
# # print("Distribuzione delle predizioni per label==7 (valore, conteggio):")
# # print(np.unique(preds_labels7, return_counts=True))

# preds_labels8 = preds[label == 8]
# print("Predizioni corrispondenti agli indici in cui label==8:")
# print(preds_labels8.mean())
# # print("Distribuzione delle predizioni per label==8 (valore, conteggio):")
# # print(np.unique(preds_labels8, return_counts=True))

# preds_labels9 = preds[label == 9]
# print("Predizioni corrispondenti agli indici in cui label==9:")
# print(preds_labels9.mean())
# # print("Distribuzione delle predizioni per label==9 (valore, conteggio):")
# # print(np.unique(preds_labels9, return_counts=True))

# preds_labels10 = preds[label == 10]
# print("Predizioni corrispondenti agli indici in cui label==10:")
# print(preds_labels10.mean())
# # print("Distribuzione delle predizioni per label==10 (valore, conteggio):")
# # print(np.unique(preds_labels10, return_counts=True))

# preds_labels11 = preds[label == 11]
# print("Predizioni corrispondenti agli indici in cui label==11:")
# print(preds_labels11.mean())
# # print("Distribuzione delle predizioni per label==11 (valore, conteggio):")
# # print(np.unique(preds_labels11, return_counts=True))

# preds_labels12 = preds[label == 12]
# print("Predizioni corrispondenti agli indici in cui label==12:")
# print(preds_labels12.mean())
# # print("Distribuzione delle predizioni per label==12 (valore, conteggio):")
# # print(np.unique(preds_labels12, return_counts=True))

# preds_labels13 = preds[label == 13]
# print("Predizioni corrispondenti agli indici in cui label==13:")
# print(preds_labels13.mean())
# # print("Distribuzione delle predizioni per label==13 (valore, conteggio):")
# # print(np.unique(preds_labels13, return_counts=True))

# preds_labels14 = preds[label == 14]
# print("Predizioni corrispondenti agli indici in cui label==14:")
# print(preds_labels14.mean())
# # print("Distribuzione delle predizioni per label==14 (valore, conteggio):")
# # print(np.unique(preds_labels14, return_counts=True))

# preds_labels15 = preds[label == 15]
# print("Predizioni corrispondenti agli indici in cui label==15:")
# print(preds_labels15.mean())
# # print("Distribuzione delle predizioni per label==15 (valore, conteggio):")
# # print(np.unique(preds_labels15, return_counts=True))

# preds_labels16 = preds[label == 16]
# print("Predizioni corrispondenti agli indici in cui label==16:")
# print(preds_labels16.mean())
# # print("Distribuzione delle predizioni per label==16 (valore, conteggio):")
# # print(np.unique(preds_labels16, return_counts=True))
