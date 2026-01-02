import glob
import csv
import os
from collections import defaultdict
import random

# Imposta il seed per riproducibilità
random.seed(42)

# Lista per contenere tutti i dati del CSV
csv_data = []

# ============================================================================
# 1. FF4ALL - dividi per classe con proporzione 70-15-15 (train-val-test)
# ============================================================================
print("=" * 60)
print("Processando FF4ALL (con classi)")
print("=" * 60)

path_pattern_ff4all = "/media/orazio_mattia_group/ad4dd/FF4ALL/*/*/*.png"
all_files_ff4all = glob.glob(path_pattern_ff4all)

# Organizza i file per classe
files_by_class = defaultdict(list)

for file_path in all_files_ff4all:
    parts = file_path.split(os.sep)
    class_name = parts[-2]
    files_by_class[class_name].append(file_path)

# Per ogni classe, dividi in train (70%), val (15%) e test (15%)
for class_name, files in files_by_class.items():
    random.shuffle(files)
    n_train = int(len(files) * 0.7)
    n_val = int(len(files) * 0.15)
    
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    
    for file_path in train_files:
        csv_data.append([file_path, "train"])
    
    for file_path in val_files:
        csv_data.append([file_path, "val"])
    
    for file_path in test_files:
        csv_data.append([file_path, "test"])
    
    print(f"Classe: {class_name}")
    print(f"  Train: {len(train_files)} immagini ({len(train_files)/len(files)*100:.1f}%)")
    print(f"  Val: {len(val_files)} immagini ({len(val_files)/len(files)*100:.1f}%)")
    print(f"  Test: {len(test_files)} immagini ({len(test_files)/len(files)*100:.1f}%)")
    print(f"  Totale: {len(files)} immagini\n")

# ============================================================================
# 2. Altri file senza classe - dividi con proporzione 70-15-15 (train-val-test)
# ============================================================================
print("=" * 60)
print("Processando altri file (senza classi)")
print("=" * 60)

path_pattern_other = "/media/orazio_mattia_group/ad4dd/*/*/*.png"
all_files_other = glob.glob(path_pattern_other)

# Escludi i file di FF4ALL già processati
all_files_other = [f for f in all_files_other if "/FF4ALL/" not in f]

# Dividi in train (70%), val (15%) e test (15%)
random.shuffle(all_files_other)
n_train_other = int(len(all_files_other) * 0.7)
n_val_other = int(len(all_files_other) * 0.15)

train_files_other = all_files_other[:n_train_other]
val_files_other = all_files_other[n_train_other:n_train_other + n_val_other]
test_files_other = all_files_other[n_train_other + n_val_other:]

for file_path in train_files_other:
    csv_data.append([file_path, "train"])

for file_path in val_files_other:
    csv_data.append([file_path, "val"])

for file_path in test_files_other:
    csv_data.append([file_path, "test"])

print(f"File senza classe:")
print(f"  Train: {len(train_files_other)} immagini ({len(train_files_other)/len(all_files_other)*100:.1f}%)")
print(f"  Val: {len(val_files_other)} immagini ({len(val_files_other)/len(all_files_other)*100:.1f}%)")
print(f"  Test: {len(test_files_other)} immagini ({len(test_files_other)/len(all_files_other)*100:.1f}%)")
print(f"  Totale: {len(all_files_other)} immagini\n")

# ============================================================================
# 3. CelebA HQ - mantieni la divisione originale train/val, val diventa "val"
# ============================================================================
print("=" * 60)
print("Processando CelebA HQ")
print("=" * 60)

# Train
celeba_train_pattern = "/media/orazio_mattia_group/ad4dd/celeba_hq/train/*/*.jpg"
celeba_train_files = glob.glob(celeba_train_pattern)

for file_path in celeba_train_files:
    csv_data.append([file_path, "train"])

print(f"CelebA HQ Train: {len(celeba_train_files)} immagini")

# Val (ora lo mettiamo come "val" invece di "test")
celeba_val_pattern = "/media/orazio_mattia_group/ad4dd/celeba_hq/val/*/*.jpg"
celeba_val_files = glob.glob(celeba_val_pattern)

for file_path in celeba_val_files:
    csv_data.append([file_path, "val"])

print(f"CelebA HQ Val: {len(celeba_val_files)} immagini")
print(f"CelebA HQ Totale: {len(celeba_train_files) + len(celeba_val_files)} immagini\n")

# ============================================================================
# Salva il CSV
# ============================================================================
output_file = '/media/orazio_mattia_group/ad4dd/dataset_split_rand.csv'
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["path", "split"])
    writer.writerows(csv_data)

print("=" * 60)
print("RIEPILOGO FINALE")
print("=" * 60)
total_train = sum(1 for row in csv_data if row[1] == "train")
total_val = sum(1 for row in csv_data if row[1] == "val")
total_test = sum(1 for row in csv_data if row[1] == "test")

print(f"CSV salvato in: {output_file}")
print(f"Totale immagini: {len(csv_data)}")
print(f"  Train: {total_train} immagini ({total_train/len(csv_data)*100:.1f}%)")
print(f"  Val: {total_val} immagini ({total_val/len(csv_data)*100:.1f}%)")
print(f"  Test: {total_test} immagini ({total_test/len(csv_data)*100:.1f}%)")