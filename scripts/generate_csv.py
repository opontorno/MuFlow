import glob
import csv
import os
from collections import defaultdict
import random

from muflow import constants as const

# Set seed for reproducibility
random.seed(42)

ROOT = os.path.join(const.DATA_DIR, "datasets")

# List to contain all CSV data
csv_data = []

# ============================================================================
# 1. WILD - split by class with proportion 70-15-15 (train-val-test)
# ============================================================================
print("=" * 60)
print("Processing WILD (with classes)")
print("=" * 60)

path_pattern_ff4all = os.path.join(ROOT, "WILD", "*", "*", "*.png")
all_files_ff4all = glob.glob(path_pattern_ff4all)

# Organize files by class
files_by_class = defaultdict(list)

for file_path in all_files_ff4all:
    parts = file_path.split(os.sep)
    class_name = parts[-2]
    files_by_class[class_name].append(file_path)

# For each class, split into train (70%), val (15%) and test (15%)
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
    
    print(f"Class: {class_name}")
    print(f"  Train: {len(train_files)} images ({len(train_files)/len(files)*100:.1f}%)")
    print(f"  Val: {len(val_files)} images ({len(val_files)/len(files)*100:.1f}%)")
    print(f"  Test: {len(test_files)} images ({len(test_files)/len(files)*100:.1f}%)")
    print(f"  Total: {len(files)} images\n")

# ============================================================================
# 2. Other files without class - split with proportion 70-15-15 (train-val-test)
# ============================================================================
print("=" * 60)
print("Processing other files (without classes)")
print("=" * 60)

path_pattern_other = os.path.join(ROOT, "*", "*", "*.png")
all_files_other = glob.glob(path_pattern_other)

# Exclude WILD files already processed
all_files_other = [f for f in all_files_other if "/WILD/" not in f]

# Split into train (70%), val (15%) and test (15%)
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

print(f"Files without class:")
print(f"  Train: {len(train_files_other)} images ({len(train_files_other)/len(all_files_other)*100:.1f}%)")
print(f"  Val: {len(val_files_other)} images ({len(val_files_other)/len(all_files_other)*100:.1f}%)")
print(f"  Test: {len(test_files_other)} images ({len(test_files_other)/len(all_files_other)*100:.1f}%)")
print(f"  Total: {len(all_files_other)} images\n")

# ============================================================================
# 3. datasets_DFX - split with proportion 70-15-15 (train-val-test)
# ============================================================================
print("=" * 60)
print("Processing datasets_DFX")
print("=" * 60)

path_pattern_dfx = os.path.join(ROOT, "datasets_DFX", "*", "*.png")
all_files_dfx = glob.glob(path_pattern_dfx)

# Split into train (70%), val (15%) and test (15%)
random.shuffle(all_files_dfx)
n_train_dfx = int(len(all_files_dfx) * 0.7)
n_val_dfx = int(len(all_files_dfx) * 0.15)

train_files_dfx = all_files_dfx[:n_train_dfx]
val_files_dfx = all_files_dfx[n_train_dfx:n_train_dfx + n_val_dfx]
test_files_dfx = all_files_dfx[n_train_dfx + n_val_dfx:]

for file_path in train_files_dfx:
    csv_data.append([file_path, "train"])

for file_path in val_files_dfx:
    csv_data.append([file_path, "val"])

for file_path in test_files_dfx:
    csv_data.append([file_path, "test"])

print(f"datasets_DFX:")
print(f"  Train: {len(train_files_dfx)} images ({len(train_files_dfx)/len(all_files_dfx)*100:.1f}%)")
print(f"  Val: {len(val_files_dfx)} images ({len(val_files_dfx)/len(all_files_dfx)*100:.1f}%)")
print(f"  Test: {len(test_files_dfx)} images ({len(test_files_dfx)/len(all_files_dfx)*100:.1f}%)")
print(f"  Total: {len(all_files_dfx)} images\n")

# ============================================================================
# 4. CelebA HQ - keep original train/val split, val becomes "val"
# ============================================================================
print("=" * 60)
print("Processing CelebA HQ")
print("=" * 60)

# Train
celeba_train_pattern = os.path.join(ROOT, "celeba_hq", "train", "*", "*.jpg")
celeba_train_files = glob.glob(celeba_train_pattern)

random.shuffle(celeba_train_files)
n_train_celeba = int(len(celeba_train_files) * 0.8)
n_val_celeba = len(celeba_train_files) - n_train_celeba

for file_path in celeba_train_files[:n_train_celeba]:
    csv_data.append([file_path, "train"])
for file_path in celeba_train_files[n_train_celeba:]:
    csv_data.append([file_path, "val"])

print(f"CelebA HQ Train: {n_train_celeba} images")
print(f"CelebA HQ Val: {n_val_celeba} images")

celeba_test_pattern = os.path.join(ROOT, "celeba_hq", "val", "*", "*.jpg")
celeba_test_files = glob.glob(celeba_test_pattern)

for file_path in celeba_test_files:
    csv_data.append([file_path, "test"])

print(f"CelebA HQ Test: {len(celeba_test_files)} images")
print(f"CelebA HQ Total: {len(celeba_train_files) + len(celeba_test_files)} images\n")
# ============================================================================
# Save the CSV
# ============================================================================
output_file = os.path.join(ROOT, "dataset_split_rand.csv")
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["path", "split"])
    writer.writerows(csv_data)

print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
total_train = sum(1 for row in csv_data if row[1] == "train")
total_val = sum(1 for row in csv_data if row[1] == "val")
total_test = sum(1 for row in csv_data if row[1] == "test")

print(f"CSV saved in: {output_file}")
print(f"Total images: {len(csv_data)}")
print(f"  Train: {total_train} images ({total_train/len(csv_data)*100:.1f}%)")
print(f"  Val: {total_val} images ({total_val/len(csv_data)*100:.1f}%)")
print(f"  Test: {total_test} images ({total_test/len(csv_data)*100:.1f}%)")