import glob
import csv
import os
from collections import defaultdict
import random

from muflow import constants as const

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

random.seed(42)

ROOT = os.path.join(const.DATA_DIR, "datasets")
output_file = os.path.join(const.WORKING_DIR, "data", "dataset_split_rand.csv")

csv_data = []


N_FAKE_PER_CLASS = cfg.FAKE_PER_CLASS
R_TRAIN, R_VAL   = cfg.SPLIT[0], cfg.SPLIT[1]


def split_by_class(pattern, label="class"):
    """Split files grouped by generator, capped per class, by the configured ratios."""
    files_by_class = defaultdict(list)
    for f in glob.glob(pattern):
        files_by_class[f.split(os.sep)[-2]].append(f)

    print(f"{'='*60}\nProcessing {label} ({len(files_by_class)} classes)")
    rows = []
    for cls, files in sorted(files_by_class.items()):
        random.shuffle(files)
        files = files[:N_FAKE_PER_CLASS]
        n_train = int(len(files) * R_TRAIN)
        n_val   = int(len(files) * R_VAL)
        splits  = (["train"] * n_train
                 + ["val"]   * n_val
                 + ["test"]  * (len(files) - n_train - n_val))
        rows += [[f, s] for f, s in zip(files, splits)]
        print(f"  {cls}: {len(files)} used  "
              f"(train={n_train} / val={n_val} / test={len(files)-n_train-n_val})")
    return rows


def split_random(pattern, label=""):
    """Split files into a random split (no class grouping) by the configured ratios."""
    files = glob.glob(pattern, recursive=True)
    random.shuffle(files)
    n_train = int(len(files) * R_TRAIN)
    n_val   = int(len(files) * R_VAL)
    rows = (  [[f, "train"] for f in files[:n_train]]
            + [[f, "val"]   for f in files[n_train:n_train + n_val]]
            + [[f, "test"]  for f in files[n_train + n_val:]])
    print(f"{'='*60}\nProcessing {label}: {len(files)} images  "
          f"(train={n_train} / val={n_val} / test={len(files)-n_train-n_val})")
    return rows


csv_data += split_random(
    os.path.join(ROOT, "ffhq", "*", "*.png"),
    label="FFHQ (reals)",
)

csv_data += split_by_class(
    os.path.join(ROOT, "WILD", "*", "*", "*.png"),
    label="WILD (Closed_Set + Open_Set)",
)

csv_data += split_by_class(
    os.path.join(ROOT, "datasets_DFX", "*", "*.png"),
    label="datasets_DFX",
)

print("=" * 60)
print("Processing CelebA-HQ (OOD reals)")

celeba_train = glob.glob(os.path.join(ROOT, "celeba_hq", "train", "*", "*.jpg"))
random.shuffle(celeba_train)
n_train_c = int(len(celeba_train) * 0.80)
csv_data += [[f, "train"] for f in celeba_train[:n_train_c]]
csv_data += [[f, "val"]   for f in celeba_train[n_train_c:]]

celeba_test = glob.glob(os.path.join(ROOT, "celeba_hq", "val", "*", "*.jpg"))
csv_data += [[f, "test"] for f in celeba_test]

print(f"  train folder → train={n_train_c} / val={len(celeba_train)-n_train_c}")
print(f"  val   folder → test={len(celeba_test)}")

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "split"])
    writer.writerows(csv_data)

total_train = sum(1 for r in csv_data if r[1] == "train")
total_val   = sum(1 for r in csv_data if r[1] == "val")
total_test  = sum(1 for r in csv_data if r[1] == "test")

print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"CSV saved in : {output_file}")
print(f"Total images : {len(csv_data)}")
print(f"  Train : {total_train} ({total_train/len(csv_data)*100:.1f}%)")
print(f"  Val   : {total_val}   ({total_val/len(csv_data)*100:.1f}%)")
print(f"  Test  : {total_test}  ({total_test/len(csv_data)*100:.1f}%)")
