import pandas as pd

CSV_PATH = '/media/orazio_mattia_group/ad4dd/dataset_split.csv'

old_root = "/media/orazio_mattia_group/ad4dd/celeba_hq/"
new_root = "/mnt/storage/ad4dd/celeba_hq/"

# Load CSV
df = pd.read_csv(CSV_PATH)

# Replace the path
df["path"] = df["path"].str.replace(old_root, new_root, regex=False)

# Save new CSV
df.to_csv("dataset_split.csv", index=False)
print("Updated CSV saved as 'dataset_split.csv'")