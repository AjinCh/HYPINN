# This code is used to cout the sample size of datasets 
import os
import pandas as pd
from pathlib import Path
# ---------------------------------------------------------------------
# automatically detect project root
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_A = REPO_ROOT / 'data' / 'Dataset_A'
DATASET_B = REPO_ROOT / 'data' / 'Dataset_B'

roots = [DATASET_A, DATASET_B]

total_samples = 0
total_files = 0

for root in roots:
    sub_total = 0
    files = os.listdir(root)
    num_files = len(files)

    for filename in files:
        file_path = os.path.join(root, filename)
        try:
            df = pd.read_csv(file_path, encoding="latin1")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="ISO-8859-1")

        sub_total += df.shape[0]

    print(f"Dataset '{os.path.basename(root)}' has {num_files} files and {sub_total} samples.")
    total_samples += sub_total
    total_files += num_files

print("\nSummary:")
print(f"  Total samples: {total_samples}")
print(f"  Total files (batteries): {total_files}")
