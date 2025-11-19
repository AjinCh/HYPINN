#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Correlation analysis on battery degradation dataset Dataset_A
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
from Data_loading.dataloader import Dataset_A as DatasetAData, Dataset_B as DatasetBData

BATCHES = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
# ---------------------------------------------------------------------
# automatically detect project root for portable execution
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

def resolve_data_root(dataset: str, cli_root: Optional[str] = None) -> Path:
    if cli_root:
        p = Path(cli_root).expanduser().resolve()
        candidates = [p, p / dataset, p / "data" / dataset]
        for c in candidates:
            if c.name == dataset and c.is_dir():
                return c
        tried = ", ".join(str(c) for c in candidates)
        raise FileNotFoundError("--data_root did not resolve to '{0}'. Tried: {1}".format(dataset, tried))
# use automatic repo detection for portability
    search_bases = [REPO_ROOT, REPO_ROOT / "data", Path.cwd(), Path.cwd().parent]
    for base in search_bases:
        cand = base / "data" / dataset if (base / "data").is_dir() else base / dataset
        if cand.is_dir():
            return cand
    tried = ", ".join(str(base / 'data' / dataset) for base in search_bases)
    raise FileNotFoundError("Could not find data/{0}. Tried: {1}".format(dataset, tried))

def project_root_from_data_root(data_root: Path) -> Path:
    # automatically resolve root for both data and results folders
    return REPO_ROOT

def build_args(normalization: str, batch_size: int):
    return argparse.Namespace(
        batch_size=batch_size,
        normalization_method=normalization,
        log_dir='log.txt',
        save_folder='.'
    )

def list_all_files(root: Path, batch_label: str) -> List[Path]:
    paths = []
    for fname in os.listdir(root):
        if batch_label.lower() in fname.lower():
            fpath = root / fname
            if fpath.is_file():
                paths.append(fpath)
    paths.sort(key=lambda p: (0 if p.suffix.lower() == '.csv' else 1, str(p)))
    return paths

def load_xy_with_dataloader(data_cls, root: Path, one_file: Path, args_for_loader) -> Tuple[np.ndarray, np.ndarray]:
    data = data_cls(root=str(root), args=args_for_loader)
    loaders = data.read_all(specific_path_list=[str(one_file)])
    Xs, ys = [], []
    for split in ("train", "valid", "test"):
        for x1, _, y1, _ in loaders[split]:
            x_np = x1.detach().cpu().numpy() if hasattr(x1, "detach") else np.asarray(x1)
            y_np = y1.detach().cpu().numpy().reshape(-1) if hasattr(y1, "detach") else np.asarray(y1).reshape(-1)
            Xs.append(x_np)
            ys.append(y_np)
    return np.vstack(Xs), np.concatenate(ys)

# computes pearson correlation coefficient for each feature
def compute_pearson(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y)
    for j in range(X.shape[1]):
        mask &= np.isfinite(X[:, j])
    Xc = X[mask] - X[mask].mean(axis=0, keepdims=True)
    yc = y[mask] - y[mask].mean()
    denom_y = np.sqrt((yc ** 2).sum())
    denom_x = np.sqrt((Xc ** 2).sum(axis=0))
    r = np.zeros(Xc.shape[1], dtype=float)
    if denom_y > 0:
        numer = (Xc * yc.reshape(-1, 1)).sum(axis=0)
        safe = (denom_x > 0) & (denom_y > 0)
        r[safe] = numer[safe] / (denom_x[safe] * denom_y)
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["Dataset_A", "Dataset_B"], default="Dataset_A")
    ap.add_argument("--normalization_method", default="min-max", choices=["min-max", "z-score"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--out_root", type=str, default="results for analysis")
    args = ap.parse_args()

    # automatically resolve data and project roots for any environment
    data_root = resolve_data_root(args.dataset, args.data_root)
    project_root = project_root_from_data_root(data_root)
    outdir = project_root / args.out_root / "correlation_results" / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)

    print("[info] dataset   : {0}".format(args.dataset))
    print("[info] data_root : {0}".format(data_root))
    print("[info] outdir    : {0}".format(outdir))

    DataClass = DatasetAData if args.dataset == "Dataset_A" else DatasetBData
    all_rows = []
    files_used = {b: [] for b in BATCHES}
    feature_dim = None

    for batch in BATCHES:
        loader_args = build_args(args.normalization_method, args.batch_size)
        batch_files = list_all_files(data_root, batch)
        if not batch_files:
            raise FileNotFoundError("No files found in {0} for batch '{1}'.".format(data_root, batch))
        X_accum, y_accum = [], []
        for f in batch_files:
            try:
                X, y = load_xy_with_dataloader(DataClass, data_root, f, loader_args)
                if feature_dim is None:
                    feature_dim = X.shape[1]
                elif X.shape[1] != feature_dim:
                    raise ValueError("Feature dimension mismatch in file {0}: got {1} vs expected {2}".format(f, X.shape[1], feature_dim))
                X_accum.append(X)
                y_accum.append(y)
                files_used[batch].append(str(f))
                print("[{0}] included -> {1} (X: {2}, y: {3})".format(batch, f.name, X.shape, y.shape))
            except Exception as e:
                print("[{0}] Skipping file due to error: {1}\n  {2}".format(batch, f, e))
        if not X_accum:
            raise RuntimeError("No valid files were loaded for batch {0} after filtering.".format(batch))
        X_all = np.vstack(X_accum)
        y_all = np.concatenate(y_accum)
        all_rows.append(compute_pearson(X_all, y_all))

    corr_matrix = np.vstack(all_rows)
    feature_labels = ["F{0}".format(i) for i in range(1, corr_matrix.shape[1] + 1)]
    df = pd.DataFrame(corr_matrix, index=BATCHES, columns=feature_labels)
    csv_path  = outdir / "pearson_all_batches_matrix.csv"
    used_path = outdir / "pearson_all_batches_files_used.txt"
    png_path  = outdir / "pearson_all_batches_heatmap.png"

    df.to_csv(csv_path, float_format="%.6f")
    print(" Saved CSV -> {0}".format(csv_path))

    with open(used_path, "w", encoding="utf-8") as f:
        for b in BATCHES:
            f.write("[{0}]\n".format(b))
            for p in files_used[b]:
                f.write("  {0}\n".format(p))
    print(" Saved files list -> {0}".format(used_path))

    fig = plt.figure(figsize=(corr_matrix.shape[1] * 0.5 + 2.5, len(BATCHES) * 0.7 + 2.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(corr_matrix, aspect='auto', vmin=-1.0, vmax=1.0, cmap='coolwarm')
    ax.set_xticks(range(len(feature_labels)))
    ax.set_xticklabels([lbl.replace("F", "") for lbl in feature_labels], rotation=45, ha='right')
    ax.set_yticks(range(len(BATCHES)))
    ax.set_yticklabels(BATCHES)
    ax.set_title("Pearson r (Features vs SOH) — {0}".format(args.dataset))
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            ax.text(j, i, "{0:.2f}".format(corr_matrix[i, j]), ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close(fig)

    print(" Saved heatmap -> {0}".format(png_path))
    print("\nCorrelation matrix (rounded to 2 decimals):")
    print(df.round(2))

if __name__ == "__main__":
    main()
