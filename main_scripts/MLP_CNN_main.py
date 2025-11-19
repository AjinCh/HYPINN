#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Training of baseline models MLP and CNN on Dataset_A saving the results under results for analysis
from __future__ import annotations

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Tuple

from Data_loading.dataloader import Dataset_A as DatasetAData, Dataset_B as DatasetBData
from Model.Comparing_Models import MLP, CNN
# detect GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# automatically detect repository root for portability
REPO_DIR = Path(__file__).resolve().parents[1]
LOG_FILENAME = "logging.txt"
# ---------------------------------------------------------------------
# defining of batches and assigning numeric indices
# ---------------------------------------------------------------------
BATCHES = ["2C", "3C", "R2.5", "R3", "RW", "satellite"]
BATCH_TO_INDEX = {b: i for i, b in enumerate(BATCHES)}

# creates unique folder for experiments
def exp_dir(save_base: Path, dataset: str, model: str, batch_idx: int, exp_id: int) -> Path:
    path = save_base / f"{dataset}-{model} results" / f"{batch_idx}-{batch_idx}" / f"Experiment{exp_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path

def write_log(path: Path, text: str) -> None:
    with open(path / LOG_FILENAME, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
# ---------------------------------------------------------------------
# computes rmse between true vs predicted
# ---------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return float("inf")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))
# ---------------------------------------------------------------------
# data loader and splitting
# ---------------------------------------------------------------------
def load_data(args, small_sample: int | None = None) -> Dict[str, torch.utils.data.DataLoader]:
    if args.dataset == "Dataset_A":
        root = args.Dataset_A_root
        DataClass = DatasetAData
    else:
        root = args.Dataset_B_root
        DataClass = DatasetBData

    data = DataClass(root=root, args=args)

    train_list, test_list = [], []
    for fname in os.listdir(root):
        if args.batch in fname:
            if ("4" in fname) or ("8" in fname):
                test_list.append(os.path.join(root, fname))
            else:
                train_list.append(os.path.join(root, fname))

    if small_sample:
        train_list = train_list[:small_sample]

    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    return {
        "train": train_loader["train"],
        "valid": train_loader["valid"],
        "test": test_loader["test"],
    }
# ---------------------------------------------------------------------
# model building
# ---------------------------------------------------------------------
def build_model(model_name: str) -> nn.Module:
    if model_name == "MLP":
        return MLP()
    if model_name == "CNN":
        return CNN()
    raise ValueError("Unknown model name. Choose from: 'MLP', 'CNN'.")

# training, testing and evaluation for one model
def train_one_model(
    model: nn.Module,
    loaders: Dict[str, torch.utils.data.DataLoader],
    epochs: int,
    lr: float,
    weight_decay: float,
) -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device(DEVICE)
    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x1, _, y1, _ in loaders["train"]:
            x1 = x1.to(device)
            y1 = y1.to(device)
            pred = model(x1)
            loss = loss_fn(pred, y1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch:03d}] Loss: {total_loss / max(1, len(loaders['train'])):.6f}")

    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for x1, _, y1, _ in loaders["test"]:
            x1 = x1.to(device)
            pred = model(x1)
            trues.append(y1.cpu().numpy())
            preds.append(pred.cpu().numpy())

    y_true = np.concatenate(trues, axis=0).ravel()
    y_pred = np.concatenate(preds, axis=0).ravel()
    return y_true, y_pred
# ---------------------------------------------------------------------
# arguments for parameter defining
# ---------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["Dataset_A", "Dataset_B"], default="Dataset_A")
    p.add_argument("--batches", nargs="*", default=BATCHES)
    p.add_argument("--batch", type=str, default="2C")  # set per-iteration
    p.add_argument("--models", nargs="+", choices=["MLP", "CNN"], default=["MLP", "CNN"])

    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)

    # loader options
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--normalization_method", type=str, default="min-max")

    # experiments / saving
    p.add_argument("--experiments_per", type=int, default=10)
    p.add_argument("--save_base", type=str, default=str(REPO_DIR / "results for analysis"))

    # dataset roots (portable defaults)
    p.add_argument("--Dataset_A_root", type=str, default=str(REPO_DIR / "data" / "Dataset_A"))
    p.add_argument("--Dataset_B_root", type=str, default=str(REPO_DIR / "data" / "Dataset_B"))

    # passthroughs expected by DataClass
    p.add_argument("--log_dir", type=str, default="logging.txt")
    p.add_argument("--save_folder", type=str, default=".")
    return p.parse_args()
# ---------------------------------------------------------------------
# training loop for complete batch
# ---------------------------------------------------------------------
def train_all_batches():
    args = get_args()
    dataset_tag = args.dataset
    save_base = Path(args.save_base)

    for batch in args.batches:
        if batch not in BATCH_TO_INDEX:
            print(f"  Unknown batch '{batch}' — skipping.")
            continue
        bidx = BATCH_TO_INDEX[batch]
        print(f"\n Training Baseline Models for {dataset_tag} | batch: {batch} (idx {bidx})")

        a = get_args()
        setattr(a, "dataset", dataset_tag)
        setattr(a, "batch", batch)
        setattr(a, "save_base", str(save_base))

        loaders = load_data(a)

        for model_name in args.models:
            for e in range(1, args.experiments_per + 1):
                exp_path = exp_dir(save_base, dataset_tag, model_name, bidx, e)
                setattr(a, "save_folder", str(exp_path))
                print(f"→ Saving to: {exp_path.resolve()}")

                model = build_model(model_name)
                y_true, y_pred = train_one_model(
                    model, loaders, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay
                )

                np.save(exp_path / "true_soh.npy", y_true)
                np.save(exp_path / "predicted_soh.npy", y_pred)
                try:
                    torch.save(model.state_dict(), exp_path / "model.pth")
                except Exception:
                    pass

                score = rmse(y_true, y_pred)
                write_log(exp_path, f"Dataset={dataset_tag} Batch={batch} Exp={e} RMSE={score:.8f}")
                print(f" Saved | {model_name} | RMSE={score:.6f}")


if __name__ == "__main__":
    train_all_batches()
