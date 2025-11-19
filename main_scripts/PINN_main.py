#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Training script of Hybrid PINN on Dataset_A and saving the results under results for analysis
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from Data_loading.dataloader import Dataset_A as DatasetAData, Dataset_B as DatasetBData
from Model.Model import PINN, MLP, HybridPINN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ---------------------------------------------------------------------
# automatically detect repository root for portability
# ---------------------------------------------------------------------
REPO_DIR = Path(__file__).resolve().parents[1]
# ---------------------------------------------------------------------
# creating result formats
# ---------------------------------------------------------------------
def exp_dir(save_base: Path, dataset: str, model: str, batch_idx: int, exp_id: int) -> Path:
    path = save_base / f"{dataset}-{model} results" / f"{batch_idx}-{batch_idx}" / f"Experiment{exp_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path

def write_log(path: Path, text: str):
    with open(path / "logging.txt", "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return float("inf")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))

# loading data
def load_data(args, small_sample=None):
    if args.dataset == 'Dataset_A':
        root = args.Dataset_A_root
        DataClass = DatasetAData
    else:
        root = args.Dataset_B_root
        DataClass = DatasetBData

    data = DataClass(root=root, args=args)
    train_list, test_list = [], []

    for fname in os.listdir(root):
        if args.batch in fname:
            # the file name with number 4 or 8 chosen as test
            if ('4' in fname) or ('8' in fname):
                test_list.append(os.path.join(root, fname))
            else:
                train_list.append(os.path.join(root, fname))

    if small_sample:
        train_list = train_list[:small_sample]

    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    return {
        'train': train_loader['train'],
        'valid': train_loader['valid'],
        'test': test_loader['test']
    }
# ---------------------------------------------------------------------
# arguments for parameter defining
# ---------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    # About datasets and we use Dataset_A for training and testing as default.
    p.add_argument('--dataset', choices=['Dataset_A', 'Dataset_B'], default='Dataset_A')
    p.add_argument('--batches', nargs='*', default=['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite'])
    p.add_argument('--batch', type=str, default='2C')

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--normalization_method', type=str, default='min-max')

    # Parameters
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--early_stop', type=int, default=20)
    p.add_argument('--warmup_epochs', type=int, default=30)
    p.add_argument('--warmup_lr', type=float, default=0.002)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--final_lr', type=float, default=0.0002)
    p.add_argument('--lr_F', type=float, default=0.001)
    p.add_argument('--F_layers_num', type=int, default=3)
    p.add_argument('--F_hidden_dim', type=int, default=60)
    p.add_argument('--alpha', type=float, default=0.7)
    p.add_argument('--beta', type=float, default=0.2)

    # Result saving
    p.add_argument('--save_base', type=str, default=str(REPO_DIR / 'results for analysis'))
    p.add_argument('--model_tag', type=str, default='PINN')

    # roots for data loading (portable)
    p.add_argument('--Dataset_A_root', type=str, default=str(REPO_DIR / 'data' / 'Dataset_A'))
    p.add_argument('--Dataset_B_root', type=str, default=str(REPO_DIR / 'data' / 'Dataset_B'))
    p.add_argument('--log_dir', type=str, default='log.txt')
    p.add_argument('--save_folder', type=str, default='.')
    return p.parse_args()
# ---------------------------------------------------------------------
# Training of base PINN
# ---------------------------------------------------------------------
def train_pinn(args, loaders, input_dim: int):
    setattr(args, 'input_dim', input_dim)  # used by your PINN implementation
    model = PINN(args).to(DEVICE)
    model.Train(loaders['train'], loaders['valid'], loaders['test'])
    model.eval()
    return model

def train_all_batches():
    args = get_args()
    dataset_tag = args.dataset
    model_tag = args.model_tag
    save_base = Path(args.save_base)

    # batch name matching
    batch_list = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    batch_to_index = {b: i for i, b in enumerate(batch_list)}

    # iteration over all batches and batch name mapping to indices
    for batch in args.batches:
        if batch not in batch_to_index:
            print(f"  Unknown batch '{batch}' — skipping.")
            continue
        bidx = batch_to_index[batch]
        print(f"\n Training Hybrid Model for {dataset_tag} | batch: {batch} (idx {bidx})")

        a = get_args()
        setattr(a, 'dataset', dataset_tag)
        setattr(a, 'batch', batch)
        setattr(a, 'save_base', str(save_base))
        setattr(a, 'model_tag', model_tag)

        loaders = load_data(a)

        x1, _, _, _ = next(iter(loaders['train']))
        input_dim = x1.shape[1]

        base_pinn = train_pinn(a, loaders, input_dim)

        for e in range(1, 11):
            exp_path = exp_dir(save_base, dataset_tag, model_tag, bidx, e)
            setattr(a, 'save_folder', str(exp_path))  # allow PINN to save here if needed
            print(f"→ Saving to: {exp_path.resolve()}")

            correction_net = MLP().to(DEVICE)
            hybrid_model = HybridPINN(base_pinn, correction_net).to(DEVICE)
            optimizer = optim.Adam(hybrid_model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            for epoch in range(1, 21):
                hybrid_model.train()
                total_loss = 0.0
                for x1, _, y1, _ in loaders['train']:
                    x1, y1 = x1.to(DEVICE), y1.to(DEVICE)
                    preds = hybrid_model(x1)
                    loss = loss_fn(preds, y1)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"[Batch {batch} - Exp {e} - Epoch {epoch}] "
                      f"Loss: {total_loss / len(loaders['train']):.6f}")
# ---------------------------------------------------------------------
# evaluation and saving
# ---------------------------------------------------------------------
            trues, preds = [], []
            hybrid_model.eval()
            with torch.no_grad():
                for x1, _, y1, _ in loaders['test']:
                    x1 = x1.to(DEVICE)
                    pred = hybrid_model(x1)
                    trues.append(y1.numpy())
                    preds.append(pred.cpu().numpy())
            trues = np.concatenate(trues).ravel()
            preds = np.concatenate(preds).ravel()

            np.save(exp_path / 'true_soh.npy', trues)
            np.save(exp_path / 'predicted_soh.npy', preds)
            base_pinn.save_model(exp_path / 'model.pth')

            score = rmse(trues, preds)
            write_log(exp_path, f"Dataset={dataset_tag} Batch={batch} Exp={e} RMSE={score:.8f}")
            print(f" Saved | RMSE={score:.6f}")


if __name__ == '__main__':
    train_all_batches()
