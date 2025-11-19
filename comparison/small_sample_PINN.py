#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This code is used to run small sample regime on Hybrid PINN
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Optional, Tuple
from Data_loading.dataloader import Dataset_A as DatasetAData, Dataset_B as DatasetBData
from Model.Model import PINN, MLP, HybridPINN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ---------------------------------------------------------------------
# for resolving paths — automatically detects repository root for portability
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
# ---------------------------------------------------------------------
# to maintain consistent access to data
# ---------------------------------------------------------------------
def _resolve(root_str: str) -> str:
    p = Path(root_str).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    return str((REPO_ROOT / p).resolve())
# ---------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------
def mae(y_true, y_pred):  return float(np.mean(np.abs(y_true - y_pred)))
def mse(y_true, y_pred):  return float(np.mean((y_true - y_pred) ** 2))
def rmse(y_true, y_pred): return float(np.sqrt(mse(y_true, y_pred)))
def mape(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))
# ---------------------------------------------------------------------
# creating and returning paths for experiment
# ---------------------------------------------------------------------
def exp_dir(save_base: Path, dataset: str, model: str, n_train: int, batch_idx: int, exp_id: int) -> Path:
    path = (
        save_base
        / f"{dataset}-{model} results (small sample {n_train})"
        / f"{batch_idx}-{batch_idx}"
        / f"Experiment{exp_id}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path
# ---------------------------------------------------------------------
# appending evaluation to logging.txt file
# ---------------------------------------------------------------------
def write_log(path: Path, text: str):
    with open(path / "logging.txt", "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
# ---------------------------------------------------------------------
# loading data and datas with “4” or “8” filenames chosen for testing in each batches
# ---------------------------------------------------------------------
def load_data(args, small_sample: Optional[int] = None):
    if args.dataset == 'Dataset_A':
        root = _resolve(args.Dataset_A_root)
        DataClass = DatasetAData
    else:
        root = _resolve(args.Dataset_B_root)
        DataClass = DatasetBData
    if not Path(root).exists():
        raise FileNotFoundError("Data root not found: {}".format(root))
    data = DataClass(root=root, args=args)
    train_list, test_list = [], []
    for fname in os.listdir(root):
        if args.batch in fname:
            (test_list if ('4' in fname or '8' in fname) else train_list).append(os.path.join(root, fname))
    if small_sample:
        train_list = train_list[:small_sample]
    train_loader = data.read_all(specific_path_list=train_list)
    test_loader  = data.read_all(specific_path_list=test_list)
    return {'train': train_loader['train'], 'valid': train_loader['valid'], 'test': test_loader['test']}
# ---------------------------------------------------------------------
# parameters defining
# ---------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="HybridPINN small-sample experiments")
    p.add_argument('--dataset', choices=['Dataset_A', 'Dataset_B'], default='Dataset_A')
    p.add_argument('--batches', nargs='*', default=['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite'])
    p.add_argument('--batch', type=str, default='2C')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--normalization_method', type=str, default='min-max')
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
    p.add_argument('--save_base', type=str, default=str(REPO_ROOT / 'results for analysis'))
    p.add_argument('--model_tag', type=str, default='PINN')
    p.add_argument('--Dataset_A_root', type=str, default=str(REPO_ROOT / 'data' / 'Dataset_A'))
    p.add_argument('--Dataset_B_root', type=str, default=str(REPO_ROOT / 'data' / 'Dataset_B'))
    p.add_argument('--log_dir', type=str, default='log.txt')
    p.add_argument('--save_folder', type=str, default='.')
    p.add_argument('--n_values', nargs='*', type=int, default=[1, 2, 3, 4])
    p.add_argument('--num_exps', type=int, default=10)
    p.add_argument('--corr_epochs', type=int, default=20)
    return p.parse_args()
# ---------------------------------------------------------------------
# trains on base model using full training data
# ---------------------------------------------------------------------
def train_base_pinn(args, loaders, input_dim: int):
    setattr(args, 'input_dim', input_dim)
    model = PINN(args).to(DEVICE)
    model.Train(loaders['train'], loaders['valid'], loaders['test'])
    model.eval()
    return model
# ---------------------------------------------------------------------
# testing and evaluation
# ---------------------------------------------------------------------
def evaluate_to_arrays(model: nn.Module, test_loader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for x1, _, y1, _ in test_loader:
            x1 = x1.to(DEVICE)
            pred = model(x1)
            trues.append(y1.numpy())
            preds.append(pred.cpu().numpy())
    return np.concatenate(trues).ravel(), np.concatenate(preds).ravel()
# ---------------------------------------------------------------------
# runs the model for small sample regime
# ---------------------------------------------------------------------
def run_small_sample_for_batch(args_base, batch_name: str, batch_idx: int, n_train: int):
    a = get_args()
    for k, v in vars(args_base).items():
        setattr(a, k, v)
    setattr(a, 'batch', batch_name)
    a.Dataset_A_root = _resolve(a.Dataset_A_root)
    a.Dataset_B_root = _resolve(a.Dataset_B_root)
    a.save_base = _resolve(a.save_base)
    loaders = load_data(a, small_sample=n_train)
    x1, _, _, _ = next(iter(loaders['train']))
    input_dim = x1.shape[1]
    print("  • Training base PINN (batch={}, n={})".format(batch_name, n_train))
    base_pinn = train_base_pinn(a, loaders, input_dim)
    for e in range(1, a.num_exps + 1):
        exp_path = exp_dir(Path(a.save_base), a.dataset, a.model_tag, n_train, batch_idx, e)
        setattr(a, 'save_folder', str(exp_path))
        print("    → Saving to:", exp_path.resolve())
        correction_net = MLP().to(DEVICE)
        hybrid_model = HybridPINN(base_pinn, correction_net).to(DEVICE)
        optimizer = optim.Adam(hybrid_model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        for epoch in range(1, a.corr_epochs + 1):
            hybrid_model.train()
            total_loss = 0.0
            for x1, _, y1, _ in loaders['train']:
                x1, y1 = x1.to(DEVICE), y1.to(DEVICE)
                preds = hybrid_model(x1)
                loss = loss_fn(preds, y1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            print("      [n={} | Exp {:02d} | Epoch {:02d}] Loss: {:.6f}".format(n_train, e, epoch, total_loss / len(loaders['train'])))
        y_true, y_pred = evaluate_to_arrays(hybrid_model, loaders['test'])
        np.save(exp_path / 'true_soh.npy', y_true)
        np.save(exp_path / 'predicted_soh.npy', y_pred)
        try:
            base_pinn.save_model(str(exp_path / 'model.pth'))
        except Exception:
            pass
        write_log(exp_path, "Dataset={} Batch={} n={} Exp={}".format(a.dataset, batch_name, n_train, e))
        write_log(exp_path, "MAE={:.8f}".format(mae(y_true, y_pred)))
        write_log(exp_path, "MAPE={:.8f}".format(mape(y_true, y_pred)))
        write_log(exp_path, "MSE={:.8f}".format(mse(y_true, y_pred)))
        write_log(exp_path, "RMSE={:.8f}".format(rmse(y_true, y_pred)))
        print("      ✓ RMSE={:.6f}".format(rmse(y_true, y_pred)))
# ---------------------------------------------------------------------
# run across all batches
# ---------------------------------------------------------------------
def run_all_small_samples():
    args = get_args()
    batch_order = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    batch_to_index = {b: i for i, b in enumerate(batch_order)}
    for batch in args.batches:
        if batch not in batch_to_index:
            print("  Unknown batch '{}' — skipping.".format(batch))
            continue
        bidx = batch_to_index[batch]
        print("\n Small-sample runs for {} | batch: {} (idx {})".format(args.dataset, batch, bidx))
        for n in args.n_values:
            print("  ▶ Train batteries n = {}".format(n))
            run_small_sample_for_batch(args, batch, bidx, n)

if __name__ == "__main__":
    run_all_small_samples()
