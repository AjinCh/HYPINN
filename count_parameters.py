#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script prints the number of trainable parameters for MLP,CNN,PINN
"""

import sys
from pathlib import Path
import argparse

# ---------------------------------------------------------------------
# auto-locate repository root
# ---------------------------------------------------------------------
def add_repo_root_to_sys_path() -> Path:
    """Ensure repo root (containing Model/, utils/, Data_loading/) is on sys.path."""
    here = Path(__file__).resolve().parent
    for base in [here] + list(here.parents):
        if all((base / d).is_dir() for d in ["Model", "utils", "Data_loading"]):
            if str(base) not in sys.path:
                sys.path.insert(0, str(base))
            return base
    raise RuntimeError("❌ Could not locate repo root (requires Model/, utils/, Data_loading/).")

REPO_ROOT = add_repo_root_to_sys_path()

# ---------------------------------------------------------------------
# imports (after repo root is added)
# ---------------------------------------------------------------------
from Model.Model import PINN
from Model.Comparing_Models import MLP, CNN

# ---------------------------------------------------------------------
# utility
# ---------------------------------------------------------------------
def num_trainable_params(model) -> int:
    """Return total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Model size inspector for HYPINN architectures")
    parser.add_argument('--dataset', type=str, default='Dataset_A', choices=['Dataset_A', 'Dataset_B'])
    parser.add_argument('--batch', type=str, default='2C')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--normalization_method', type=str, default='z-score')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--warmup_lr', type=float, default=5e-4)
    parser.add_argument('--final_lr', type=float, default=1e-4)
    parser.add_argument('--lr_F', type=float, default=1e-3)
    parser.add_argument('--iter_per_epoch', type=int, default=1)
    parser.add_argument('--F_layers_num', type=int, default=3)
    parser.add_argument('--F_hidden_dim', type=int, default=60)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--save_folder', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    return parser.parse_args()

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    args = get_args()

    pinn = PINN(args)
    print(f"🔹 PINN (solution_u) parameter count: {num_trainable_params(pinn.solution_u):,}")

    mlp = MLP()
    print(f"🔹 MLP parameter count: {num_trainable_params(mlp):,}")

    cnn = CNN()
    print(f"🔹 CNN parameter count: {num_trainable_params(cnn):,}")

    print("\n✅ Parameter count computation completed successfully.")

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == '__main__':
    main()
