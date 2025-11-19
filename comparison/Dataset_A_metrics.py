#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This code is used to produce the average metric values produced by Hybrid PINN and saves as xlsx file
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

BATCHES = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
# ---------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------
def rmse_np(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return np.nan
    y_true, y_pred = y_true[:n], y_pred[:n]
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
# ---------------------------------------------------------------------
# computes MAPE ratio with avoidal of zero denominator 
# ---------------------------------------------------------------------
def mape_ratio(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return np.nan
    y_true, y_pred = y_true[:n], y_pred[:n]
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))
# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def load_pair(exp_dir: Path):
    candidates = [
        ("true_soh.npy", "predicted_soh.npy"),
        ("true_label.npy", "pred_label.npy"),
    ]
    for t, p in candidates:
        t_path = exp_dir / t
        p_path = exp_dir / p
        if t_path.is_file() and p_path.is_file():
            yt = np.load(t_path)
            yp = np.load(p_path)
            n = min(len(yt), len(yp))
            return yt[:n], yp[:n]
    return None, None

def collect_batch_df(batch_index: int, batch_root: Path) -> pd.DataFrame:
    if not batch_root.is_dir():
        raise FileNotFoundError(f"Batch folder not found: {batch_root}")

    rows = []
    for exp_dir in sorted(batch_root.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.lower().startswith("experiment"):
            continue
        yt, yp = load_pair(exp_dir)
        if yt is None or yp is None or len(yt) == 0:
            continue

        mae = float(mean_absolute_error(yt, yp))
        rmse = rmse_np(yt, yp)
        mape = mape_ratio(yt, yp)
        try:
            r2 = float(r2_score(yt, yp))
        except Exception:
            r2 = np.nan

        exp_name = exp_dir.name
        try:
            exp_id = int(''.join(ch for ch in exp_name if ch.isdigit()))
        except ValueError:
            exp_id = exp_name

        rows.append({"experiment": exp_id, "MAE": mae, "MAPE": mape, "RMSE": rmse, "R2": r2})

    if not rows:
        raise RuntimeError(f"No experiments found with valid pairs under {batch_root}")

    df = pd.DataFrame(rows).sort_values(by="experiment").reset_index(drop=True)
    return df


def write_sheet(out_path: Path, sheet_name: str, df: pd.DataFrame, first_sheet: bool):
    if first_sheet and out_path.exists():
        out_path.unlink()
    mode = "w" if first_sheet else "a"
    with pd.ExcelWriter(out_path, engine="openpyxl", mode=mode) as xl:
        df.to_excel(xl, index=False, sheet_name=sheet_name)


def create_excel(dataset: str, repo_root: Path, out_path: Path):
    in_root = repo_root / "results for analysis" / f"{dataset}-PINN results"
    if not in_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {in_root}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    any_written = False
    for idx, _batch_name in enumerate(BATCHES):
        batch_dir = in_root / f"{idx}-{idx}"
        try:
            df = collect_batch_df(idx, batch_dir)
        except Exception as e:
            print(f"⚠ Skipping batch {idx} ({_batch_name}): {e}")
            continue

        sheet = f"battery_mean_{idx}"
        write_sheet(out_path, sheet, df, first_sheet=(not any_written))
        any_written = True
        print(f"✓ Wrote {sheet}: {len(df)} experiment rows from {batch_dir}")

    if not any_written:
        raise SystemExit("No batch sheets were written (no valid experiments found).")
    print(f"\n Excel saved → {out_path.resolve()}")
def main():
    # Always resolve the repository root dynamically (no hardcoded path)
    here = Path(__file__).resolve()
    repo_root = here.parents[1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["Dataset_A", "Dataset_B"], default="Dataset_A")
    ap.add_argument("--repo_root", default=str(repo_root))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    default_out = repo_root / "results for analysis" / "processed results" / f"PINN-{args.dataset}-results.xlsx"
    out_path = Path(args.out or default_out)

    create_excel(dataset=args.dataset, repo_root=repo_root, out_path=out_path)


if __name__ == "__main__":
    main()
