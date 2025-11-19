#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This script visualizes the true vs predicted SOH of Hybrid PINN on Dataset_A using scatter plot
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------
# RMSE computation
# ---------------------------------------------------------------------
def rmse(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return np.inf
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))

# ---------------------------------------------------------------------
# Listing experiment results
# ---------------------------------------------------------------------
def list_experiments(batch_dir: Path):
    if not batch_dir.is_dir():
        return []
    exps = [d for d in batch_dir.iterdir() if d.is_dir() and re.match(r"^Experiment\d+$", d.name)]
    exps.sort(key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    return exps

# ---------------------------------------------------------------------
# Loading true vs predicted SOH pairs
# ---------------------------------------------------------------------
def load_pair(exp_dir: Path):
    pairs = [
        (exp_dir / "true_soh.npy", exp_dir / "predicted_soh.npy"),
        (exp_dir / "true_label.npy", exp_dir / "pred_label.npy"),
    ]
    for t, p in pairs:
        if t.exists() and p.exists():
            yt = np.ravel(np.load(str(t)).astype(float))
            yp = np.ravel(np.load(str(p)).astype(float))
            n = min(len(yt), len(yp))
            if n > 0:
                return yt[:n], yp[:n]
    return None

# ---------------------------------------------------------------------
# Finds the best experiment (lowest RMSE)
# ---------------------------------------------------------------------
def best_experiment_for_batch(root: Path, batch_index: int):
    batch_dir = root / f"{batch_index}-{batch_index}"
    exps = list_experiments(batch_dir)
    best = (None, None, None, np.inf)
    for exp in exps:
        pair = load_pair(exp)
        if pair is None:
            continue
        yt, yp = pair
        score = rmse(yt, yp)
        if score < best[3]:
            best = (exp, yt, yp, score)
    if best[0] is None:
        return (None, None)
    return (best[1], best[2])

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default=None)
    ap.add_argument("--dataset", type=str, choices=["Dataset_A", "Dataset_B"], default="Dataset_A")
    ap.add_argument("--model", type=str, choices=["PINN", "MLP", "CNN"], default="PINN")
    ap.add_argument("--save", action="store_true", help="Save figure to results/figures/")
    args = ap.parse_args()

    # -----------------------------------------------------------------
    # Reproducible path handling
    # -----------------------------------------------------------------
    REPO_ROOT = Path(args.repo_root or Path(__file__).resolve().parents[1]).resolve()
    ROOT = REPO_ROOT / "results for analysis" / f"{args.dataset}-{args.model} results"
    FIG_DIR = REPO_ROOT / "results for analysis" / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Batch names and titles
    # -----------------------------------------------------------------
    batch_names = ["2C", "3C", "R2.5", "R3", "RW", "satellite"]
    batch_titles = [f"{args.dataset} batch {i+1}" for i in range(len(batch_names))]

    print(f"[info] Reading from: {ROOT}")
    if not ROOT.exists():
        print("[error] Results root does not exist. Check --repo_root/--dataset/--model.")
        return

    # -----------------------------------------------------------------
    # Plot settings
    # -----------------------------------------------------------------
    LIMS = (0.80, 1.00)
    TICKS = np.round(np.linspace(*LIMS, 6), 2)
    CMAP = plt.cm.colors.LinearSegmentedColormap.from_list(
        "custom_cmap", ["#74AED4", "#7BDFF2", "#FBDD85", "#F46F43", "#CF3D3E"], N=256
    )
    NORM = plt.Normalize(vmin=0, vmax=0.10)

    # -----------------------------------------------------------------
    # Load best experiments
    # -----------------------------------------------------------------
    batches = []
    missing = []
    for i, title in enumerate(batch_titles):
        yt, yp = best_experiment_for_batch(ROOT, i)
        if yt is None:
            batches.append((f"{title} (missing)", None, None))
            missing.append(i + 1)
        else:
            batches.append((title, yt, yp))
    if missing:
        print(f"[warn] No experiments found for batch indices (1-based): {missing}")

    # -----------------------------------------------------------------
    # Scatter plot
    # -----------------------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), dpi=160, constrained_layout=True)
    for i, (title, y_true, y_pred) in enumerate(batches):
        ax = axs[i // 3, i % 3]
        if y_true is None:
            ax.set_title(title, fontsize=12)
            ax.axis("off")
            continue
        m = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[m], y_pred[m]
        err = np.abs(y_pred - y_true)
        ax.scatter(y_true, y_pred, c=err, cmap=CMAP, norm=NORM, s=22, alpha=0.9, linewidths=0)
        ax.plot(LIMS, LIMS, "--", c="#ff4d4e", lw=1)
        ax.set_aspect("equal")
        ax.set_xlim(LIMS)
        ax.set_ylim(LIMS)
        ax.set_xticks(TICKS)
        ax.set_yticks(TICKS)
        ax.set_xlabel("True SOH", fontsize=12)
        ax.set_ylabel("Predicted SOH", fontsize=12)
        ax.set_title(title, fontsize=12)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), location="right", shrink=0.9, pad=0.02)
    cbar.set_label("Absolute error", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # -----------------------------------------------------------------
    # Save reproducibly
    # -----------------------------------------------------------------
    if args.save:
        out_path = FIG_DIR / f"{args.model}_true_vs_predicted_{args.dataset}.png"
        fig.savefig(out_path, dpi=400, bbox_inches="tight")
        print(f"[info] Saved figure: {out_path}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
