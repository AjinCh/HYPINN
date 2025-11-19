#!/usr/bin/env python3
# This code is used to draw violin plot for comparing three models with RMSE as metric
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# RMSE computation
# ---------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# ---------------------------------------------------------------------
# Loading predictions and logs
# ---------------------------------------------------------------------
def _load_pair(exp_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    pairs = [
        (exp_dir / "true_soh.npy", exp_dir / "predicted_soh.npy"),
        (exp_dir / "true_label.npy", exp_dir / "pred_label.npy"),
    ]
    for t, p in pairs:
        if t.exists() and p.exists():
            try:
                yt, yp = np.ravel(np.load(t)), np.ravel(np.load(p))
                n = min(len(yt), len(yp))
                if n > 0:
                    return yt[:n], yp[:n]
            except Exception:
                pass
    return None


LOG_RE = re.compile(r"RMSE[^0-9eE\.\+\-]*([eE\+\-\.0-9]+)")


def _parse_rmse_from_log(exp_dir: Path) -> Optional[float]:
    log = exp_dir / "logging.txt"
    if not log.exists():
        return None
    try:
        txt = log.read_text(errors="ignore")
        vals = LOG_RE.findall(txt)
        if vals:
            return float(vals[-1])
    except Exception:
        pass
    return None


def small_sample_dir(base_dir: Path, dataset: str, model: str, n: int) -> Path:
    return base_dir / f"{dataset}-{model} results (small sample {n})"


def collect_rmse_for(base_dir: Path, dataset: str, model: str, batch_idx: int, n_train: int) -> List[float]:
    base = small_sample_dir(base_dir, dataset, model, n_train) / f"{batch_idx}-{batch_idx}"
    vals: List[float] = []
    if not base.exists():
        return vals
    for exp in sorted([p for p in base.iterdir() if p.is_dir() and p.name.lower().startswith("experiment")]):
        pair = _load_pair(exp)
        if pair is not None:
            yt, yp = pair
            vals.append(rmse(yt, yp))
        else:
            r = _parse_rmse_from_log(exp)
            if r is not None:
                vals.append(r)
    return vals

# ---------------------------------------------------------------------
# Violin plot drawing
# ---------------------------------------------------------------------
def draw_grouped_violins(ax, rmse_dict, models, small_samples, title, show_legend=True):
    base_positions = {m: i for i, m in enumerate(models)}
    all_vals = [v for m in rmse_dict for n in rmse_dict[m] for v in rmse_dict[m][n]]
    if not all_vals:
        ax.set_title(title + " (no data)")
        ax.axis("off")
        return

    HUE_COLORS = {1: "#cfe8f3", 2: "#9ed0e6", 3: "#66b2d9", 4: "#2f8fbf"}
    OFFSETS = {n: o for n, o in zip(small_samples, np.linspace(-0.36, 0.36, len(small_samples)))}
    VIOLIN_WIDTH, SCATTER_JITTER, SCATTER_SIZE = 0.19, 0.010, 12
    y_top = np.quantile(all_vals, 0.99) * 1.10 if max(all_vals) > 0 else 1.0

    for m in models:
        m_pos = base_positions[m]
        for n in small_samples:
            vals = rmse_dict.get(m, {}).get(n, [])
            if not vals:
                continue
            pos = m_pos + OFFSETS[n]
            vp = ax.violinplot([vals], positions=[pos], widths=VIOLIN_WIDTH,
                               showmeans=False, showmedians=False, showextrema=False)
            for b in vp['bodies']:
                b.set_facecolor(HUE_COLORS.get(n, "#cccccc"))
                b.set_edgecolor('none')
                b.set_alpha(0.95)
            xs = np.random.normal(loc=pos, scale=SCATTER_JITTER, size=len(vals))
            ax.scatter(xs, vals, s=SCATTER_SIZE, color='black', alpha=0.85, zorder=3)
            mu, sd = float(np.mean(vals)), float(np.std(vals))
            ax.plot([pos, pos], [mu - sd, mu + sd], color='black', lw=1.1, zorder=4)
            ax.plot([pos - 0.09, pos + 0.09], [mu, mu], color='red', lw=1.5, zorder=4)

    ax.set_xlim(-0.6, len(models) - 1 + 0.6)
    ax.set_ylim(0, y_top)
    ax.set_xticks([base_positions[m] for m in models])
    ax.set_xticklabels(models)
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle=':', linewidth=0.7, alpha=0.75)

    if show_legend:
        patches = [
            Patch(facecolor="#cfe8f3", edgecolor="none", label="1 battery"),
            Patch(facecolor="#9ed0e6", edgecolor="none", label="2 batteries"),
            Patch(facecolor="#66b2d9", edgecolor="none", label="3 batteries"),
            Patch(facecolor="#2f8fbf", edgecolor="none", label="4 batteries"),
        ]
        mean_line = Line2D([0], [0], color='red', lw=1.5, label='mean')
        std_line = Line2D([0], [0], color='black', lw=1.5, label='mean ± std')
        ax.legend(handles=patches + [mean_line, std_line],
                  loc="upper right", frameon=True, title="Train size")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Per-batch violin plots of RMSE.")
    ap.add_argument("--repo_root", type=str, default=None)
    ap.add_argument("--dataset", type=str, choices=["Dataset_A", "Dataset_B"], default="Dataset_A")
    ap.add_argument("--models", nargs="*", default=["PINN", "MLP", "CNN"])
    ap.add_argument("--small_samples", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--batches", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    # -----------------------------------------------------------------
    # Reproducible path handling
    # -----------------------------------------------------------------
    REPO_ROOT = Path(args.repo_root or Path(__file__).resolve().parents[1]).resolve()
    BASE_DIR = REPO_ROOT / "results for analysis"
    FIG_DIR = BASE_DIR / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.dpi": 400,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    for batch_no in args.batches:
        bidx = int(batch_no) - 1
        if bidx < 0:
            continue
        title = f"{args.dataset} Batch {bidx + 1}"

        rmse_dict = {
            m: {n: collect_rmse_for(BASE_DIR, args.dataset, m, bidx, n) for n in args.small_samples}
            for m in args.models
        }

        for m in args.models:
            counts = ", ".join(f"n={n}:{len(rmse_dict[m][n])}" for n in args.small_samples)
            print(f"[info] {title} | {m}: {counts}")

        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        draw_grouped_violins(ax, rmse_dict, args.models, args.small_samples, title, show_legend=True)
        plt.tight_layout()

        # Save reproducibly in results/figures
        if args.save:
            out = FIG_DIR / f"violin_rmse_batch{bidx + 1}_{args.dataset}.png"
            fig.savefig(out, dpi=400, bbox_inches="tight")
            print(f"[info] Saved: {out}")

        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
