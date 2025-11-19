# This script is for transfer learning following transfer utilities
import argparse
from pathlib import Path
from Model.Model import PINN
from Data_loading.dataloader import Dataset_B
from transfer_utils import fine_tune_and_evaluate_hybrid, plot_predictions

# ---------------------------------------------------------------------
# Default pretrained model path (reproducible)
# ---------------------------------------------------------------------
def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

DEFAULT_REPO_ROOT = _default_repo_root()

DEFAULT_PRETRAIN = (
    DEFAULT_REPO_ROOT
    / "results for analysis"
    / "Dataset_A-PINN results"
    / "0-0"
    / "Experiment1"
    / "model.pth"
)

# ---------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser("Run Hybrid transfer: Dataset_A -> Dataset_B")
    p.add_argument('--normalization_method', type=str, default='min-max')
    p.add_argument('--F_layers_num', type=int, default=3)
    p.add_argument('--F_hidden_dim', type=int, default=60)
    p.add_argument('--alpha', type=float, default=0.7)
    p.add_argument('--beta', type=float, default=0.2)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--early_stop', type=int, default=10)
    p.add_argument('--use_physics', action='store_true')
    p.add_argument('--xjtu_root', type=str, default='data/Dataset_A')
    p.add_argument('--hust_root', type=str, default='data/Dataset_B')
    p.add_argument('--pretrain_model', type=str, default=str(DEFAULT_PRETRAIN))
    p.add_argument('--repo_root', type=str, default=str(DEFAULT_REPO_ROOT))
    p.add_argument('--result_dir', type=str, default=None)
    p.add_argument('--exp_name', type=str, default=None)
    p.add_argument('--small_sample', type=int, default=None)
    p.add_argument('--group_by', type=str, choices=['batch', 'file'], default='batch')
    p.add_argument('--take_first_n', type=int, default=None)
    return p.parse_args()

# ---------------------------------------------------------------------
# Resolve relative or absolute paths
# ---------------------------------------------------------------------
def _resolve_path(base: Path, maybe_path: str) -> Path:
    p = Path(maybe_path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()

# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------
def main():
    args = get_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    xjtu_root = _resolve_path(repo_root, args.xjtu_root)
    hust_root = _resolve_path(repo_root, args.hust_root)
    model_path = Path(args.pretrain_model).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Pretrained model not found: {model_path}")

    print(" Using pretrained model:", model_path)

    # -----------------------------------------------------------------
    # Initialize pretrained PINN
    # -----------------------------------------------------------------
    pinn_args = argparse.Namespace(
        normalization_method=args.normalization_method,
        F_layers_num=args.F_layers_num,
        F_hidden_dim=args.F_hidden_dim,
        alpha=args.alpha,
        beta=args.beta,
        batch_size=args.batch_size,
        epochs=200, early_stop=10, warmup_epochs=30, warmup_lr=0.002,
        lr=0.01, final_lr=0.0002, lr_F=0.01,
    )

    base_pinn = PINN(pinn_args)
    base_pinn.load_model(str(model_path))

    # -----------------------------------------------------------------
    # Dataset_B loading and grouping
    # -----------------------------------------------------------------
    all_csv_paths = sorted([p for p in hust_root.rglob("*.csv")])
    if not all_csv_paths:
        raise RuntimeError(f"No CSVs found in {hust_root}")

    if not args.exp_name:
        try:
            exp = model_path.parent.name
            batch = model_path.parent.parent.name
            args.exp_name = f"{batch}_{exp}"
        except Exception:
            args.exp_name = "transfer_run"

    # -----------------------------------------------------------------
    # Group CSVs by file or batch
    # -----------------------------------------------------------------
    if args.group_by == "file":
        items = [(i, [str(p)], p.stem) for i, p in enumerate(all_csv_paths, start=1)]
    else:
        from collections import defaultdict
        groups = defaultdict(list)
        for p in all_csv_paths:
            stem = p.stem
            batch_id = stem.split("-")[0]
            groups[batch_id].append(str(p))
        items = [(int(b), sorted(paths), f"batch_{b}") for b, paths in groups.items()]
        items = sorted(items, key=lambda t: t[0])

    if args.take_first_n is not None:
        items = items[:args.take_first_n]

    # -----------------------------------------------------------------
    # Run transfer for each batch or file
    # -----------------------------------------------------------------
    for idx, file_list, label in items:
        print(f"\n Transfer for {label}: ({len(file_list)} file(s))")

        hust_args = argparse.Namespace(
            batch_size=args.batch_size,
            normalization_method=args.normalization_method,
            log_dir=None,
            save_folder=None,
            Dataset_A_root=str(xjtu_root),
        )

        # Load Dataset_B subset
        ds_b = Dataset_B(root=str(hust_root), args=hust_args)
        dl = ds_b.read_all(specific_path_list=file_list)

        step_exp = f"{args.exp_name}_{label}"

        # -----------------------------------------------------------------
        # Fine-tune and evaluate Hybrid PINN
        # -----------------------------------------------------------------
        trues, preds, out_dir = fine_tune_and_evaluate_hybrid(
            base_pinn,
            dl,
            save_dir=args.result_dir,
            epochs=args.epochs,
            lr=args.lr,
            alpha=args.alpha,
            beta=args.beta,
            early_stop=args.early_stop,
            use_physics=args.use_physics,
            repo_root=str(repo_root),
            small_sample=args.small_sample,
            exp_name=step_exp,
        )

        # -----------------------------------------------------------------
        # Plot predictions
        # -----------------------------------------------------------------
        if trues is not None and preds is not None:
            plot_predictions(trues, preds, battery_id=idx, save_dir=out_dir, title_prefix="HybridPINN Transfer")
            print(" Results saved in:", out_dir)

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
