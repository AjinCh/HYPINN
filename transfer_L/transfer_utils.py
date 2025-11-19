# This coding script provides transfer learning utilities for Hybrid PINN
import os
from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from Model.Model import HybridPINN
from utils.util import eval_metrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PathLike = Union[str, Path]

# ---------------------------------------------------------------------
# Reproducible repo root resolution
# ---------------------------------------------------------------------
def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------
# Output directory creation (dynamic)
# ---------------------------------------------------------------------
def resolve_transfer_results_dir(
    repo_root: Optional[PathLike] = None,
    small_sample: Optional[int] = None,
    exp_name: Optional[str] = None,
    explicit_dir: Optional[PathLike] = None,
) -> Path:
    if explicit_dir is not None:
        out = Path(explicit_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out

    repo = Path(repo_root).expanduser().resolve() if repo_root else _default_repo_root()
    base = repo / "results for analysis"
    if small_sample and small_sample > 0:
        base = base / f"Dataset_B-transfer results (small sample {small_sample})"
    else:
        base = base / "Dataset_B-transfer results"

    if exp_name:
        base = base / exp_name

    base.mkdir(parents=True, exist_ok=True)
    return base

# ---------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------
def _infer_input_dim(dataloader) -> int:
    for x1, _, _, _ in dataloader:
        return x1.shape[1]
    raise RuntimeError("Empty Data_loading; cannot infer input dimension.")

@torch.no_grad()
def _evaluate_hybrid(hybrid_model, loader):
    """Evaluate hybrid PINN on a dataloader."""
    hybrid_model.eval()
    ys, ps = [], []
    for x1, _, y1, _ in loader:
        x1 = x1.to(device)
        y1 = y1.to(device)
        p = hybrid_model(x1)
        ys.append(y1.detach().cpu().numpy())
        ps.append(p.detach().cpu().numpy())
    y = np.concatenate(ys).ravel()
    p = np.concatenate(ps).ravel()
    mse = float(np.mean((p - y) ** 2))
    return mse, y, p

def _evaluate_base(base_pinn, loader):
    """Evaluate base PINN before transfer learning."""
    base_pinn.eval()
    ys, ps = [], []
    for x1, _, y1, _ in loader:
        x1 = x1.to(device).requires_grad_(True)
        y1 = y1.to(device)
        with torch.enable_grad():
            u_base, _ = base_pinn.forward(x1)
        ys.append(y1.detach().cpu().numpy())
        ps.append(u_base.detach().cpu().numpy())
    y = np.concatenate(ys).ravel()
    p = np.concatenate(ps).ravel()
    mse = float(np.mean((p - y) ** 2))
    return mse, y, p

# ---------------------------------------------------------------------
# Main fine-tuning procedure
# ---------------------------------------------------------------------
def fine_tune_and_evaluate_hybrid(
    pinn_model,
    dataloader,
    save_dir: Optional[PathLike] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    alpha: float = 0.7,
    beta: float = 0.2,
    early_stop: int = 10,
    use_physics: bool = True,
    repo_root: Optional[PathLike] = None,
    small_sample: Optional[int] = None,
    exp_name: Optional[str] = None,
):
    out_dir = resolve_transfer_results_dir(
        repo_root=repo_root, small_sample=small_sample, exp_name=exp_name, explicit_dir=save_dir
    )
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    train_loader = dataloader['train']
    valid_loader = dataloader.get('valid', None)
    test_loader = dataloader['test']

    if len(train_loader) == 0:
        print(" No training data found. Skipping…")
        return None, None, out_dir

    base_mse, base_true, base_pred = _evaluate_base(pinn_model, test_loader)
    base_MAE, base_MAPE, base_MSE, base_RMSE = eval_metrix(base_pred, base_true)
    print("Before adaptation (frozen base on target):")
    print("  MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}".format(
        base_MSE, base_MAE, base_MAPE, base_RMSE
    ))

    pinn_model.to(device)
    pinn_model.eval()
    for p in pinn_model.parameters():
        p.requires_grad = False

    in_dim = _infer_input_dim(train_loader)
    correction_net = nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)

    hybrid_model = HybridPINN(pinn_model, correction_net).to(device)
    optimizer = optim.Adam(correction_net.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    relu = nn.ReLU()

    best_val = float('inf')
    patience = 0
    best_true = None
    best_pred = None
    best_state = None

    for epoch in range(1, epochs + 1):
        hybrid_model.train()
        pinn_model.eval()
        total_data = total_pde = total_mono = 0.0

        for x1, x2, y1, y2 in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            y1, y2 = y1.to(device), y2.to(device)

            u1_hat = hybrid_model(x1)
            u2_hat = hybrid_model(x2)
            loss1 = 0.5 * mse_loss(u1_hat, y1) + 0.5 * mse_loss(u2_hat, y2)

            if use_physics:
                x1.requires_grad_(True)
                x2.requires_grad_(True)
                with torch.enable_grad():
                    _, f1 = pinn_model.forward(x1)
                    _, f2 = pinn_model.forward(x2)
                loss2 = 0.5 * mse_loss(f1, torch.zeros_like(f1)) + 0.5 * mse_loss(f2, torch.zeros_like(f2))
                loss3 = relu((u2_hat - u1_hat) * (y1 - y2)).sum()
                loss = loss1 + alpha * loss2 + beta * loss3
            else:
                loss2 = torch.tensor(0.0, device=device)
                loss3 = torch.tensor(0.0, device=device)
                loss = loss1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_data += loss1.item()
            total_pde += loss2.item()
            total_mono += loss3.item()

        if valid_loader:
            val_mse, _, _ = _evaluate_hybrid(hybrid_model, valid_loader)
        else:
            val_mse = float('nan')

        print("[{}] Epoch {:03d} | data:{:.6f} PDE:{:.6f} mono:{:.6f} | valMSE:{:.6f}".format(
            out_dir.name,
            epoch,
            total_data / len(train_loader),
            total_pde / len(train_loader),
            total_mono / len(train_loader),
            val_mse
        ))

        improved = (val_mse < best_val) if valid_loader else True
        if improved:
            patience = 0
            best_val = val_mse
            _, y_true, y_pred = _evaluate_hybrid(hybrid_model, test_loader)
            best_true, best_pred = y_true, y_pred
            best_state = correction_net.state_dict()
        else:
            patience += 1
            if early_stop and patience > early_stop:
                print("Early stop at epoch {}".format(epoch))
                break

    if best_true is not None and best_pred is not None:
        np.save(out_dir / 'true_soh.npy', best_true)
        np.save(out_dir / 'predicted_soh.npy', best_pred)
        torch.save({'correction_mlp': best_state}, out_dir / 'hybrid_head.pth')
        rmse = float(np.sqrt(mean_squared_error(best_true, best_pred)))
        mae, mape, mse, _ = eval_metrix(best_pred, best_true)
        with open(out_dir / "metrics.txt", "w") as f:
            f.write("MAE: {}\nMAPE: {}\nMSE: {}\nRMSE: {}\n".format(mae, mape, mse, rmse))
        print(" Saved predictions to {} | RMSE: {:.4f}".format(out_dir, rmse))
        return best_true, best_pred, out_dir

    return None, None, out_dir

# ---------------------------------------------------------------------
# Plot predictions (reproducible save)
# ---------------------------------------------------------------------
def plot_predictions(trues, preds, battery_id=0, save_dir: Optional[PathLike] = None, title_prefix="HybridPINN Transfer"):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    trues = np.array(trues).ravel()
    preds = np.array(preds).ravel()
    errors = np.abs(trues - preds)

    color_list = ['#74AED4', '#7BDFF2', '#FBDD85', '#F46F43', '#CF3D3E']
    colors = mcolors.LinearSegmentedColormap.from_list('custom_cmap', color_list, N=256)

    lims = [min(trues.min(), preds.min()) - 0.02, max(trues.max(), preds.max()) + 0.02]

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(trues, preds, c=errors, cmap=colors, s=10, alpha=0.7, vmin=0, vmax=0.1)
    plt.plot(lims, lims, '--', lw=1)
    plt.xlabel("True SOH")
    plt.ylabel("Predicted SOH")
    plt.title(f"{title_prefix} (Battery {battery_id})")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.gca().set_aspect('equal')
    plt.grid(True)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Absolute Error")

    base = Path(save_dir) if save_dir else resolve_transfer_results_dir()
    (base / "plots").mkdir(parents=True, exist_ok=True)

    save_path = base / "plots" / f"battery_{battery_id}_plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()

    print(f"✅ Saved plot at: {save_path.resolve()}")
