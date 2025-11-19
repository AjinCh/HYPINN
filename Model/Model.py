# Defines the architecture of Hybrid PINN
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from utils.util import AverageMeter, eval_metrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------
# activation Function
# ---------------------------------------------------------------------
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


# ---------------------------------------------------------------------
# general MLP architecture
# ---------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=17, output_dim=1, layers_num=4, hidden_dim=50, dropout=0.2):
        super().__init__()
        assert layers_num >= 2
        layers = []
        for i in range(layers_num):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(Sin())
            elif i == layers_num - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(Sin())
                layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    # Neural network parameter initialization
    def _init_weights(self):
        g = torch.Generator().manual_seed(1337)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, generator=g)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------
# predictor maps embeddings to scalar SOH output
# ---------------------------------------------------------------------
class Predictor(nn.Module):
    def __init__(self, input_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, 32),
            Sin(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------
# data-driven approximation of SOH (Solution_u network)
# ---------------------------------------------------------------------
class Solution_u(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MLP(input_dim=17, output_dim=32, layers_num=3, hidden_dim=60, dropout=0.2)
        self.predictor = Predictor(input_dim=32)
        self._init_weights()

    def _init_weights(self):
        g = torch.Generator().manual_seed(1337)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, generator=g)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        emb = self.encoder(x)
        return self.predictor(emb)


# ---------------------------------------------------------------------
# Customizing Learning Rate Scheduler
# ---------------------------------------------------------------------
class LR_Scheduler:
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch=1):
        self.optimizer = optimizer
        self.warm_iters = max(1, int(warmup_epochs) * max(1, int(iter_per_epoch)))
        self.total_iters = max(self.warm_iters + 1, int(num_epochs) * max(1, int(iter_per_epoch)))
        self.base_lr = float(base_lr)
        self.final_lr = float(final_lr)
        self.warmup_lr = float(warmup_lr)
        self.t = 0

    def _lr_at(self, t):
        if t < self.warm_iters:
            r = t / float(self.warm_iters)
            return self.warmup_lr + (self.base_lr - self.warmup_lr) * (r ** 3)
        remain = self.total_iters - self.warm_iters
        r = (t - self.warm_iters) / float(remain)
        cos = 0.5 * (1.0 + np.cos(np.pi * r))
        mix = 0.95 * cos + 0.05 * (1.0 - r)
        return self.final_lr + (self.base_lr - self.final_lr) * mix

    def step(self):
        lr = float(self._lr_at(self.t))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.t += 1
        return lr


# ---------------------------------------------------------------------
# physics-Informed Neural Network (PINN)
# ---------------------------------------------------------------------
class PINN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.solution_u = Solution_u().to(device)
        self.dynamical_F = MLP(
            input_dim=35, output_dim=1,
            layers_num=args.F_layers_num,
            hidden_dim=args.F_hidden_dim,
            dropout=0.2
        ).to(device)
        self.opt_u = torch.optim.Adam(self.solution_u.parameters(), lr=args.warmup_lr)
        self.opt_F = torch.optim.Adam(self.dynamical_F.parameters(), lr=args.lr_F)
        self.scheduler = LR_Scheduler(
            self.opt_u,
            args.warmup_epochs, args.warmup_lr,
            args.epochs, args.lr, args.final_lr
        )
        self.loss_fn = nn.MSELoss()
        self.relu = nn.ReLU()
        self.alpha = float(args.alpha)
        self.beta = float(args.beta)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        self.solution_u.load_state_dict(checkpoint['solution_u'])
        self.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        self.eval()

    def save_model(self, model_path):
        torch.save({
            'solution_u': self.solution_u.state_dict(),
            'dynamical_F': self.dynamical_F.state_dict()
        }, model_path)

    def forward(self, xt):
        xt = xt.requires_grad_(True)
        x, t = xt[:, :-1], xt[:, -1:]
        u = self.solution_u(torch.cat([x, t], dim=1))
        u_t = grad(u.sum(), t, create_graph=True)[0]
        u_x = grad(u.sum(), x, create_graph=True)[0]
        F = self.dynamical_F(torch.cat([xt, u, u_x, u_t], dim=1))
        f = u_t - F
        return u, f

    # Training one epoch
    def train_epoch_pinn(self, loader):
        self.solution_u.train()
        self.dynamical_F.train()
        m_data, m_pde, m_mono = AverageMeter(), AverageMeter(), AverageMeter()
        for x1, x2, y1, y2 in loader:
            x1, x2 = x1.to(device), x2.to(device)
            y1, y2 = y1.to(device), y2.to(device)
            u1, f1 = self.forward(x1)
            u2, f2 = self.forward(x2)
            loss_data = 0.5 * (self.loss_fn(u1, y1) + self.loss_fn(u2, y2))
            zero = torch.zeros_like(f1)
            loss_pde = 0.5 * (self.loss_fn(f1, zero) + self.loss_fn(f2, zero))
            loss_mono = self.relu((u2 - u1) * (y1 - y2)).mean()
            loss = loss_data + self.alpha * loss_pde + self.beta * loss_mono
            self.opt_u.zero_grad()
            self.opt_F.zero_grad()
            loss.backward()
            self.opt_u.step()
            self.opt_F.step()
            m_data.update(loss_data.item())
            m_pde.update(loss_pde.item())
            m_mono.update(loss_mono.item())
        return m_data.avg, m_pde.avg, m_mono.avg

    def validate_epoch(self, loader):
        self.solution_u.eval()
        total = 0.0
        with torch.no_grad():
            for x1, _, y1, _ in loader:
                pred = self.solution_u(x1.to(device))
                total += self.loss_fn(pred, y1.to(device)).item()
        return total / max(1, len(loader))

    def predict_on_loader(self, loader):
        self.solution_u.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x1, _, y1, _ in loader:
                p = self.solution_u(x1.to(device)).cpu().numpy()
                ys.append(y1.numpy())
                ps.append(p)
        return np.concatenate(ys), np.concatenate(ps)

    def Train(self, train_loader, valid_loader=None, test_loader=None):
        best_val = float('inf')
        patience = 0
        save_dir = getattr(self.args, 'save_folder', '.')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'model.pth')

        iters_per_epoch = max(1, len(train_loader))
        self.scheduler.total_iters = max(self.scheduler.total_iters, iters_per_epoch * int(self.args.epochs))
        self.scheduler.warm_iters = min(self.scheduler.warm_iters, self.scheduler.total_iters - 1)

        for epoch in range(1, int(self.args.epochs) + 1):
            l_data, l_pde, l_mono = self.train_epoch_pinn(train_loader)
            lr = self.scheduler.step()
            val_mse = self.validate_epoch(valid_loader) if valid_loader else float('inf')

            improved = val_mse < best_val if valid_loader else True
            if improved:
                best_val = val_mse
                patience = 0
                self.save_model(save_path)
                manifest = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "best_val_mse": best_val,
                    "alpha": self.alpha,
                    "beta": self.beta,
                    "F_layers_num": int(self.args.F_layers_num),
                    "F_hidden_dim": int(self.args.F_hidden_dim),
                    "lr": float(self.args.lr),
                    "final_lr": float(self.args.final_lr),
                    "warmup_lr": float(self.args.warmup_lr),
                }
                try:
                    with open(os.path.join(save_dir, "manifest.json"), "w") as f:
                        json.dump(manifest, f, indent=2)
                except Exception:
                    pass
            else:
                patience += 1

            total_loss = l_data + self.alpha * l_pde + self.beta * l_mono
            print(f"[Train] Epoch {epoch:03d} | LR {lr:.6f} | data {l_data:.6f} | pde {l_pde:.6f} | mono {l_mono:.6f} | total {total_loss:.6f}")
            if valid_loader:
                print(f"[Valid] Epoch {epoch:03d} | MSE {val_mse:.8f}")
            if getattr(self.args, 'early_stop', 0) and patience > int(self.args.early_stop):
                print(f"Early stopping at epoch {epoch}")
                break

        if test_loader:
            y_true, y_pred = self.predict_on_loader(test_loader)
            mae, mape, mse, rmse = eval_metrix(y_pred, y_true)
            print(f"[Test] MAE {mae:.6f} | MAPE {mape:.6f} | MSE {mse:.6f} | RMSE {rmse:.6f}")


# ---------------------------------------------------------------------
# adds correction network to base PINN
# ---------------------------------------------------------------------
class HybridPINN(nn.Module):
    def __init__(self, base_pinn: PINN, correction_net: nn.Module):
        super().__init__()
        self.base_pinn = base_pinn
        self.correction_net = correction_net

    def forward(self, x):
        with torch.no_grad():
            pinn_pred = self.base_pinn.solution_u(x)
        correction = self.correction_net(x)
        return pinn_pred + correction


# ---------------------------------------------------------------------
# print number of trainable parameters
# ---------------------------------------------------------------------
def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Model has {total:,} trainable parameters")
    return total
