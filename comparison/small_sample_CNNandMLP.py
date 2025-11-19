#!/usr/bin/env python3
# This code is for small sample run of MLP and CNN.
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from utils.util import AverageMeter, get_logger
from Model.Comparing_Models import MLP, CNN
from Model.Model import LR_Scheduler
from Data_loading.dataloader import Dataset_A as DatasetAData, Dataset_B as DatasetBData
# ---------------------------------------------------------------------
# defining what models and where should the result save.
# ---------------------------------------------------------------------
MODELS = ['MLP', 'CNN']
BATCHES = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
SMALL_SAMPLES = [1, 2, 3, 4]
EXPERIMENTS_PER = 10
EPOCHS = 100
PATIENCE = 10
# ---------------------------------------------------------------------
# Automatically detect project root (portable across systems)
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]  

SAVE_BASE = REPO_ROOT / "results for analysis"
DATASET = 'Dataset_A'
DATA_ROOTS = {
    'Dataset_A': REPO_ROOT / 'data' / 'Dataset_A',
    'Dataset_B': REPO_ROOT / 'data' / 'Dataset_B',
}

LOG_FILENAME = 'logging.txt'
NORM_METHOD = 'min-max'
# ---------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------
def mae(a, b): return float(np.mean(np.abs(a - b)))
def mse(a, b): return float(np.mean((a - b) ** 2))
def rmse(a, b): return float(np.sqrt(mse(a, b)))
def mape(a, b, eps=1e-8):
    denom = np.maximum(np.abs(a), eps)
    return float(np.mean(np.abs((a - b) / denom)))
# ---------------------------------------------------------------------
# loads the training, testing and validation dataset of each batches
# ---------------------------------------------------------------------
def load_data(dataset: str, batch_name: str, batch_size: int, normalization_method: str, small_sample=None):
    root = str(DATA_ROOTS[dataset])
    DataClass = DatasetAData if dataset == 'Dataset_A' else DatasetBData
    class A: pass
    args = A()
    args.batch_size = batch_size
    args.normalization_method = normalization_method
    args.log_dir = LOG_FILENAME
    args.save_folder = '.'
    data = DataClass(root=root, args=args)
    train_list, test_list = [], []
    for fname in os.listdir(root):
        if batch_name in fname:
            (test_list if ('4' in fname or '8' in fname) else train_list).append(os.path.join(root, fname))
    if small_sample is not None:
        train_list = train_list[:small_sample]
    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    return {'train': train_loader['train'], 'valid': train_loader['valid'], 'test': test_loader['test']}
# ---------------------------------------------------------------------
# for both MLP and CNN
# ---------------------------------------------------------------------
class Trainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.save_dir = args.save_folder
        os.makedirs(self.save_dir, exist_ok=True)
        self.epochs = args.epochs
        self.logger = get_logger(os.path.join(args.save_folder, args.log_dir))
        self.loss_meter = AverageMeter()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.warmup_lr)
        self.scheduler = LR_Scheduler(
            optimizer=self.optimizer,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr,
            num_epochs=args.epochs,
            base_lr=args.lr,
            final_lr=args.final_lr
        )

    def clear_logger(self):
        try:
            self.logger.removeHandler(self.logger.handlers[0])
            self.logger.handlers.clear()
        except Exception:
            pass

    def train_one_epoch(self, epoch):
        self.model.train()
        self.loss_meter.reset()
        for (x1, _, y1, _) in self.train_loader:
            x1 = x1.to(self.device)
            y1 = y1.to(self.device)
            y_pred = self.model(x1)
            loss = self.loss_func(y_pred, y1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_meter.update(loss.item())
        info = f'[Train] epoch:{epoch:0>3d}, data loss:{self.loss_meter.avg:.6f}'
        self.logger.info(info)
        return self.loss_meter.avg

    def valid(self, epoch):
        self.model.eval()
        self.loss_meter.reset()
        with torch.no_grad():
            for (x1, _, y1, _) in self.valid_loader:
                x1 = x1.to(self.device)
                y1 = y1.to(self.device)
                y_pred = self.model(x1)
                loss = self.loss_func(y_pred, y1)
                self.loss_meter.update(loss.item())
        info = f'[Valid] epoch:{epoch:0>3d}, data loss:{self.loss_meter.avg:.6f}'
        self.logger.info(info)
        return self.loss_meter.avg

    # saves true and predicted soh as .npy files
    def test(self):
        self.model.eval()
        trues, preds = [], []
        with torch.no_grad():
            for (x1, _, y1, _) in self.test_loader:
                x1 = x1.to(self.device)
                y_pred = self.model(x1)
                trues.append(y1.cpu().numpy())
                preds.append(y_pred.detach().cpu().numpy())
        true_label = np.concatenate(trues, axis=0)
        pred_label = np.concatenate(preds, axis=0)
        np.save(os.path.join(self.save_dir, 'true_label.npy'), true_label)
        np.save(os.path.join(self.save_dir, 'pred_label.npy'), pred_label)
        np.save(os.path.join(self.save_dir, 'true_soh.npy'), true_label.ravel())
        np.save(os.path.join(self.save_dir, 'predicted_soh.npy'), pred_label.ravel())
        return true_label, pred_label
# ---------------------------------------------------------------------
# loops over each epoch with scheduled learning rate, stores best prediction
# ---------------------------------------------------------------------
    def train_loop(self, patience=PATIENCE):
        min_loss = float('inf')
        no_improve = 0
        best_pair = None
        for epoch in range(1, self.epochs + 1):
            _ = self.train_one_epoch(epoch)
            _ = self.scheduler.step()
            vloss = self.valid(epoch)
            if vloss < min_loss and self.test_loader is not None:
                min_loss = vloss
                best_pair = self.test()
                no_improve = 0
            else:
                no_improve += 1
            if no_improve > patience:
                self.logger.info(f'[EarlyStop] patience {patience} exceeded at epoch {epoch}')
                break
        if best_pair is None and self.test_loader is not None:
            best_pair = self.test()
        self.clear_logger()
        return best_pair

def load_model(name: str):
    if name == 'MLP':
        return MLP()
    elif name == 'CNN':
        return CNN()
# ---------------------------------------------------------------------
# computes all metrics and append to logging.txt
# ---------------------------------------------------------------------
def write_metrics(exp_dir: Path, y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.ravel(y_true); y_pred = np.ravel(y_pred)
    with open(exp_dir / LOG_FILENAME, 'a') as f:
        f.write(f"Final MAE:  {mae(y_true, y_pred):.6f}\n")
        f.write(f"Final MAPE: {mape(y_true, y_pred):.6f}\n")
        f.write(f"Final MSE:  {mse(y_true, y_pred):.6f}\n")
        f.write(f"Final RMSE: {rmse(y_true, y_pred):.6f}\n")

# full experimental run
def run_one_experiment(dataset: str, model_name: str, batch_name: str, batch_idx: int, n_train: int, k_exp: int):
    exp_dir = (
        SAVE_BASE
        / f'{dataset}-{model_name} results (small sample {n_train})'
        / f'{batch_idx}-{batch_idx}'
        / f'Experiment{k_exp}'
    )
    exp_dir.mkdir(parents=True, exist_ok=True)
    class A: pass
    a = A()
    a.model = model_name
    a.dataset = dataset
    a.normalization_method = NORM_METHOD
    a.batch = batch_name
    a.epochs = EPOCHS
    a.early_stop = PATIENCE
    a.warmup_epochs = 30
    a.warmup_lr = 2e-3
    a.lr = 1e-2
    a.final_lr = 2e-4
    a.lr_F = 5e-4
    a.batch_size = 512
    a.save_folder = str(exp_dir)
    a.log_dir = LOG_FILENAME
    loaders = load_data(dataset, batch_name, a.batch_size, a.normalization_method, small_sample=n_train)
    model = load_model(model_name)
    trainer = Trainer(model, loaders['train'], loaders['valid'], loaders['test'], a)
    true_label, pred_label = trainer.train_loop(patience=PATIENCE)
    write_metrics(exp_dir, true_label, pred_label)
    print(f"[DONE] {dataset} | {model_name} | Batch {batch_name} (idx {batch_idx}) | n={n_train} | Exp{k_exp} → {exp_dir.resolve()}")
# ---------------------------------------------------------------------
# for full experiment
# ---------------------------------------------------------------------
def main():
    batch_to_index = {b: i for i, b in enumerate(BATCHES)}
    for model_name in MODELS:
        for batch_name in BATCHES:
            i = batch_to_index[batch_name]
            print(f"\n=== {DATASET} | {model_name} | Batch {batch_name} (idx {i}) ===")
            for n in SMALL_SAMPLES:
                print(f"  -- small sample n = {n}")
                for k in range(1, EXPERIMENTS_PER + 1):
                    run_one_experiment(DATASET, model_name, batch_name, i, n, k)

# runner
if __name__ == '__main__':
    main()
