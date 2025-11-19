# This code does the data preprocessing and loading of battery degradation datasets Dataset_A and Dataset_B
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
from sklearn.model_selection import train_test_split
from utils.util import write_to_txt
from pathlib import Path
# ---------------------------------------------------------------------
# automatically detect repository root (portable on any system)
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
# ---------------------------------------------------------------------
# data preprocessing steps
# ---------------------------------------------------------------------
class DF():
    def __init__(self, args):
        self.normalization = True
        self.normalization_method = args.normalization_method
        self.args = args

    def _3_sigma(self, Ser1):
        rule = (Ser1.mean() - 3 * Ser1.std() > Ser1) | (Ser1.mean() + 3 * Ser1.std() < Ser1)
        return np.arange(Ser1.shape[0])[rule]
# ---------------------------------------------------------------------
# outlier removal from data
# ---------------------------------------------------------------------
    def delete_3_sigma(self, df):
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        out_index = list(set([idx for col in df.columns for idx in self._3_sigma(df[col])]))
        return df.drop(out_index, axis=0).reset_index(drop=True)
# ---------------------------------------------------------------------
# this part introduced to ensuring capacity column exists
# ---------------------------------------------------------------------
    def _ensure_capacity_column(self, df):
        if 'capacity' in df.columns:
            return df
        candidates = ['Discharge_Capacity(mAh)', 'discharge_capacity', 'Discharge_Capacity',
                      'Q', 'q', 'capacity_mAh', 'Capacity', 'capacity(Ah)', 'capacity_ah']
        for cand in candidates:
            if cand in df.columns:
                return df.rename(columns={cand: 'capacity'})
        raise KeyError("No 'capacity' column found.")
# ---------------------------------------------------------------------
# reads one csv and insert cycle index, data cleaning and feature normalization
# ---------------------------------------------------------------------
    def read_one_csv(self, file_name, nominal_capacity=None):
        df = pd.read_csv(file_name)
        df = self._ensure_capacity_column(df)
        insert_pos = df.shape[1] - 1
        df.insert(insert_pos, 'cycle index', np.arange(df.shape[0], dtype=np.float64))
        df = self.delete_3_sigma(df)
        if nominal_capacity is not None:
            df['capacity'] = df['capacity'] / nominal_capacity
        feature_cols = df.columns[:-1]
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
        if self.normalization:
            f_df = df.loc[:, feature_cols].copy()
            if self.normalization_method == 'min-max':
                denom = (f_df.max() - f_df.min()).replace(0, np.nan)
                f_df = 2 * (f_df - f_df.min()) / denom - 1
            elif self.normalization_method == 'z-score':
                denom = f_df.std().replace(0, np.nan)
                f_df = (f_df - f_df.mean()) / denom
            df.loc[:, feature_cols] = f_df.astype(np.float64).values
        else:
            df.loc[:, feature_cols] = df.loc[:, feature_cols].astype(np.float64).values
        return df
# ---------------------------------------------------------------------
# produces training pairs in sequential way
# ---------------------------------------------------------------------
    def load_one_battery(self, path, nominal_capacity=None):
        df = self.read_one_csv(path, nominal_capacity)
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return (x[:-1], y[:-1]), (x[1:], y[1:])
# ---------------------------------------------------------------------
# combining multiple csv files, convert to tensors and prepares train/test/validation split
# ---------------------------------------------------------------------
    def load_all_battery(self, path_list, nominal_capacity):
        X1, X2, Y1, Y2 = [], [], [], []
        if self.args.log_dir and self.args.save_folder:
            save_name = os.path.join(self.args.save_folder, self.args.log_dir)
            write_to_txt(save_name, 'Data paths:\n' + '\n'.join(path_list))
        for path in path_list:
            (x1, y1), (x2, y2) = self.load_one_battery(path, nominal_capacity)
            X1.append(x1); X2.append(x2); Y1.append(y1); Y2.append(y2)
        X1, X2 = np.concatenate(X1, axis=0), np.concatenate(X2, axis=0)
        Y1, Y2 = np.concatenate(Y1, axis=0), np.concatenate(Y2, axis=0)
        tensor_X1 = torch.tensor(X1, dtype=torch.float32)
        tensor_X2 = torch.tensor(X2, dtype=torch.float32)
        tensor_Y1 = torch.tensor(Y1, dtype=torch.float32).view(-1, 1)
        tensor_Y2 = torch.tensor(Y2, dtype=torch.float32).view(-1, 1)
        split = int(len(tensor_X1) * 0.8)
        train_X1, test_X1 = tensor_X1[:split], tensor_X1[split:]
        train_X2, test_X2 = tensor_X2[:split], tensor_X2[split:]
        train_Y1, test_Y1 = tensor_Y1[:split], tensor_Y1[split:]
        train_Y2, test_Y2 = tensor_Y2[:split], tensor_Y2[split:]
        (train_X1, valid_X1, train_X2, valid_X2,
         train_Y1, valid_Y1, train_Y2, valid_Y2) = train_test_split(
            train_X1, train_X2, train_Y1, train_Y2, test_size=0.2, random_state=420
        )
        train_loader = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
                                  batch_size=self.args.batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
                                  batch_size=self.args.batch_size, shuffle=True)
        test_loader  = DataLoader(TensorDataset(test_X1, test_X2, test_Y1, test_Y2),
                                  batch_size=self.args.batch_size, shuffle=False)
        return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

# ---------------------------------------------------------------------
# handling Dataset-A
# ---------------------------------------------------------------------
class Dataset_A(DF):
    def __init__(self, root, args):
        super(Dataset_A, self).__init__(args)
        self.root = root
        self.file_list = [f for f in os.listdir(root) if f.endswith('.csv')]
        self.num = len(self.file_list)
        self.batch_names = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
        self.batch_size = args.batch_size
        self.nominal_capacity = 2.0 if self.normalization else None

    def read_one_batch(self, batch='2C'):
        if isinstance(batch, int):
            batch = self.batch_names[batch]
        assert batch in self.batch_names
        file_list = [os.path.join(self.root, f) for f in self.file_list if batch in f]
        return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)

    def read_all(self, specific_path_list=None):
        file_list = ([os.path.join(self.root, f) for f in self.file_list]
                     if specific_path_list is None else specific_path_list)
        return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)

# ---------------------------------------------------------------------
# handling Dataset_B
# ---------------------------------------------------------------------
class Dataset_B(DF):
    def __init__(self, root='../data/Dataset_B', args=None):
        super(Dataset_B, self).__init__(args)
        self.root = root
        self.nominal_capacity = 1.1 if self.normalization else None
        self.reference_features = None
        try:
            Dataset_A_root = getattr(self.args, 'Dataset_A_root', None)
            if Dataset_A_root and os.path.isdir(Dataset_A_root):
                ref_csvs = [f for f in os.listdir(Dataset_A_root) if f.endswith('.csv')]
                if ref_csvs:
                    ref_path = os.path.join(Dataset_A_root, ref_csvs[0])
                    ref_df = pd.read_csv(ref_path)
                    ref_df = self._ensure_capacity_column(ref_df)
                    insert_pos = ref_df.shape[1] - 1
                    if 'cycle index' not in ref_df.columns:
                        ref_df.insert(insert_pos, 'cycle index',
                                      np.arange(ref_df.shape[0], dtype=np.float64))
                    self.reference_features = [c for c in ref_df.columns if c != 'capacity']
        except Exception as e:
            print(f"[Dataset_B] Warning: could not derive reference features from Dataset_A: {e}")

    def read_one_csv(self, file_name, nominal_capacity=None):
        df = pd.read_csv(file_name)
        df = self._ensure_capacity_column(df)
        other_cols = [c for c in df.columns if c != 'capacity']
        df = df[other_cols + ['capacity']]
        insert_pos = df.shape[1] - 1
        if 'cycle index' not in df.columns:
            df.insert(insert_pos, 'cycle index', np.arange(df.shape[0], dtype=np.float64))
        df = self.delete_3_sigma(df)
        if nominal_capacity is not None:
            df['capacity'] = df['capacity'] / nominal_capacity
        if self.reference_features is not None:
            for col in self.reference_features:
                if col not in df.columns:
                    df[col] = 0.0
            keep = set(self.reference_features + ['capacity'])
            extra = [c for c in df.columns if c not in keep]
            if extra:
                df = df.drop(columns=extra)
            df = df[self.reference_features + ['capacity']]
        feature_cols = [c for c in df.columns if c != 'capacity']
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
        if self.normalization:
            f_df = df.loc[:, feature_cols].copy()
            if self.normalization_method == 'min-max':
                denom = (f_df.max() - f_df.min()).replace(0, np.nan)
                f_df = 2 * (f_df - f_df.min()) / denom - 1
            elif self.normalization_method == 'z-score':
                denom = f_df.std().replace(0, np.nan)
                f_df = (f_df - f_df.mean()) / denom
            f_df = f_df.fillna(0.0)
            df.loc[:, feature_cols] = f_df.astype(np.float64).values
        else:
            df.loc[:, feature_cols] = df.loc[:, feature_cols].astype(np.float64).values
        return df

    def load_one_battery(self, path, nominal_capacity=None):
        df = self.read_one_csv(path, nominal_capacity)
        X = df.drop(columns=['capacity']).values
        y = df['capacity'].values
        return (X[:-1], y[:-1]), (X[1:], y[1:])

    def read_all(self, specific_path_list=None):
        if specific_path_list is None:
            file_list = [os.path.join(self.root, f) for f in os.listdir(self.root) if f.endswith('.csv')]
        else:
            file_list = specific_path_list
        return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)


if __name__ == '__main__':
# ---------------------------------------------------------------------
# following part is used for ensuring everything goes well
# ---------------------------------------------------------------------
    import argparse

    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--normalization_method', type=str, default='z-score')
        parser.add_argument('--log_dir', type=str, default='test.txt')
        parser.add_argument('--save_folder', type=str, default=None)
        # Portable dataset paths (automatically based on repo root)
        parser.add_argument('--Dataset_A_root', type=str, default=str(REPO_ROOT / 'data' / 'Dataset_A'))
        parser.add_argument('--Dataset_B_root', type=str, default=str(REPO_ROOT / 'data' / 'Dataset_B'))
        return parser.parse_args()

    args = get_args()
    ds_a = Dataset_A(root=args.Dataset_A_root, args=args)
    loader_x = ds_a.read_all()
    print(f'[Dataset_A] Train: {len(loader_x["train"])}, Valid: {len(loader_x["valid"])}, Test: {len(loader_x["test"])}')
    for x1, x2, y1, y2 in loader_x['train']:
        print('[Dataset_A] shapes:', x1.shape, x2.shape, y1.shape, y2.shape)
        print('[Dataset_A] Max SOH in batch:', float(y1.max()))
        break

    ds_b = Dataset_B(root=args.Dataset_B_root, args=args)
    loader_h = ds_b.read_all()
    print(f'[Dataset_B] Train: {len(loader_h["train"])}, Valid: {len(loader_h["valid"])}, Test: {len(loader_h["test"])}')
    for x1, x2, y1, y2 in loader_h['train']:
        print('[Dataset_B] shapes:', x1.shape, x2.shape, y1.shape, y2.shape)
        print('[Dataset_B] Max SOH in batch:', float(y1.max()))
        break
