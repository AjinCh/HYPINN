#!/usr/bin/env python3
# This code produces .csv and a .tx file with small sample metric evaluation result of all models
import os, re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

def mae(y_true, y_pred): return float(np.mean(np.abs(y_true - y_pred)))
def mse(y_true, y_pred): return float(np.mean((y_true - y_pred)**2))
def rmse(y_true, y_pred): return float(np.sqrt(mse(y_true, y_pred)))
def mape(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred)/denom)))
# ---------------------------------------------------------------------
# loading true and predicted datas from the saved results
# ---------------------------------------------------------------------
def load_pair(exp_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    for t, p in [
        (exp_dir / "true_soh.npy", exp_dir / "predicted_soh.npy"),
        (exp_dir / "true_label.npy", exp_dir / "pred_label.npy"),
    ]:
        if t.exists() and p.exists():
            try:
                yt = np.ravel(np.load(t))
                yp = np.ravel(np.load(p))
                n = min(len(yt), len(yp))
                if n > 0:
                    return yt[:n], yp[:n]
            except Exception:
                pass
    return None
# ---------------------------------------------------------------------
# defining metrics
# ---------------------------------------------------------------------
LOG_PAT = {
    "MAE":  re.compile(r"MAE[^0-9eE\.\+\-]*([eE\+\-\.0-9]+)"),
    "MAPE": re.compile(r"MAPE[^0-9eE\.\+\-]*([eE\+\-\.0-9]+)"),
    "MSE":  re.compile(r"MSE[^0-9eE\.\+\-]*([eE\+\-\.0-9]+)"),
    "RMSE": re.compile(r"RMSE[^0-9eE\.\+\-]*([eE\+\-\.0-9]+)"),
}

def parse_log(log_path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        txt = log_path.read_text(errors="ignore")
    except Exception:
        return out
    for k, rx in LOG_PAT.items():
        vals = rx.findall(txt)
        if vals:
            try:
                out[k] = float(vals[-1])
            except Exception:
                pass
    return out
# ---------------------------------------------------------------------
# looks for experiment directories
# ---------------------------------------------------------------------
def small_sample_dirc(repo_root: Path, dataset: str, model: str, n_train: int) -> List[Path]:
    cands: List[Path] = []
    new_dir = repo_root / "results for analysis" / f"{dataset}-{model} results (small sample {n_train})"
    if new_dir.exists():
        cands.append(new_dir)
    legacy_map = {
        "PINN": "results",
        "MLP":  "results_mlp",
        "CNN":  "results_cnn",
    }
    legacy_root = repo_root / legacy_map.get(model, "")
    if legacy_root.exists():
        old_dir = legacy_root / f"{dataset} results (small sample {n_train})"
        if old_dir.exists():
            cands.append(old_dir)
    return cands
# ---------------------------------------------------------------------
# loops though each experiment folders to gather experiment metrics
# ---------------------------------------------------------------------
def collect_model(repo_root: Path, dataset: str, model: str, n_list: List[int]) -> pd.DataFrame:
    rows: List[dict] = []
    for n_train in n_list:
        bases = small_sample_dirc(repo_root, dataset, model, n_train)
        if not bases:
            continue
        for base in bases:
            for bdir in sorted([p for p in base.iterdir() if p.is_dir() and re.match(r"^\d+-\d+$", p.name)]):
                try:
                    bidx = int(bdir.name.split("-")[0])
                except Exception:
                    continue
                batch_no = bidx + 1
                for exp in sorted([p for p in bdir.iterdir() if p.is_dir() and p.name.lower().startswith("experiment")]):
                    metrics: Dict[str, float] = {}
                    pair = load_pair(exp)
                    if pair is not None:
                        yt, yp = pair
                        metrics = {
                            "MAE":  mae(yt, yp),
                            "MAPE": mape(yt, yp),
                            "MSE":  mse(yt, yp),
                            "RMSE": rmse(yt, yp),
                        }
                    else:
                        log = exp / "logging.txt"
                        if log.exists():
                            metrics = parse_log(log)
                    if not metrics:
                        continue
                    rows.append({
                        "Model": model,
                        "Batch": batch_no,
                        "TrainBatteries": n_train,
                        "Experiment": exp.name,
                        "MAE":  metrics.get("MAE",  np.nan),
                        "MAPE": metrics.get("MAPE", np.nan),
                        "MSE":  metrics.get("MSE",  np.nan),
                        "RMSE": metrics.get("RMSE", np.nan),
                    })
    return pd.DataFrame(rows)
# ---------------------------------------------------------------------
# computes the mean of metrics
# ---------------------------------------------------------------------
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    return (df.groupby(["Batch", "TrainBatteries", "Model"], as_index=False)
              .agg(MAE=("MAE","mean"),
                   MAPE=("MAPE","mean"),
                   MSE=("MSE","mean"),
                   RMSE=("RMSE","mean")))
# ---------------------------------------------------------------------
# For side by side comparison
# ---------------------------------------------------------------------
def pivot_wide(agg: pd.DataFrame, models_order: List[str], metrics: List[str]) -> pd.DataFrame:
    if agg.empty: return agg
    wide = (agg.pivot_table(index=["Batch","TrainBatteries"], columns="Model",
                            values=metrics, aggfunc="first").sort_index())
    ordered = []
    for m in models_order:
        for met in metrics:
            col = (met, m)
            if col in wide.columns:
                ordered.append(col)
    if ordered:
        wide = wide[ordered]
    wide.columns = pd.MultiIndex.from_tuples([(m, met) for (met, m) in wide.columns],
                                             names=["Model","Metric"])
    return wide
# ---------------------------------------------------------------------
# .tex file generation
# ---------------------------------------------------------------------
def latex_table(dfw: pd.DataFrame, metrics: List[str], models: List[str], dataset: str) -> str:
    if dfw.empty:
        return "% No data found. Check folder structure and names.\n"
    colspec = "cc" + "".join([" " + "r"*len(metrics) for _ in models])
    lines = []
    lines += [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{Small-sample SOH performance on {dataset} (mean over experiments).}}",
        "\\label{tab:small-sample-all-models}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        "\\multicolumn{2}{c}{} " + " ".join(
            [f"& \\multicolumn{{{len(metrics)}}}{{c}}{{\\textbf{{{m}}}}}" for m in models]
        ) + " \\\\",
    ]
    start = 3
    cmids = []
    for k in range(len(models)):
        s = start + k*len(metrics)
        e = s + len(metrics) - 1
        cmids.append(f"\\cmidrule(lr){{{s}-{e}}}")
    lines.append("".join(cmids))
    sub = ["\\textbf{Batch}", "\\textbf{Train Batteries}"]
    for _ in models:
        for met in metrics:
            sub.append(f"\\textbf{{{met}}}")
    lines.append(" & ".join(sub) + " \\\\")
    lines.append("\\midrule")
    idx = dfw.index
    batches = sorted(set(idx.get_level_values(0)))
    for b in batches:
        tbs = sorted(set(idx.get_level_values(1)[idx.get_level_values(0)==b]))
        for j, tb in enumerate(tbs):
            row = dfw.loc[(b, tb)]
            cells = []
            cells.append(f"\\multirow{{{len(tbs)}}}{{*}}{{\\textbf{{{b}}}}}" if j==0 else "")
            cells.append(str(tb))
            for m in models:
                for met in metrics:
                    val = row.get((m, met))
                    cells.append(f"{val:.4f}" if pd.notna(val) else "")
            lines.append(" & ".join(cells) + " \\\\")
        if b != batches[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)

# runner
def main():
    ap = argparse.ArgumentParser("Small-sample table aggregator")
    ap.add_argument("--repo_root", default=None)
    ap.add_argument("--dataset", default="Dataset_A", choices=["Dataset_A", "Dataset_B"])
    ap.add_argument("--models", default="Ours,MLP,CNN")
    ap.add_argument("--small_samples", default="1,2,3,4")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--out_csv", default="small_sample_all_models.csv")
    ap.add_argument("--out_tex", default="small_sample_all_models.tex")
    args = ap.parse_args()
# ---------------------------------------------------------------------
# Automatically detect repository root path to make the script portable
# ---------------------------------------------------------------------
    here = Path(__file__).resolve()
    repo_root = Path(args.repo_root or here.parents[1]).resolve()
    out_dir = Path(args.out_dir or (repo_root / "results for analysis" / "processed results")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / args.out_csv
    out_tex = out_dir / args.out_tex

    dataset = args.dataset
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    n_list = [int(x) for x in args.small_samples.split(",") if x.strip()]
    frames = []

    for model in models:
        dfm = collect_model(repo_root, dataset, model, n_list)
        if not dfm.empty:
            frames.append(dfm)

    df_long = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df_long.empty:
        print("No experiments found. Check paths and that results are under "
              f"'{repo_root}/results for analysis/<DATASET>-<MODEL> results (small sample n)/...'")
        return

    agg = aggregate(df_long)
    dfw = pivot_wide(agg, models_order=models, metrics=["MAE","MAPE","RMSE"])
    csv_df = dfw.copy()
    csv_df.columns = [f"{m}_{met}" for (m, met) in csv_df.columns]
    csv_df.to_csv(out_csv)
    print(f"Saved CSV: {out_csv}")

    tex = latex_table(dfw, metrics=["MAE","MAPE","RMSE"], models=models, dataset=dataset)
    out_tex.write_text(tex, encoding="utf-8")
    print(f"Saved LaTeX: {out_tex}")

    with pd.option_context("display.max_columns", None, "display.width", 180):
        print("\nPreview:")
        print(dfw.round(4))

if __name__ == "__main__":
    main()
