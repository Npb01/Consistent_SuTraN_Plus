"""Collect finished-run artifacts into one tidy table -- the backbone of the
analysis pipeline.

    python -m experiments.collect_results <sweep> [<sweep> ...] \
        [--root .] [--out collected_results.csv]

One row per run (config + metrics). It is sweep-driven: for each config in the
sweep(s) it resolves the run dir the same way the pipeline does
(`grid.resolve_run_dir`, so no fragile name parsing), globs for the (nested)
`backup_results.csv`, and reads:

  - backup_results.csv   -> best epoch (via the run's own `select_best_epoch`),
                            the best-epoch validation metrics, and the axiom
                            contribution SHARE (contrib / composite loss) used
                            for lambda calibration.
  - compute_cost.csv     -> per-epoch train time + peak memory summaries.
  - TEST_SET_RESULTS/*.pkl-> predictive metrics (averaged_results_{IB,CB}) and
                            the impl-agnostic consistency yardstick
                            (consistency_diagnostics).

Missing runs are kept as rows with `found=False` so gaps are visible. Analysis
scripts should import `collect_sweeps(...)` and work off the returned DataFrame
rather than re-walking the result dirs.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import pickle

import pandas as pd

from experiments.grid import configs_for, resolve_run_dir
from experiments.sweeps import SWEEPS
from Utils.callback_selection import get_target_metrics_dict, select_best_epoch

# Config keys surfaced as columns, with the run_mto_experiment defaults so a
# config that omits a key still gets the value the run actually used.
CFG_DEFAULTS = {
    "log_name": None, "mto_technique": None, "seed": None,
    "subset_fraction": 1.0, "val_subset_fraction": 1.0,
    "lambda_ltn": 0.0, "detach_mode": "none", "axiom1_impl": "ltn_smooth_eq",
    "lambda_ltn_outcome": 0.0, "detach_mode_outcome": "none",
    "axiom2_impl": "collapsed_q",
    "batch_size": 128, "num_epochs": None, "validate_every": 1, "patience": None,
}

# best-epoch validation columns worth carrying (present-only).
_VAL_COLS = {
    "val_dl": "Activity suffix: 1-DL (validation)",
    "val_mae_ttne_min": "TTNE - minutes MAE validation",
    "val_mae_rrt_min": "RRT - mintues MAE validation",
    "val_outcome_acc": "Multi-Class Outcome - Accuracy",
    "val_outcome_macro_f1": "Multi-Class Outcome - Macro-F1 score",
}
_AXIOM_COLS = ["ltn_ax1_term", "ltn_ax1_contrib", "ltn_ax2_term",
               "ltn_ax2_contrib", "ltn_ax2_residual_mass", "ltn_ax2_min_survival"]

# Which target keys get a select_best_epoch entry if their column is present.
_TASK_COL = {
    "activity_suffix": "Activity suffix: 1-DL (validation)",
    "timestamp_suffix": "TTNE - minutes MAE validation",
    "remaining_runtime": "RRT - mintues MAE validation",
    "multiclass_outcome": "Multi-Class Outcome - Macro-F1 score",
    "binary_outcome": "Binary Outcome - AUC-ROC validation",
}


def _find_backup_dir(top_dir):
    """The directory holding backup_results.csv, somewhere under `top_dir`
    (the UW run nests it under CaLenDiR_training/UncertaintyWeighting/...)."""
    hits = sorted(glob.glob(os.path.join(top_dir, "**", "backup_results.csv"),
                            recursive=True))
    return os.path.dirname(hits[0]) if hits else None


def _load_pickle(path):
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def _best_epoch_row(df):
    """Reproduce the run's best-epoch choice; fall back to the last row."""
    task_list = [t for t, col in _TASK_COL.items() if col in df.columns]
    try:
        best_epoch, _ = select_best_epoch(df, get_target_metrics_dict(task_list))
        row = df[df["epoch"] == int(best_epoch)].iloc[0]
        return int(best_epoch), row
    except Exception:
        return int(df["epoch"].iloc[-1]), df.iloc[-1]


def _num(x):
    try:
        v = float(x)
        return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


def collect_run(cfg, root="."):
    """One run's row: config fields + best-epoch metrics + compute + test metrics."""
    row = {k: cfg.get(k, d) for k, d in CFG_DEFAULTS.items()}
    # Which axiom this run exercises, and its active impl/lambda (convenience).
    row["axiom1_on"] = row["lambda_ltn"] > 0
    row["axiom2_on"] = row["lambda_ltn_outcome"] > 0
    top_dir = os.path.join(root, str(resolve_run_dir(cfg)))
    row["run_dir"] = str(resolve_run_dir(cfg))

    backup_dir = _find_backup_dir(top_dir)
    row["found"] = backup_dir is not None
    if backup_dir is None:
        return row

    # --- per-epoch table: best epoch, val metrics, lambda contribution share ---
    df = pd.read_csv(os.path.join(backup_dir, "backup_results.csv"))
    best_epoch, best = _best_epoch_row(df)
    row["best_epoch"] = best_epoch
    row["n_validated_epochs"] = len(df)
    composite = _num(best.get("composite training loss"))
    row["composite_loss_best"] = composite
    for out_key, col in _VAL_COLS.items():
        if col in df.columns:
            row[out_key] = _num(best.get(col))
    for col in _AXIOM_COLS:
        if col in df.columns:
            row[col] = _num(best.get(col))
    # Contribution share = axiom contrib / composite loss, at the best epoch.
    for ax in ("1", "2"):
        contrib = _num(best.get(f"ltn_ax{ax}_contrib"))
        row[f"ax{ax}_contrib_share"] = (contrib / composite) if (
            contrib is not None and composite not in (None, 0)) else None

    # --- compute cost ---
    cc_path = os.path.join(backup_dir, "compute_cost.csv")
    if os.path.exists(cc_path):
        cc = pd.read_csv(cc_path)
        if len(cc):
            row["train_seconds_mean"] = _num(cc["train_seconds"].mean())
            row["train_seconds_median"] = _num(cc["train_seconds"].median())
            row["train_seconds_total"] = _num(cc["train_seconds"].sum())
            row["peak_gpu_mem_mb_max"] = _num(cc["peak_gpu_mem_mb"].max())
            # Optimization stability (present only on runs with grad-norm
            # logging): overall norm, plus per-head means if logged.
            if "grad_norm_orig_mean" in cc.columns:
                row["grad_norm_orig_mean"] = _num(cc["grad_norm_orig_mean"].mean())
                row["grad_norm_orig_max"] = _num(cc["grad_norm_orig_max"].max())
            for col in cc.columns:
                if col.startswith("grad_norm_head_"):
                    row[col] = _num(cc[col].mean())

    # --- test-set metrics (predictive + consistency yardstick) ---
    tsr = os.path.join(backup_dir, "TEST_SET_RESULTS")
    for prefix, fname in (("ib", "averaged_results_IB.pkl"),
                          ("cb", "averaged_results_CB.pkl")):
        for k, v in _load_pickle(os.path.join(tsr, fname)).items():
            row[f"{prefix}_{k}"] = _num(v)
    for k, v in _load_pickle(os.path.join(tsr, "consistency_diagnostics.pkl")).items():
        row[f"cons_{k}"] = _num(v)

    return row


def collect_sweeps(names, root="."):
    """Tidy DataFrame, one row per (deduped) config across the given sweeps.

    This is the per-run SUMMARY (best epoch + test metrics + compute). For the
    per-epoch training/validation CURVES, use `collect_history`.
    """
    rows = []
    for name in names:
        if name not in SWEEPS:
            raise KeyError(f"unknown sweep {name!r}; have {sorted(SWEEPS)}")
        for cfg in configs_for(SWEEPS[name]):
            r = collect_run(cfg, root=root)
            r["sweep"] = name
            rows.append(r)
    return pd.DataFrame(rows)


def collect_history(names, root="."):
    """Long per-epoch table across runs, for training/validation curve plots.

    One row per (run, epoch), with the config fields attached and the compute
    cost merged in. Columns come straight from `backup_results.csv` -- which
    holds BOTH the per-epoch training losses (composite / act CE / ttne MAE /
    rrt MAE / outcome CE) AND the validation metrics (val TTNE-MAE, 1-DL, RRT-MAE,
    outcome acc / macro-F1). Note rows exist only for VALIDATED epochs (every
    `validate_every`); the merged compute-cost columns cover every epoch, so
    non-validated epochs (if any) carry timing with NaN metrics.
    """
    frames = []
    for name in names:
        if name not in SWEEPS:
            raise KeyError(f"unknown sweep {name!r}; have {sorted(SWEEPS)}")
        for cfg in configs_for(SWEEPS[name]):
            top_dir = os.path.join(root, str(resolve_run_dir(cfg)))
            backup_dir = _find_backup_dir(top_dir)
            if backup_dir is None:
                continue
            df = pd.read_csv(os.path.join(backup_dir, "backup_results.csv"))
            cc_path = os.path.join(backup_dir, "compute_cost.csv")
            if os.path.exists(cc_path):
                # compute_cost has every epoch; backup_results only validated
                # ones -> outer merge keeps all epochs.
                df = pd.read_csv(cc_path).merge(df, on="epoch", how="outer")
            df.insert(0, "sweep", name)
            for k, d in CFG_DEFAULTS.items():
                df[k] = cfg.get(k, d)
            df["run_dir"] = str(resolve_run_dir(cfg))
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    p = argparse.ArgumentParser(description="Collect run artifacts into one CSV.")
    p.add_argument("sweeps", nargs="+", help="sweep name(s) from experiments/sweeps.py")
    p.add_argument("--root", default=".", help="results root holding <LOG>/<model_string> (default: .)")
    p.add_argument("--out", default="collected_results.csv", help="per-run summary CSV path")
    p.add_argument("--history", default=None,
                   help="also write the per-epoch curve table to this CSV path")
    args = p.parse_args()

    df = collect_sweeps(args.sweeps, root=args.root)
    df.to_csv(args.out, index=False)
    n_found = int(df["found"].sum()) if "found" in df else 0
    print(f"{n_found}/{len(df)} runs found; wrote {len(df.columns)} columns to {args.out}")

    if args.history:
        hist = collect_history(args.sweeps, root=args.root)
        hist.to_csv(args.history, index=False)
        print(f"per-epoch history: {len(hist)} rows -> {args.history}")
    if n_found < len(df):
        missing = df.loc[~df["found"], "run_dir"].tolist()
        print(f"  missing ({len(missing)}):")
        for d in missing[:10]:
            print(f"    {d}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")


if __name__ == "__main__":
    main()
