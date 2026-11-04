"""Collect the hard-enforcement results of runs into tidy tables.

    python -m experiments.collect_hard <sweep> [<sweep> ...] --out hard.csv \\
        [--summary hard_summary.csv] [--tag <overrides>] [--root .]

Reads what `experiments.hard_enforcement` wrote per run and split
(``<SPLIT>_SET_RESULTS_HARD[_<tag>]/``) and writes

  - ``--out``: one row per run, split and arm (``raw``: the stored predictions
    re-scored, ``enforced``: after hard enforcement), with the run's
    configuration, the kept side of each relation, the number of changed
    outcome predictions, the averaged metrics (``CB ...`` / ``IB ...``) and the
    scalar consistency diagnostics (gaps in hours).
  - ``--summary``: mean and standard deviation over seeds per configuration,
    split and arm.

Sweep-driven like `experiments.collect_results`: run folders are resolved with
`grid.resolve_run_dir`; runs or splits without enforcement output are skipped
and reported.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import pandas as pd

from experiments.collect_results import CFG_DEFAULTS, _find_backup_dir
from experiments.grid import configs_for, resolve_run_dir
from experiments.hard_enforcement import SPLITS
from experiments.sweeps import SWEEPS

ARMS = ("raw", "enforced")
_SECONDS_PER_HOUR = 3600


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _scalar(value):
    return isinstance(value, (int, float, str, bool)) or value is None


def collect_split(cfg, hard_dir):
    """The raw and enforced rows of one run and split."""
    with open(os.path.join(hard_dir, "enforcement.json")) as f:
        record = json.load(f)
    rows = []
    for arm in ARMS:
        row = {k: cfg.get(k, default) for k, default in CFG_DEFAULTS.items()}
        row.update(arm=arm, time_keep=record.get("time_keep"),
                   outcome_keep=record.get("outcome_keep"),
                   num_instances=record.get("num_instances"),
                   outcome_changed=record.get("outcome_changed", 0) if arm == "enforced" else 0)
        for basis in ("CB", "IB"):
            path = os.path.join(hard_dir, arm, f"averaged_results_{basis}.pkl")
            if os.path.exists(path):
                row.update({f"{basis} {k}": v for k, v in _load_pickle(path).items()})
        diagnostics = _load_pickle(os.path.join(hard_dir, arm, "consistency_diagnostics.pkl"))
        for k, v in diagnostics.items():
            if not _scalar(v):
                continue
            if k.startswith("mean_abs_gap"):
                row[k.replace("mean_abs_gap", "mean_abs_gap_hours")] = v / _SECONDS_PER_HOUR
            else:
                row[k] = v
        rows.append(row)
    return rows


def collect_hard(names, tag=None, root="."):
    """One row per run, split and arm of the sweeps' hard-enforcement output."""
    rows, missing = [], []
    for name in names:
        if name not in SWEEPS:
            raise KeyError(f"unknown sweep {name!r}; have {sorted(SWEEPS)}")
        for cfg in configs_for(SWEEPS[name]):
            run_dir = os.path.join(root, str(resolve_run_dir(cfg)))
            backup_dir = _find_backup_dir(run_dir)
            for split, (folder, _) in SPLITS.items():
                hard_dir = (os.path.join(backup_dir, f"{folder}_HARD" + (f"_{tag}" if tag else ""))
                            if backup_dir else None)
                if hard_dir is None or not os.path.exists(os.path.join(hard_dir, "enforcement.json")):
                    missing.append(f"{run_dir} {split}")
                    continue
                for row in collect_split(cfg, hard_dir):
                    rows.append({"sweep": name, "split": split, **row})
    return pd.DataFrame(rows), missing


def summarize(df):
    """Mean and SD over seeds per configuration, split and arm."""
    keys = ["split", "arm"] + [k for k in CFG_DEFAULTS if k != "seed" and k in df]
    numeric = [c for c in df.select_dtypes("number").columns if c not in keys + ["seed"]]
    grouped = df.groupby(keys, sort=False, dropna=False)
    mean = grouped[numeric].mean().add_suffix(" mean")
    sd = grouped[numeric].std().add_suffix(" sd")
    summary = pd.concat([mean, sd], axis=1)[[f"{c} {s}" for c in numeric for s in ("mean", "sd")]]
    summary.insert(0, "n_seeds", grouped.size())
    return summary.reset_index()


def main():
    p = argparse.ArgumentParser(description="Collect hard-enforcement results.")
    p.add_argument("sweeps", nargs="+", help="sweep name(s) from experiments/sweeps.py")
    p.add_argument("--out", required=True, help="CSV with one row per run, split and arm")
    p.add_argument("--summary", default=None, help="CSV with seed means and SDs")
    p.add_argument("--tag", default=None,
                   help="override tag of the enforcement folders (e.g. time_ttne_sum)")
    p.add_argument("--root", default=".", help="results root (default: .)")
    args = p.parse_args()

    df, missing = collect_hard(args.sweeps, tag=args.tag, root=args.root)
    for m in missing:
        print(f"  no hard-enforcement output: {m}")
    if df.empty:
        raise SystemExit("Nothing collected.")
    df.to_csv(args.out, index=False)
    print(f"{len(df)} rows -> {args.out}")
    if args.summary:
        summary = summarize(df)
        summary.to_csv(args.summary, index=False)
        print(f"{len(summary)} rows -> {args.summary}")


if __name__ == "__main__":
    main()
