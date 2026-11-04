"""Check that the phase-1 checkpoints of two-phase sweeps are in place.

    python -m experiments.check_phase1 <sweep> [<sweep> ...] [--root .]

For every run of the sweeps that starts from a phase-1 run (``phase1_epoch``),
resolves the phase-1 epoch exactly as the run will (`phase_resume.
phase1_checkpoint_epoch`, on the baseline folder of the same log, seed and
batch size) and reports whether its checkpoint exists. Runs that share a
baseline and an epoch spec are checked once. Run it on the machine that will
train, from the results root. Exits with status 1 when a checkpoint is missing.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from experiments.collect_results import _find_backup_dir
from experiments.grid import configs_for, resolve_run_dir
from experiments.sweeps import SWEEPS
from phase_resume import _STOPPING_METRICS, phase1_checkpoint_epoch

# The config keys that name the phase-1 run a phase-2 run starts from.
_BASELINE_KEYS = ("log_name", "seed", "subset_fraction", "batch_size", "out_string")


def baseline_config(cfg):
    """The configuration of the phase-1 run of a phase-2 config."""
    return {k: cfg[k] for k in _BASELINE_KEYS if k in cfg}


def phase1_tasks(backup_dir):
    """The run's tasks, read off the validation columns it logged."""
    columns = pd.read_csv(os.path.join(backup_dir, "backup_results.csv"), nrows=0).columns
    return [task for task, metrics in _STOPPING_METRICS.items() if metrics[0][0] in columns]


def check(names, root="."):
    """One row per (baseline, epoch spec): the resolved epoch or the problem."""
    rows, seen = [], set()
    for name in names:
        if name not in SWEEPS:
            raise KeyError(f"unknown sweep {name!r}; have {sorted(SWEEPS)}")
        for cfg in configs_for(SWEEPS[name]):
            spec = cfg.get("phase1_epoch")
            if spec is None:
                continue
            baseline_dir = os.path.join(root, str(resolve_run_dir(baseline_config(cfg))))
            if (baseline_dir, spec) in seen:
                continue
            seen.add((baseline_dir, spec))
            row = {"sweep": name, "log_name": cfg["log_name"], "seed": cfg["seed"],
                   "phase1_epoch": spec, "epoch": None, "status": "ok"}
            backup_dir = _find_backup_dir(baseline_dir)
            if backup_dir is None:
                row["status"] = f"no baseline run at {baseline_dir}"
            else:
                try:
                    row["epoch"] = phase1_checkpoint_epoch(backup_dir, spec, phase1_tasks(backup_dir))
                except (FileNotFoundError, ValueError) as e:
                    row["status"] = str(e)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Check the phase-1 checkpoints of two-phase sweeps.")
    p.add_argument("sweeps", nargs="+", help="sweep name(s) from experiments/sweeps.py")
    p.add_argument("--root", default=".", help="results root (default: .)")
    args = p.parse_args()
    df = check(args.sweeps, root=args.root)
    if df.empty:
        raise SystemExit("No run in these sweeps starts from a phase-1 run.")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(df.to_string(index=False))
    missing = (df["status"] != "ok").sum()
    print(f"\n{len(df) - missing} of {len(df)} phase-1 checkpoints in place.")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
