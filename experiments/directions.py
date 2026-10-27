"""Which head the two-phase rule froze, per log and seed.

    python -m experiments.directions <sweep> [<sweep> ...] [--root .] [--out directions.csv]

Each phase-2 run with a detach mode of "auto" records its decision in
``phase1.json``: both predictions of each relation scored on the validation set
with the phase-1 checkpoint, the better one, and the detach mode that freezes
it (see `phase_resume.direction_from_diagnostics`). The decision depends only
on the phase-1 checkpoint, so all phase-2 runs that start from the same one
must agree; this is checked. Prints one row per (log, seed, relation) and a
per-log summary of how often each side was chosen.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from experiments.collect_results import _find_backup_dir, read_phase1_record
from experiments.grid import configs_for, resolve_run_dir
from experiments.sweeps import SWEEPS

_SCORES = {"axiom1": ("rrt_head", "ttne_sum"), "axiom2": ("outcome_head", "suffix_implied")}
_DETACH_KEY = {"axiom1": "detach_mode", "axiom2": "detach_mode_outcome"}


def collect_directions(names, root="."):
    """One row per (log, seed, phase-1 epoch, relation) with the recorded decision."""
    rows = {}
    for name in names:
        if name not in SWEEPS:
            raise KeyError(f"unknown sweep {name!r}; have {sorted(SWEEPS)}")
        for cfg in configs_for(SWEEPS[name]):
            backup_dir = _find_backup_dir(os.path.join(root, str(resolve_run_dir(cfg))))
            record = read_phase1_record(backup_dir) if backup_dir else None
            if not record or not record.get("direction"):
                continue
            for relation, choice in record["direction"].items():
                a, b = _SCORES[relation]
                key = (cfg["log_name"], cfg["seed"], record["phase1_epoch"], relation)
                row = {"log_name": key[0], "seed": key[1], "phase1_epoch": key[2],
                       "relation": relation, a: choice[a], b: choice[b],
                       "better": choice["better"], "frozen": choice[_DETACH_KEY[relation]]}
                if key in rows and rows[key]["better"] != row["better"]:
                    raise ValueError(f"Phase-2 runs from the same checkpoint disagree: {key}")
                rows[key] = row
    return pd.DataFrame(list(rows.values()))


def main():
    p = argparse.ArgumentParser(description="Tabulate the two-phase direction decisions.")
    p.add_argument("sweeps", nargs="+", help="phase-2 sweep name(s) from experiments/sweeps.py")
    p.add_argument("--root", default=".", help="results root (default: .)")
    p.add_argument("--out", default=None, help="also write the table to this CSV path")
    args = p.parse_args()

    df = collect_directions(args.sweeps, root=args.root)
    if df.empty:
        print("No phase-2 run with a recorded direction found.")
        return
    df = df.sort_values(["relation", "log_name", "seed"])
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False))
        print("\nchosen side per log (count over seeds):")
        print(df.groupby(["relation", "log_name"])["better"].value_counts()
                .unstack(fill_value=0).to_string())
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nwrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
