"""The enforcement direction: which side of each relation to freeze.

    python -m experiments.directions --baselines <sweep> [...] [--root .] [--out f.csv]
    python -m experiments.directions <sweep> [<sweep> ...] [--root .] [--out f.csv]

With ``--baselines``, one direction per log and relation from the baseline
sweeps: the validation diagnostics of the selected checkpoints
(``VAL_SET_RESULTS/consistency_diagnostics.pkl``), case-based, averaged over
the seeds, with one seed standard deviation of the paired difference between
the two sides as the tie margin (see `phase_resume.direction_from_diagnostics`).
This is the direction every enforcing run of that log uses.

Without it, the per-checkpoint decisions that phase-2 runs with a detach mode
of "auto" recorded in ``phase1.json``. The decision depends only on the
phase-1 checkpoint, so all phase-2 runs that start from the same one must
agree; this is checked. Prints one row per (log, seed, relation) and a per-log
summary of how often each side was chosen.
"""

from __future__ import annotations

import argparse
import os
import pickle

import pandas as pd

from experiments.collect_results import _find_backup_dir, read_phase1_record
from experiments.grid import configs_for, resolve_run_dir
from experiments.sweeps import SWEEPS
from phase_resume import direction_from_diagnostics

# The two sides of each relation in the validation diagnostics, per basis.
_SIDES = {"axiom1": ("mae_rrt_{}", "mae_ttne_sum_{}"),
          "axiom2": ("outcome_head_macro_f1_{}", "outcome_suffix_macro_f1_{}")}


def baseline_directions(names, root=".", basis="CB"):
    """One row per (log, relation): the seed-mean scores and the direction."""
    rows = []
    for name in names:
        if name not in SWEEPS:
            raise KeyError(f"unknown sweep {name!r}; have {sorted(SWEEPS)}")
        by_log = {}
        for cfg in configs_for(SWEEPS[name]):
            backup_dir = _find_backup_dir(os.path.join(root, str(resolve_run_dir(cfg))))
            path = backup_dir and os.path.join(backup_dir, "VAL_SET_RESULTS",
                                               "consistency_diagnostics.pkl")
            if not path or not os.path.exists(path):
                print(f"  missing validation diagnostics: {resolve_run_dir(cfg)}")
                continue
            with open(path, "rb") as f:
                by_log.setdefault(cfg["log_name"], []).append(pickle.load(f))
        for log_name, diags in by_log.items():
            df = pd.DataFrame(diags)
            mean = df.mean(numeric_only=True).to_dict()
            margins = {}
            for relation, (a, b) in _SIDES.items():
                a, b = a.format(basis), b.format(basis)
                if a in df and b in df:
                    diff = df[a] - df[b]
                    margins[relation] = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
            for relation, choice in direction_from_diagnostics(mean, basis, margins).items():
                rows.append({"log_name": log_name, "relation": relation,
                             "num_seeds": len(df), **choice})
    return pd.DataFrame(rows)

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
    p = argparse.ArgumentParser(description="Tabulate the enforcement directions.")
    p.add_argument("sweeps", nargs="+", help="sweep name(s) from experiments/sweeps.py")
    p.add_argument("--baselines", action="store_true",
                   help="the sweeps are baselines: one direction per log from their "
                        "validation diagnostics")
    p.add_argument("--basis", default="CB", choices=["CB", "IB"],
                   help="case-based (default) or instance-based scores (--baselines only)")
    p.add_argument("--root", default=".", help="results root (default: .)")
    p.add_argument("--out", default=None, help="also write the table to this CSV path")
    args = p.parse_args()

    if args.baselines:
        df = baseline_directions(args.sweeps, root=args.root, basis=args.basis)
        if df.empty:
            print("No baseline with validation diagnostics found.")
            return
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(df.drop(columns="metric").to_string(index=False))
        if args.out:
            df.to_csv(args.out, index=False)
            print(f"\nwrote {len(df)} rows to {args.out}")
        return

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
