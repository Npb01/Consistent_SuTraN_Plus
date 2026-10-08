"""Minimal analysis stub: read a collected summary CSV (from collect_results)
and print the per-impl comparison table. A starting point -- extend with
best-of-lambda selection, significance tests, and plots as results land.

    uv run python -m experiments.analyze collected_results.csv
    uv run python -m experiments.analyze collected_results.csv --axiom 2

Rows are grouped as `baseline (lambda=0)` and `<impl> @ lambda=<x>`, averaged over
seeds. Remember the comparison is best-of-lambda PER IMPL on the shared yardstick
(consistency + predictive) -- not at a shared lambda.
"""

from __future__ import annotations

import argparse

import pandas as pd

# Curated columns shown when present (collect_results names them like this).
_METRICS = [
    # predictive (instance-based test set)
    "ib_DL sim", "ib_MAE TTNE minutes", "ib_MAE RRT minutes",
    "ib_Multi-Class Accuracy", "ib_Macro-F1",
    # impl-agnostic consistency yardstick (test set)
    "cons_mean_abs_gap_IB",
    "cons_outcome_head_accuracy_IB", "cons_outcome_suffix_accuracy_IB",
    "cons_outcome_disagreement_rate_IB",
    # optimization / compute
    "best_epoch", "train_seconds_mean", "peak_gpu_mem_mb_max",
    "ax1_contrib_share", "ax2_contrib_share",
]


def summarize(df, axiom):
    """Per-(impl, lambda) means over seeds for the given axiom (1 or 2)."""
    if "found" in df.columns:
        df = df[df["found"] == True].copy()      # noqa: E712 (pandas mask)
    else:
        df = df.copy()

    lam = "lambda_ltn" if axiom == 1 else "lambda_ltn_outcome"
    impl = "axiom1_impl" if axiom == 1 else "axiom2_impl"

    # Rows for this axiom: its lambda > 0, plus the shared no-axiom baseline.
    is_baseline = (df["lambda_ltn"] == 0) & (df["lambda_ltn_outcome"] == 0)
    df = df[(df[lam] > 0) | is_baseline]
    if df.empty:
        return None

    df["config"] = [
        "baseline (lambda=0)" if l == 0 else f"{i} @ lambda={l}"
        for i, l in zip(df[impl], df[lam])
    ]
    metrics = [m for m in _METRICS if m in df.columns]
    grouped = df.groupby("config")
    out = grouped[metrics].mean(numeric_only=True)
    out.insert(0, "n_seeds", grouped.size())
    return out.sort_index()


def main():
    p = argparse.ArgumentParser(description="Per-impl comparison table from a collected CSV.")
    p.add_argument("csv", help="collected summary CSV (from collect_results --out)")
    p.add_argument("--axiom", type=int, choices=[1, 2], default=None,
                   help="which axiom to summarize; default: whichever has active runs")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.4g}")

    axes = [args.axiom] if args.axiom else [1, 2]
    printed = False
    for ax in axes:
        table = summarize(df, ax)
        if table is None or table.empty:
            continue
        printed = True
        print(f"\n=== Axiom {ax} - mean over seeds ===")
        print(table.to_string())
    if not printed:
        print("No runs to summarize (nothing found, or no active-axiom rows).")


if __name__ == "__main__":
    main()
