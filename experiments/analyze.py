"""Minimal analysis stub: read a collected summary CSV (from collect_results)
and print the per-impl comparison table, and optionally plot the
consistency-vs-accuracy frontier. A starting point -- extend with best-of-lambda
selection and significance tests as results land.

    uv run python -m experiments.analyze collected_results.csv
    uv run python -m experiments.analyze collected_results.csv --axiom 2
    uv run python -m experiments.analyze collected_results.csv --axiom 2 --frontier frontier

Rows are grouped as `baseline (lambda=0)` and `<impl> @ lambda=<x>`, averaged over
seeds. Remember the comparison is best-of-lambda PER IMPL on the shared yardstick
(consistency + predictive) -- not at a shared lambda.
"""

from __future__ import annotations

import argparse
import math

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


# Frontier axes per axiom: (consistency metric, accuracy metric). Consistency is
# always "lower is better"; the accuracy metric's direction is given too, so the
# "better" corner can be annotated. Defaults can be overridden on the CLI.
_FRONTIER_AXES = {
    1: {"x": "cons_mean_abs_gap_IB", "x_label": "mean |sum dt - RRT| gap (lower better)",
        "y": "ib_MAE TTNE minutes", "y_label": "TTNE MAE minutes", "y_lower_better": True},
    2: {"x": "cons_outcome_disagreement_rate_IB", "x_label": "outcome head-vs-suffix disagreement (lower better)",
        "y": "ib_Macro-F1", "y_label": "outcome macro-F1", "y_lower_better": False},
}


def _is_set(v):
    """True for a meaningful scheduled value (not None / NaN / empty / 'none-ish')."""
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    return str(v) not in ("", "nan", "None")


def _setup_label(row, axiom):
    """A short label grouping runs into one frontier curve.

    Runs that share a label differ only in their target weight, so they form the
    points of a single curve. Encodes the impl, the base detach mode, and any
    lambda-schedule / detach-switch, so a scheduled setup is distinct from the
    fixed-weight one with the same impl/detach.
    """
    lam = row["lambda_ltn"] if axiom == 1 else row["lambda_ltn_outcome"]
    if lam == 0:
        return "baseline"
    if axiom == 1:
        impl, base = row["axiom1_impl"], row["detach_mode"]
        late, switch = row.get("detach_mode_late"), row.get("detach_switch_epoch")
        sched, warm = row.get("lambda_ltn_schedule"), row.get("lambda_ltn_warmup_epochs")
    else:
        impl, base = row["axiom2_impl"], row["detach_mode_outcome"]
        late, switch = row.get("detach_mode_outcome_late"), row.get("detach_switch_outcome_epoch")
        sched, warm = row.get("lambda_ltn_outcome_schedule"), row.get("lambda_ltn_outcome_warmup_epochs")

    parts = [str(impl), f"detach={base}"]
    if _is_set(late):
        parts.append(f"->{late}@{int(float(switch))}")
    if _is_set(sched) and str(sched) != "constant":
        parts.append(f"{sched}@{int(float(warm))}")
    return " ".join(parts)


def frontier(df, axiom, out_path, x_col=None, y_col=None):
    """Scatter the (consistency, accuracy) frontier for one axiom, saved to PNG.

    Each setup (see `_setup_label`) is one curve of weight points, averaged over
    seeds with +/- 1 std whiskers; the baseline is a single marker. Returns the
    path written, or None if there is nothing to plot.
    """
    import matplotlib
    matplotlib.use("Agg")                      # file output, no display needed
    import matplotlib.pyplot as plt

    axcfg = _FRONTIER_AXES[axiom]
    x_col = x_col or axcfg["x"]
    y_col = y_col or axcfg["y"]
    lam_col = "lambda_ltn" if axiom == 1 else "lambda_ltn_outcome"

    if "found" in df.columns:
        df = df[df["found"] == True].copy()    # noqa: E712 (pandas mask)
    else:
        df = df.copy()
    is_baseline = (df["lambda_ltn"] == 0) & (df["lambda_ltn_outcome"] == 0)
    df = df[(df[lam_col] > 0) | is_baseline]
    df = df[df[x_col].notna() & df[y_col].notna()]
    if df.empty:
        return None

    df["setup"] = [_setup_label(r, axiom) for _, r in df.iterrows()]
    # Mean + std over seeds, per (setup, weight).
    agg = (df.groupby(["setup", lam_col])[[x_col, y_col]]
             .agg(["mean", "std"]).reset_index())
    agg.columns = ["setup", "lam", "x", "x_std", "y", "y_std", ][:len(agg.columns)]

    fig, ax = plt.subplots(figsize=(8, 6))
    for setup, g in agg.groupby("setup"):
        g = g.sort_values("lam")
        if setup == "baseline":
            ax.errorbar(g["x"], g["y"], xerr=g["x_std"], yerr=g["y_std"],
                        marker="*", markersize=16, color="black", linestyle="none",
                        label="baseline", zorder=5)
            continue
        line = ax.errorbar(g["x"], g["y"], xerr=g["x_std"], yerr=g["y_std"],
                           marker="o", markersize=5, capsize=2, label=setup)
        # Annotate each point with its target weight.
        for _, r in g.iterrows():
            ax.annotate(f"{r['lam']:g}", (r["x"], r["y"]), fontsize=7,
                        textcoords="offset points", xytext=(4, 4),
                        color=line[0].get_color())

    ax.set_xlabel(axcfg["x_label"])
    ax.set_ylabel(axcfg["y_label"])
    better = "top-left" if not axcfg["y_lower_better"] else "bottom-left"
    ax.set_title(f"Axiom {axiom} consistency-accuracy frontier "
                 f"(better = {better}); points = weight, mean +/- std over seeds")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Per-impl comparison table from a collected CSV.")
    p.add_argument("csv", help="collected summary CSV (from collect_results --out)")
    p.add_argument("--axiom", type=int, choices=[1, 2], default=None,
                   help="which axiom to summarize; default: whichever has active runs")
    p.add_argument("--frontier", default=None, metavar="PREFIX",
                   help="also save the consistency-accuracy frontier plot(s) to "
                        "<PREFIX>_axiom<N>.png")
    p.add_argument("--x-col", default=None,
                   help="override the frontier consistency (x) column")
    p.add_argument("--y-col", default=None,
                   help="override the frontier accuracy (y) column")
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

    if args.frontier:
        for ax in axes:
            out = f"{args.frontier}_axiom{ax}.png"
            written = frontier(df, ax, out, x_col=args.x_col, y_col=args.y_col)
            print(f"frontier axiom {ax}: {'wrote ' + written if written else 'no data to plot'}")


if __name__ == "__main__":
    main()
