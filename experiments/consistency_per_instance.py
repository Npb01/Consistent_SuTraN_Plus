"""Per-instance inconsistency-vs-error analysis (RQ4).

    python -m experiments.consistency_per_instance <sweep> [<sweep> ...] \
        [--root .] [--out-ax1 ax1_per_instance.csv] [--out-ax2 ax2_per_instance.csv] \
        [--summary reliability_summary.csv]

Loads each run's `*_SET_RESULTS/consistency_per_instance.pt` (raw predictions +
truths, written by `SuTraN/inference_procedure.py`) and turns it into tidy tables
for the question "does per-instance inconsistency predict per-instance error?":

  * ax1 long table -- one row per (run, instance): the between-route time gap and
    both route errors, in SECONDS (and minutes), with prefix/suffix lengths.
  * ax2 long table -- one row per (run, non-leaky instance): head correctness,
    suffix-route agreement, head confidence, has-determining-act.
  * per-run reliability summary -- the headline readouts: ax1 rank/linear
    correlation of |gap| with |rrt error| plus a gap-decile error lift; ax2 head
    accuracy conditional on head/suffix agreement, and a confidence cross-check.

Sweep-driven and import-friendly, same as `collect_results`: resolve each run
dir via `grid.resolve_run_dir`, no name parsing. Runs without the dump are
skipped (reported in the CLI).
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

from experiments.grid import configs_for, resolve_run_dir
from experiments.sweeps import SWEEPS
from experiments.collect_results import CFG_DEFAULTS, _find_backup_dir

_N_DECILES = 10


def _cfg_cols(cfg):
    """The same config fields collect_results surfaces, for grouping/joining."""
    return {k: cfg.get(k, d) for k, d in CFG_DEFAULTS.items()}


def _load_dump(cfg, root="."):
    """The run's per-instance dict, or None if the run/dump is absent."""
    top_dir = os.path.join(root, str(resolve_run_dir(cfg)))
    backup_dir = _find_backup_dir(top_dir)
    if backup_dir is None:
        return None
    path = os.path.join(backup_dir, "TEST_SET_RESULTS", "consistency_per_instance.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _np(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


def _corr(a, b, method="pearson"):
    """Correlation that returns NaN (no warning) when either side is constant."""
    a, b = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float))
    if a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
        return float("nan")
    return a.corr(b, method=method)


def ax1_frame(names, root="."):
    """Long per-instance ax1 table (seconds; minutes for the headline errors)."""
    frames = []
    for name in names:
        if name not in SWEEPS:
            raise KeyError(f"unknown sweep {name!r}; have {sorted(SWEEPS)}")
        for cfg in configs_for(SWEEPS[name]):
            d = _load_dump(cfg, root)
            if d is None or "ax1_rt_pred" not in d:
                continue
            rt_pred, rt_true = _np(d["ax1_rt_pred"]), _np(d["ax1_rt_true"])
            sum_ts_pred, sum_ts_true = _np(d["ax1_sum_ts_pred"]), _np(d["ax1_sum_ts_true"])
            gap = sum_ts_pred - rt_pred                      # between-route disagreement
            df = pd.DataFrame({
                "signed_gap_s": gap,
                "abs_gap_s": np.abs(gap),
                "err_rt_s": rt_pred - rt_true,               # rrt-head error
                "err_ts_s": sum_ts_pred - sum_ts_true,       # summed-ttne error
                "abs_err_rt_min": np.abs(rt_pred - rt_true) / 60.0,
                "abs_err_ts_min": np.abs(sum_ts_pred - sum_ts_true) / 60.0,
                "abs_gap_min": np.abs(gap) / 60.0,
                "dam_lev_similarity": _np(d["dam_lev_similarity"]),
                "pref_len": _np(d["pref_len"]),
                "suf_len": _np(d["suf_len"]),
            })
            df.insert(0, "sweep", name)
            df["run_dir"] = str(resolve_run_dir(cfg))
            for k, v in _cfg_cols(cfg).items():
                df[k] = v
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ax2_frame(names, root="."):
    """Long per-instance ax2 table over the non-leaky subset."""
    frames = []
    for name in names:
        if name not in SWEEPS:
            raise KeyError(f"unknown sweep {name!r}; have {sorted(SWEEPS)}")
        for cfg in configs_for(SWEEPS[name]):
            d = _load_dump(cfg, root)
            if d is None or "ax2_head_logits" not in d:
                continue
            logits = _np(d["ax2_head_logits"])               # (N, C)
            # Softmax max = head confidence; argmax = head prediction.
            z = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
            head_pred = probs.argmax(axis=1)
            head_conf = probs.max(axis=1)
            implied = _np(d["ax2_implied"])
            has_det = _np(d["ax2_has_det"]).astype(bool)
            labels = _np(d["ax2_outcome_labels"])
            df = pd.DataFrame({
                "head_pred": head_pred,
                "head_conf": head_conf,
                "implied": implied,
                "has_det": has_det,
                "label": labels,
                "head_wrong": (head_pred != labels).astype(int),
                # Suffix route scored on the same denominator: no determining act
                # counts as wrong (it failed to produce an answer).
                "suffix_wrong": (~((implied == labels) & has_det)).astype(int),
                "disagree": (head_pred != implied).astype(int),
            })
            df.insert(0, "sweep", name)
            df["run_dir"] = str(resolve_run_dir(cfg))
            for k, v in _cfg_cols(cfg).items():
                df[k] = v
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _decile_lift(incons, err):
    """Mean error in the top vs bottom inconsistency decile (nan if too few)."""
    if len(incons) < _N_DECILES * 2:
        return np.nan, np.nan, np.nan
    q = pd.qcut(pd.Series(incons).rank(method="first"), _N_DECILES, labels=False)
    err = pd.Series(err).values
    bottom = err[q == 0].mean()
    top = err[q == _N_DECILES - 1].mean()
    lift = (top / bottom) if bottom not in (0, np.nan) and not np.isnan(bottom) else np.nan
    return bottom, top, lift


def reliability_summary(names, root="."):
    """One row per run: the headline inconsistency-vs-error readouts per axiom.

    ax1 -- does the between-route gap predict the rrt-head error?
      pearson/spearman(|gap|, |err_rt|), and the error lift from the lowest to
      the highest |gap| decile (>1 means more error where the routes disagree).
    ax2 -- does head/suffix disagreement mark head unreliability?
      head accuracy overall and conditional on agree/disagree, the accuracy drop,
      the disagreement rate, and spearman((1-confidence), head_wrong).
    """
    a1 = ax1_frame(names, root)
    a2 = ax2_frame(names, root)
    rows = {}

    def _row(df_row):
        r = {"sweep": df_row["sweep"], "run_dir": df_row["run_dir"]}
        for k in ("seed", "lambda_ltn", "detach_mode", "axiom1_impl",
                  "lambda_ltn_outcome", "detach_mode_outcome", "axiom2_impl"):
            r[k] = df_row.get(k)
        return r

    for run_dir, g in a1.groupby("run_dir") if len(a1) else []:
        r = rows.setdefault(run_dir, _row(g.iloc[0]))
        incons, err = g["abs_gap_s"], g["abs_err_rt_min"]
        r["ax1_n"] = len(g)
        r["ax1_pearson_gap_errrt"] = _corr(incons, err)
        r["ax1_spearman_gap_errrt"] = _corr(incons, err, method="spearman")
        r["ax1_spearman_gap_errts"] = _corr(g["abs_gap_s"], g["abs_err_ts_min"], method="spearman")
        lo, hi, lift = _decile_lift(incons.values, err.values)
        r["ax1_err_lowdecile_min"], r["ax1_err_topdecile_min"], r["ax1_err_decile_lift"] = lo, hi, lift

    for run_dir, g in a2.groupby("run_dir") if len(a2) else []:
        r = rows.setdefault(run_dir, _row(g.iloc[0]))
        agree, dis = g[g["disagree"] == 0], g[g["disagree"] == 1]
        r["ax2_n"] = len(g)
        r["ax2_disagreement_rate"] = g["disagree"].mean()
        r["ax2_head_acc"] = 1.0 - g["head_wrong"].mean()
        r["ax2_head_acc_agree"] = (1.0 - agree["head_wrong"].mean()) if len(agree) else np.nan
        r["ax2_head_acc_disagree"] = (1.0 - dis["head_wrong"].mean()) if len(dis) else np.nan
        r["ax2_acc_drop_on_disagree"] = r["ax2_head_acc_agree"] - r["ax2_head_acc_disagree"]
        # Low confidence should track head error if confidence is a reliability cue.
        r["ax2_spearman_unconf_wrong"] = _corr(
            1.0 - g["head_conf"], g["head_wrong"].astype(float), method="spearman")

    return pd.DataFrame(list(rows.values()))


def main():
    p = argparse.ArgumentParser(description="Per-instance inconsistency-vs-error analysis.")
    p.add_argument("sweeps", nargs="+", help="sweep name(s) from experiments/sweeps.py")
    p.add_argument("--root", default=".", help="results root (default: .)")
    p.add_argument("--out-ax1", default=None, help="write the ax1 per-instance long table here")
    p.add_argument("--out-ax2", default=None, help="write the ax2 per-instance long table here")
    p.add_argument("--summary", default="reliability_summary.csv",
                   help="write the per-run reliability summary here")
    args = p.parse_args()

    summ = reliability_summary(args.sweeps, root=args.root)
    summ.to_csv(args.summary, index=False)
    print(f"reliability summary: {len(summ)} runs -> {args.summary}")
    if args.out_ax1:
        a1 = ax1_frame(args.sweeps, root=args.root)
        a1.to_csv(args.out_ax1, index=False)
        print(f"ax1 per-instance: {len(a1)} rows -> {args.out_ax1}")
    if args.out_ax2:
        a2 = ax2_frame(args.sweeps, root=args.root)
        a2.to_csv(args.out_ax2, index=False)
        print(f"ax2 per-instance: {len(a2)} rows -> {args.out_ax2}")
    if summ.empty:
        print("  (no runs had consistency_per_instance.pt -- rerun inference "
              "after redeploying, or check the sweep names)")


if __name__ == "__main__":
    main()
