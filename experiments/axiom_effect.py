"""Paired per-instance effect of a consistency term against its lambda=0 control.

    python -m experiments.axiom_effect <sweep> [<sweep> ...] [--root .] \
        [--out axiom_effect.csv] [--pairs-time FILE] [--pairs-outcome FILE]

Every weighted run is paired with the lambda=0 run of the same log, seed,
switch epoch and phase-2 learning-rate mode, and the two test-set dumps
(`TEST_SET_RESULTS/consistency_per_instance.pt`) are compared row by row. Both
runs score the same test prefixes, so each change is attributed to an
instance. All averages are case-based: a prefix is weighted by the inverse of
the number of prefixes of its case, as in the CB metrics.

Time relation (summed TTNE suffix vs RRT). The true TTNE suffix sums to the
true remaining time, so the gap equals the difference of the two route errors.
The gap is reported on all cases and, per run, on the cases whose predicted
suffix contains an END (without one the sum runs to the window length), with
the share without an END for both runs. The summary separates
  * which route moved: the mean change of the summed TTNE suffix (split into
    the steps before the true END and the steps after it, which the per-step
    TTNE metric does not score) and of the RRT prediction;
  * how concentrated the gap reduction is: the share of the total reduction
    carried by the top 1/10/25% of the case weight;
  * whether it moved toward the truth: quadrants of gap change x summed-TTNE
    error change, and the summed-TTNE error change split by whether the RRT
    head was the more accurate route in the control;
  * how much of each head metric's change comes from the top-10% instances.

Outcome relation (outcome head vs the outcome the suffix implies), on the
non-leaky instances. Each instance is in one of three states: agree, conflict
(a determining activity that implies another class) or no determining
activity. The summary gives the state transition matrix control -> run, which
side changed in the resolved conflicts, whether the resolved instances are
correct, and how much of the DL and head-accuracy change comes from the
instances whose state changed.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from CaLenDiR_Utils.weighted_metrics_utils import get_weight_tensor
from experiments.collect_results import CFG_DEFAULTS, _find_backup_dir
from experiments.grid import configs_for, resolve_run_dir
from experiments.sweeps import SWEEPS
from experiments.time_gap import has_pred_end, window_size

_SECONDS_PER_DAY = 86400.0
_TOP_SHARES = (0.01, 0.10, 0.25)
_BRIDGE_SHARE = 0.10
_STATES = ("agree", "conflict", "nodet")
_PAIR_KEYS = ("log_name", "seed", "phase1_epoch", "phase2_lr", "batch_size",
              "subset_fraction")


def _cfg(cfg):
    return {k: cfg.get(k, d) for k, d in CFG_DEFAULTS.items()}


def _relation(cfg):
    """'time', 'outcome', or None for a control (no consistency term)."""
    c = _cfg(cfg)
    if c["lambda_ltn"] > 0 and c["lambda_ltn_outcome"] > 0:
        raise ValueError(f"both relations weighted in {resolve_run_dir(cfg)}")
    if c["lambda_ltn"] > 0:
        return "time"
    if c["lambda_ltn_outcome"] > 0:
        return "outcome"
    return None


def pair_runs(names):
    """(relation, run cfg, control cfg) for every weighted run with a control."""
    cfgs = []
    for name in names:
        if name not in SWEEPS:
            raise KeyError(f"unknown sweep {name!r}; have {sorted(SWEEPS)}")
        cfgs.extend(configs_for(SWEEPS[name]))
    key = lambda c: tuple(_cfg(c)[k] for k in _PAIR_KEYS)
    controls = {key(c): c for c in cfgs if _relation(c) is None}
    pairs, unpaired = [], []
    for c in cfgs:
        rel = _relation(c)
        if rel is None:
            continue
        if key(c) in controls:
            pairs.append((rel, c, controls[key(c)]))
        else:
            unpaired.append(c)
    return pairs, unpaired


def _load_dump(cfg, root):
    top_dir = os.path.join(root, str(resolve_run_dir(cfg)))
    backup_dir = _find_backup_dir(top_dir)
    if backup_dir is None:
        return None
    path = os.path.join(backup_dir, "TEST_SET_RESULTS", "consistency_per_instance.pt")
    if not os.path.exists(path):
        return None
    d = torch.load(path, map_location="cpu", weights_only=False)
    return {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in d.items()}


_WEIGHT_CACHE = {}


def _case_weights(root, log_name, mask=None):
    """Per-instance CB weights over the test set (or its masked subset),
    normalised so that a weighted mean is a plain weighted sum."""
    cache_key = (root, log_name, None if mask is None else mask.tobytes())
    if cache_key not in _WEIGHT_CACHE:
        og_caseint = torch.load(os.path.join(root, log_name, "og_caseint_test.pt"), weights_only=True)
        if mask is not None:
            og_caseint = og_caseint[torch.as_tensor(mask, dtype=torch.bool)]
        weights, num_cases = get_weight_tensor(og_caseint)
        _WEIGHT_CACHE[cache_key] = weights.double().numpy() / num_cases
    return _WEIGHT_CACHE[cache_key]


def _top_share(reduction, w, q):
    """Share of the total weighted reduction carried by the instances with the
    largest reductions that together hold a fraction `q` of the case weight."""
    total = np.sum(w * reduction)
    if total == 0:
        return np.nan
    order = np.argsort(-reduction)
    cum_w = np.cumsum(w[order])
    top = order[cum_w <= q * cum_w[-1]]
    return np.sum(w[top] * reduction[top]) / total


def _top_mask(reduction, w, q):
    order = np.argsort(-reduction)
    cum_w = np.cumsum(w[order])
    mask = np.zeros(len(reduction), dtype=bool)
    mask[order[cum_w <= q * cum_w[-1]]] = True
    return mask


# --------------------------------------------------------------------- time

def time_pairs(ctrl, run, w, window):
    """One row per test prefix: control and run quantities in days."""
    day = lambda x: np.asarray(x, dtype=np.float64) / _SECONDS_PER_DAY
    truth = day(ctrl["ax1_rt_true"])
    df = pd.DataFrame({"weight": w, "pref_len": ctrl["pref_len"], "suf_len": ctrl["suf_len"],
                       "rt_true_d": truth})
    for tag, d in (("ctrl", ctrl), ("run", run)):
        ts, ts_within, rt = day(d["ax1_sum_ts_pred"]), day(d["ax1_sum_ts_pred_capped"]), day(d["ax1_rt_pred"])
        df[f"sum_ts_{tag}_d"] = ts
        df[f"sum_ts_within_{tag}_d"] = ts_within
        df[f"sum_ts_after_end_{tag}_d"] = ts - ts_within
        df[f"rt_{tag}_d"] = rt
        df[f"gap_{tag}_d"] = ts - rt
        df[f"gap_within_{tag}_d"] = ts_within - rt
        df[f"err_ts_{tag}_d"] = ts - day(d["ax1_sum_ts_true"])
        df[f"err_rt_{tag}_d"] = rt - truth
        df[f"pred_len_{tag}"] = d["pred_len"]
        df[f"has_end_{tag}"] = has_pred_end(d, window)
        df[f"dl_{tag}"] = d["dam_lev_similarity"]
    return df


def time_summary(p):
    w = p["weight"].values
    wm = lambda x: float(np.sum(w * np.asarray(x, dtype=np.float64)))
    s = {}
    for tag in ("ctrl", "run"):
        s[f"abs_gap_{tag}_d"] = wm(p[f"gap_{tag}_d"].abs())
        s[f"abs_gap_within_{tag}_d"] = wm(p[f"gap_within_{tag}_d"].abs())
        s[f"bias_sum_ts_{tag}_d"] = wm(p[f"err_ts_{tag}_d"])
        s[f"bias_rt_{tag}_d"] = wm(p[f"err_rt_{tag}_d"])
        s[f"mae_sum_ts_{tag}_d"] = wm(p[f"err_ts_{tag}_d"].abs())
        s[f"mae_rt_{tag}_d"] = wm(p[f"err_rt_{tag}_d"].abs())
        s[f"dl_{tag}"] = wm(p[f"dl_{tag}"])
        s[f"frac_overlong_{tag}"] = wm(p[f"pred_len_{tag}"] > p["suf_len"])
        # Each run on its own cases with a predicted END; the no-END share of
        # both runs shows whether that case set shifted.
        has_end = p[f"has_end_{tag}"].values
        s[f"frac_no_end_{tag}"] = wm(~has_end)
        if has_end.any():
            w_end = np.sum(w * has_end)
            s[f"abs_gap_defined_{tag}_d"] = wm(p[f"gap_{tag}_d"].abs() * has_end) / w_end
            s[f"mae_sum_ts_defined_{tag}_d"] = wm(p[f"err_ts_{tag}_d"].abs() * has_end) / w_end
    s["abs_gap_rel_change"] = s["abs_gap_run_d"] / s["abs_gap_ctrl_d"] - 1.0
    if "abs_gap_defined_ctrl_d" in s and "abs_gap_defined_run_d" in s:
        s["abs_gap_defined_rel_change"] = s["abs_gap_defined_run_d"] / s["abs_gap_defined_ctrl_d"] - 1.0

    # Which route moved.
    s["d_sum_ts_within_d"] = wm(p["sum_ts_within_run_d"] - p["sum_ts_within_ctrl_d"])
    s["d_sum_ts_after_end_d"] = wm(p["sum_ts_after_end_run_d"] - p["sum_ts_after_end_ctrl_d"])
    s["d_rt_d"] = wm(p["rt_run_d"] - p["rt_ctrl_d"])

    # Concentration of the gap reduction.
    reduction = (p["gap_ctrl_d"].abs() - p["gap_run_d"].abs()).values
    for q in _TOP_SHARES:
        s[f"reduction_share_top{int(q * 100)}"] = _top_share(reduction, w, q)
    s["frac_gap_up"] = wm(reduction < 0)

    # Toward the truth or toward the other route.
    gap_down = reduction > 0
    err_down = (p["err_ts_run_d"].abs() < p["err_ts_ctrl_d"].abs()).values
    for g_name, g in (("gap_down", gap_down), ("gap_up", ~gap_down)):
        for e_name, e in (("err_down", err_down), ("err_up", ~err_down)):
            s[f"frac_{g_name}_{e_name}"] = wm(g & e)
    d_abs_err_ts = (p["err_ts_run_d"].abs() - p["err_ts_ctrl_d"].abs()).values
    rt_better = (p["err_rt_ctrl_d"].abs() < p["err_ts_ctrl_d"].abs()).values
    s["frac_rt_better_ctrl"] = wm(rt_better)
    s["d_mae_sum_ts_from_rt_better_d"] = wm(d_abs_err_ts * rt_better)
    s["d_mae_sum_ts_from_rt_worse_d"] = wm(d_abs_err_ts * ~rt_better)

    # How much of each metric's change the top-10% instances carry.
    top = _top_mask(reduction, w, _BRIDGE_SHARE)
    deltas = {
        "mae_sum_ts_d": d_abs_err_ts,
        "mae_rt_d": (p["err_rt_run_d"].abs() - p["err_rt_ctrl_d"].abs()).values,
        "dl": (p["dl_run"] - p["dl_ctrl"]).values,
    }
    tag = f"top{int(_BRIDGE_SHARE * 100)}"
    for name, d in deltas.items():
        s[f"d_{name}_{tag}"] = wm(d * top)
        s[f"d_{name}_rest"] = wm(d * ~top)
    return s


# ------------------------------------------------------------------ outcome

def _states(implied, has_det, head_pred):
    return np.where(~has_det, "nodet", np.where(implied == head_pred, "agree", "conflict"))


def outcome_pairs(ctrl, run, w):
    """One row per non-leaky test prefix: control and run states."""
    mask = ctrl["ax2_retain_bool_out"]
    rows = np.ones(len(ctrl["dam_lev_similarity"]), dtype=bool) if mask is None else mask.astype(bool)
    df = pd.DataFrame({"weight": w, "label": ctrl["ax2_outcome_labels"],
                       "pref_len": ctrl["pref_len"][rows], "suf_len": ctrl["suf_len"][rows]})
    for tag, d in (("ctrl", ctrl), ("run", run)):
        head = d["ax2_head_logits"].argmax(axis=1)
        has_det = d["ax2_has_det"].astype(bool)
        df[f"head_{tag}"] = head
        df[f"implied_{tag}"] = d["ax2_implied"]
        df[f"has_det_{tag}"] = has_det
        df[f"state_{tag}"] = _states(d["ax2_implied"], has_det, head)
        df[f"dl_{tag}"] = d["dam_lev_similarity"][rows]
    if not (df["label"].values == run["ax2_outcome_labels"]).all():
        raise ValueError("control and run outcome labels are not row-aligned")
    return df


def outcome_summary(p):
    w = p["weight"].values
    wm = lambda x: float(np.sum(w * np.asarray(x, dtype=np.float64)))
    label = p["label"].values
    s = {}
    for tag in ("ctrl", "run"):
        st = p[f"state_{tag}"].values
        s[f"conflict_{tag}"] = wm(st == "conflict")
        s[f"nodet_{tag}"] = wm(st == "nodet")
        s[f"head_acc_{tag}"] = wm(p[f"head_{tag}"].values == label)
        s[f"suffix_acc_{tag}"] = wm(p[f"has_det_{tag}"].values & (p[f"implied_{tag}"].values == label))
        s[f"dl_{tag}"] = wm(p[f"dl_{tag}"])
    s["conflict_rel_change"] = (s["conflict_run"] / s["conflict_ctrl"] - 1.0
                                if s["conflict_ctrl"] > 0 else np.nan)

    st0, st1 = p["state_ctrl"].values, p["state_run"].values
    for a in _STATES:
        for b in _STATES:
            s[f"tr_{a}_{b}"] = wm((st0 == a) & (st1 == b))

    head_changed = p["head_ctrl"].values != p["head_run"].values
    suffix_changed = ((p["implied_ctrl"].values != p["implied_run"].values)
                      | (p["has_det_ctrl"].values != p["has_det_run"].values))
    s["frac_head_changed"] = wm(head_changed)
    s["frac_suffix_changed"] = wm(suffix_changed)

    # Resolved conflicts: which side moved, and are they right afterwards.
    to_agree = (st0 == "conflict") & (st1 == "agree")
    to_nodet = (st0 == "conflict") & (st1 == "nodet")
    cond = lambda x, m: wm(x & m) / wm(m) if wm(m) > 0 else np.nan
    s["to_agree_by_suffix"] = cond(suffix_changed & ~head_changed, to_agree)
    s["to_agree_by_head"] = cond(head_changed & ~suffix_changed, to_agree)
    s["to_agree_correct"] = cond(p["head_run"].values == label, to_agree)
    s["to_agree_suffix_was_correct"] = cond(p["implied_ctrl"].values == label, to_agree)
    s["to_agree_head_was_correct"] = cond(p["head_ctrl"].values == label, to_agree)
    s["to_nodet_head_correct"] = cond(p["head_run"].values == label, to_nodet)

    # How much of the DL and head-accuracy change the state changes carry.
    changed = st0 != st1
    d_dl = (p["dl_run"] - p["dl_ctrl"]).values
    d_acc = (p["head_run"].values == label).astype(float) - (p["head_ctrl"].values == label)
    s["frac_state_changed"] = wm(changed)
    s["d_dl_state_changed"] = wm(d_dl * changed)
    s["d_dl_state_same"] = wm(d_dl * ~changed)
    s["d_head_acc_state_changed"] = wm(d_acc * changed)
    s["d_head_acc_state_same"] = wm(d_acc * ~changed)
    return s


# --------------------------------------------------------------------- main

def axiom_effect(names, root="."):
    """Per-pair summary table, plus the per-instance tables of both relations."""
    pairs, unpaired = pair_runs(names)
    rows, per_instance, missing = [], {"time": [], "outcome": []}, []
    for rel, cfg, ctrl_cfg in pairs:
        ctrl, run = _load_dump(ctrl_cfg, root), _load_dump(cfg, root)
        if ctrl is None or run is None:
            missing.append(cfg)
            continue
        c = _cfg(cfg)
        if len(ctrl["pref_len"]) != len(run["pref_len"]) or not (
                (ctrl["suf_len"] == run["suf_len"]).all() and (ctrl["pref_len"] == run["pref_len"]).all()):
            raise ValueError(f"test rows not aligned: {resolve_run_dir(cfg)}")
        if rel == "time":
            p = time_pairs(ctrl, run, _case_weights(root, c["log_name"]),
                           window_size(root, c["log_name"]))
            s = time_summary(p)
            lam, impl, detach = c["lambda_ltn"], c["axiom1_impl"], c["detach_mode"]
        else:
            mask = ctrl["ax2_retain_bool_out"]
            p = outcome_pairs(ctrl, run, _case_weights(root, c["log_name"], mask))
            s = outcome_summary(p)
            lam, impl, detach = c["lambda_ltn_outcome"], c["axiom2_impl"], c["detach_mode_outcome"]
        meta = {"relation": rel, "log_name": c["log_name"], "seed": c["seed"], "lambda": lam,
                "impl": impl, "detach": detach, "phase1_epoch": c["phase1_epoch"],
                "phase2_lr": c["phase2_lr"], "run_dir": str(resolve_run_dir(cfg)),
                "ctrl_dir": str(resolve_run_dir(ctrl_cfg))}
        rows.append({**meta, **s})
        per_instance[rel].append(p.assign(**{k: meta[k] for k in
                                             ("log_name", "seed", "lambda", "impl")}))
    tables = {rel: pd.concat(f, ignore_index=True) if f else pd.DataFrame()
              for rel, f in per_instance.items()}
    return pd.DataFrame(rows), tables, unpaired, missing


def main():
    p = argparse.ArgumentParser(description="Paired effect of a consistency term vs its control.")
    p.add_argument("sweeps", nargs="+", help="sweep name(s) from experiments/sweeps.py; "
                   "include the sweep holding the lambda=0 controls")
    p.add_argument("--root", default=".", help="results root (default: .)")
    p.add_argument("--out", default="axiom_effect.csv", help="per-pair summary CSV")
    p.add_argument("--pairs-time", default=None, help="per-instance time-relation CSV")
    p.add_argument("--pairs-outcome", default=None, help="per-instance outcome-relation CSV")
    args = p.parse_args()

    summary, tables, unpaired, missing = axiom_effect(args.sweeps, root=args.root)
    summary.to_csv(args.out, index=False)
    print(f"{len(summary)} pairs -> {args.out}")
    for rel, path in (("time", args.pairs_time), ("outcome", args.pairs_outcome)):
        if path:
            df = tables[rel]
            df.to_csv(path, index=False)
            print(f"{rel} per-instance: {len(df)} rows -> {path}")
    if unpaired:
        print(f"{len(unpaired)} weighted runs without a lambda=0 control in these sweeps")
    if missing:
        print(f"{len(missing)} pairs skipped (a run or its dump is missing locally)")


if __name__ == "__main__":
    main()
