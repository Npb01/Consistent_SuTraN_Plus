"""Can cross-task inconsistency flag wrong predictions at inference?

    python -m experiments.consistency_per_instance <sweep> [<sweep> ...] \\
        [--root .] [--out-dir .]

For every run of the sweeps, loads the per-instance dumps of the validation and
test sets (`*_SET_RESULTS/consistency_per_instance.pt`, written by
`SuTraN.inference_procedure`) and scores how well the inconsistency between two
routes to the same quantity detects the errors of the predictions. Both
relations are scored on every run:

  time     inconsistency |gap| = |summed TTNE - RRT|; errors |RRT error| and
           |summed TTNE error|.
  outcome  inconsistency between the outcome head and the outcome implied by
           the decoded activity suffix: binary disagreement and the graded
           1 - p_head(implied class); error: head wrong. The head's
           unconfidence (1 - max softmax) is the certainty reference.

Both relations are also scored against the activity-suffix quality, 1 - DLS.

Runs are the phase-1 baseline (trained relation "none"), the two-phase
lambda=0 control of a relation and the weight `SELECTED_LAMBDAS` picks for it;
other weights are skipped.

Inconsistency is only defined where both routes give an answer: a predicted
END for time (for dumps without the END flag, a predicted length below the
window, as in `experiments.time_gap`), a determining activity in the decoded
suffix for outcome. The
detection metrics are computed on that defined subset ("own"); its coverage and
the error on the undefined rest are reported as a separate flag. Because the
defined subset differs between runs, the control and the selected run of a
seed are also scored on the instances defined in both ("common").

All statistics are case-based: an instance weighs the inverse of the number of
instances of its case in the split, also within a subset, so a subset mean
times its coverage plus the rest's mean times the rest's share gives the
split's case-based mean. Thresholds and fitted combinations are set on the
validation set and scored on the test set. Readouts:

  spearman       weighted rank correlation; the ranks are mid-ranks of the
                 cumulative case weight.
  AUROC          the score as a ranking for a binary error (head wrong; time:
                 |error| above the validation 75th percentile).
  aurc_gain      the area under the risk-coverage curve (abstaining on the
                 highest scores first), as the fraction of the gap between
                 random abstention (0) and abstention by the true error (1).
  flag lift      error of the instances above the validation 90th percentile
                 of the score, over the error of the rest.
  incremental    for time, a logistic model on features available at
                 inference (predicted RRT, prefix length, predicted suffix
                 length) with and without |gap|; for outcome, on unconfidence
                 with and without the inconsistency scores.

Writes, into --out-dir, one row per run, scored relation and subset
(flagging_runs.csv); the mean and SD over seeds with the paired difference to
the lambda=0 control (flagging_summary.csv); and the outcome accuracy per
confidence x agreement cell (flagging_cells.csv).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from experiments.collect_results import _find_backup_dir
from experiments.grid import configs_for, resolve_run_dir
from experiments.hard_enforcement import load_log_data, split_data
from experiments.sweeps import SELECTED_LAMBDAS, SWEEPS
from experiments.time_gap import has_pred_end, window_size

SPLITS = {"VAL": ("VAL_SET_RESULTS", "val"), "TEST": ("TEST_SET_RESULTS", "test")}
TRAINED_RELATIONS = {"rapid_time": "time", "rapid_outcome": "outcome", None: "none"}
LAMBDA_KEYS = {"time": "lambda_ltn", "outcome": "lambda_ltn_outcome"}
LARGE_ERROR_QUANTILE = 0.75
FLAG_QUANTILE = 0.90
N_RRT_STRATA = 10
_SECONDS_PER_DAY = 86400.0


# --- case-based statistics -------------------------------------------------

def case_weights(case_ids):
    """Inverse of the number of instances of each instance's case."""
    _, inverse, counts = np.unique(np.asarray(case_ids), return_inverse=True,
                                   return_counts=True)
    return 1.0 / counts[inverse]


def wmean(x, w):
    w = np.asarray(w, float)
    return float(np.sum(np.asarray(x, float) * w) / w.sum()) if w.sum() > 0 else np.nan


def wquantile(x, w, q):
    order = np.argsort(x)
    x, w = np.asarray(x, float)[order], np.asarray(w, float)[order]
    cum = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), cum, x))


def wrank(x, w):
    """Mid-rank of each value in cumulative weight; tied values share one."""
    df = pd.DataFrame({"x": np.asarray(x, float), "w": np.asarray(w, float)})
    mass = df.groupby("x")["w"].sum().sort_index()
    mid = mass.cumsum() - 0.5 * mass
    return df["x"].map(mid).to_numpy(float)


def wspearman(a, b, w):
    """Weighted Pearson correlation of the weighted ranks."""
    w = np.asarray(w, float)
    if len(w) < 3:
        return np.nan
    ra, rb = wrank(a, w), wrank(b, w)
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return np.nan
    ma, mb = wmean(ra, w), wmean(rb, w)
    cov = wmean((ra - ma) * (rb - mb), w)
    return float(cov / np.sqrt(wmean((ra - ma) ** 2, w) * wmean((rb - mb) ** 2, w)))


def wauroc(score, label, w):
    label = np.asarray(label).astype(int)
    if len(label) == 0 or label.min() == label.max():
        return np.nan
    return float(roc_auc_score(label, score, sample_weight=w))


def _aurc(score, err, w):
    """Area under the weighted risk-coverage curve when the instances are
    retained from the lowest score up. Tied scores are retained in random
    order, so their errors are replaced by the block's mean (the expected
    curve)."""
    df = pd.DataFrame({"s": np.asarray(score, float), "e": np.asarray(err, float),
                       "w": np.asarray(w, float)})
    df["we"] = df["e"] * df["w"]
    blocks = df.groupby("s")[["we", "w"]].transform("sum")
    df["e"] = blocks["we"] / blocks["w"]
    df = df.sort_values("s", kind="stable")
    w_sorted = df["w"].to_numpy()
    risk = np.cumsum(df["e"].to_numpy() * w_sorted) / np.cumsum(w_sorted)
    return float(np.sum(risk * w_sorted) / w_sorted.sum())


def aurc_gain(score, err, w):
    """Fraction of the AURC gap between random abstention and the oracle."""
    if len(err) == 0:
        return np.nan
    full = wmean(err, w)
    oracle = _aurc(err, err, w)
    if full - oracle <= 0:
        return np.nan
    return float((full - _aurc(score, err, w)) / (full - oracle))


def flag_lift(score_val, w_val, score_test, err_test, w_test):
    """Test error of the flagged instances (score above the validation
    FLAG_QUANTILE) and of the rest, their ratio and the flagged share."""
    threshold = wquantile(score_val, w_val, FLAG_QUANTILE)
    flagged = np.asarray(score_test) > threshold
    share = wmean(flagged, w_test)
    if flagged.all() or not flagged.any():
        return np.nan, np.nan, np.nan, share
    e_f = wmean(err_test[flagged], w_test[flagged])
    e_r = wmean(err_test[~flagged], w_test[~flagged])
    return e_f, e_r, (e_f / e_r if e_r > 0 else np.nan), share


def fitted_auroc(x_val, y_val, w_val, x_test, y_test, w_test):
    """Test AUROC of a case-weighted logistic model fitted on validation."""
    y_val = np.asarray(y_val).astype(int)
    if len(y_val) == 0 or y_val.min() == y_val.max():
        return np.nan
    scaler = StandardScaler().fit(x_val)
    model = LogisticRegression(max_iter=1000)
    model.fit(scaler.transform(x_val), y_val, sample_weight=w_val)
    return wauroc(model.decision_function(scaler.transform(x_test)), y_test, w_test)


# --- per-instance frames ---------------------------------------------------

def _np(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


def _load_dump(cfg, split, root="."):
    backup_dir = _find_backup_dir(os.path.join(root, str(resolve_run_dir(cfg))))
    if backup_dir is None:
        return None
    path = os.path.join(backup_dir, SPLITS[split][0], "consistency_per_instance.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def time_frame(dump, case_ids, window):
    """One row per instance: the time-route gap, the errors and the
    inference-time features, in days."""
    rt_pred, rt_true = _np(dump["ax1_rt_pred"]), _np(dump["ax1_rt_true"])
    ts_pred, ts_true = _np(dump["ax1_sum_ts_pred"]), _np(dump["ax1_sum_ts_true"])
    return pd.DataFrame({
        "gap": np.abs(ts_pred - rt_pred) / _SECONDS_PER_DAY,
        "err_rt": np.abs(rt_pred - rt_true) / _SECONDS_PER_DAY,
        "err_ts": np.abs(ts_pred - ts_true) / _SECONDS_PER_DAY,
        "suffix_bad": 1.0 - _np(dump["dam_lev_similarity"]),
        "rt_pred": rt_pred / _SECONDS_PER_DAY,
        "pref_len": _np(dump["pref_len"]),
        "pred_len": _np(dump["pred_len"]),
        "defined": has_pred_end(dump, window),
        "w": case_weights(case_ids),
    })


def outcome_frame(dump, case_ids, window=None):
    """One row per non-leaky instance: head correctness, suffix quality,
    head/suffix inconsistency and head unconfidence."""
    retain = dump.get("ax2_retain_bool_out")
    dls = _np(dump["dam_lev_similarity"])
    if retain is not None:
        keep = _np(retain).astype(bool)
        case_ids, dls = np.asarray(case_ids)[keep], dls[keep]
    logits = _np(dump["ax2_head_logits"]).astype(float)
    z = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = z / z.sum(axis=1, keepdims=True)
    head = probs.argmax(axis=1)
    implied = _np(dump["ax2_implied"]).astype(int)
    defined = _np(dump["ax2_has_det"]).astype(bool)
    labels = _np(dump["ax2_outcome_labels"]).astype(int)
    p_implied = np.where(defined, probs[np.arange(len(probs)), np.clip(implied, 0, None)], np.nan)
    return pd.DataFrame({
        "head_wrong": (head != labels).astype(int),
        "suffix_wrong": (implied != labels).astype(int),
        "suffix_bad": 1.0 - dls,
        "disagree": (head != implied).astype(int),
        "graded": 1.0 - p_implied,
        "unconf": 1.0 - probs.max(axis=1),
        "defined": defined,
        "w": case_weights(case_ids),
    })


FRAMES = {"time": time_frame, "outcome": outcome_frame}


# --- detection metrics -----------------------------------------------------

def _undefined_flag(df, err_col):
    d, u = df[df["defined"]], df[~df["defined"]]
    return {
        "coverage": wmean(df["defined"], df["w"]),
        f"{err_col}_defined": wmean(d[err_col], d["w"]) if len(d) else np.nan,
        f"{err_col}_undefined": wmean(u[err_col], u["w"]) if len(u) else np.nan,
    }


def _suffix_quality(t, scores):
    """Rank correlation and AURC gain of each score against 1 - DLS."""
    r = {"suffix_bad_mean": wmean(t["suffix_bad"], t["w"])}
    for s in scores:
        r[f"suffix_bad_spearman_{s}"] = wspearman(t[s], t["suffix_bad"], t["w"])
        r[f"suffix_bad_aurc_gain_{s}"] = aurc_gain(t[s].to_numpy(), t["suffix_bad"].to_numpy(),
                                                   t["w"].to_numpy())
    return r


def time_metrics(val, test):
    """Detection of the time-route errors and of poor suffixes by |gap|, on
    the defined subset."""
    r = _undefined_flag(test, "err_rt")
    v, t = val[val["defined"]].copy(), test[test["defined"]].copy()
    features = ["log_rt_pred", "pref_len", "pred_len"]
    for df in (v, t):
        df["log_rt_pred"] = np.log1p(df["rt_pred"].clip(lower=0))
        df["log_gap"] = np.log1p(df["gap"])
    r["gap_mean"] = wmean(t["gap"], t["w"])
    for err in ("err_rt", "err_ts"):
        cut = wquantile(v[err], v["w"], LARGE_ERROR_QUANTILE)
        large_v, large_t = v[err] > cut, t[err] > cut
        e_t, w_t = t[err].to_numpy(), t["w"].to_numpy()
        r[f"{err}_mean"] = wmean(e_t, w_t)
        r[f"{err}_spearman_gap"] = wspearman(t["gap"], e_t, w_t)
        r[f"{err}_auroc_gap"] = wauroc(t["gap"], large_t, w_t)
        r[f"{err}_aurc_gain_gap"] = aurc_gain(t["gap"].to_numpy(), e_t, w_t)
        e_f, e_r, lift, share = flag_lift(v["gap"].to_numpy(), v["w"].to_numpy(),
                                          t["gap"].to_numpy(), e_t, w_t)
        r[f"{err}_flagged"], r[f"{err}_unflagged"], r[f"{err}_flag_lift"] = e_f, e_r, lift
        r["flag_share"] = share
        # Features available at inference without the second route.
        r[f"{err}_auroc_rt_pred"] = wauroc(t["rt_pred"], large_t, w_t)
        r[f"{err}_aurc_gain_rt_pred"] = aurc_gain(t["rt_pred"].to_numpy(), e_t, w_t)
        base = fitted_auroc(v[features], large_v, v["w"], t[features], large_t, w_t)
        full = fitted_auroc(v[features + ["log_gap"]], large_v, v["w"],
                            t[features + ["log_gap"]], large_t, w_t)
        r[f"{err}_auroc_features"], r[f"{err}_auroc_features_gap"] = base, full
        r[f"{err}_auroc_gap_increment"] = full - base
        # Rank correlation of |gap| and error within strata of the predicted RRT.
        strata = pd.qcut(t["rt_pred"].rank(method="first"), N_RRT_STRATA, labels=False)
        per_stratum = np.array([(wspearman(g["gap"], g[err], g["w"]), g["w"].sum())
                                for _, g in t.groupby(strata)])
        rho, mass = per_stratum[:, 0], per_stratum[:, 1]
        ok = ~np.isnan(rho)
        r[f"{err}_spearman_gap_within_rt_pred"] = (
            float(np.sum(rho[ok] * mass[ok]) / mass[ok].sum()) if ok.any() else np.nan)
    r.update(_suffix_quality(t, ["gap"]))
    return r, []


def outcome_metrics(val, test):
    """Detection of the head's errors and of poor suffixes by head/suffix
    inconsistency and by unconfidence on the defined subset, plus the
    per-cell accuracies."""
    r = _undefined_flag(test, "head_wrong")
    v, t = val[val["defined"]], test[test["defined"]]
    w, y = t["w"].to_numpy(), t["head_wrong"].to_numpy()
    agree, dis = t[t["disagree"] == 0], t[t["disagree"] == 1]
    r["disagree_rate"] = wmean(t["disagree"], w)
    r["head_acc"] = 1.0 - wmean(y, w)
    r["head_acc_agree"] = 1.0 - wmean(agree["head_wrong"], agree["w"]) if len(agree) else np.nan
    r["head_acc_disagree"] = 1.0 - wmean(dis["head_wrong"], dis["w"]) if len(dis) else np.nan
    r["disagree_precision"] = 1.0 - r["head_acc_disagree"]
    r["disagree_recall"] = wmean(t["disagree"][y == 1], w[y == 1]) if y.any() else np.nan
    for score in ("disagree", "graded", "unconf"):
        r[f"auroc_{score}"] = wauroc(t[score], y, w)
        r[f"aurc_gain_{score}"] = aurc_gain(t[score].to_numpy(), y, w)
    e_f, e_r, lift, share = flag_lift(v["graded"].to_numpy(), v["w"].to_numpy(),
                                      t["graded"].to_numpy(), y.astype(float), w)
    r["graded_flagged_err"], r["graded_unflagged_err"], r["graded_flag_lift"] = e_f, e_r, lift
    r["graded_flag_share"] = share
    # Which route the flag points at: the suffix-implied class.
    r["suffix_class_acc"] = 1.0 - wmean(t["suffix_wrong"], w)
    for score in ("disagree", "graded"):
        r[f"suffix_wrong_auroc_{score}"] = wauroc(t[score], t["suffix_wrong"], w)
    r.update(_suffix_quality(t, ["disagree", "graded"]))
    # Certainty and inconsistency: does unconfidence find the disagreements,
    # and does inconsistency add to unconfidence in finding the errors?
    r["auroc_unconf_for_disagree"] = wauroc(t["unconf"], t["disagree"], w)
    r["spearman_unconf_graded"] = wspearman(t["unconf"], t["graded"], w)
    both = ["unconf", "graded", "disagree"]
    r["auroc_unconf_fitted"] = fitted_auroc(v[["unconf"]], v["head_wrong"], v["w"],
                                            t[["unconf"]], y, w)
    r["auroc_combined"] = fitted_auroc(v[both], v["head_wrong"], v["w"], t[both], y, w)
    r["auroc_combined_increment"] = r["auroc_combined"] - r["auroc_unconf_fitted"]

    median_unconf = wquantile(v["unconf"], v["w"], 0.5)
    cells = []
    for low_conf in (False, True):
        for disagree in (0, 1):
            c = t[((t["unconf"] > median_unconf) == low_conf) & (t["disagree"] == disagree)]
            cells.append({"confidence": "low" if low_conf else "high",
                          "agreement": "disagree" if disagree else "agree",
                          "share": c["w"].sum() / w.sum(),
                          "head_acc": 1.0 - wmean(c["head_wrong"], c["w"]) if len(c) else np.nan})
    return r, cells


METRICS = {"time": time_metrics, "outcome": outcome_metrics}


# --- sweep driver ----------------------------------------------------------

def _run_info(cfg):
    """Trained relation and weight of a run, or None when it is neither a
    baseline, a control nor the selected weight."""
    trained = TRAINED_RELATIONS.get(cfg.get("phase1_epoch"))
    if trained is None:
        raise ValueError(f"no relation for phase1_epoch {cfg.get('phase1_epoch')!r}")
    if trained == "none":
        return trained, 0.0
    lam = float(cfg.get(LAMBDA_KEYS[trained], 0.0))
    if lam != 0.0 and SELECTED_LAMBDAS.get((cfg["log_name"], trained)) != lam:
        return None
    return trained, lam


def _run_frames(cfg, logs, root):
    """Validation and test frames of both relations, or None without dumps."""
    dumps = {s: _load_dump(cfg, s, root) for s in SPLITS}
    if any(d is None for d in dumps.values()):
        return None
    log = cfg["log_name"]
    if log not in logs:
        logs[log] = load_log_data(log, root)
    window = window_size(root, log)
    frames = {rel: {} for rel in FRAMES}
    for s, (_, split) in SPLITS.items():
        case_ids = _np(split_data(logs[log], split, cfg)[1])
        for rel, build in FRAMES.items():
            if rel == "outcome" and "ax2_head_logits" not in dumps[s]:
                continue
            frames[rel][s] = build(dumps[s], case_ids, window)
    return {rel: f for rel, f in frames.items() if f}


def _score(frames, meta, subset, rows, cell_rows):
    for rel, f in frames.items():
        metrics, cells = METRICS[rel](f["VAL"], f["TEST"])
        tag = {**meta, "scored_relation": rel, "subset": subset}
        rows.append({**tag, **metrics})
        cell_rows += [{**tag, **c} for c in cells]


def _common(frames_a, frames_b):
    """Both runs' frames restricted to the instances defined in both."""
    out_a, out_b = {}, {}
    for rel in frames_a.keys() & frames_b.keys():
        out_a[rel], out_b[rel] = {}, {}
        for s in SPLITS:
            both = frames_a[rel][s]["defined"].to_numpy() & frames_b[rel][s]["defined"].to_numpy()
            out_a[rel][s] = frames_a[rel][s].assign(defined=both)
            out_b[rel][s] = frames_b[rel][s].assign(defined=both)
    return out_a, out_b


def collect(names, root="."):
    """Per-run detection metrics and outcome cells of the sweeps' runs."""
    runs = {}
    for name in names:
        if name not in SWEEPS:
            raise KeyError(f"unknown sweep {name!r}; have {sorted(SWEEPS)}")
        for cfg in configs_for(SWEEPS[name]):
            info = _run_info(cfg)
            if info is None:
                continue
            trained, lam = info
            runs.setdefault((trained, cfg["log_name"], cfg["seed"]), {}).setdefault(lam, (name, cfg))

    rows, cell_rows, logs = [], [], {}
    for (trained, log, seed), by_lambda in sorted(runs.items()):
        frames = {}
        for lam, (name, cfg) in by_lambda.items():
            f = _run_frames(cfg, logs, root)
            if f is None:
                print(f"  skip (no dump): {resolve_run_dir(cfg)}")
                continue
            frames[lam] = f
            meta = {"sweep": name, "trained_relation": trained, "log_name": log,
                    "seed": seed, "lambda": lam, "run_dir": str(resolve_run_dir(cfg))}
            _score(f, meta, "own", rows, cell_rows)
        selected = [lam for lam in frames if lam != 0.0]
        if trained != "none" and 0.0 in frames and selected:
            control, chosen = _common(frames[0.0], frames[selected[0]])
            for lam, f in ((0.0, control), (selected[0], chosen)):
                name, cfg = by_lambda[lam]
                meta = {"sweep": name, "trained_relation": trained, "log_name": log,
                        "seed": seed, "lambda": lam, "run_dir": str(resolve_run_dir(cfg))}
                _score(f, meta, "common", rows, cell_rows)
    return pd.DataFrame(rows), pd.DataFrame(cell_rows)


_GROUP = ["trained_relation", "scored_relation", "subset", "log_name"]


def summarize(runs):
    """Mean and SD over seeds per trained relation, scored relation, subset,
    log and weight, with the mean and SD of the per-seed difference to the
    lambda=0 control."""
    if runs.empty:
        return runs
    metrics = [c for c in runs.columns
               if c not in _GROUP + ["lambda", "seed", "sweep", "run_dir"]
               and runs[c].dtype.kind == "f"]
    out = []
    for keys, g in runs.groupby(_GROUP + ["lambda"]):
        *group, lam = keys
        same = (runs[_GROUP] == pd.Series(dict(zip(_GROUP, group)))).all(axis=1)
        control = runs[same & (runs["lambda"] == 0.0)].set_index("seed")
        row = {**dict(zip(_GROUP, group)), "lambda": lam, "n_seeds": len(g)}
        for m in metrics:
            vals = g[m].dropna()
            row[f"{m}_mean"], row[f"{m}_sd"] = vals.mean(), vals.std()
            if lam != 0.0 and len(control):
                diff = (g.set_index("seed")[m] - control[m]).dropna()
                row[f"{m}_vs_control_mean"], row[f"{m}_vs_control_sd"] = diff.mean(), diff.std()
        out.append(row)
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(description="Inconsistency as an error flag at inference.")
    p.add_argument("sweeps", nargs="+", help="sweep name(s) from experiments/sweeps.py")
    p.add_argument("--root", default=".", help="results root (default: .)")
    p.add_argument("--out-dir", default=".", help="folder for the three CSVs (default: .)")
    args = p.parse_args()

    runs, cells = collect(args.sweeps, root=args.root)
    os.makedirs(args.out_dir, exist_ok=True)
    runs.to_csv(os.path.join(args.out_dir, "flagging_runs.csv"), index=False)
    summarize(runs).to_csv(os.path.join(args.out_dir, "flagging_summary.csv"), index=False)
    cells.to_csv(os.path.join(args.out_dir, "flagging_cells.csv"), index=False)
    print(f"{len(runs)} rows -> {args.out_dir}")


if __name__ == "__main__":
    main()
