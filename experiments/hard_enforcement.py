"""Hard enforcement of the cross-task relations on the stored predictions of runs.

    python -m experiments.hard_enforcement <sweep> [<sweep> ...] [--splits VAL TEST] \\
        [--time-keep rrt_head|ttne_sum] [--outcome-keep outcome_head|suffix_implied] \\
        [--root .]

For every run of the sweeps and every split whose predictions were stored
(``store_preds``), applies `hard_enforcement` and scores the predictions before
and after with `SuTraN.inference_procedure.evaluate_predictions`, into

    <SPLIT>_HARD[_<overrides>]/raw/        the stored predictions, re-scored
    <SPLIT>_HARD[_<overrides>]/enforced/   the enforced predictions
    <SPLIT>_HARD[_<overrides>]/enforcement.json

Each folder holds ``averaged_results_{IB,CB}.pkl`` and the consistency
diagnostics, as the split folder of the run does. The side each relation keeps
is chosen per seed, as in two-phase training: the side that is better
(`phase_resume.direction_from_diagnostics`) on the validation diagnostics of
the seed's baseline at the relation's switch checkpoint (the end of rapid
learning, `phase_resume.rapid_learning_end`; evaluated by the
``baseline_rapid_*_eval`` sweeps). Every run of a seed (baseline, control or
weighted) thus keeps the side its soft enforcement freezes.
``--time-keep`` / ``--outcome-keep`` override it. Where the outcome predictions
are replaced by classes, the cross-entropy is left out.

The re-scored test predictions are compared with the run's own test results,
which they must reproduce.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle

import pandas as pd
import torch

from experiments.collect_results import _find_backup_dir
from experiments.grid import configs_for, resolve_run_dir
from experiments.sweeps import SWEEPS
from hard_enforcement import (outcome_to_suffix, rrt_to_ttne, suffix_to_outcome,
                              ttne_to_rrt)
from outcome_consistency_metrics import resolve_determining_ids
from phase_resume import direction_from_diagnostics, rapid_learning_end, rapid_metrics
from SuTraN.inference_procedure import evaluate_predictions
from TRAIN_EVAL_FUNCTIONALITY import log_configs
from Utils.subset import subset_split_by_case

SPLITS = {"VAL": ("VAL_SET_RESULTS", "val"), "TEST": ("TEST_SET_RESULTS", "test")}
TIME_SIDES = ("rrt_head", "ttne_sum")
OUTCOME_SIDES = ("outcome_head", "suffix_implied")
# Relation key of `direction_from_diagnostics` -> name of its switch point.
_SWITCH = {"axiom1": "time", "axiom2": "outcome"}
# Config keys that identify a run's phase-1 baseline (all else at its default).
_BASELINE_KEYS = ("log_name", "seed", "subset_fraction", "batch_size", "out_string")
_MC_KEYS = {"accuracy": "Multi-Class Accuracy", "macro_f1": "Macro-F1",
            "weighted_f1": "Weighted-F1", "macro_precision": "Macro-Precision",
            "weighted_precision": "Weighted-Precision", "macro_recall": "Macro-Recall",
            "weighted_recall": "Weighted-Recall"}
_BIN_KEYS = {"accuracy": "Binary Accuracy (th 0.5)", "f1": "Binary F1 (th 0.5)",
             "precision": "Binary Precision (th 0.5)", "recall": "Binary Recall (th 0.5)",
             "balanced_accuracy": "Binary Bal. Accuracy (th 0.5)"}


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_log_data(log_name, root="."):
    """The settings and per-split tensors of a preprocessed log."""
    data_path = os.path.join(root, log_name)

    def _dict(suffix):
        return _load_pickle(os.path.join(data_path, f"{log_name}_{suffix}.pkl"))

    means, stds = _dict("train_means_dict"), _dict("train_std_dict")
    outcome_bool = log_configs.outcome_bools_dict[log_name]
    out_mask = outcome_bool and log_configs.out_masks_dict[log_name]
    data = {
        "outcome_bool": outcome_bool,
        "out_mask": out_mask,
        "out_type": log_configs.out_types_dict[log_name],
        "num_outclasses": log_configs.num_outclasses_dict[log_name],
        "num_categoricals_pref": len(_dict("cat_cols_dict")["prefix_df"]),
        "mean_std_ttne": [means["timeLabel_df"][0], stds["timeLabel_df"][0]],
        "mean_std_tsp": [means["suffix_df"][1], stds["suffix_df"][1]],
        "mean_std_tss": [means["suffix_df"][0], stds["suffix_df"][0]],
        "mean_std_rrt": [means["timeLabel_df"][1], stds["timeLabel_df"][1]],
        "determining_ids": None,
        "end_token": _dict("cardin_dict")["concept:name"] + 1,
    }
    det_names = log_configs.outcome_determining_activities_dict.get(log_name)
    if outcome_bool and det_names:
        categ_mapping = _dict("categ_mapping")["concept:name"]
        data["determining_ids"], data["end_token"] = resolve_determining_ids(
            categ_mapping, det_names)
    for split in ("val", "test"):
        data[split] = {
            "dataset": torch.load(os.path.join(data_path, f"{split}_tensordataset.pt")),
            "og_caseint": torch.load(os.path.join(data_path, f"og_caseint_{split}.pt")),
            "instance_mask_out": (torch.load(os.path.join(data_path, f"instance_mask_out_{split}.pt"))
                                  if out_mask else None),
        }
    return data


def split_data(data, split, cfg):
    """The tensors of a split as the run evaluated them (validation subset)."""
    d = data[split]
    fraction = cfg.get("val_subset_fraction", 1.0) if split == "val" else 1.0
    dataset, og_caseint, keep, _ = subset_split_by_case(
        d["dataset"], d["og_caseint"], fraction, cfg["seed"] + 2)
    mask = d["instance_mask_out"]
    return dataset, og_caseint, (mask[keep] if mask is not None else None)


def load_predictions(split_dir, remaining_runtime_head=True, outcome_bool=True):
    """The stored predictions of a split, or None when they were not stored."""
    def _load(name):
        path = os.path.join(split_dir, name)
        return torch.load(path) if os.path.exists(path) else None

    preds = {"suffix_acts_decoded": _load("suffix_acts_decoded.pt"),
             "suffix_ttne_preds": _load("suffix_ttne_preds.pt"),
             "rrt_pred": _load("rrt_pred.pt") if remaining_runtime_head else None,
             "out_pred": _load("out_pred.pt") if outcome_bool else None}
    required = ["suffix_acts_decoded", "suffix_ttne_preds"]
    required += ["rrt_pred"] if remaining_runtime_head else []
    required += ["out_pred"] if outcome_bool else []
    if any(preds[k] is None for k in required):
        return None
    return preds


def averaged_results(results_IB, results_CB, remaining_runtime_head, out_type):
    """The ``averaged_results_{IB,CB}`` dicts from the lists `evaluate_predictions`
    returns, with the keys of the run's own result files."""
    dicts = []
    for results in (results_IB, results_CB):
        out = {"MAE TTNE minutes": results[1], "DL sim": results[2]}
        rest = results[3:]
        if remaining_runtime_head:
            out["MAE RRT minutes"] = rest[1]
            rest = rest[2:]
        if out_type == "multiclass_outcome":
            out["CE"] = rest[0]
            out.update({name: rest[1][key] for key, name in _MC_KEYS.items()})
        elif out_type == "binary_outcome":
            out.update({"BCE": rest[0], "AUC-ROC": rest[1], "AUC-PR": rest[2]})
            out.update({name: rest[3][key] for key, name in _BIN_KEYS.items()})
        dicts.append(out)
    return dicts


def score(preds, data, split_tensors, results_path, store_preds=False):
    """Evaluate predictions; write the result files into `results_path`."""
    dataset, og_caseint, instance_mask_out = split_tensors
    os.makedirs(results_path, exist_ok=True)
    results_IB, results_CB, pref_suf = evaluate_predictions(
        **preds,
        inference_dataset=dataset,
        remaining_runtime_head=True,
        outcome_bool=data["outcome_bool"],
        out_mask=data["out_mask"],
        out_type=data["out_type"],
        num_outclasses=data["num_outclasses"],
        num_categoricals_pref=data["num_categoricals_pref"],
        mean_std_ttne=data["mean_std_ttne"],
        mean_std_tsp=data["mean_std_tsp"],
        mean_std_tss=data["mean_std_tss"],
        mean_std_rrt=data["mean_std_rrt"],
        og_caseint=og_caseint,
        instance_mask_out=instance_mask_out,
        results_path=results_path,
        outcome_determining_ids=data["determining_ids"],
        store_preds=store_preds)
    return averaged_results(results_IB, results_CB, True, data["out_type"]), pref_suf


def enforce(preds, data, eligible, time_keep, outcome_keep):
    """Enforced predictions and how many instances each relation changed."""
    preds = dict(preds)
    acts, end = preds["suffix_acts_decoded"], data["end_token"]
    counts = {"num_instances": int(acts.shape[0])}

    # The outcome relation first: when the suffix gives way, the time relation
    # is then enforced on the edited suffix (the edit keeps its length).
    if outcome_keep == "outcome_head":
        preds["suffix_acts_decoded"], changed = suffix_to_outcome(
            acts, preds["out_pred"], data["determining_ids"], end, eligible)
        counts["outcome_changed"] = int(changed.sum())
    elif outcome_keep == "suffix_implied":
        out_class, changed = outcome_to_suffix(
            preds["out_pred"], acts, data["determining_ids"], end, eligible)
        preds["out_pred"] = torch.nn.functional.one_hot(
            out_class, num_classes=preds["out_pred"].shape[-1]).float()
        counts["outcome_changed"] = int(changed.sum())

    if time_keep == "rrt_head":
        preds["suffix_ttne_preds"], preds["rrt_pred"] = ttne_to_rrt(
            preds["suffix_ttne_preds"], preds["rrt_pred"], preds["suffix_acts_decoded"], end,
            data["mean_std_ttne"], data["mean_std_rrt"])
    elif time_keep == "ttne_sum":
        preds["rrt_pred"] = rrt_to_ttne(
            preds["suffix_ttne_preds"], preds["suffix_acts_decoded"], end,
            data["mean_std_ttne"], data["mean_std_rrt"])
    return preds, counts


def _max_rel_diff(a, b):
    """Largest relative difference over the numeric entries both dicts share."""
    worst = 0.0
    for k in set(a) & set(b):
        x, y = a[k], b[k]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) \
                and math.isfinite(x) and math.isfinite(y):
            worst = max(worst, abs(x - y) / max(abs(y), 1e-12))
    return worst


def _phase1_tasks(log_name):
    """The targets of a log's baseline, as two-phase training lists them."""
    tasks = ["activity_suffix", "timestamp_suffix", "remaining_runtime"]
    if log_configs.outcome_bools_dict[log_name]:
        tasks.append(log_configs.out_types_dict[log_name])
    return tasks


def switch_direction(cfg, relation, root="."):
    """The kept side of `relation` for the seed of `cfg`, and its switch epoch.

    The side that is better on the validation diagnostics of the seed's
    baseline at the end of rapid learning on the relation's heads (the
    checkpoint two-phase training switches at).
    """
    base = {k: cfg[k] for k in _BASELINE_KEYS if k in cfg}
    backup_dir = _find_backup_dir(os.path.join(root, str(resolve_run_dir(base))))
    if backup_dir is None:
        raise FileNotFoundError(f"No baseline run for {base}.")
    log_name = cfg["log_name"].upper()
    metrics = rapid_metrics(_SWITCH[relation], _phase1_tasks(log_name))
    epoch = rapid_learning_end(pd.read_csv(os.path.join(backup_dir, "backup_results.csv")), metrics)
    path = os.path.join(backup_dir, f"VAL_SET_RESULTS_EPOCH_{epoch}", "consistency_diagnostics.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing; evaluate the checkpoint with the "
            f"baseline_rapid_{_SWITCH[relation]}_eval sweep of {log_name}.")
    choice = direction_from_diagnostics(_load_pickle(path)).get(relation)
    if choice is None or choice["better"] is None:
        raise ValueError(
            f"{log_name} seed {cfg['seed']} {_SWITCH[relation]}: no direction at epoch "
            f"{epoch} (a tie or no diagnostics); pass --{_SWITCH[relation]}-keep.")
    return choice["better"], epoch


def run(names, splits=("VAL", "TEST"), time_keep=None, outcome_keep=None, root="."):
    tag = "_".join(f"{rel}_{side}" for rel, side in (("time", time_keep), ("outcome", outcome_keep))
                   if side)
    cache = {}
    for name in names:
        if name not in SWEEPS:
            raise KeyError(f"unknown sweep {name!r}; have {sorted(SWEEPS)}")
        for cfg in configs_for(SWEEPS[name]):
            log_name = cfg["log_name"].upper()
            backup_dir = _find_backup_dir(os.path.join(root, str(resolve_run_dir(cfg))))
            if backup_dir is None:
                print(f"  no run: {resolve_run_dir(cfg)}")
                continue
            if log_name not in cache:
                cache = {log_name: load_log_data(log_name, root)}   # one log in memory
            data = cache[log_name]
            switch_epochs = {}
            time_side = time_keep
            if time_side is None:
                time_side, switch_epochs["time"] = switch_direction(cfg, "axiom1", root)
            outcome_side = None
            if data["determining_ids"]:
                outcome_side = outcome_keep
                if outcome_side is None:
                    outcome_side, switch_epochs["outcome"] = switch_direction(cfg, "axiom2", root)

            for split in splits:
                folder, key = SPLITS[split]
                preds = load_predictions(os.path.join(backup_dir, folder),
                                         outcome_bool=data["outcome_bool"])
                if preds is None:
                    print(f"  no stored predictions: {backup_dir}/{folder}")
                    continue
                tensors = split_data(data, key, cfg)
                mask = tensors[2]
                eligible = (~mask if mask is not None
                            else torch.ones(preds["suffix_acts_decoded"].shape[0], dtype=torch.bool))
                out_dir = os.path.join(backup_dir, f"{folder}_HARD" + (f"_{tag}" if tag else ""))

                (raw_IB, raw_CB), _ = score(preds, data, tensors, os.path.join(out_dir, "raw"))
                enforced, counts = enforce(preds, data, eligible, time_side, outcome_side)
                (enf_IB, enf_CB), pref_suf = score(enforced, data, tensors,
                                                   os.path.join(out_dir, "enforced"),
                                                   store_preds=True)
                if outcome_side == "suffix_implied":
                    enf_IB.pop("CE", None)
                    enf_CB.pop("CE", None)

                record = {"time_keep": time_side, "outcome_keep": outcome_side,
                          "direction_switch_epochs": switch_epochs, **counts}
                original = os.path.join(backup_dir, folder, "averaged_results_CB.pkl")
                checks = {"raw_vs_run_diagnostics": _max_rel_diff(
                    _load_pickle(os.path.join(out_dir, "raw", "consistency_diagnostics.pkl")),
                    _load_pickle(os.path.join(backup_dir, folder, "consistency_diagnostics.pkl")))}
                if os.path.exists(original):
                    checks["raw_vs_run_averaged_CB"] = _max_rel_diff(raw_CB, _load_pickle(original))
                record["max_rel_diff_to_run"] = checks

                for sub, ib, cb in (("raw", raw_IB, raw_CB), ("enforced", enf_IB, enf_CB)):
                    for basis, d in (("IB", ib), ("CB", cb)):
                        with open(os.path.join(out_dir, sub, f"averaged_results_{basis}.pkl"), "wb") as f:
                            pickle.dump(d, f)
                with open(os.path.join(out_dir, "enforcement.json"), "w") as f:
                    json.dump(record, f, indent=2)
                print(f"{log_name} seed {cfg['seed']} {split}: {record}")


def main():
    p = argparse.ArgumentParser(description="Hard enforcement on stored predictions.")
    p.add_argument("sweeps", nargs="+", help="sweep name(s) from experiments/sweeps.py")
    p.add_argument("--splits", nargs="+", default=["VAL", "TEST"], choices=sorted(SPLITS))
    p.add_argument("--time-keep", choices=TIME_SIDES, default=None,
                   help="override the kept side of the time relation")
    p.add_argument("--outcome-keep", choices=OUTCOME_SIDES, default=None,
                   help="override the kept side of the outcome relation")
    p.add_argument("--root", default=".", help="results root (default: .)")
    args = p.parse_args()
    run(args.sweeps, splits=args.splits, time_keep=args.time_keep,
        outcome_keep=args.outcome_keep, root=args.root)


if __name__ == "__main__":
    main()
