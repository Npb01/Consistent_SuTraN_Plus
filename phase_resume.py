"""Two-phase training: continuing a phase-1 run as phase 2.

Phase 1 is an already-trained run with the consistency terms off (the
baseline). Phase 2 starts from one of its checkpoints, switches a consistency
term on, and freezes the head that predicts better on the validation set, so
the constraint pulls the weaker prediction towards the stronger one.

This module holds the pieces of that procedure that do not need the training
loop: which phase-1 checkpoint to start from, the UW loss weights belonging to
it, and the direction rule.
"""

from __future__ import annotations

import glob
import os

import pandas as pd
import torch

from Utils.callback_selection import get_target_metrics_dict, select_best_epoch

PHASE2_LR_POLICIES = ("restart", "continue")

# Two-phase training switches at the first validation where the phase-1 run's
# early-stopping counter reaches this value: two consecutive validations
# without improvement in any tracked metric.
PLATEAU_COUNT = 2

# The validation metrics the training loop's early-stopping counter tracks, per
# task, with the direction that counts as an improvement. The counter resets
# when any of them beats its best so far.
_STOPPING_METRICS = {
    "activity_suffix": [("Activity suffix: 1-DL (validation)", "max")],
    "timestamp_suffix": [("TTNE - standardized MAE validation", "min")],
    "remaining_runtime": [("RRT - standardized MAE validation", "min")],
    "binary_outcome": [("Binary Outcome - BCE validation", "min"),
                       ("Binary Outcome - AUC-PR validation", "max"),
                       ("Binary Outcome - AUC-ROC validation", "max")],
    "multiclass_outcome": [("Multi-Class Outcome - CE validation", "min"),
                           ("Multi-Class Outcome - Macro-F1 score", "max"),
                           ("Multi-Class Outcome - Weighted-F1 score", "max")],
}


# The early switch: the end of rapid learning on the heads a relation connects.
# A validation is slow when every one of those heads improves its best value so
# far by less than RAPID_TOLERANCE (relative); the switch is at the validation
# closing RAPID_ROUNDS consecutive slow ones.
RAPID_TOLERANCE = 0.02
RAPID_ROUNDS = 2

# Per relation, the validation metric of each head it connects, with the
# direction that counts as an improvement. Tasks absent from a run are skipped.
RAPID_METRICS = {
    "time": {"timestamp_suffix": ("TTNE - standardized MAE validation", "min"),
             "remaining_runtime": ("RRT - standardized MAE validation", "min")},
    "outcome": {"activity_suffix": ("Activity suffix: 1-DL (validation)", "max"),
                "multiclass_outcome": ("Multi-Class Outcome - CE validation", "min"),
                "binary_outcome": ("Binary Outcome - BCE validation", "min")},
}

# The named phase-1 epochs; any other spec is an explicit epoch number.
PHASE1_EPOCH_SPECS = ("best", "plateau") + tuple(f"rapid_{r}" for r in RAPID_METRICS)


def stopping_counter(df, task_list):
    """The early-stopping counter after each validated epoch of a run.

    Replays the training loop's rule on its ``backup_results.csv``: the counter
    resets to 0 at a validation where any tracked metric strictly beats its
    best so far, and increases by 1 otherwise. A run stops once it reaches the
    patience.

    Parameters
    ----------
    df : pandas.DataFrame
        The run's ``backup_results.csv``, one row per validated epoch.
    task_list : list of str
        The run's targets, as passed to `get_target_metrics_dict`.

    Returns
    -------
    pandas.Series
        The counter per row, indexed like ``df``.
    """
    metrics = [m for task in task_list for m in _STOPPING_METRICS[task]]
    best = {col: None for col, _ in metrics}
    counts, count = [], 0
    for _, row in df.iterrows():
        improved = False
        for col, objective in metrics:
            value = row[col]
            if best[col] is None or (value < best[col] if objective == "min"
                                     else value > best[col]):
                best[col] = value
                improved = True
        count = 0 if improved else count + 1
        counts.append(count)
    return pd.Series(counts, index=df.index, name="stopping_counter")


def plateau_epoch(df, task_list, count):
    """The first validated epoch at which the stopping counter reaches `count`.

    Uses only the validation scores up to that epoch, so a run could apply it
    while training. Returns None when the counter never gets there.
    """
    counter = stopping_counter(df, task_list)
    hits = df.loc[counter >= count, "epoch"]
    return int(hits.iloc[0]) if len(hits) else None


def rapid_learning_end(df, metrics, tolerance=RAPID_TOLERANCE, rounds=RAPID_ROUNDS):
    """The first validated epoch that closes `rounds` consecutive slow validations.

    A validation is slow when none of `metrics` improves its best value so far
    by `tolerance` or more, relative to that best. The first validation sets
    the bests and is never slow. Uses only the validation scores up to that
    epoch, so a run could apply it while training.

    Parameters
    ----------
    df : pandas.DataFrame
        The run's ``backup_results.csv``, one row per validated epoch.
    metrics : list of (str, {"min", "max"})
        Column and improvement direction of each watched metric.
    tolerance : float
        Relative improvement below which a metric counts as not improving.
    rounds : int
        Consecutive slow validations needed.

    Returns
    -------
    int or None
        None when the run never gets there.
    """
    best = {col: None for col, _ in metrics}
    slow_streak = 0
    for _, row in df.iterrows():
        slow = all(best[col] is not None for col, _ in metrics)
        for col, objective in metrics:
            value, previous = row[col], best[col]
            if previous is None:
                best[col] = value
                continue
            gain = previous - value if objective == "min" else value - previous
            if gain > 0:
                best[col] = value
            if gain >= tolerance * abs(previous):
                slow = False
        slow_streak = slow_streak + 1 if slow else 0
        if slow_streak >= rounds:
            return int(row["epoch"])
    return None


def rapid_metrics(relation, task_list):
    """The watched metrics of `relation` ("time" or "outcome") for a run's tasks."""
    if relation not in RAPID_METRICS:
        raise ValueError(f"Unknown relation {relation!r}; have {sorted(RAPID_METRICS)}.")
    metrics = [m for task, m in RAPID_METRICS[relation].items() if task in task_list]
    if not metrics:
        raise ValueError(f"The run predicts none of the heads of the {relation} relation.")
    return metrics


def switch_rule(spec, task_list):
    """The settings that resolve a ``"rapid_<relation>"`` spec, for the run's record.

    Returns
    -------
    dict or None
        ``tolerance``, ``rounds`` and the watched ``metrics`` (column,
        direction); None for the other specs, which have no settings.
    """
    if not (isinstance(spec, str) and spec.startswith("rapid_")):
        return None
    metrics = rapid_metrics(spec[len("rapid_"):], task_list)
    return {"tolerance": RAPID_TOLERANCE, "rounds": RAPID_ROUNDS,
            "metrics": [list(m) for m in metrics]}


def phase1_checkpoint_epoch(phase1_backup_dir, spec, task_list):
    """The phase-1 epoch whose checkpoint phase 2 starts from.

    Parameters
    ----------
    phase1_backup_dir : str
        Folder of the phase-1 run holding ``backup_results.csv`` and the
        ``model_epoch_<N>.pt`` checkpoints.
    spec : {"best", "plateau", "rapid_time", "rapid_outcome"} or int
        ``"best"`` is the epoch the phase-1 run selects as its own final model
        (`select_best_epoch` over its validated epochs). ``"plateau"`` is the
        first validated epoch at which its early-stopping counter reaches
        `PLATEAU_COUNT` (`plateau_epoch`). ``"rapid_<relation>"`` is the end
        of rapid learning on the heads that relation connects
        (`rapid_learning_end` with `RAPID_METRICS`); the relation is named
        rather than taken from the consistency weights so that the control
        without a consistency term starts from the same epoch. An int is
        used as is.
    task_list : list of str
        The run's targets, as passed to `get_target_metrics_dict`.

    Returns
    -------
    int
    """
    results = os.path.join(phase1_backup_dir, "backup_results.csv")
    if not os.path.exists(results):
        raise FileNotFoundError(
            f"No phase-1 run at {phase1_backup_dir} (missing backup_results.csv); "
            "phase 2 needs the finished baseline of the same configuration.")
    if spec == "best":
        df = pd.read_csv(results)
        epoch, _ = select_best_epoch(df, get_target_metrics_dict(task_list))
        epoch = int(epoch)
    elif spec == "plateau":
        epoch = plateau_epoch(pd.read_csv(results), task_list, PLATEAU_COUNT)
        if epoch is None:
            raise ValueError(
                f"The early-stopping counter of {phase1_backup_dir} never reaches "
                f"{PLATEAU_COUNT}; there is no plateau to switch at.")
    elif isinstance(spec, str) and spec.startswith("rapid_"):
        relation = spec[len("rapid_"):]
        epoch = rapid_learning_end(pd.read_csv(results), rapid_metrics(relation, task_list))
        if epoch is None:
            raise ValueError(
                f"Rapid learning of the {relation} heads of {phase1_backup_dir} never "
                f"ends ({RAPID_ROUNDS} validations below {RAPID_TOLERANCE:.0%}).")
    else:
        epoch = int(spec)
    checkpoint = os.path.join(phase1_backup_dir, f"model_epoch_{epoch}.pt")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Phase-1 checkpoint not found: {checkpoint}")
    return epoch


def phase1_log_sigmas(phase1_backup_dir, epoch, components, checkpoint):
    """The UW log-sigmas of the phase-1 run at the end of ``epoch``.

    Read from the checkpoint when it stores them. Older checkpoints do not, so
    the values are then taken from the run's TensorBoard log, which records
    them (as ``LogSigma_Epoch/<component>``) after every validated epoch --
    with no optimizer step between the checkpoint and that log entry, so they
    are the same values.

    Parameters
    ----------
    phase1_backup_dir : str
    epoch : int
    components : list of str
        Task names in log-sigma order, e.g. ``["ActivitySuffix",
        "TimestampSuffix", "RemainingRuntime", "Outcome"]``.
    checkpoint : dict
        The loaded phase-1 checkpoint.

    Returns
    -------
    values : list of float
    source : {"checkpoint", "tensorboard"}
    """
    if "log_sigmas" in checkpoint:
        return [float(v) for v in checkpoint["log_sigmas"]], "checkpoint"

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    found = {}
    for path in sorted(glob.glob(os.path.join(phase1_backup_dir, "events.out.tfevents*"))):
        acc = EventAccumulator(path, size_guidance={"scalars": 0})
        acc.Reload()
        tags = set(acc.Tags()["scalars"])
        for comp in components:
            tag = f"LogSigma_Epoch/{comp}"
            if tag not in tags:
                continue
            for event in acc.Scalars(tag):
                # A restarted run can log the same step twice; keep the latest.
                if event.step == epoch and (
                        comp not in found or event.wall_time >= found[comp][0]):
                    found[comp] = (event.wall_time, event.value)
    missing = [c for c in components if c not in found]
    if missing:
        raise ValueError(
            f"UW log-sigmas for epoch {epoch} not in the checkpoint nor in the "
            f"TensorBoard log of {phase1_backup_dir} (missing {missing}). They are "
            "only logged on validated epochs; start phase 2 from one of those.")
    return [found[c][1] for c in components], "tensorboard"


def direction_from_diagnostics(diag, basis="CB", margins=None):
    """Which head to freeze in phase 2, per consistency relation.

    The rule: freeze the prediction that is better on the validation set, so
    the constraint moves the weaker one. Both predictions of a relation are
    scored on the quantity the relation is about, so they are directly
    comparable:

    - time relation (axiom 1): the remaining time, predicted directly by the
      RRT head and as the sum of the predicted time-till-next-event suffix;
      better = lower MAE.
    - outcome relation (axiom 2): the case outcome, predicted directly by the
      outcome head and implied by the decoded activity suffix; better = higher
      macro-F1 (the outcome classes are imbalanced, so accuracy would favour
      the majority classes).

    A relation is skipped when its diagnostics are absent (e.g. no outcome
    head). A difference no larger than the relation's margin gives no basis to
    choose, so neither side is frozen.

    Parameters
    ----------
    diag : dict
        Consistency diagnostics on the validation set, as written by
        `inference_loop` (``consistency_diagnostics.pkl``), or their mean over
        seeds.
    basis : {"CB", "IB"}, optional
        Case-based (default) or instance-based scores.
    margins : dict, optional
        Tie margin per relation, ``{"axiom1": seconds, "axiom2": macro-F1}``,
        e.g. one seed standard deviation of the difference. Missing relations
        use 0, so only an exact tie freezes neither side.

    Returns
    -------
    dict
        ``{"axiom1": {...}, "axiom2": {...}}`` with the two scores, the better
        prediction, and the detach mode that freezes it.
    """
    if basis not in ("CB", "IB"):
        raise ValueError(f"basis must be 'CB' or 'IB', got {basis!r}")
    margins = margins or {}
    label = {"CB": "case-based", "IB": "instance-based"}[basis]
    out = {}
    rrt_key, ttne_key = f"mae_rrt_{basis}", f"mae_ttne_sum_{basis}"
    if rrt_key in diag and ttne_key in diag:
        rrt, ttne = float(diag[rrt_key]), float(diag[ttne_key])
        margin = float(margins.get("axiom1", 0.0))
        better = ("rrt_head" if ttne - rrt > margin
                  else "ttne_sum" if rrt - ttne > margin else None)
        out["axiom1"] = {
            "metric": f"remaining-time MAE (seconds, {label}), lower is better",
            "rrt_head": rrt, "ttne_sum": ttne, "margin": margin, "better": better,
            "detach_mode": {"rrt_head": "rrt", "ttne_sum": "ttne", None: "none"}[better],
        }
    head_key, suffix_key = f"outcome_head_macro_f1_{basis}", f"outcome_suffix_macro_f1_{basis}"
    if head_key in diag and suffix_key in diag:
        head, suffix = float(diag[head_key]), float(diag[suffix_key])
        margin = float(margins.get("axiom2", 0.0))
        better = ("outcome_head" if head - suffix > margin
                  else "suffix_implied" if suffix - head > margin else None)
        out["axiom2"] = {
            "metric": f"outcome macro-F1 ({label}), higher is better",
            "outcome_head": head, "suffix_implied": suffix, "margin": margin,
            "better": better,
            "detach_mode_outcome": {"outcome_head": "outcome",
                                    "suffix_implied": "act", None: "none"}[better],
        }
    return out


def load_phase1_checkpoint(phase1_backup_dir, epoch, device):
    """The phase-1 checkpoint dict for ``epoch``."""
    path = os.path.join(phase1_backup_dir, f"model_epoch_{epoch}.pt")
    return torch.load(path, map_location=device)
