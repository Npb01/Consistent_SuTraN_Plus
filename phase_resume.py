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


def phase1_checkpoint_epoch(phase1_backup_dir, spec, task_list):
    """The phase-1 epoch whose checkpoint phase 2 starts from.

    Parameters
    ----------
    phase1_backup_dir : str
        Folder of the phase-1 run holding ``backup_results.csv`` and the
        ``model_epoch_<N>.pt`` checkpoints.
    spec : {"best"} or int
        ``"best"`` is the epoch the phase-1 run selects as its own final model
        (`select_best_epoch` over its validated epochs); an int is used as is.
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


def direction_from_diagnostics(diag):
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
    head). An exact tie gives no basis to choose, so neither side is frozen.

    Parameters
    ----------
    diag : dict
        Consistency diagnostics of the phase-1 checkpoint on the validation
        set, as written by `inference_loop` (``consistency_diagnostics.pkl``).

    Returns
    -------
    dict
        ``{"axiom1": {...}, "axiom2": {...}}`` with the two scores, the better
        prediction, and the detach mode that freezes it.
    """
    out = {}
    if "mae_rrt_IB" in diag and "mae_ttne_sum_IB" in diag:
        rrt, ttne = float(diag["mae_rrt_IB"]), float(diag["mae_ttne_sum_IB"])
        better = "rrt_head" if rrt < ttne else "ttne_sum" if ttne < rrt else None
        out["axiom1"] = {
            "metric": "remaining-time MAE (seconds, instance-based), lower is better",
            "rrt_head": rrt, "ttne_sum": ttne, "better": better,
            "detach_mode": {"rrt_head": "rrt", "ttne_sum": "ttne", None: "none"}[better],
        }
    if "outcome_head_macro_f1_IB" in diag and "outcome_suffix_macro_f1_IB" in diag:
        head = float(diag["outcome_head_macro_f1_IB"])
        suffix = float(diag["outcome_suffix_macro_f1_IB"])
        better = ("outcome_head" if head > suffix
                  else "suffix_implied" if suffix > head else None)
        out["axiom2"] = {
            "metric": "outcome macro-F1 (instance-based), higher is better",
            "outcome_head": head, "suffix_implied": suffix, "better": better,
            "detach_mode_outcome": {"outcome_head": "outcome",
                                    "suffix_implied": "act", None: "none"}[better],
        }
    return out


def load_phase1_checkpoint(phase1_backup_dir, epoch, device):
    """The phase-1 checkpoint dict for ``epoch``."""
    path = os.path.join(phase1_backup_dir, f"model_epoch_{epoch}.pt")
    return torch.load(path, map_location=device)
