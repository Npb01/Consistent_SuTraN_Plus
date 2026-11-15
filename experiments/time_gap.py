"""Time-relation gap on the cases whose predicted suffix ends.

A predicted suffix without an END token has no completion time: its summed
TTNE runs to the end of the window, so its gap with the RRT prediction depends
on the window length rather than on the model. Inference therefore also
reports the gap on the cases with a predicted END (`*_defined_{IB,CB}`), next
to the share without one (`frac_no_pred_end_*`).

This module computes the same metrics from a run's stored per-instance dump
(`consistency_per_instance.pt`), for runs evaluated before inference reported
them. Dumps without the `ax1_has_pred_end` flag count a predicted length equal
to the window as "no END"; that also counts a suffix whose END falls exactly
on the last window position, which the window choice makes rare.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from CaLenDiR_Utils.weighted_metrics_utils import get_weight_tensor
from Utils.subset import subset_split_by_case

DEFINED_METRICS = ("mean_abs_gap_defined", "mean_signed_gap_defined", "mae_ttne_sum_defined")

_WINDOW_CACHE = {}
_WEIGHT_CACHE = {}


def window_size(root, log_name, split="test"):
    """Window length of a log's preprocessed split (read without loading the data)."""
    key = (root, log_name)
    if key not in _WINDOW_CACHE:
        path = os.path.join(root, log_name, f"{split}_tensordataset.pt")
        dataset = torch.load(path, mmap=True, weights_only=False)
        _WINDOW_CACHE[key] = int(dataset[0].shape[1])
    return _WINDOW_CACHE[key]


def case_weights(root, log_name, split="test", subset_fraction=1.0, seed=None):
    """CB weight per instance of a split, as the run evaluated it.

    The validation split of a run may be a case subset (`val_subset_fraction`,
    drawn with seed + 2 as in training); the test split is always complete.
    """
    key = (root, log_name, split, subset_fraction, seed)
    if key not in _WEIGHT_CACHE:
        og_caseint = torch.load(os.path.join(root, log_name, f"og_caseint_{split}.pt"), weights_only=True)
        if subset_fraction < 1.0:
            _, og_caseint, _, _ = subset_split_by_case((), og_caseint, subset_fraction, seed + 2)
        weights, _ = get_weight_tensor(og_caseint)
        _WEIGHT_CACHE[key] = weights.double().numpy()
    return _WEIGHT_CACHE[key]


def has_pred_end(dump, window):
    """Per instance: does the predicted suffix contain an END token?"""
    if "ax1_has_pred_end" in dump:
        return np.asarray(dump["ax1_has_pred_end"], dtype=bool)
    return np.asarray(dump["pred_len"]) < window


def defined_gap_metrics(dump, has_end, weights):
    """The `*_defined_{IB,CB}` metrics of one dump, in seconds."""
    to_np = lambda k: np.asarray(dump[k], dtype=np.float64)
    gap = to_np("ax1_sum_ts_pred") - to_np("ax1_rt_pred")
    err_ts = to_np("ax1_sum_ts_pred") - to_np("ax1_sum_ts_true")
    if len(weights) != len(gap):
        raise ValueError(f"{len(weights)} weights for {len(gap)} instances")
    m = has_end.astype(np.float64)
    out = {}
    for name, x in zip(DEFINED_METRICS, (np.abs(gap), gap, np.abs(err_ts))):
        if m.sum() == 0:
            out[f"{name}_IB"] = out[f"{name}_CB"] = float("nan")
            continue
        out[f"{name}_IB"] = float(np.sum(x * m) / m.sum())
        out[f"{name}_CB"] = float(np.sum(x * weights * m) / np.sum(weights * m))
    return out
