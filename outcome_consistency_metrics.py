"""Outcome / activity-suffix cross-task consistency metrics (axiom 2).

Pure tensor functions, deliberately free of any model or inference-loop state so
they can be unit-tested on synthetic inputs without a GPU. `inference_procedure`
calls `outcome_consistency_diagnostics()`; nothing here imports from SuTraN.

The quantity of interest is the outcome IMPLIED by the predicted activity suffix:
the LAST activity in the suffix drawn from the determining set
{O_Accepted, O_Cancelled, O_Refused} fixes the case outcome. That identity holds
exactly on the non-leaky instances.

The any-occurrence form is NOT valid here: `O_Accepted in suffix => Accepted`
holds only ~0.70 of the time, because an accepted offer can be cancelled later.

These metrics answer the question that gates the axiom-2 study: given two routes
to the case outcome -- the outcome head, and the decoded activity suffix -- which
is more accurate, and how often do they disagree? (The direct analogue of the
sum-ttne vs RRT comparison for axiom 1.)
"""

from __future__ import annotations

import torch


def implied_outcome_from_suffix(act_suffix, determining_ids, end_token,
                                pad_token=0):
    """Outcome implied by each predicted activity suffix.

    Parameters
    ----------
    act_suffix : torch.Tensor
        (N, W) int64 activity ids, as produced by greedy decoding.
    determining_ids : dict[int, int]
        Maps outcome class index -> the activity id that implies it.
    end_token : int
        Activity id of the END token. Positions from the first END onward are
        ignored.
    pad_token : int, optional
        Padding id, ignored as well. Default 0.

    Returns
    -------
    implied : torch.Tensor
        (N,) int64, outcome class per instance, or -1 where the suffix contains
        no determining activity at all.
    has_determining : torch.Tensor
        (N,) bool, True where an implied outcome could be derived.
    """
    if act_suffix.dim() != 2:
        raise ValueError(f"act_suffix must be (N, W); got {tuple(act_suffix.shape)}")
    n, w = act_suffix.shape
    device = act_suffix.device

    # Everything at or after the first END token is not part of the suffix.
    # cummax over the END indicator marks those positions in one pass; a row with
    # no END stays fully valid (a decode that ran to the window limit).
    is_end = (act_suffix == end_token)
    after_end = torch.cummax(is_end.to(torch.int64), dim=1).values.bool()
    valid = (~after_end) & (act_suffix != pad_token)

    ids = torch.tensor(sorted(determining_ids.values()), device=device)
    is_det = torch.isin(act_suffix, ids) & valid          # (N, W)
    has_determining = is_det.any(dim=1)                    # (N,)

    # Index of the LAST determining position. Rows with none give 0, harmless
    # because they are masked out by `has_determining`.
    positions = torch.arange(w, device=device).unsqueeze(0)
    idx_last = (positions * is_det.to(torch.int64)).max(dim=1).values
    last_act = act_suffix.gather(1, idx_last.unsqueeze(1)).squeeze(1)

    implied = torch.full((n,), -1, dtype=torch.int64, device=device)
    for outcome_class, act_id in determining_ids.items():
        implied[(last_act == act_id) & has_determining] = outcome_class

    return implied, has_determining


def outcome_consistency_diagnostics(act_suffix, outcome_logits, outcome_labels,
                                    determining_ids, end_token, pad_token=0,
                                    weights=None, num_cases=None,
                                    corrected_avg_fn=None,
                                    return_per_instance=False):
    """Head-vs-suffix outcome metrics.

    `act_suffix`, `outcome_logits` and `outcome_labels` must all be restricted to
    the SAME instances -- the non-leaky subset (`instance_mask_out == False`)
    when the log uses outcome masking. Mismatched subsetting is the easiest way
    to get plausible-looking wrong numbers, so the row counts are asserted.

    Parameters
    ----------
    act_suffix : torch.Tensor
        (N, W) decoded activity suffix.
    outcome_logits : torch.Tensor
        (N, C) raw outcome-head outputs. argmax only, so logits or probabilities
        both work.
    outcome_labels : torch.Tensor
        (N,) or (N, 1) ground-truth outcome class indices.
    determining_ids : dict[int, int]
        Outcome class -> determining activity id.
    weights, num_cases, corrected_avg_fn : optional
        If all three are given, case-based (CB) variants are added alongside the
        instance-based (IB) ones.
    return_per_instance : bool, optional
        When True, also return a dict of the per-instance 1-D tensors the
        aggregates are built from (for the inconsistency-vs-error analysis).

    Returns
    -------
    dict of float, or (dict of float, dict of torch.Tensor) when
    `return_per_instance` is True.
    """
    outcome_labels = outcome_labels.squeeze(-1) if outcome_labels.dim() == 2 else outcome_labels
    n = act_suffix.shape[0]
    if not (outcome_logits.shape[0] == outcome_labels.shape[0] == n):
        raise ValueError(
            "act_suffix, outcome_logits and outcome_labels must cover the same "
            f"instances; got {n}, {outcome_logits.shape[0]}, {outcome_labels.shape[0]}. "
            "Check that all three were subset by the same non-leaky mask."
        )

    head_pred = outcome_logits.argmax(dim=-1).to(torch.int64)
    implied, has_det = implied_outcome_from_suffix(
        act_suffix, determining_ids, end_token, pad_token)
    outcome_labels = outcome_labels.to(torch.int64)

    head_correct = (head_pred == outcome_labels).float()
    # Where the suffix implies nothing, the suffix route has failed to produce an
    # answer -- scored as incorrect rather than dropped, so the two routes are
    # compared on the same denominator.
    implied_correct = ((implied == outcome_labels) & has_det).float()
    agree = (head_pred == implied).float()

    out = {
        "outcome_head_accuracy_IB": head_correct.mean().item(),
        "outcome_suffix_accuracy_IB": implied_correct.mean().item(),
        "outcome_head_vs_suffix_agreement_IB": agree.mean().item(),
        "outcome_disagreement_rate_IB": (1.0 - agree).mean().item(),
        "frac_suffix_no_determining_act": (~has_det).float().mean().item(),
        "num_outcome_instances_evaluated": float(n),
    }

    # Macro-F1 over the classes present in the labels: the outcome classes are
    # imbalanced, so accuracy alone would hide a model that ignores a rare class.
    classes = sorted(set(outcome_labels.unique().tolist()))
    for name, pred in (("head", head_pred), ("suffix", implied)):
        f1s = []
        for c in classes:
            tp = ((pred == c) & (outcome_labels == c)).sum().item()
            fp = ((pred == c) & (outcome_labels != c)).sum().item()
            fn = ((pred != c) & (outcome_labels == c)).sum().item()
            denom = 2 * tp + fp + fn
            f1s.append((2 * tp / denom) if denom > 0 else 0.0)
        out[f"outcome_{name}_macro_f1_IB"] = sum(f1s) / len(f1s) if f1s else float("nan")

    if weights is not None and num_cases is not None and corrected_avg_fn is not None:
        out["outcome_head_accuracy_CB"] = corrected_avg_fn(
            head_correct, weight_tens=weights, num_cases=num_cases)
        out["outcome_suffix_accuracy_CB"] = corrected_avg_fn(
            implied_correct, weight_tens=weights, num_cases=num_cases)
        out["outcome_disagreement_rate_CB"] = corrected_avg_fn(
            1.0 - agree, weight_tens=weights, num_cases=num_cases)

    if return_per_instance:
        # Raw per-instance quantities, so any error/agreement metric can be
        # recomputed offline. The caller additionally saves the full head logits
        # (head confidence) and the non-leaky realignment mask.
        per_instance = {
            "ax2_implied": implied,              # suffix-implied class, -1 if none
            "ax2_has_det": has_det,              # suffix had a determining act
            "ax2_outcome_labels": outcome_labels,  # ground-truth outcome class
        }
        return out, per_instance
    return out


def resolve_determining_ids(categ_mapping, activity_names_by_class):
    """Turn activity NAMES into the integer ids used in the tensors.

    Activity integers in the tensor datasets are `categ_mapping + 1`, because 0
    is reserved as the padding index. Resolving names at runtime rather than
    hardcoding ids means a re-preprocessed log with a different vocabulary fails
    loudly here instead of silently scoring the wrong activity.

    Parameters
    ----------
    categ_mapping : dict[str, int]
        The log's `<LOG>_categ_mapping.pkl['concept:name']`.
    activity_names_by_class : dict[int, str]
        Outcome class index -> activity name.

    Returns
    -------
    determining_ids : dict[int, int]
    end_token : int
    """
    missing = [n for n in activity_names_by_class.values() if n not in categ_mapping]
    if missing:
        raise KeyError(
            f"Determining activities absent from this log's activity vocabulary: "
            f"{missing}. Available: {sorted(categ_mapping)}"
        )
    determining_ids = {c: categ_mapping[name] + 1
                       for c, name in activity_names_by_class.items()}
    end_token = len(categ_mapping) + 1
    return determining_ids, end_token
