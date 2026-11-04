"""Hard enforcement of the cross-task relations on stored predictions.

Post-hoc: maps the predictions of a trained model onto predictions that satisfy
a relation exactly. One side of the relation is kept (the more reliable one, see
`phase_resume.direction_from_diagnostics`) and the other is changed to agree
with it.

Time relation: the TTNE suffix, summed up to and including the predicted END
(the whole window when the decoder predicts none), equals the remaining runtime.
Durations are destandardized and clamped at zero, as in the evaluation.

- RRT kept: the TTNE steps are rescaled proportionally so that they sum to the
  RRT; when they sum to zero, the RRT is spread evenly over the steps.
- TTNE kept: the RRT becomes the summed TTNE suffix.

Outcome relation: the last determining activity of the activity suffix implies
the outcome. It is enforced only where the suffix contains a determining
activity (the guard; elsewhere the relation makes no claim) and only on the
instances the outcome is evaluated on.

- Suffix kept: the predicted outcome becomes the implied class. The result is
  a class, not a distribution, so only class-based outcome metrics apply.
- Outcome kept: the last determining activity is replaced by the determining
  activity of the predicted class. This needs one determining activity per
  class.

Pure tensor functions on CPU tensors; nothing here imports from SuTraN.
"""

from __future__ import annotations

import torch

from outcome_consistency_metrics import (implied_outcome_from_suffix,
                                         last_determining_position,
                                         normalize_determining_ids)


def predicted_suffix_mask(act_suffix, end_token):
    """(N, W) bool, True up to and including the first predicted END, or on the
    whole window when the decoder predicts none."""
    w = act_suffix.shape[1]
    is_end = act_suffix == end_token
    has_end = is_end.any(dim=1)
    end_pos = torch.where(has_end, is_end.to(torch.int64).argmax(dim=1),
                          torch.full_like(has_end, w - 1, dtype=torch.int64))
    return torch.arange(w).unsqueeze(0) <= end_pos.unsqueeze(1)


def _to_seconds(values, mean_std):
    return (values.double() * mean_std[1] + mean_std[0]).clamp(min=0.0)


def _to_standardized(seconds, mean_std):
    return ((seconds - mean_std[0]) / mean_std[1]).float()


def ttne_to_rrt(ttne_pred, rrt_pred, act_suffix, end_token, mean_std_ttne,
                mean_std_rrt):
    """Rescale the TTNE suffix so that it sums to the remaining runtime.

    Parameters
    ----------
    ttne_pred : torch.Tensor
        (N, W) TTNE suffix predictions, standardized.
    rrt_pred : torch.Tensor
        (N,) remaining runtime predictions, standardized.
    act_suffix : torch.Tensor
        (N, W) decoded activity suffix; fixes the predicted suffix length.
    end_token : int
    mean_std_ttne, mean_std_rrt : sequence of float
        Training mean and standard deviation (seconds) of the two targets.

    Returns
    -------
    ttne_new : torch.Tensor
        (N, W) standardized; positions after the predicted END are unchanged.
    rrt_new : torch.Tensor
        (N,) standardized; the input clamped at a duration of zero.
    """
    mask = predicted_suffix_mask(act_suffix, end_token)
    steps = _to_seconds(ttne_pred, mean_std_ttne) * mask
    rrt = _to_seconds(rrt_pred, mean_std_rrt)
    total = steps.sum(dim=1, keepdim=True)
    num_steps = mask.sum(dim=1, keepdim=True)
    scaled = torch.where(total > 0, steps * (rrt.unsqueeze(1) / total.clamp(min=1e-12)),
                         (rrt.unsqueeze(1) / num_steps).expand_as(steps))
    ttne_new = torch.where(mask, _to_standardized(scaled, mean_std_ttne), ttne_pred)
    return ttne_new, _to_standardized(rrt, mean_std_rrt)


def rrt_to_ttne(ttne_pred, act_suffix, end_token, mean_std_ttne, mean_std_rrt):
    """The remaining runtime that equals the summed TTNE suffix.

    Returns
    -------
    rrt_new : torch.Tensor
        (N,) standardized.
    """
    mask = predicted_suffix_mask(act_suffix, end_token)
    total = (_to_seconds(ttne_pred, mean_std_ttne) * mask).sum(dim=1)
    return _to_standardized(total, mean_std_rrt)


def outcome_to_suffix(out_logits, act_suffix, determining_ids, end_token, eligible):
    """Make the predicted outcome the class the activity suffix implies.

    Parameters
    ----------
    out_logits : torch.Tensor
        (N, C) outcome logits.
    act_suffix : torch.Tensor
        (N, W) decoded activity suffix.
    determining_ids : dict[int, int | list[int]]
        Outcome class -> determining activity id (or ids).
    end_token : int
    eligible : torch.Tensor
        (N,) bool, the instances the outcome is evaluated on.

    Returns
    -------
    out_class : torch.Tensor
        (N,) int64 predicted outcome class: the implied class where the
        suffix implies one, the head's class elsewhere.
    changed : torch.Tensor
        (N,) bool, the instances whose predicted outcome changed.
    """
    implied, has_det = implied_outcome_from_suffix(act_suffix, determining_ids, end_token)
    head = out_logits.argmax(dim=-1)
    changed = eligible & has_det & (implied != head)
    return torch.where(changed, implied, head), changed


def suffix_to_outcome(act_suffix, out_logits, determining_ids, end_token, eligible):
    """Make the last determining activity of the suffix imply the predicted outcome.

    Parameters
    ----------
    act_suffix, out_logits, determining_ids, end_token, eligible
        As in `outcome_to_suffix`.

    Returns
    -------
    act_new : torch.Tensor
        (N, W) activity suffix with the last determining activity replaced
        where it implied a different class than the predicted one.
    changed : torch.Tensor
        (N,) bool, the instances whose suffix changed.
    """
    determining_ids = normalize_determining_ids(determining_ids)
    several = {c: acts for c, acts in determining_ids.items() if len(acts) > 1}
    if several:
        raise ValueError(
            "Replacing the last determining activity needs one determining "
            f"activity per outcome class; these classes have several: {several}")
    class_act = torch.tensor([determining_ids[c][0] for c in sorted(determining_ids)])

    implied, has_det = implied_outcome_from_suffix(act_suffix, determining_ids, end_token)
    idx_last, _ = last_determining_position(act_suffix, determining_ids, end_token)
    head = out_logits.argmax(dim=-1)
    changed = eligible & has_det & (implied != head)
    rows = changed.nonzero(as_tuple=True)[0]
    act_new = act_suffix.clone()
    act_new[rows, idx_last[rows]] = class_act[head[rows]]
    return act_new, changed
