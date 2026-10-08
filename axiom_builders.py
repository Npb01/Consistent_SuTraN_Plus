"""Factory for the swappable axiom-loss implementations.

Each implementation exposes the same forward signature and return shape as the
current module, so the training loop is agnostic to which one is selected:

    axiom 1 : module(ts_suffix_pred_std, ts_suffix_mask, rt_pred_std) -> (loss, sat)
    axiom 2 : module(act_logits, act_labels, outcome_logits, valid_mask)
              -> (loss, sat, diagnostics)

The `*_impl` names are the values accepted by the `axiom1_impl` / `axiom2_impl`
config knob (and hence a sweep dimension in `experiments/sweeps.py`).
"""

from __future__ import annotations

AXIOM1_IMPLS = ("ltn_smooth_eq", "naive_penalty")
AXIOM2_IMPLS = ("collapsed_q", "native_last", "native_exists", "brier_mean")

AXIOM1_DEFAULT = "ltn_smooth_eq"
AXIOM2_DEFAULT = "collapsed_q"


def build_axiom1_module(impl, *, ts_mean, ts_std, rt_mean, rt_std, detach_mode):
    """Construct the axiom-1 (cross-task time consistency) loss module."""
    if impl not in AXIOM1_IMPLS:
        raise ValueError(f"axiom1_impl must be one of {AXIOM1_IMPLS}, got {impl!r}")

    if impl == "ltn_smooth_eq":
        from ltn_consistency import CrossTaskConsistencyLoss
        return CrossTaskConsistencyLoss(
            ts_mean=ts_mean, ts_std=ts_std, rt_mean=rt_mean, rt_std=rt_std,
            detach_mode=detach_mode,
        )

    if impl == "naive_penalty":
        from ltn_consistency import NaivePenaltyConsistencyLoss
        return NaivePenaltyConsistencyLoss(
            ts_mean=ts_mean, ts_std=ts_std, rt_mean=rt_mean, rt_std=rt_std,
            detach_mode=detach_mode,
        )

    raise ValueError(f"unhandled axiom1_impl={impl!r}")


def build_axiom2_module(impl, *, determining_ids, end_token, num_outclasses, detach_mode):
    """Construct the axiom-2 (outcome/activity-suffix consistency) loss module."""
    if impl not in AXIOM2_IMPLS:
        raise ValueError(f"axiom2_impl must be one of {AXIOM2_IMPLS}, got {impl!r}")

    if impl == "collapsed_q":
        from ltn_outcome_consistency import OutcomeConsistencyLoss
        return OutcomeConsistencyLoss(
            determining_ids=determining_ids, end_token=end_token,
            num_outclasses=num_outclasses, detach_mode=detach_mode,
        )

    if impl == "native_last":
        from ltn_outcome_consistency import NativeLastOutcomeConsistencyLoss
        return NativeLastOutcomeConsistencyLoss(
            determining_ids=determining_ids, end_token=end_token,
            num_outclasses=num_outclasses, detach_mode=detach_mode,
        )

    if impl == "native_exists":
        from ltn_outcome_consistency import NativeExistsOutcomeConsistencyLoss
        return NativeExistsOutcomeConsistencyLoss(
            determining_ids=determining_ids, end_token=end_token,
            num_outclasses=num_outclasses, detach_mode=detach_mode,
        )

    if impl == "brier_mean":
        from ltn_outcome_consistency import BrierMeanOutcomeConsistencyLoss
        return BrierMeanOutcomeConsistencyLoss(
            determining_ids=determining_ids, end_token=end_token,
            num_outclasses=num_outclasses, detach_mode=detach_mode,
        )

    raise NotImplementedError(
        f"axiom2_impl={impl!r} is registered but not implemented yet "
        "(see AXIOM_COMPARISON_PLAN.md §3a)."
    )
