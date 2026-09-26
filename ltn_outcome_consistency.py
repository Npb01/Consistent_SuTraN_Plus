"""Axiom 2: outcome / activity-suffix consistency as an LTN constraint.

Formulation A: the LAST activity in the suffix drawn from the determining set
{O_Accepted, O_Cancelled, O_Refused} fixes the case outcome. That identity holds
exactly in the labels (and exactly per class) on the non-leaky instances.

Structurally parallel to `ltn_consistency.py` (axiom 1):

    grounding   ordinary tensor code building the term the logic talks about
                -- here, the soft "last determining activity" distribution q(o)
    predicate   smooth truth degree in [0, 1]
    quantifier  ltn Forall with AggregPMeanError(p=2)

Only the last two are LTN operations. The product inside the grounding is a fuzzy
conjunction under the PRODUCT T-NORM ("a(o) occurs at t AND nothing determining
occurs after t", where AND is multiplication and NOT x is 1 - x):

    p_t(a(o)) * prod_{s > t} (1 - d_s)

and q(o) is the fuzzy existential over t.

Why not any-occurrence: `O_Accepted in suffix => Accepted` holds only ~0.70 of
the time -- an accepted offer can still be cancelled later. Only the last
occurrence is exact. The one-directional any-occurrence form also has a
degenerate solution (predict all determining activities everywhere).

Numerics: the product is computed directly, not in log space. With ~1.2
determining activities per suffix, prod(1 - d_s) ~= exp(-1.2) ~= 0.30, far above
float32 underflow. `min_survival` is returned so this can be monitored; switch to
log space only if it ever approaches ~1e-6.
"""

from __future__ import annotations

import torch
import ltn


class _OutcomeAgreementPredicate(torch.nn.Module):
    """Truth degree that the suffix-implied outcome and the outcome head agree.

    The probability that two independent draws -- one from each distribution --
    land on the same class. Equals 1 iff both are the same one-hot, smooth
    everywhere, and needs no scale parameter (both arguments are probabilities).
    """

    def forward(self, q: torch.Tensor, outcome_probs: torch.Tensor) -> torch.Tensor:
        return (q * outcome_probs).sum(dim=-1)


class OutcomeConsistencyLoss(torch.nn.Module):
    """Enforces: outcome head == outcome implied by the predicted activity suffix.

    Parameters
    ----------
    determining_ids : dict[int, int]
        Outcome class index -> activity id whose last occurrence implies it.
        Resolve from activity NAMES with
        `outcome_consistency_metrics.resolve_determining_ids`.
    end_token : int
        Activity id of the END token; positions from the first END onward are
        excluded from both the sum and the product.
    num_outclasses : int
    pad_token : int, optional
    p : int, optional
        Exponent of the p-mean-error aggregator. p=2 mildly emphasises the
        worst-satisfied instances, matching axiom 1.
    detach_mode : {'none', 'act', 'outcome'}
        Which side receives gradient. 'act' freezes the suffix-implied
        distribution so only the outcome head moves; 'outcome' does the reverse.
        This detaches a TERM, not a head: the encoder/decoder stack is shared,
        so the frozen side still moves indirectly.
    """

    def __init__(self, determining_ids, end_token, num_outclasses,
                 pad_token=0, p=2, detach_mode="none"):
        super().__init__()
        if detach_mode not in ("none", "act", "outcome"):
            raise ValueError(
                "detach_mode must be 'none', 'act' (freeze the suffix side) or "
                f"'outcome' (freeze the outcome head); got {detach_mode!r}"
            )
        if sorted(determining_ids) != list(range(num_outclasses)):
            raise ValueError(
                f"determining_ids must cover outcome classes 0..{num_outclasses - 1} "
                f"exactly once; got keys {sorted(determining_ids)}"
            )
        self.detach_mode = detach_mode
        self.num_outclasses = num_outclasses
        self.end_token = int(end_token)
        self.pad_token = int(pad_token)

        # Column c of this index vector is the activity id implying outcome c,
        # so p_t[..., act_of_class] gathers all classes in one shot.
        self.register_buffer(
            "act_of_class",
            torch.tensor([determining_ids[c] for c in range(num_outclasses)],
                         dtype=torch.long),
        )

        self.predicate = ltn.Predicate(_OutcomeAgreementPredicate())
        self.Forall = ltn.Quantifier(
            ltn.fuzzy_ops.AggregPMeanError(p=p), quantifier="f"
        )

    def implied_distribution(self, act_logits, act_labels):
        """Soft distribution over the outcome the predicted suffix implies.

        Parameters
        ----------
        act_logits : (B, W, C_act)
        act_labels : (B, W)
            Ground-truth activity labels, used ONLY to mask padding and anything
            at or after the END token -- the same mild train-time use of label
            information that axiom 1 makes via its ttne mask.

        Returns
        -------
        q : (B, num_outclasses)      unnormalised, sums to <= 1
        survival : (B, W)            prod_{s > t}(1 - d_s), for monitoring
        """
        probs = torch.softmax(act_logits, dim=-1)                     # (B, W, C)

        # Valid = before the first END token and not padding. cummax over the
        # END indicator marks the END position and everything after it, matching
        # outcome_consistency_metrics.implied_outcome_from_suffix so the training
        # and evaluation definitions cannot drift apart.
        is_end = (act_labels == self.end_token)
        after_end = torch.cummax(is_end.to(torch.int64), dim=1).values.bool()
        valid = (~after_end) & (act_labels != self.pad_token)          # (B, W)
        mask = valid.to(probs.dtype)

        # p_det[b, t, c] = P(step t is the activity implying outcome c)
        p_det = probs[..., self.act_of_class] * mask.unsqueeze(-1)     # (B, W, K)
        d = p_det.sum(dim=-1)                                          # (B, W)

        # survival[b, t] = prod_{s > t} (1 - d[b, s]). A reversed cumulative
        # product of (1 - d) shifted by one gives every t in a single pass; the
        # final position gets 1.0 (nothing after it).
        one_minus = (1.0 - d).clamp(min=0.0)
        rev_cumprod = torch.flip(torch.cumprod(torch.flip(one_minus, [1]), dim=1), [1])
        survival = torch.ones_like(rev_cumprod)
        survival[:, :-1] = rev_cumprod[:, 1:]

        q = (p_det * survival.unsqueeze(-1)).sum(dim=1)                # (B, K)
        return q, survival

    def forward(self, act_logits, act_labels, outcome_logits, valid_mask=None):
        """
        Parameters
        ----------
        act_logits : (B, W, C_act)
        act_labels : (B, W)
        outcome_logits : (B, num_outclasses)
            Raw outcome-head outputs, already sliced to the first decoding step.
        valid_mask : (B,) bool, optional
            True for instances the axiom applies to -- i.e. `instance_mask_out
            == False`, the non-leaky subset. Outside it the prefix already
            reveals the outcome and the identity is not guaranteed.

        Returns
        -------
        loss : scalar tensor         1 - sat
        sat : detached scalar
        diagnostics : dict[str, float]
        """
        q, survival = self.implied_distribution(act_logits, act_labels)
        outcome_probs = torch.softmax(outcome_logits, dim=-1)

        if valid_mask is not None:
            if valid_mask.dtype != torch.bool:
                valid_mask = valid_mask.bool()
            q = q[valid_mask]
            outcome_probs = outcome_probs[valid_mask]
            survival = survival[valid_mask]

        if q.shape[0] == 0:
            zero = outcome_logits.sum() * 0.0        # keeps the graph connected
            return zero, zero.detach(), {"num_instances": 0.0}

        # Mass NOT assigned to any determining activity: the model's belief that
        # no determining event occurs at all. Exactly 0 in the ground truth, so a
        # large value means the axiom has little to bite on -- monitor it.
        total = q.sum(dim=-1)
        residual = (1.0 - total).clamp(min=0.0)
        q_norm = q / total.clamp(min=1e-8).unsqueeze(-1)

        if self.detach_mode == "act":
            q_norm = q_norm.detach()                 # only the outcome head moves
        elif self.detach_mode == "outcome":
            outcome_probs = outcome_probs.detach()   # only the activity head moves

        x_q = ltn.Variable("q_implied", q_norm)
        x_y = ltn.Variable("outcome_pred", outcome_probs)
        sat_agg = self.Forall(ltn.diag(x_q, x_y), self.predicate(x_q, x_y)).value

        diagnostics = {
            "num_instances": float(q.shape[0]),
            "mean_residual_mass": residual.mean().item(),
            "min_survival": survival.min().item(),
            "mean_agreement": (q_norm.detach() * outcome_probs.detach()).sum(-1).mean().item(),
        }
        return 1.0 - sat_agg, sat_agg.detach(), diagnostics
