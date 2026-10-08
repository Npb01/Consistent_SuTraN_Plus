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


class _OutcomeBrierPredicate(torch.nn.Module):
    """Brier equality between the suffix-implied outcome and the outcome head:

        Eq(q, y_hat) = 1 - 1/2 * sum_o (q_o - y_hat_o)^2

    Equals 1 iff the distributions are identical, 0 for opposite one-hots, stays
    in [0, 1], and needs no scale parameter (both arguments are probabilities).
    """

    def forward(self, q: torch.Tensor, outcome_probs: torch.Tensor) -> torch.Tensor:
        return 1.0 - 0.5 * (q - outcome_probs).pow(2).sum(dim=-1)


class _TruthPredicate(torch.nn.Module):
    """Lifts a precomputed per-individual truth degree, shape (N, 1), into a
    predicate output of shape (N,), so it can enter an LTN connective."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(-1)


def _determining_survival(act_logits, act_labels, act_of_class, end_token, pad_token):
    """Shared grounding for the axiom-2 impls.

    Valid positions are those before the first END token and not padding. The
    END mask uses ground-truth labels and mirrors
    `outcome_consistency_metrics.implied_outcome_from_suffix`, so the training
    and evaluation definitions cannot drift apart.

    Returns
    -------
    p_det : (B, W, K)      masked prob mass on each determining class per position
    det : (B, W)           prob mass on the determining set (0 outside the suffix)
    survival : (B, W)      prod_{s > t} (1 - det_s); 1.0 at the last position
    pos_valid : (B, W) bool
    """
    probs = torch.softmax(act_logits, dim=-1)
    is_end = (act_labels == end_token)
    after_end = torch.cummax(is_end.to(torch.int64), dim=1).values.bool()
    pos_valid = (~after_end) & (act_labels != pad_token)
    mask = pos_valid.to(probs.dtype)

    p_det = probs[..., act_of_class] * mask.unsqueeze(-1)              # (B, W, K)
    det = p_det.sum(dim=-1)                                            # (B, W)

    one_minus = (1.0 - det).clamp(min=0.0)
    rev_cumprod = torch.flip(torch.cumprod(torch.flip(one_minus, [1]), dim=1), [1])
    survival = torch.ones_like(rev_cumprod)
    survival[:, :-1] = rev_cumprod[:, 1:]
    return p_det, det, survival, pos_valid


class _OutcomeConsistencyModule(torch.nn.Module):
    """Shared setup for the axiom-2 outcome-consistency impls.

    Holds the determining-activity index buffer, the Brier equality predicate,
    and the p-mean-error Forall. Subclasses implement `forward`.

    Parameters
    ----------
    determining_ids : dict[int, int]
        Outcome class index -> activity id whose last occurrence implies it.
        Resolve from activity NAMES with
        `outcome_consistency_metrics.resolve_determining_ids`.
    end_token : int
        Activity id of the END token; positions from the first END onward are
        excluded from the grounding.
    num_outclasses : int
    pad_token : int, optional
    p : int, optional
        Exponent of the p-mean-error aggregator. p=2 mildly emphasises the
        worst-satisfied individuals, matching axiom 1.
    detach_mode : {'none', 'act', 'outcome'}
        Which side receives gradient. 'act' freezes the suffix-implied side so
        only the outcome head moves; 'outcome' does the reverse. This detaches a
        TERM, not a head: the encoder/decoder stack is shared, so the frozen
        side still moves indirectly.
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
        # so probs[..., act_of_class] gathers all classes in one shot.
        self.register_buffer(
            "act_of_class",
            torch.tensor([determining_ids[c] for c in range(num_outclasses)],
                         dtype=torch.long),
        )

        self.predicate = ltn.Predicate(_OutcomeBrierPredicate())
        self.Forall = ltn.Quantifier(
            ltn.fuzzy_ops.AggregPMeanError(p=p), quantifier="f"
        )

    def _collapsed_diagnostics(self, p_det, survival, outcome_probs,
                               valid_mask, num_positions=None):
        """Implementation-agnostic diagnostics shared across the axiom-2 impls.

        Scored on the collapsed distribution q so every impl reports the same
        yardstick regardless of how it aggregates internally. `num_positions` is
        included only for the per-position impls (native_last / native_exists).
        """
        q = (p_det * survival.unsqueeze(-1)).sum(dim=1)[valid_mask]    # (Bv, K)
        total = q.sum(dim=-1)
        residual = (1.0 - total).clamp(min=0.0)
        q_norm = q / total.clamp(min=1e-8).unsqueeze(-1)
        y_inst = outcome_probs[valid_mask]
        diag = {
            "num_instances": float(valid_mask.sum().item()),
            "mean_residual_mass": residual.mean().item(),
            "min_survival": survival[valid_mask].min().item(),
            "mean_agreement": (q_norm * y_inst).sum(-1).mean().item(),
        }
        if num_positions is not None:
            diag["num_positions"] = float(num_positions)
        return diag


class OutcomeConsistencyLoss(_OutcomeConsistencyModule):
    """Enforces: outcome head == outcome implied by the predicted activity suffix.

    Collapses `forall t` into one per-instance distribution q(o) via the
    product-t-norm survival term, then scores a single Brier equality per
    instance. See `_OutcomeConsistencyModule` for the constructor parameters.
    """

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
        p_det, _, survival, _ = _determining_survival(
            act_logits, act_labels, self.act_of_class,
            self.end_token, self.pad_token,
        )
        # q(o) = sum_t P(step t is the last determining activity implying o).
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


class NativeLastOutcomeConsistencyLoss(_OutcomeConsistencyModule):
    """Same axiom, per-position `Last` instead of a collapsed distribution:

        forall sigma. forall t. [ Last(sigma, t) -> out(act(sigma, t)) = O(sigma) ]

    with `Last(sigma, t) = Det(act(sigma, t)) AND prod_{s > t}(1 - Det(act(sigma, s)))`
    (product t-norm; the product is the per-position "nothing determining after
    t" survival term). The implication is Reichenbach, the equality Brier, and
    both quantifiers are p-mean-error. There is one truth value per valid
    (sigma, t) pair -- non-last positions have `Last ~ 0` and are vacuously
    satisfied -- in contrast to `OutcomeConsistencyLoss`, which collapses
    `forall t` into one Brier score per instance.

    See `_OutcomeConsistencyModule` for the constructor parameters.
    """

    def __init__(self, determining_ids, end_token, num_outclasses,
                 pad_token=0, p=2, detach_mode="none"):
        super().__init__(determining_ids, end_token, num_outclasses,
                         pad_token=pad_token, p=p, detach_mode=detach_mode)
        self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        self.last_predicate = ltn.Predicate(_TruthPredicate())

    def forward(self, act_logits, act_labels, outcome_logits, valid_mask=None):
        """Signature and return shape match `OutcomeConsistencyLoss.forward`."""
        p_det, det, survival, pos_valid = _determining_survival(
            act_logits, act_labels, self.act_of_class,
            self.end_token, self.pad_token,
        )
        # Last(sigma, t): a determining activity at t AND none strictly after it.
        last = det * survival                                          # (B, W)
        # out(act(sigma, t)): the outcome distribution the determining mass at t
        # implies, renormalised to a distribution. Well defined wherever det > 0;
        # elsewhere last ~ 0 makes the implication vacuous, so the 1e-8 floor
        # only guards the arithmetic.
        out_implied = p_det / det.clamp(min=1e-8).unsqueeze(-1)        # (B, W, K)
        outcome_probs = torch.softmax(outcome_logits, dim=-1)         # (B, K)

        if valid_mask is None:
            valid_mask = torch.ones(act_labels.shape[0], dtype=torch.bool,
                                    device=act_labels.device)
        elif valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()

        # One individual per valid (sigma, t): sigma in the non-leaky subset and
        # t a real suffix position (before END, not padding).
        keep = pos_valid & valid_mask.unsqueeze(1)                    # (B, W)
        if keep.sum() == 0:
            zero = outcome_logits.sum() * 0.0        # keeps the graph connected
            return zero, zero.detach(), {"num_instances": 0.0, "num_positions": 0.0}

        W = act_labels.shape[1]
        last_flat = last[keep]                                        # (N,)
        out_flat = out_implied[keep]                                 # (N, K)
        y_flat = outcome_probs.unsqueeze(1).expand(-1, W, -1)[keep]   # (N, K)

        if self.detach_mode == "act":
            last_flat = last_flat.detach()          # only the outcome head moves
            out_flat = out_flat.detach()
        elif self.detach_mode == "outcome":
            y_flat = y_flat.detach()                # only the activity head moves

        x_last = ltn.Variable("last", last_flat.unsqueeze(-1))
        x_out = ltn.Variable("out_implied", out_flat)
        x_y = ltn.Variable("outcome_pred", y_flat)
        # diag ties the three variables so they are iterated together (one
        # individual per valid position); it must be applied before the
        # predicates are evaluated, hence the inline first argument.
        sat_agg = self.Forall(
            ltn.diag(x_last, x_out, x_y),
            self.Implies(self.last_predicate(x_last),
                         self.predicate(x_out, x_y)),
        ).value

        diagnostics = self._collapsed_diagnostics(
            p_det, survival, outcome_probs, valid_mask, keep.sum().item()
        )
        return 1.0 - sat_agg, sat_agg.detach(), diagnostics


class NativeExistsOutcomeConsistencyLoss(_OutcomeConsistencyModule):
    """Same axiom, "excuse non-last" existential form (crisp-equivalent to
    native_last):

        forall sigma. forall t.
            [ Det(sigma, t) -> ( out(act(sigma, t)) = O(sigma)
                                 OR  exists s > t. Det(sigma, s) ) ]

    A determining position t must either imply the outcome or be excused by some
    later determining position; only the last determining position has no later
    one, so its equality is forced. The inner `exists s > t. Det` is a p-mean
    (p=5, max-like) over the later valid positions, computed with reverse-
    cumulative ops (O(W)) rather than by building "no determining after t"
    explicitly. Operators: product t-norm, Reichenbach implication, product
    t-conorm (probabilistic sum) for OR, Brier equality, p-mean-error (p=2) for
    the outer forall. One truth value per valid (sigma, t) pair.

    See `_OutcomeConsistencyModule` for the constructor parameters. `exists_p`
    is the exponent of the existential p-mean (default 5).
    """

    def __init__(self, determining_ids, end_token, num_outclasses,
                 pad_token=0, p=2, exists_p=5, detach_mode="none"):
        super().__init__(determining_ids, end_token, num_outclasses,
                         pad_token=pad_token, p=p, detach_mode=detach_mode)
        self.exists_p = exists_p
        self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        self.Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
        self.det_predicate = ltn.Predicate(_TruthPredicate())
        self.exists_predicate = ltn.Predicate(_TruthPredicate())

    def _exists_after(self, det, pos_valid):
        """exists s > t. Det(s), as a p-mean over the later VALID positions.

        p-mean = (mean over the domain of value^p)^(1/p); the domain of s is the
        valid positions strictly after t, so the denominator is their count.
        Computed with reverse-cumulative ops (O(W)). Uses ltn's stable p-mean
        semantics -- `pi_0` maps values into [eps, 1] before the power so the
        1/p root has a finite gradient -- and returns 0 where there is no later
        valid position (the existential is then vacuously false).
        """
        p = self.exists_p
        mask = pos_valid.to(det.dtype)
        num = ltn.fuzzy_ops.pi_0(det).pow(p) * mask                  # (B, W)

        def _tail(x):
            rev = torch.flip(torch.cumsum(torch.flip(x, [1]), dim=1), [1])
            tail = torch.zeros_like(rev)
            tail[:, :-1] = rev[:, 1:]                                 # strictly after t
            return tail

        tail_num = _tail(num)
        tail_cnt = _tail(mask)
        has_later = tail_cnt > 0
        base = tail_num / tail_cnt.clamp(min=1.0)
        # Dummy 1.0 where there is no later position so pow(1/p) never sees 0
        # (its gradient there would be infinite); forced back to 0 afterwards.
        base = torch.where(has_later, base, torch.ones_like(base))
        exists = base.pow(1.0 / p)
        return torch.where(has_later, exists, torch.zeros_like(exists))

    def forward(self, act_logits, act_labels, outcome_logits, valid_mask=None):
        """Signature and return shape match `OutcomeConsistencyLoss.forward`."""
        p_det, det, survival, pos_valid = _determining_survival(
            act_logits, act_labels, self.act_of_class,
            self.end_token, self.pad_token,
        )
        exists_after = self._exists_after(det, pos_valid)             # (B, W)
        out_implied = p_det / det.clamp(min=1e-8).unsqueeze(-1)       # (B, W, K)
        outcome_probs = torch.softmax(outcome_logits, dim=-1)        # (B, K)

        if valid_mask is None:
            valid_mask = torch.ones(act_labels.shape[0], dtype=torch.bool,
                                    device=act_labels.device)
        elif valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()

        keep = pos_valid & valid_mask.unsqueeze(1)                   # (B, W)
        if keep.sum() == 0:
            zero = outcome_logits.sum() * 0.0        # keeps the graph connected
            return zero, zero.detach(), {"num_instances": 0.0, "num_positions": 0.0}

        W = act_labels.shape[1]
        det_flat = det[keep]                                         # (N,)
        exists_flat = exists_after[keep]                            # (N,)
        out_flat = out_implied[keep]                                # (N, K)
        y_flat = outcome_probs.unsqueeze(1).expand(-1, W, -1)[keep]  # (N, K)

        if self.detach_mode == "act":
            det_flat = det_flat.detach()            # only the outcome head moves
            exists_flat = exists_flat.detach()
            out_flat = out_flat.detach()
        elif self.detach_mode == "outcome":
            y_flat = y_flat.detach()                # only the activity head moves

        x_det = ltn.Variable("det", det_flat.unsqueeze(-1))
        x_exists = ltn.Variable("exists_after", exists_flat.unsqueeze(-1))
        x_out = ltn.Variable("out_implied", out_flat)
        x_y = ltn.Variable("outcome_pred", y_flat)
        # diag must precede the predicates (see NativeLast). Body per position:
        # Det -> ( Eq OR exists-later-determining ).
        sat_agg = self.Forall(
            ltn.diag(x_det, x_exists, x_out, x_y),
            self.Implies(
                self.det_predicate(x_det),
                self.Or(self.predicate(x_out, x_y),
                        self.exists_predicate(x_exists)),
            ),
        ).value

        diagnostics = self._collapsed_diagnostics(
            p_det, survival, outcome_probs, valid_mask, keep.sum().item()
        )
        return 1.0 - sat_agg, sat_agg.detach(), diagnostics


class BrierMeanOutcomeConsistencyLoss(_OutcomeConsistencyModule):
    """Naive control for axiom 2: the collapsed distribution q and the same Brier
    equality as `OutcomeConsistencyLoss`, but aggregated with a plain arithmetic
    mean over instances instead of the ltn `Forall`/`AggregPMeanError` -- and
    computed directly, without the ltn Predicate/Quantifier wrapping. It isolates
    what the LTN quantifier buys over a vanilla per-instance Brier mean.

    Same logical content as `collapsed_q`; only the aggregator differs
    (arithmetic mean is more lenient than p-mean-error, which up-weights the
    worst-satisfied instances). See `_OutcomeConsistencyModule` for the
    constructor parameters (`p` is unused here).
    """

    def forward(self, act_logits, act_labels, outcome_logits, valid_mask=None):
        """Signature and return shape match `OutcomeConsistencyLoss.forward`."""
        p_det, _, survival, _ = _determining_survival(
            act_logits, act_labels, self.act_of_class,
            self.end_token, self.pad_token,
        )
        q = (p_det * survival.unsqueeze(-1)).sum(dim=1)               # (B, K)
        outcome_probs = torch.softmax(outcome_logits, dim=-1)        # (B, K)

        if valid_mask is None:
            valid_mask = torch.ones(act_labels.shape[0], dtype=torch.bool,
                                    device=act_labels.device)
        elif valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()

        q_v = q[valid_mask]
        y_v = outcome_probs[valid_mask]
        if q_v.shape[0] == 0:
            zero = outcome_logits.sum() * 0.0        # keeps the graph connected
            return zero, zero.detach(), {"num_instances": 0.0}

        total = q_v.sum(dim=-1)
        q_norm = q_v / total.clamp(min=1e-8).unsqueeze(-1)

        if self.detach_mode == "act":
            q_norm = q_norm.detach()                 # only the outcome head moves
        elif self.detach_mode == "outcome":
            y_v = y_v.detach()                       # only the activity head moves

        # Brier equality per instance, then a PLAIN arithmetic mean (the control:
        # no ltn Predicate, no p-mean-error quantifier).
        brier = 1.0 - 0.5 * (q_norm - y_v).pow(2).sum(dim=-1)        # (Bv,)
        sat_agg = brier.mean()

        diagnostics = self._collapsed_diagnostics(
            p_det, survival, outcome_probs, valid_mask
        )
        return 1.0 - sat_agg, sat_agg.detach(), diagnostics
