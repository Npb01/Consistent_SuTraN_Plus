"""Soft guarded quantification: a universal quantifier with a fuzzy guard.

LTN's guarded quantifier evaluates `forall x : A(x). B(x)` by aggregating B
over the individuals for which the crisp guard A holds; the guard is a Boolean
mask and carries no gradient. This module extends it to a fuzzy guard
A(x) in [0, 1]: every individual contributes to the aggregate in proportion to
its guard value,

    forall x : A(x). B(x)  =  1 - ( sum_x A(x) (1 - B(x))^p / sum_x A(x) )^(1/p),

the p-mean-error (LTN's universal aggregator) weighted by A. For p = 1 it is
the A-weighted mean of B, the degree to which B holds given A.

Properties:
  - For a crisp guard it equals LTN's guarded quantification with
    `AggregPMeanError` (same stabilisation).
  - The guard is a weight, not a premise: it receives no gradient. The
    constraint can only be satisfied by making B hold where A holds, not by
    making A false (vacuous satisfaction). A is learned through other losses.
  - The value does not change when A is scaled uniformly, and individuals with
    A(x) = 0 do not contribute.
  - An empty guard (sum_x A(x) = 0) gives 1, as LTN's guarded forall over an
    empty domain, without gradient.

The rule is read as a conditional ("B, given A") rather than as the material
implication A -> B.
"""

from __future__ import annotations

import torch
import ltn


class AggregSoftGuardedPMeanError(ltn.fuzzy_ops.AggregationOperator):
    """p-mean-error aggregation weighted by a fuzzy guard.

    Parameters
    ----------
    p : int, optional
        Exponent of the p-mean-error. p = 1 is the guard-weighted mean; larger
        p emphasises the worst-satisfied individuals (p -> inf: the minimum
        over the individuals with a non-zero guard).
    stable : bool, optional
        Apply LTN's `pi_1` projection so the 1/p root keeps a finite gradient
        when every individual is satisfied. Not needed for p = 1.

    The guard is always fully normalised: however small its total mass, the
    individuals it selects carry the full weight of the aggregate. Only an
    empty guard (total mass 0) gives the vacuous value 1.
    """

    def __init__(self, p=2, stable=True):
        self.p = p
        self.stable = stable

    def __repr__(self):
        return f"AggregSoftGuardedPMeanError(p={self.p}, stable={self.stable})"

    def __call__(self, xs, guard, dim=None, keepdim=False):
        """
        Parameters
        ----------
        xs : torch.Tensor
            Truth values of the consequent B, in [0, 1].
        guard : torch.Tensor
            Truth values of the guard A, in [0, 1], broadcastable to `xs`.
            Detached here: it only weights the individuals.
        dim : int or tuple of int, optional
            Dimension(s) quantified over; all of them when None.
        keepdim : bool, optional

        Returns
        -------
        torch.Tensor
            Truth value of the guarded universal, in [0, 1].
        """
        guard = guard.detach().expand_as(xs)
        if self.stable:
            xs = ltn.fuzzy_ops.pi_1(xs)
        err = (1.0 - xs).pow(self.p)
        if dim is None:
            dim = tuple(range(xs.dim()))
        num = (guard * err).sum(dim=dim, keepdim=keepdim)
        mass = guard.sum(dim=dim, keepdim=keepdim)
        # An empty guard gives 1. The inner where keeps the division and the
        # root away from 0 there, so the gradient stays finite (0 * inf would
        # give NaN).
        empty = num <= 0
        ratio = torch.where(empty, torch.ones_like(num), num / torch.where(empty, torch.ones_like(mass), mass))
        return torch.where(empty, torch.ones_like(num), 1.0 - ratio.pow(1.0 / self.p))
