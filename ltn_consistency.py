import torch
import ltn


class _ConsistencyPredicateModel(torch.nn.Module):
    """Smooth equality predicate: truth degree decays with the absolute error
    between the predicted timestamp-suffix sum and the predicted remaining
    runtime, both in original (unstandardized) units."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale  # normalizer; defaults to the remaining-time target's std

    def forward(self, sum_ts: torch.Tensor, rt_pred: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(sum_ts - rt_pred)
        return torch.exp(-diff / self.scale)


class _CrossTaskConsistencyBase(torch.nn.Module):
    """Shared setup for the axiom-1 impls: the mean/std buffers used to
    reconstruct predictions in unstandardized units, and the detach-mode
    handling. Subclasses implement `forward`.

    Both quantities are reconstructed in real units from the standardized model
    outputs and the log's stored mean/std, so the axiom is enforced on real
    durations rather than z-scores.
    """

    def __init__(self, ts_mean, ts_std, rt_mean, rt_std, detach_mode="none"):
        super().__init__()
        if detach_mode not in ("none", "ttne", "rrt"):
            raise ValueError(
                "detach_mode must be 'none', 'ttne' (freeze the ttne-sum), "
                "or 'rrt' (freeze rrt)"
            )
        self.detach_mode = detach_mode
        self.register_buffer("ts_mean", torch.tensor(float(ts_mean)))
        self.register_buffer("ts_std", torch.tensor(float(ts_std)))
        self.register_buffer("rt_mean", torch.tensor(float(rt_mean)))
        self.register_buffer("rt_std", torch.tensor(float(rt_std)))

    def _reconstruct(self, ts_suffix_pred_std, ts_suffix_mask, rt_pred_std):
        """Return (sum_ts, rt) in unstandardized units, with the detach mode
        applied: 'ttne' freezes the ttne-sum, 'rrt' freezes rrt."""
        ts_unstd = (ts_suffix_pred_std * self.ts_std + self.ts_mean) * ts_suffix_mask
        sum_ts = ts_unstd.sum(dim=1)
        rt_unstd = rt_pred_std * self.rt_std + self.rt_mean

        if self.detach_mode == "ttne":
            sum_ts = sum_ts.detach()      # only rrt moves toward the ttne-sum
        elif self.detach_mode == "rrt":
            rt_unstd = rt_unstd.detach()  # only ttne moves toward rrt
        return sum_ts, rt_unstd


class CrossTaskConsistencyLoss(_CrossTaskConsistencyBase):
    """Axiom 1 (LTN): sum(timestamp-suffix predictions) ~= remaining-time
    prediction, via a smooth-equality predicate aggregated by a p-mean-error
    Forall. See `_CrossTaskConsistencyBase` for the constructor parameters.
    """

    def __init__(self, ts_mean, ts_std, rt_mean, rt_std, scale=None, p=2, detach_mode="none"):
        super().__init__(ts_mean, ts_std, rt_mean, rt_std, detach_mode=detach_mode)
        pred_scale = scale if scale is not None else float(rt_std)
        self.predicate = ltn.Predicate(_ConsistencyPredicateModel(scale=pred_scale))
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=p), quantifier="f")

    def forward(self, ts_suffix_pred_std, ts_suffix_mask, rt_pred_std):
        """
        ts_suffix_pred_std : (B, window_size)  standardized Delta-t predictions
        ts_suffix_mask     : (B, window_size)  1 = real event, 0 = padding
        rt_pred_std        : (B,)              standardized remaining-time prediction

        Returns (loss, sat_degree), where loss = 1 - satisfaction.
        """
        sum_ts, rt_unstd = self._reconstruct(ts_suffix_pred_std, ts_suffix_mask, rt_pred_std)

        x_sum = ltn.Variable("sum_ts", sum_ts)
        x_rt = ltn.Variable("rt_pred", rt_unstd)
        sat_agg = self.Forall(ltn.diag(x_sum, x_rt), self.predicate(x_sum, x_rt)).value

        return 1.0 - sat_agg, sat_agg.detach()


class NaivePenaltyConsistencyLoss(_CrossTaskConsistencyBase):
    """Axiom 1 naive control (no LTN): a plain differentiable penalty on the
    sigma_R-normalized gap between the timestamp-suffix sum and the remaining-
    time prediction.

        loss = mean( ((sum_ts - rt) / sigma_R)^2 )

    Added the same lambda-weighted way as the LTN impl, with the same detach
    modes. It shows whether the LTN wrapper buys anything over a vanilla
    regression penalty for this scalar equality. The returned satisfaction
    reuses the LTN smooth equality exp(-|sum_ts - rt| / sigma_R) (plain mean) so
    it stays a comparable diagnostic across the two impls; only `loss` is used by
    the training loop. See `_CrossTaskConsistencyBase` for the constructor
    parameters.
    """

    def forward(self, ts_suffix_pred_std, ts_suffix_mask, rt_pred_std):
        """Signature and return shape match `CrossTaskConsistencyLoss.forward`;
        here `loss` is the penalty (not 1 - sat)."""
        sum_ts, rt_unstd = self._reconstruct(ts_suffix_pred_std, ts_suffix_mask, rt_pred_std)

        gap = (sum_ts - rt_unstd) / self.rt_std          # sigma_R-normalized
        loss = gap.pow(2).mean()
        sat = torch.exp(-gap.abs()).mean().detach()
        return loss, sat
