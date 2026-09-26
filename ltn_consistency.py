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


class CrossTaskConsistencyLoss(torch.nn.Module):
    """Axiom 1: sum(timestamp-suffix predictions) ~= remaining-time prediction.

    Both quantities are reconstructed in unstandardized units from the
    standardized model outputs and the log's stored mean/std, so the axiom is
    enforced on real durations rather than z-scores.
    """

    def __init__(self, ts_mean, ts_std, rt_mean, rt_std, scale=None, p=2, detach_mode="none"):
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
        ts_unstd = (ts_suffix_pred_std * self.ts_std + self.ts_mean) * ts_suffix_mask
        sum_ts = ts_unstd.sum(dim=1)
        rt_unstd = rt_pred_std * self.rt_std + self.rt_mean

        if self.detach_mode == "ttne":
            sum_ts = sum_ts.detach()      # only rrt moves toward the ttne-sum
        elif self.detach_mode == "rrt":
            rt_unstd = rt_unstd.detach()  # only ttne moves toward rrt

        x_sum = ltn.Variable("sum_ts", sum_ts)
        x_rt = ltn.Variable("rt_pred", rt_unstd)
        sat_agg = self.Forall(ltn.diag(x_sum, x_rt), self.predicate(x_sum, x_rt)).value

        return 1.0 - sat_agg, sat_agg.detach()
