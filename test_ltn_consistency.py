"""Tests for the axiom-1 modules (LTN smooth-eq and the naive-penalty control).
CPU only, no data files, seconds to run.

    python test_ltn_consistency.py

Axiom 1: the sum of the timestamp-suffix (Delta-t) predictions must equal the
remaining-time prediction, both reconstructed in unstandardized units. The LTN
impl scores this with a smooth-equality predicate + p-mean-error Forall
(loss = 1 - sat); the naive control uses a plain sigma_R-normalized squared-gap
penalty (loss = the penalty). Both share the reconstruction + detach handling.
"""

import sys

import torch

from ltn_consistency import CrossTaskConsistencyLoss, NaivePenaltyConsistencyLoss

# Arbitrary but non-trivial standardization stats, to exercise reconstruction.
TS_MEAN, TS_STD = 10.0, 5.0
RT_MEAN, RT_STD = 100.0, 50.0

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'OK  ' if condition else 'FAIL'} {name}{('  -- ' + detail) if detail else ''}")


def make(ts_std_vals, mask_vals, rt_std_val):
    ts = torch.tensor([ts_std_vals], dtype=torch.float32)
    mask = torch.tensor([mask_vals], dtype=torch.float32)
    rt = torch.tensor([rt_std_val], dtype=torch.float32)
    return ts, mask, rt


def ltn_module(detach_mode="none"):
    return CrossTaskConsistencyLoss(TS_MEAN, TS_STD, RT_MEAN, RT_STD, detach_mode=detach_mode)


def naive_module(detach_mode="none"):
    return NaivePenaltyConsistencyLoss(TS_MEAN, TS_STD, RT_MEAN, RT_STD, detach_mode=detach_mode)


# A consistent instance: 4 steps each unstd = 0*5+10 = 10 -> sum = 40; rt must
# reconstruct to 40, i.e. rt_std = (40 - 100) / 50 = -1.2.
CONSISTENT = make([0.0, 0.0, 0.0, 0.0], [1, 1, 1, 1], -1.2)
# Same suffix sum (40) but rt reconstructs to 140 (rt_std = 0.8): gap = 100.
INCONSISTENT = make([0.0, 0.0, 0.0, 0.0], [1, 1, 1, 1], 0.8)

# ------------------------------------------------------- reconstruction check --
print("\nreconstruction in unstandardized units (mask + mean/std applied)")
lm = ltn_module()
sum_ts, rt = lm._reconstruct(*CONSISTENT)
check("sum of unstandardized Delta-t = 40", abs(sum_ts.item() - 40.0) < 1e-4,
      f"sum_ts={sum_ts.item():.3f}")
check("remaining-time reconstructs to 40", abs(rt.item() - 40.0) < 1e-4,
      f"rt={rt.item():.3f}")
# padding must be excluded: only 2 real steps -> sum = 20
sum_masked, _ = lm._reconstruct(*make([0.0, 0.0, 0.0, 0.0], [1, 1, 0, 0], 0.0))
check("padding excluded from the sum", abs(sum_masked.item() - 20.0) < 1e-4,
      f"sum_ts={sum_masked.item():.3f}")

# --------------------------------------------------------------- LTN smooth-eq --
print("\nLTN smooth-eq: satisfaction responds to the gap; loss = 1 - sat")
_, sat_c = lm(*CONSISTENT)
_, sat_i = lm(*INCONSISTENT)
check("consistent -> sat ~ 1", sat_c.item() > 0.99, f"sat={sat_c.item():.4f}")
check("inconsistent -> sat low", sat_i.item() < 0.2, f"sat={sat_i.item():.4f}")
check("consistent more satisfied than inconsistent", sat_c.item() > sat_i.item())
loss_c, sat_c2 = lm(*CONSISTENT)
check("loss == 1 - sat", abs(loss_c.item() - (1.0 - sat_c2.item())) < 1e-6)

# ----------------------------------------------------------- naive penalty ----
print("\nnaive penalty: loss is the sigma_R-normalized squared gap")
loss_c, nsat_c = naive_module()(*CONSISTENT)
loss_i, nsat_i = naive_module()(*INCONSISTENT)
check("consistent -> penalty ~ 0", loss_c.item() < 1e-6, f"loss={loss_c.item():.3e}")
# gap = (40 - 140)/50 = -2  ->  penalty = 4
check("inconsistent -> penalty = (gap/sigma_R)^2 = 4", abs(loss_i.item() - 4.0) < 1e-4,
      f"loss={loss_i.item():.4f}")
check("penalty grows with the gap", loss_i.item() > loss_c.item())
check("satisfaction diagnostic in [0,1], ~1 when consistent",
      0.0 <= nsat_c.item() <= 1.0 and nsat_c.item() > 0.99, f"sat={nsat_c.item():.4f}")
check("naive loss is NOT bounded by 1 (unlike 1 - sat)", loss_i.item() > 1.0)

# --------------------------------------------------------- differentiability --
print("\nboth impls are differentiable through the predictions")
for name, mod in (("ltn", ltn_module()), ("naive", naive_module())):
    ts = torch.tensor([[0.5, 0.5, 0.5, 0.5]], requires_grad=True)
    rt = torch.tensor([0.3], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    loss, _ = mod(ts, mask, rt)
    loss.backward()
    check(f"{name}: gradient reaches ttne and rrt",
          ts.grad is not None and ts.grad.abs().sum() > 0
          and rt.grad is not None and rt.grad.abs().sum() > 0)

# ------------------------------------------------------------- detach modes ---
print("\ndetach modes route gradient correctly (both impls)")
for name, factory in (("ltn", ltn_module), ("naive", naive_module)):
    for mode, ttne_should, rrt_should in (("none", True, True),
                                          ("ttne", False, True),
                                          ("rrt", True, False)):
        mod = factory(detach_mode=mode)
        ts = torch.tensor([[0.5, 0.5, 0.5, 0.5]], requires_grad=True)
        rt = torch.tensor([0.3], requires_grad=True)
        mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        loss, _ = mod(ts, mask, rt)
        loss.backward()
        ttne_grad = ts.grad is not None and ts.grad.abs().sum().item() > 1e-9
        rrt_grad = rt.grad is not None and rt.grad.abs().sum().item() > 1e-9
        check(f"{name} detach={mode!r}: ttne grad={ttne_grad}, rrt grad={rrt_grad}",
              ttne_grad == ttne_should and rrt_grad == rrt_should)

# ---------------------------------------------------- invalid detach rejected --
print("\ninvalid detach_mode is rejected")
try:
    ltn_module(detach_mode="bogus")
    check("bad detach_mode raises", False)
except ValueError:
    check("bad detach_mode raises", True)

# ------------------------------------------------------------------ summary --
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name in FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
print("ALL TESTS PASSED")
