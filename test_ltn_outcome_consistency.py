"""Tests for the axiom-2 LTN module. CPU only, no data files, seconds to run.

    python test_ltn_outcome_consistency.py

Run this before any GPU time. The failure modes it targets -- a wrong mask, an
off-by-one in the "strictly after t" product, gradient reaching a head that was
supposed to be frozen -- all produce plausible numbers rather than errors.
"""

import sys

import torch

from ltn_outcome_consistency import OutcomeConsistencyLoss, _OutcomeBrierPredicate

# Mirrors BPIC_17_DR: ids are categ_mapping + 1, END is the highest id.
DET_IDS = {0: 17, 1: 21, 2: 20}      # Accepted / Canceled / Refused
END = 27
PAD = 0
OTHER = 3                            # an ordinary, non-determining activity
NUM_ACT = 28
NUM_OUT = 3

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'OK  ' if condition else 'FAIL'} {name}{('  -- ' + detail) if detail else ''}")


def one_hot_logits(seq, scale=20.0):
    """(1, W, NUM_ACT) logits that are effectively one-hot on `seq`."""
    t = torch.full((1, len(seq), NUM_ACT), -scale)
    for i, a in enumerate(seq):
        t[0, i, a] = scale
    return t


def outcome_logits(cls, scale=20.0):
    t = torch.full((1, NUM_OUT), -scale)
    t[0, cls] = scale
    return t


module = OutcomeConsistencyLoss(DET_IDS, END, NUM_OUT, pad_token=PAD)

# ---------------------------------------------------------------- grounding --
print("\nimplied distribution q(o) on hard one-hot suffixes")
CASES = [
    ([17, OTHER, END, PAD], 0, "accepted only"),
    ([21, OTHER, END, PAD], 1, "cancelled only"),
    ([20, END, PAD, PAD], 2, "refused only"),
    ([17, OTHER, 21, END], 1, "ACCEPT then CANCEL -> Canceled"),
    ([21, OTHER, 17, END], 0, "CANCEL then ACCEPT -> Accepted"),
    ([17, END, 21, OTHER], 0, "determining act after END is ignored"),
    ([17, OTHER, OTHER, OTHER], 0, "no END (window-limited decode)"),
]
for seq, expected, desc in CASES:
    labels = torch.tensor([seq])
    q, _ = module.implied_distribution(one_hot_logits(seq), labels)
    got = int(q.argmax(-1))
    check(desc, got == expected and q.max() > 0.9,
          f"q={[round(v, 3) for v in q[0].tolist()]}")

print("\nno determining activity -> q is near zero everywhere")
seq = [OTHER, OTHER, END, PAD]
q, _ = module.implied_distribution(one_hot_logits(seq), torch.tensor([seq]))
check("empty q", q.sum().item() < 0.01, f"sum(q)={q.sum().item():.5f}")

# --------------------------------------------------------- Brier equality ---
print("\nBrier equality predicate Eq(q, y) = 1 - 1/2 sum (q-y)^2")
brier = _OutcomeBrierPredicate()
BRIER_CASES = [
    ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0, "identical one-hot -> 1"),
    ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0, "opposite one-hot -> 0"),
    ([0.5, 0.5, 0.0], [0.5, 0.5, 0.0], 1.0, "identical diffuse -> 1 (dot would give 0.5)"),
    ([1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3], 2 / 3, "one-hot vs uniform -> 2/3"),
]
for q_v, y_v, expected, desc in BRIER_CASES:
    got = brier(torch.tensor([q_v]), torch.tensor([y_v])).item()
    check(desc, abs(got - expected) < 1e-6, f"got={got:.4f} expected={expected:.4f}")
check("Brier stays in [0, 1] on random distributions", (
    lambda v: (0.0 <= v.min().item()) and (v.max().item() <= 1.0)
)(brier(torch.softmax(torch.randn(256, NUM_OUT), -1),
        torch.softmax(torch.randn(256, NUM_OUT), -1))))

# ------------------------------------------------------------- satisfaction --
print("\nsatisfaction responds to agreement")
seq = [17, OTHER, END, PAD]
labels = torch.tensor([seq])
_, sat_agree, _ = module(one_hot_logits(seq), labels, outcome_logits(0))
_, sat_disagree, _ = module(one_hot_logits(seq), labels, outcome_logits(1))
check("agreement -> sat ~ 1", sat_agree.item() > 0.99, f"sat={sat_agree.item():.4f}")
check("disagreement -> sat ~ 0", sat_disagree.item() < 0.01, f"sat={sat_disagree.item():.4f}")
check("agree > disagree", sat_agree.item() > sat_disagree.item())

print("\nloss = 1 - sat, and is differentiable")
logits = one_hot_logits(seq).requires_grad_(True)
loss, sat, diag = module(logits, labels, outcome_logits(1))
check("loss == 1 - sat", abs(loss.item() - (1.0 - sat.item())) < 1e-6)
loss.backward()
check("gradient reaches act logits", logits.grad is not None and logits.grad.abs().sum() > 0)

# ------------------------------------------------------------- detach modes --
print("\ndetach modes route gradient correctly")
# Deliberately UNSATURATED logits (scale=2, not 20). At scale=20 the softmax is
# effectively one-hot, dp/dlogit = p(1-p) ~ 4e-18, and every gradient looks like
# zero -- the saturation effect, not a routing bug. Testing routing needs logits
# where gradients are visible.
SOFT = 2.0
for mode, act_should, out_should in (("none", True, True),
                                     ("act", False, True),
                                     ("outcome", True, False)):
    m = OutcomeConsistencyLoss(DET_IDS, END, NUM_OUT, pad_token=PAD, detach_mode=mode)
    a = one_hot_logits(seq, scale=SOFT).requires_grad_(True)
    o = outcome_logits(1, scale=SOFT).requires_grad_(True)
    loss, _, _ = m(a, labels, o)
    loss.backward()
    act_grad = a.grad is not None and a.grad.abs().sum().item() > 1e-9
    out_grad = o.grad is not None and o.grad.abs().sum().item() > 1e-9
    check(f"detach_mode={mode!r}: act grad={act_grad}, outcome grad={out_grad}",
          act_grad == act_should and out_grad == out_should)

print("\nsoftmax saturation attenuates the activity-head gradient (predicted effect)")
grads = {}
for scale in (1.0, 2.0, 5.0, 20.0):
    m = OutcomeConsistencyLoss(DET_IDS, END, NUM_OUT, pad_token=PAD)
    a = one_hot_logits(seq, scale=scale).requires_grad_(True)
    loss, _, _ = m(a, labels, outcome_logits(1, scale=SOFT))
    loss.backward()
    grads[scale] = a.grad.abs().sum().item()
    print(f"       activity-logit confidence scale={scale:<5} |grad|={grads[scale]:.3e}")
check("gradient shrinks as the activity head grows confident",
      grads[1.0] > grads[5.0] > grads[20.0],
      "confirms the saturation/attenuation effect")

# -------------------------------------------------------------- valid_mask ---
print("\nvalid_mask restricts to the non-leaky subset")
seqs = [[17, OTHER, END, PAD], [21, OTHER, END, PAD]]
lab = torch.tensor(seqs)
al = torch.cat([one_hot_logits(s) for s in seqs])
ol = torch.cat([outcome_logits(0), outcome_logits(0)])   # 2nd instance disagrees
_, sat_both, d_both = module(al, lab, ol)
_, sat_first, d_first = module(al, lab, ol, valid_mask=torch.tensor([True, False]))
check("mask changes the instance count",
      d_both["num_instances"] == 2 and d_first["num_instances"] == 1)
check("masking out the disagreeing instance raises sat",
      sat_first.item() > sat_both.item(),
      f"{sat_both.item():.4f} -> {sat_first.item():.4f}")
_, _, d_empty = module(al, lab, ol, valid_mask=torch.tensor([False, False]))
check("all-masked batch is handled", d_empty["num_instances"] == 0)

# -------------------------------------------------------------- diagnostics --
print("\ndiagnostics are sane on a realistic-shaped batch")
torch.manual_seed(0)
B, W = 64, 46
rand_logits = torch.randn(B, W, NUM_ACT)
rand_labels = torch.full((B, W), OTHER)
for b in range(B):
    rand_labels[b, 9] = DET_IDS[b % 3]
    rand_labels[b, 13] = END
    rand_labels[b, 14:] = PAD
_, _, diag = module(rand_logits, rand_labels, torch.randn(B, NUM_OUT))
check("min_survival far above float32 underflow", diag["min_survival"] > 1e-6,
      f"min_survival={diag['min_survival']:.4f}")
check("residual mass in [0,1]", 0.0 <= diag["mean_residual_mass"] <= 1.0,
      f"residual={diag['mean_residual_mass']:.4f}")

print("\nlonger suffixes do not underflow the product")
long_labels = torch.full((1, 46), OTHER)
long_labels[0, 45] = END
_, surv = module.implied_distribution(torch.randn(1, 46, NUM_ACT), long_labels)
check("45-step survival still representable", surv.min().item() > 1e-6,
      f"min={surv.min().item():.3e}")

# ------------------------------------------------------------------ summary --
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name in FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
print("ALL TESTS PASSED")
