"""Tests for the axiom-2 `native_last` impl. CPU only, no data files, seconds.

    python test_ltn_outcome_native_last.py

Unlike the collapsed impl, native_last keeps one truth value per valid
(sigma, t): the last determining position is forced to agree with the outcome,
every other position is vacuously satisfied (Last ~ 0). So satisfaction is crisp
only when a single suffix position is valid; with several before-END positions
the vacuous ones dilute a disagreement rather than driving sat to 0. The tests
below check both the crisp core and that dilution direction.
"""

import sys

import torch

from ltn_outcome_consistency import NativeLastOutcomeConsistencyLoss

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


module = NativeLastOutcomeConsistencyLoss(DET_IDS, END, NUM_OUT, pad_token=PAD)


def sat_of(seq, cls):
    labels = torch.tensor([seq])
    _, sat, diag = module(one_hot_logits(seq), labels, outcome_logits(cls))
    return sat.item(), diag


# ------------------------------------------------- crisp single-position core --
print("\nsingle valid suffix position -> crisp satisfaction")
# [17, END, ...]: only t=0 is a valid position, so no vacuous positions dilute.
s_agree, d_agree = sat_of([17, END, PAD, PAD], 0)     # 17 -> Accepted, outcome Accepted
s_disag, d_disag = sat_of([17, END, PAD, PAD], 1)     # outcome disagrees
check("agreement -> sat ~ 1", s_agree > 0.99, f"sat={s_agree:.4f}")
check("disagreement -> sat ~ 0", s_disag < 0.01, f"sat={s_disag:.4f}")
check("exactly one position aggregated", d_agree["num_positions"] == 1.0)

# ------------------------------------------------------- last-occurrence wins --
print("\nthe LAST determining activity fixes the outcome")
CASES = [
    ([17, OTHER, 21, END], 1, "ACCEPT then CANCEL -> Canceled"),
    ([21, OTHER, 17, END], 0, "CANCEL then ACCEPT -> Accepted"),
    ([20, END, PAD, PAD], 2, "refused only"),
]
for seq, expected, desc in CASES:
    s_ok, _ = sat_of(seq, expected)
    others = [c for c in range(NUM_OUT) if c != expected]
    s_wrong = max(sat_of(seq, c)[0] for c in others)
    check(desc, s_ok > 0.99 and s_ok > s_wrong,
          f"sat(correct)={s_ok:.4f} > sat(best wrong)={s_wrong:.4f}")

print("\nnon-last determining positions are vacuously excused")
# Position 0 (ACCEPT -> Accepted) disagrees with the Canceled outcome, but a
# later determining position (CANCEL) excuses it: sat must stay ~ 1.
s_excuse, _ = sat_of([17, OTHER, 21, END], 1)
check("wrong non-last determining act does not lower sat", s_excuse > 0.99,
      f"sat={s_excuse:.4f}")

print("\nvacuous before-END positions dilute (not zero) a disagreement")
# Same suffix, outcome now disagrees with the LAST determining act (21). Two
# non-last positions are vacuously satisfied, so sat sits strictly between 0 and
# the agreeing case rather than at 0.
s_dilute, _ = sat_of([17, OTHER, 21, END], 0)
check("diluted disagreement is in (0, agree)", 0.0 < s_dilute < s_excuse,
      f"0 < {s_dilute:.4f} < {s_excuse:.4f}")

# ------------------------------------------------------------ loss / gradient --
print("\nloss = 1 - sat, and is differentiable")
seq = [17, END, PAD, PAD]
labels = torch.tensor([seq])
logits = one_hot_logits(seq, scale=2.0).requires_grad_(True)
loss, sat, diag = module(logits, labels, outcome_logits(1, scale=2.0))
check("loss == 1 - sat", abs(loss.item() - (1.0 - sat.item())) < 1e-6)
loss.backward()
check("gradient reaches act logits", logits.grad is not None and logits.grad.abs().sum() > 0)

# ------------------------------------------------------------- detach modes ---
print("\ndetach modes route gradient correctly")
# Unsaturated logits (scale=2) so gradients are visible; at scale=20 softmax
# saturates and every gradient looks like zero regardless of routing.
SOFT = 2.0
for mode, act_should, out_should in (("none", True, True),
                                     ("act", False, True),
                                     ("outcome", True, False)):
    m = NativeLastOutcomeConsistencyLoss(DET_IDS, END, NUM_OUT, pad_token=PAD, detach_mode=mode)
    a = one_hot_logits(seq, scale=SOFT).requires_grad_(True)
    o = outcome_logits(1, scale=SOFT).requires_grad_(True)
    loss, _, _ = m(a, labels, o)
    loss.backward()
    act_grad = a.grad is not None and a.grad.abs().sum().item() > 1e-9
    out_grad = o.grad is not None and o.grad.abs().sum().item() > 1e-9
    check(f"detach_mode={mode!r}: act grad={act_grad}, outcome grad={out_grad}",
          act_grad == act_should and out_grad == out_should)

# -------------------------------------------------------------- valid_mask ---
print("\nvalid_mask restricts to the non-leaky subset")
seqs = [[17, END, PAD, PAD], [21, END, PAD, PAD]]   # single valid position each
lab = torch.tensor(seqs)
al = torch.cat([one_hot_logits(s) for s in seqs])
ol = torch.cat([outcome_logits(0), outcome_logits(0)])   # 2nd instance disagrees
_, sat_both, d_both = module(al, lab, ol)
_, sat_first, d_first = module(al, lab, ol, valid_mask=torch.tensor([True, False]))
check("mask changes the instance count",
      d_both["num_instances"] == 2 and d_first["num_instances"] == 1)
check("mask changes the position count",
      d_both["num_positions"] == 2 and d_first["num_positions"] == 1)
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
check("positions aggregated over the whole batch", diag["num_positions"] > B,
      f"num_positions={diag['num_positions']:.0f}")

# ------------------------------------------------------------------ summary --
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name in FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
print("ALL TESTS PASSED")
