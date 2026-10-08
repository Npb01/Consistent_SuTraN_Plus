"""Tests for the axiom-2 `native_exists` impl. CPU only, no data files, seconds.

    python test_ltn_outcome_native_exists.py

native_exists is crisp-equivalent to native_last: a determining position must
imply the outcome OR be excused by a later determining position. In the soft
setting the excuse is a p-mean (p=5) over the later valid positions, so it is
sharp but not exactly 1 -- an excused disagreement scores high, not perfect. The
tests check the crisp core (single valid position matches native_last), the
excuse mechanism, and numerical stability (the 1/p root must not NaN on empty
tails).
"""

import sys

import torch

from ltn_outcome_consistency import (
    NativeExistsOutcomeConsistencyLoss,
    NativeLastOutcomeConsistencyLoss,
)

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
    t = torch.full((1, len(seq), NUM_ACT), -scale)
    for i, a in enumerate(seq):
        t[0, i, a] = scale
    return t


def outcome_logits(cls, scale=20.0):
    t = torch.full((1, NUM_OUT), -scale)
    t[0, cls] = scale
    return t


module = NativeExistsOutcomeConsistencyLoss(DET_IDS, END, NUM_OUT, pad_token=PAD)
native_last = NativeLastOutcomeConsistencyLoss(DET_IDS, END, NUM_OUT, pad_token=PAD)


def sat_of(mod, seq, cls):
    labels = torch.tensor([seq])
    _, sat, diag = mod(one_hot_logits(seq), labels, outcome_logits(cls))
    return sat.item(), diag


# ------------------------------------------------- crisp single-position core --
print("\nsingle valid suffix position -> crisp satisfaction")
s_agree, d_agree = sat_of(module, [17, END, PAD, PAD], 0)
s_disag, _ = sat_of(module, [17, END, PAD, PAD], 1)
check("agreement -> sat ~ 1", s_agree > 0.99, f"sat={s_agree:.4f}")
check("disagreement -> sat ~ 0", s_disag < 0.01, f"sat={s_disag:.4f}")
check("exactly one position aggregated", d_agree["num_positions"] == 1.0)

print("\ncrisp cases match native_last (equivalent in the hard limit)")
for seq, cls, desc in [([17, END, PAD, PAD], 0, "single agree"),
                       ([17, END, PAD, PAD], 1, "single disagree"),
                       ([20, END, PAD, PAD], 2, "refused only")]:
    se, _ = sat_of(module, seq, cls)
    sl, _ = sat_of(native_last, seq, cls)
    check(f"{desc}: exists ~ last", abs(se - sl) < 1e-3, f"{se:.4f} vs {sl:.4f}")

# ------------------------------------------------------- last-occurrence wins --
print("\nthe LAST determining activity fixes the outcome")
CASES = [
    ([17, OTHER, 21, END], 1, "ACCEPT then CANCEL -> Canceled"),
    ([21, OTHER, 17, END], 0, "CANCEL then ACCEPT -> Accepted"),
]
for seq, expected, desc in CASES:
    s_ok, _ = sat_of(module, seq, expected)
    s_wrong = max(sat_of(module, seq, c)[0] for c in range(NUM_OUT) if c != expected)
    check(desc, s_ok > s_wrong and s_ok > 0.85,
          f"sat(correct)={s_ok:.4f} > sat(best wrong)={s_wrong:.4f}")

# ------------------------------------------------------ the excuse mechanism ---
print("\na wrong NON-last determining act is excused by a later determining one")
# pos0 = ACCEPT (-> Accepted) disagrees with the Refused outcome, but pos1 =
# REFUSE is a later determining position that excuses it AND itself agrees.
s_excused, _ = sat_of(module, [17, 20, END, PAD], 2)
# Same wrong determining act, now with NO later determining position to excuse
# it (it is the last determining position) -> forced, so sat collapses.
s_forced, _ = sat_of(module, [17, END, PAD, PAD], 2)
check("excused disagreement stays high", s_excused > 0.85, f"sat={s_excused:.4f}")
check("un-excused (last) disagreement collapses", s_forced < 0.01, f"sat={s_forced:.4f}")
check("excuse >> no-excuse", s_excused > s_forced + 0.5,
      f"{s_forced:.4f} -> {s_excused:.4f}")

# ------------------------------------------------------------ loss / gradient --
print("\nloss = 1 - sat, differentiable, and finite (no 1/p-root NaN)")
seq = [17, END, PAD, PAD]
labels = torch.tensor([seq])
logits = one_hot_logits(seq, scale=2.0).requires_grad_(True)
loss, sat, diag = module(logits, labels, outcome_logits(1, scale=2.0))
check("loss == 1 - sat", abs(loss.item() - (1.0 - sat.item())) < 1e-6)
loss.backward()
check("gradient reaches act logits and is finite",
      logits.grad is not None and logits.grad.abs().sum() > 0
      and torch.isfinite(logits.grad).all())

print("\nempty-tail positions (last position) do not produce NaN gradients")
torch.manual_seed(1)
B, W = 32, 46
rl = torch.randn(B, W, NUM_ACT, requires_grad=True)
lab = torch.full((B, W), OTHER)
for b in range(B):
    lab[b, 9] = DET_IDS[b % 3]
    lab[b, 13] = END
    lab[b, 14:] = PAD
loss, _, _ = module(rl, lab, torch.randn(B, NUM_OUT))
loss.backward()
check("all activity-logit gradients finite", torch.isfinite(rl.grad).all(),
      f"|grad|={rl.grad.abs().sum().item():.3e}")

# ------------------------------------------------------------- detach modes ---
print("\ndetach modes route gradient correctly")
SOFT = 2.0
for mode, act_should, out_should in (("none", True, True),
                                     ("act", False, True),
                                     ("outcome", True, False)):
    m = NativeExistsOutcomeConsistencyLoss(DET_IDS, END, NUM_OUT, pad_token=PAD, detach_mode=mode)
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
seqs = [[17, END, PAD, PAD], [21, END, PAD, PAD]]
lab2 = torch.tensor(seqs)
al = torch.cat([one_hot_logits(s) for s in seqs])
ol = torch.cat([outcome_logits(0), outcome_logits(0)])   # 2nd instance disagrees
_, sat_both, d_both = module(al, lab2, ol)
_, sat_first, d_first = module(al, lab2, ol, valid_mask=torch.tensor([True, False]))
check("mask changes the instance count",
      d_both["num_instances"] == 2 and d_first["num_instances"] == 1)
check("masking out the disagreeing instance raises sat",
      sat_first.item() > sat_both.item(),
      f"{sat_both.item():.4f} -> {sat_first.item():.4f}")
_, _, d_empty = module(al, lab2, ol, valid_mask=torch.tensor([False, False]))
check("all-masked batch is handled", d_empty["num_instances"] == 0)

# -------------------------------------------------------------- diagnostics --
print("\ndiagnostics are sane on a realistic-shaped batch")
_, _, diag = module(rl.detach(), lab, torch.randn(B, NUM_OUT))
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
