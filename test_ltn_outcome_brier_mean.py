"""Tests for the axiom-2 `brier_mean` naive control. CPU only, seconds to run.

    python test_ltn_outcome_brier_mean.py

brier_mean is `collapsed_q` with the ltn quantifier removed: the same
collapsed q and the same Brier equality, aggregated with a plain arithmetic mean
instead of ltn `Forall`/`AggregPMeanError`. The tests pin that the per-instance
score is identical to the collapsed impl and that only the aggregation differs
(mean is more lenient than p-mean-error, which up-weights the worst instances).
"""

import sys

import torch

from ltn_outcome_consistency import (
    BrierMeanOutcomeConsistencyLoss,
    OutcomeConsistencyLoss,
)

# Mirrors BPIC_17_DR: ids are categ_mapping + 1, END is the highest id.
DET_IDS = {0: 17, 1: 21, 2: 20}      # Accepted / Canceled / Refused
END = 27
PAD = 0
OTHER = 3
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


module = BrierMeanOutcomeConsistencyLoss(DET_IDS, END, NUM_OUT, pad_token=PAD)
collapsed = OutcomeConsistencyLoss(DET_IDS, END, NUM_OUT, pad_token=PAD)


def sat_of(mod, seq, cls):
    labels = torch.tensor([seq])
    _, sat, diag = mod(one_hot_logits(seq), labels, outcome_logits(cls))
    return sat.item(), diag


# --------------------------------------------------------------- crisp core --
print("\nsingle instance -> crisp satisfaction")
s_agree, _ = sat_of(module, [17, END, PAD, PAD], 0)
s_disag, _ = sat_of(module, [17, END, PAD, PAD], 1)
check("agreement -> sat ~ 1", s_agree > 0.99, f"sat={s_agree:.4f}")
check("disagreement -> sat ~ 0", s_disag < 0.01, f"sat={s_disag:.4f}")

print("\nlast determining activity fixes the outcome (same q as collapsed)")
for seq, expected, desc in [([17, OTHER, 21, END], 1, "ACCEPT then CANCEL -> Canceled"),
                            ([21, OTHER, 17, END], 0, "CANCEL then ACCEPT -> Accepted"),
                            ([20, END, PAD, PAD], 2, "refused only")]:
    s_ok, _ = sat_of(module, seq, expected)
    s_wrong = max(sat_of(module, seq, c)[0] for c in range(NUM_OUT) if c != expected)
    check(desc, s_ok > 0.99 and s_ok > s_wrong,
          f"sat(correct)={s_ok:.4f} > sat(best wrong)={s_wrong:.4f}")

# ------------------------------------------------- aggregator is a plain mean --
print("\naggregation is a PLAIN arithmetic mean (not p-mean-error)")
# Mixed batch: one agreeing instance (Brier 1), one disagreeing (Brier 0).
seqs = [[17, END, PAD, PAD], [21, END, PAD, PAD]]
lab = torch.tensor(seqs)
al = torch.cat([one_hot_logits(s) for s in seqs])
ol = torch.cat([outcome_logits(0), outcome_logits(0)])   # 2nd instance disagrees
_, sat_bm, _ = module(al, lab, ol)
_, sat_cs, _ = collapsed(al, lab, ol)
check("mean of [1, 0] == 0.5", abs(sat_bm.item() - 0.5) < 1e-4, f"sat={sat_bm.item():.4f}")
check("mean is more lenient than collapsed's p-mean-error",
      sat_bm.item() > sat_cs.item(),
      f"brier_mean={sat_bm.item():.4f} > collapsed={sat_cs.item():.4f}")

print("\nsat equals the independently-computed mean of per-instance Briers")
torch.manual_seed(0)
B, W = 16, 20
logits = torch.randn(B, W, NUM_ACT)
labels = torch.full((B, W), OTHER)
for b in range(B):
    labels[b, 5] = DET_IDS[b % 3]
    labels[b, 8] = END
    labels[b, 9:] = PAD
out_log = torch.randn(B, NUM_OUT)
_, sat, _ = module(logits, labels, out_log)
# recompute q and Brier by hand
q, _ = collapsed.implied_distribution(logits, labels)
q_norm = q / q.sum(-1, keepdim=True).clamp(min=1e-8)
y = torch.softmax(out_log, dim=-1)
brier_manual = (1.0 - 0.5 * (q_norm - y).pow(2).sum(-1)).mean().item()
check("sat == mean of Briers", abs(sat.item() - brier_manual) < 1e-5,
      f"{sat.item():.6f} vs {brier_manual:.6f}")

# ------------------------------------------------------------ loss / gradient --
print("\nloss = 1 - sat, and is differentiable")
seq = [17, END, PAD, PAD]
lab1 = torch.tensor([seq])
g = one_hot_logits(seq, scale=2.0).requires_grad_(True)
loss, sat, _ = module(g, lab1, outcome_logits(1, scale=2.0))
check("loss == 1 - sat", abs(loss.item() - (1.0 - sat.item())) < 1e-6)
loss.backward()
check("gradient reaches act logits", g.grad is not None and g.grad.abs().sum() > 0)

# ------------------------------------------------------------- detach modes ---
print("\ndetach modes route gradient correctly")
SOFT = 2.0
for mode, act_should, out_should in (("none", True, True),
                                     ("act", False, True),
                                     ("outcome", True, False)):
    m = BrierMeanOutcomeConsistencyLoss(DET_IDS, END, NUM_OUT, pad_token=PAD, detach_mode=mode)
    a = one_hot_logits(seq, scale=SOFT).requires_grad_(True)
    o = outcome_logits(1, scale=SOFT).requires_grad_(True)
    loss, _, _ = m(a, lab1, o)
    loss.backward()
    act_grad = a.grad is not None and a.grad.abs().sum().item() > 1e-9
    out_grad = o.grad is not None and o.grad.abs().sum().item() > 1e-9
    check(f"detach_mode={mode!r}: act grad={act_grad}, outcome grad={out_grad}",
          act_grad == act_should and out_grad == out_should)

# -------------------------------------------------------------- valid_mask ---
print("\nvalid_mask restricts to the non-leaky subset")
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
print("\ndiagnostics are sane and per-instance (no num_positions)")
_, _, diag = module(logits, labels, out_log)
check("no num_positions key (per-instance impl)", "num_positions" not in diag)
check("min_survival far above float32 underflow", diag["min_survival"] > 1e-6,
      f"min_survival={diag['min_survival']:.4f}")
check("residual mass in [0,1]", 0.0 <= diag["mean_residual_mass"] <= 1.0,
      f"residual={diag['mean_residual_mass']:.4f}")

# ------------------------------------------------------------------ summary --
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name in FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
print("ALL TESTS PASSED")
