"""Tests for the axiom-2 `guarded_last` impl. CPU only, no data files, seconds.

    python test_ltn_outcome_guarded_last.py

guarded_last weights the per-position agreement by Last(sigma, t) (a
determining activity at t and none after it) and averages it per case; cases
are aggregated by a p-mean-error weighted by their total Last mass G. The tests
check the crisp behaviour (only the last determining activity counts, other
positions add nothing), that a case without determining mass leaves the
domain, a loop-based reference computation of the formula on soft inputs, that
Last and G receive no gradient, and the detach routing.
"""

import sys

import ltn
import torch

from ltn_outcome_consistency import GuardedLastOutcomeConsistencyLoss

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


module = GuardedLastOutcomeConsistencyLoss(DET_IDS, END, NUM_OUT, pad_token=PAD)


def sat_of(seq, cls, m=module):
    labels = torch.tensor([seq])
    _, sat, diag = m(one_hot_logits(seq), labels, outcome_logits(cls))
    return sat.item(), diag


# ------------------------------------------------------------------ crisp core --
print("\ncrisp satisfaction on the last determining activity")
s_agree, _ = sat_of([17, END, PAD, PAD], 0)
s_disag, _ = sat_of([17, END, PAD, PAD], 1)
check("agreement -> sat ~ 1", s_agree > 0.99, f"sat={s_agree:.4f}")
check("disagreement -> sat ~ 0", s_disag < 0.01, f"sat={s_disag:.4f}")

CASES = [
    ([17, OTHER, 21, END], 1, "ACCEPT then CANCEL -> Canceled"),
    ([21, OTHER, 17, END], 0, "CANCEL then ACCEPT -> Accepted"),
    ([20, END, PAD, PAD], 2, "refused only"),
]
for seq, expected, desc in CASES:
    s_ok, _ = sat_of(seq, expected)
    s_wrong = max(sat_of(seq, c)[0] for c in range(NUM_OUT) if c != expected)
    check(desc, s_ok > 0.99 and s_wrong < 0.01,
          f"sat(correct)={s_ok:.4f}, sat(best wrong)={s_wrong:.4f}")

print("\nother positions add nothing")
s_short, _ = sat_of([21, END, PAD, PAD, PAD, PAD, PAD], 0)
s_long, _ = sat_of([17, OTHER, OTHER, OTHER, OTHER, 21, END], 0)
check("a disagreement is not diluted by a longer suffix",
      s_long < 0.01 and abs(s_long - s_short) < 1e-4, f"{s_short:.4f} vs {s_long:.4f}")
s_excuse, _ = sat_of([17, OTHER, 21, END], 1)
check("a wrong non-last determining activity does not lower sat", s_excuse > 0.99,
      f"sat={s_excuse:.4f}")

# ------------------------------------------------- cases without determining --
print("\na case without determining mass leaves the domain")
seqs = [[17, END, PAD, PAD], [OTHER, OTHER, END, PAD]]
lab = torch.tensor(seqs)
al = torch.cat([one_hot_logits(s) for s in seqs])
_, sat_pair, diag_pair = module(al, lab, torch.cat([outcome_logits(0), outcome_logits(1)]))
check("it neither raises nor lowers sat", abs(sat_pair.item() - s_agree) < 1e-4,
      f"{sat_pair.item():.4f} vs {s_agree:.4f}")
check("mean guard reflects the empty case", abs(diag_pair["mean_guard"] - 0.5) < 1e-3,
      f"mean_guard={diag_pair['mean_guard']:.4f}")
# The weights are normalised, so a batch whose only case has (near) zero
# determining mass still scores that case; the loss must stay finite there.
none_logits = one_hot_logits([OTHER, END, PAD]).requires_grad_(True)
loss_none, _, _ = module(none_logits, torch.tensor([[OTHER, END, PAD]]), outcome_logits(0))
loss_none.backward()
check("a batch without determining mass stays finite",
      torch.isfinite(loss_none).item() and torch.isfinite(none_logits.grad).all().item())

# ------------------------------------------------------ reference computation --
print("\nmatches a loop-based reference on soft inputs")


def reference_sat(act_logits, act_labels, out_logits, p=2):
    """The formula written out case by case and position by position."""
    probs = torch.softmax(act_logits.double(), dim=-1)
    y = torch.softmax(out_logits.double(), dim=-1)
    det_ids = [DET_IDS[k] for k in range(NUM_OUT)]
    weights, sats = [], []
    for b in range(act_labels.shape[0]):
        valid = []
        for t, lab_t in enumerate(act_labels[b].tolist()):
            if lab_t == END:
                break
            if lab_t != PAD:
                valid.append(t)
        det = {t: sum(probs[b, t, a] for a in det_ids) for t in valid}
        last = {}
        for t in valid:
            surv = 1.0
            for s in valid:
                if s > t:
                    surv = surv * (1.0 - det[s])
            last[t] = det[t] * surv
        g = sum(last.values())
        num = 0.0
        for t in valid:
            out_t = torch.stack([probs[b, t, a] for a in det_ids]) / det[t]
            eq = 1.0 - 0.5 * ((out_t - y[b]) ** 2).sum()
            num = num + last[t] * eq
        weights.append(g)
        sats.append(num / g)
    w = torch.stack(weights)
    s = torch.stack(sats) * (1.0 - ltn.fuzzy_ops.eps)
    return 1.0 - ((w * (1.0 - s) ** p).sum() / w.sum()) ** (1.0 / p)


torch.manual_seed(0)
B, W = 6, 7
rand_logits = torch.randn(B, W, NUM_ACT) * 2.0
rand_labels = torch.full((B, W), OTHER)
for b in range(B):
    rand_labels[b, 2] = DET_IDS[b % 3]
    rand_labels[b, 4 + b % 2] = END
    rand_labels[b, 5 + b % 2:] = PAD
rand_out = torch.randn(B, NUM_OUT)
_, sat_mod, _ = module(rand_logits, rand_labels, rand_out)
sat_ref = reference_sat(rand_logits, rand_labels, rand_out)
check("guarded p-mean-error equals the reference", abs(sat_mod.item() - sat_ref.item()) < 1e-5,
      f"module={sat_mod.item():.6f} reference={sat_ref.item():.6f}")

# ------------------------------------------------------------ loss / gradient --
print("\nloss = 1 - sat, and Last receives no gradient")
seq = [17, END, PAD, PAD]
labels = torch.tensor([seq])
logits = one_hot_logits(seq, scale=2.0).requires_grad_(True)
loss, sat, _ = module(logits, labels, outcome_logits(1, scale=2.0))
check("loss == 1 - sat", abs(loss.item() - (1.0 - sat.item())) < 1e-6)
loss.backward()
check("gradient reaches act logits", logits.grad is not None and logits.grad.abs().sum() > 0)

# Two cases: the first is soft (determining mass 0.5 on ACCEPT, the rest on an
# ordinary activity) and disagrees; the second agrees. ACCEPT is the only
# determining activity with mass, so the implied outcome at the soft position is
# a constant one-hot and the only way the ACCEPT logit could raise sat is by
# shrinking the disagreeing case's weight. With Last and G detached it gets no
# gradient.
soft = torch.full((1, 2, NUM_ACT), -30.0)
soft[0, 0, 17] = 0.0
soft[0, 0, OTHER] = 0.0
soft[0, 1, END] = 30.0
crisp = one_hot_logits([17, END])
act = torch.cat([soft, crisp]).requires_grad_(True)
lab2 = torch.tensor([[17, END], [17, END]])
for mode in ("none", "outcome"):
    act.grad = None
    m = GuardedLastOutcomeConsistencyLoss(DET_IDS, END, NUM_OUT, pad_token=PAD, detach_mode=mode)
    loss, _, _ = m(act, lab2, torch.cat([outcome_logits(1), outcome_logits(0)]))
    loss.backward()
    g_accept = act.grad[0, 0, 17].abs().item() if act.grad is not None else 0.0
    check(f"detach_mode={mode!r}: no gradient through the weights", g_accept < 1e-6,
          f"|d loss / d ACCEPT logit| = {g_accept:.2e}")

# ------------------------------------------------------------- detach modes ---
print("\ndetach modes route gradient correctly")
# Unsaturated logits with mass on two determining activities, so the implied
# outcome itself depends on the activity logits.
SOFT = 2.0
mixed = one_hot_logits(seq, scale=SOFT)
mixed[0, 0, 21] = SOFT
for mode, act_should, out_should in (("none", True, True),
                                     ("act", False, True),
                                     ("outcome", True, False)):
    m = GuardedLastOutcomeConsistencyLoss(DET_IDS, END, NUM_OUT, pad_token=PAD, detach_mode=mode)
    a = mixed.clone().requires_grad_(True)
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
lab = torch.tensor(seqs)
al = torch.cat([one_hot_logits(s) for s in seqs])
ol = torch.cat([outcome_logits(0), outcome_logits(0)])   # 2nd instance disagrees
_, sat_both, d_both = module(al, lab, ol)
_, sat_first, d_first = module(al, lab, ol, valid_mask=torch.tensor([True, False]))
check("mask changes the instance count",
      d_both["num_instances"] == 2 and d_first["num_instances"] == 1)
check("masking out the disagreeing instance raises sat",
      sat_first.item() > sat_both.item(), f"{sat_both.item():.4f} -> {sat_first.item():.4f}")
_, _, d_empty = module(al, lab, ol, valid_mask=torch.tensor([False, False]))
check("all-masked batch is handled", d_empty["num_instances"] == 0)

# ------------------------------------------------------------------ summary --
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name in FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
print("ALL TESTS PASSED")
