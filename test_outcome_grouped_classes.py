"""Tests for axiom-2 outcome classes implied by several activities. CPU only,
no data files, seconds.

    python test_outcome_grouped_classes.py

A class may own a list of determining activities (SEPSIS: "released other than
Release A" is implied by any of Release B-E). Covered here:
  * the id normalisation and name resolution accept lists;
  * the suffix-implied outcome maps every activity of a group to its class,
    reading the LAST determining activity across groups;
  * all four loss impls give the same result for a grouped class as for a single
    activity carrying the group's summed probability (exact up to float
    rounding), so grouping changes nothing but where the mass is gathered from.
"""

import math
import sys

import torch

from axiom_builders import AXIOM2_IMPLS, build_axiom2_module
from outcome_consistency_metrics import (implied_outcome_from_suffix,
                                         normalize_determining_ids,
                                         outcome_consistency_diagnostics,
                                         resolve_determining_ids)

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'OK  ' if condition else 'FAIL'} {name}{('  -- ' + detail) if detail else ''}")


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False


# ---------------------------------------------------------------------------
print("normalize_determining_ids / resolve_determining_ids")
check("int becomes a one-element list",
      normalize_determining_ids({0: 5, 1: 6}) == {0: [5], 1: [6]})
check("list is kept, keys sorted",
      normalize_determining_ids({1: [7, 8], 0: 5}) == {0: [5], 1: [7, 8]})
check("activity in two classes raises",
      raises(lambda: normalize_determining_ids({0: [5], 1: [5, 6]}), ValueError))
check("empty class raises",
      raises(lambda: normalize_determining_ids({0: [5], 1: []}), ValueError))

cm = {"Release A": 4, "Release B": 5, "Release C": 6, "Other": 0}
ids, end = resolve_determining_ids(
    cm, {0: "Release A", 1: ["Release B", "Release C"]})
check("names resolve to ids + 1, list shape preserved",
      ids == {0: 5, 1: [6, 7]}, str(ids))
check("END is len(vocabulary) + 1", end == 5)
check("unknown name in a list raises KeyError",
      raises(lambda: resolve_determining_ids(cm, {0: "Release A", 1: ["Nope"]}),
             KeyError))

# ---------------------------------------------------------------------------
print("implied_outcome_from_suffix with a grouped class")
DET = {0: 5, 1: [6, 7, 8]}
END, OTHER = 10, 3
rows = [
    ([5, OTHER, END, 0], 0, "single-activity class"),
    ([6, END, 0, 0], 1, "group member 6"),
    ([OTHER, 8, END, 0], 1, "group member 8"),
    ([5, 7, END, 0], 1, "last determining is a group member"),
    ([8, 5, END, 0], 0, "group member then single: single is last"),
    ([OTHER, OTHER, END, 0], -1, "no determining activity"),
    ([OTHER, END, 7, 0], -1, "group member after END is ignored"),
    ([OTHER, 7, OTHER, OTHER], 1, "no END: whole window is suffix"),
]
suffix = torch.tensor([r[0] for r in rows])
implied, has_det = implied_outcome_from_suffix(suffix, DET, END)
for (_, expected, name), got in zip(rows, implied.tolist()):
    check(name, got == expected, f"got {got}, expected {expected}")
check("has_determining matches implied >= 0",
      torch.equal(has_det, implied >= 0))

logits = torch.tensor([[5.0, 0.0]] * len(rows))      # head always predicts class 0
labels = torch.tensor([max(r[1], 0) for r in rows])
diag = outcome_consistency_diagnostics(suffix, logits, labels, DET, END)
expected_disagree = sum(r[1] != 0 for r in rows) / len(rows)
check("disagreement rate counts grouped classes",
      math.isclose(diag["outcome_disagreement_rate_IB"], expected_disagree),
      f"{diag['outcome_disagreement_rate_IB']:.4f} vs {expected_disagree:.4f}")

# ---------------------------------------------------------------------------
print("loss impls: grouped class == single activity carrying the summed mass")
torch.manual_seed(0)
B, W, C_ACT, K = 16, 7, 12, 2
END_L = C_ACT - 1
GROUPED = {0: 5, 1: [6, 7, 8]}
SINGLE = {0: 5, 1: 6}

act_logits = torch.randn(B, W, C_ACT) * 2.0
probs = torch.softmax(act_logits, dim=-1)
merged = probs.clone()
merged[..., 6] = probs[..., 6] + probs[..., 7] + probs[..., 8]
merged[..., 7] = 0.0
merged[..., 8] = 0.0
merged_logits = torch.log(merged)                     # softmax(log p) == p

act_labels = torch.randint(1, END_L, (B, W))
for b in range(B):                                    # END at a varying position
    e = 1 + b % (W - 1)
    act_labels[b, e] = END_L
    act_labels[b, e + 1:] = 0
outcome_logits = torch.randn(B, K)
valid_mask = torch.rand(B) > 0.25
valid_mask[0] = True

for impl in AXIOM2_IMPLS:
    for detach in ("none", "act", "outcome"):
        grouped = build_axiom2_module(impl, determining_ids=GROUPED, end_token=END_L,
                                      num_outclasses=K, detach_mode=detach)
        single = build_axiom2_module(impl, determining_ids=SINGLE, end_token=END_L,
                                     num_outclasses=K, detach_mode=detach)
        lg, sg, _ = grouped(act_logits, act_labels, outcome_logits, valid_mask)
        ls, ss, _ = single(merged_logits, act_labels, outcome_logits, valid_mask)
        check(f"{impl:13s} detach={detach:7s} loss equal",
              torch.allclose(lg, ls, atol=1e-6), f"{lg.item():.8f} vs {ls.item():.8f}")

grouped = build_axiom2_module("native_last", determining_ids=GROUPED, end_token=END_L,
                              num_outclasses=K, detach_mode="none")
x = act_logits.clone().requires_grad_(True)
y = outcome_logits.clone().requires_grad_(True)
loss, _, _ = grouped(x, act_labels, y, valid_mask)
loss.backward()
check("gradients reach every grouped activity's logit",
      all(x.grad[..., a].abs().sum() > 0 for a in (6, 7, 8)))
check("gradients reach the outcome head", y.grad.abs().sum() > 0)

# ---------------------------------------------------------------------------
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TESTS PASSED")
