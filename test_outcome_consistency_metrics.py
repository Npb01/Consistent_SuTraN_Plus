"""Axiom-2 metric tests: synthetic unit cases plus a ground-truth check.

    python test_outcome_consistency_metrics.py

Needs the preprocessed BPIC_17_DR tensors for the ground-truth part; the
synthetic cases are pure and run without data.
"""
import sys
import pickle

sys.path.insert(0, '.')
import torch
from outcome_consistency_metrics import (implied_outcome_from_suffix,
                                         outcome_consistency_diagnostics,
                                         resolve_determining_ids)

with open('BPIC_17_DR/BPIC_17_DR_categ_mapping.pkl', 'rb') as fh:
    cm = pickle.load(fh)['concept:name']
DET_NAMES = {0: 'O_Accepted', 1: 'O_Cancelled', 2: 'O_Refused'}
det_ids, END = resolve_determining_ids(cm, DET_NAMES)
print('determining ids:', det_ids, ' END:', END)
assert det_ids == {0: 17, 1: 21, 2: 20} and END == 27, "id resolution changed!"

# ---------- synthetic unit tests (answer known by construction) ----------
A, C, R, X = det_ids[0], det_ids[1], det_ids[2], 3
cases = [
    ([A, X, END, 0, 0],  0, "accepted only"),
    ([C, X, END, 0, 0],  1, "cancelled only"),
    ([R, END, 0, 0, 0],  2, "refused only"),
    ([A, X, C, X, END],  1, "ACCEPT then CANCEL -> Canceled"),
    ([C, X, A, X, END],  0, "CANCEL then ACCEPT -> Accepted"),
    ([X, X, X, END, 0], -1, "no determining act"),
    ([A, X, X, X, X],    0, "no END token (window-limited decode)"),
    ([A, END, C, X, X],  0, "det act AFTER end token must be ignored"),
]
suf = torch.tensor([c[0] for c in cases])
imp, has = implied_outcome_from_suffix(suf, det_ids, END)
ok = True
for i, (seq, e, desc) in enumerate(cases):
    got = imp[i].item()
    if got != e:
        ok = False
    print(f"  {'OK ' if got == e else 'FAIL'} {desc:42s} expected {e:2d}  got {got:2d}")
assert ok, "synthetic cases failed"

# ---------- ground truth: suffix accuracy must be exactly 1.0 on labels ----------
for split in ['train', 'test']:
    ts = torch.load(f'BPIC_17_DR/{split}_tensordataset.pt', weights_only=False)
    act_lab, outcome = ts[12].clone(), ts[13].clone()
    del ts
    leak = torch.load(f'BPIC_17_DR/instance_mask_out_{split}.pt', weights_only=False)
    valid = ~leak
    onehot = torch.nn.functional.one_hot(outcome[valid].to(torch.int64), num_classes=3).float()
    d = outcome_consistency_diagnostics(
        act_suffix=act_lab[valid], outcome_logits=onehot,
        outcome_labels=outcome[valid], determining_ids=det_ids, end_token=END)
    print(f"\n{split}: n={int(d['num_outcome_instances_evaluated']):,}")
    print(f"  suffix-implied accuracy (ground-truth suffix): {d['outcome_suffix_accuracy_IB']:.6f}")
    print(f"  suffix macro-F1:                               {d['outcome_suffix_macro_f1_IB']:.6f}")
    print(f"  frac with no determining act:                  {d['frac_suffix_no_determining_act']:.6f}")
    assert abs(d['outcome_suffix_accuracy_IB'] - 1.0) < 1e-9, "axiom broken!"

# ---------- per-instance return: raw tensors must reproduce the aggregates ----------
suf2 = torch.tensor([c[0] for c in cases])
lab2 = torch.tensor([max(c[1], 0) for c in cases])          # -1 -> 0, harmless here
logits2 = torch.nn.functional.one_hot(lab2, num_classes=3).float()
agg, pi = outcome_consistency_diagnostics(
    act_suffix=suf2, outcome_logits=logits2, outcome_labels=lab2,
    determining_ids=det_ids, end_token=END, return_per_instance=True)
assert set(pi) == {"ax2_implied", "ax2_has_det", "ax2_outcome_labels"}
assert all(v.shape[0] == suf2.shape[0] for v in pi.values()), "per-instance row mismatch"
# head pred (argmax of the saved logits) vs suffix route reproduces agreement
head_pi = logits2.argmax(dim=-1)
agree_pi = (head_pi == pi["ax2_implied"]).float().mean().item()
assert abs(agree_pi - agg["outcome_head_vs_suffix_agreement_IB"]) < 1e-9, \
    "per-instance agreement does not reproduce the aggregate"
# suffix-route accuracy from raw tensors reproduces the aggregate
suffix_acc_pi = ((pi["ax2_implied"] == pi["ax2_outcome_labels"]) & pi["ax2_has_det"]).float().mean().item()
assert abs(suffix_acc_pi - agg["outcome_suffix_accuracy_IB"]) < 1e-9, \
    "per-instance suffix accuracy does not reproduce the aggregate"
# default call (no flag) must still return a bare dict
assert isinstance(outcome_consistency_diagnostics(
    act_suffix=suf2, outcome_logits=logits2, outcome_labels=lab2,
    determining_ids=det_ids, end_token=END), dict)
print("per-instance return: OK")

# ---------- macro-F1: weighted counts match sklearn's sample_weight ----------
from sklearn.metrics import f1_score
from outcome_consistency_metrics import macro_f1

g = torch.Generator().manual_seed(0)
ref = torch.randint(0, 3, (200,), generator=g)
pred = torch.randint(0, 3, (200,), generator=g)
w = torch.rand(200, generator=g)
for weights in (None, w):
    ours = macro_f1(pred, ref, [0, 1, 2], weights)
    theirs = f1_score(ref.numpy(), pred.numpy(), average="macro",
                      sample_weight=None if weights is None else weights.numpy())
    assert abs(ours - theirs) < 1e-6, f"macro-F1 {ours} != sklearn {theirs}"
assert abs(macro_f1(pred, ref, [0, 1, 2], w) - macro_f1(ref, pred, [0, 1, 2], w)) < 1e-9, \
    "macro-F1 must be symmetric"
print("macro-F1 matches sklearn (weighted and unweighted): OK")

# ---------- agreement macro-F1 and the split of the disagreement ----------
# head classes 0,1,2,0 ; suffix implies 0,2,none,0 -> one conflict, one none.
suf3 = torch.tensor([[A, END, 0], [C, R, END], [X, X, END], [A, X, END]])
head3 = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 0]), num_classes=3).float()
lab3 = torch.tensor([0, 1, 2, 0])
ones = torch.ones(4)


def plain_avg(metric_tens, weight_tens, num_cases):
    return (metric_tens * weight_tens).sum().item() / num_cases


d3 = outcome_consistency_diagnostics(suf3, head3, lab3, det_ids, END,
                                     weights=ones, num_cases=4, corrected_avg_fn=plain_avg)
assert abs(d3["outcome_disagreement_rate_CB"] - 0.5) < 1e-9
assert abs(d3["outcome_conflict_rate_CB"] - 0.25) < 1e-9, "one real conflict of four"
assert abs(d3["outcome_no_determining_rate_CB"] - 0.25) < 1e-9, "one suffix implies nothing"
# per class (head vs suffix): 0 -> F1 1, 1 -> F1 0 (head 1, suffix 2), 2 -> F1 0
assert abs(d3["outcome_agreement_macro_f1_CB"] - 1 / 3) < 1e-9, d3["outcome_agreement_macro_f1_CB"]
assert abs(d3["outcome_agreement_macro_f1_IB"] - d3["outcome_agreement_macro_f1_CB"]) < 1e-9, \
    "unit weights: CB equals IB"
assert abs(d3["outcome_head_macro_f1_CB"] - 1.0) < 1e-9
assert abs(d3["outcome_suffix_macro_f1_CB"] - d3["outcome_suffix_macro_f1_IB"]) < 1e-9
# full agreement scores 1
d4 = outcome_consistency_diagnostics(suf3[[0, 3]], head3[[0, 3]], lab3[[0, 3]], det_ids, END,
                                     weights=ones[:2], num_cases=2, corrected_avg_fn=plain_avg)
assert abs(d4["outcome_agreement_macro_f1_CB"] - 1.0) < 1e-9
print("agreement macro-F1 and conflict / no-determining split: OK")

print("\nALL CHECKS PASSED")
