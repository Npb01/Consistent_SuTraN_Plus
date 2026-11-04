"""Hard enforcement on synthetic predictions. CPU only, no data.

    python test_hard_enforcement.py
"""

import sys

import torch

from hard_enforcement import (outcome_to_suffix, predicted_suffix_mask, rrt_to_ttne,
                              suffix_to_outcome, ttne_to_rrt)
from outcome_consistency_metrics import implied_outcome_from_suffix

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'OK  ' if condition else 'FAIL'} {name}{('  -- ' + detail) if detail else ''}")


# Activities 1-5, END = 6. Determining activities: 3 -> class 0, 4 -> class 1,
# 5 -> class 2.
END = 6
DET = {0: 3, 1: 4, 2: 5}
MS_TTNE = (100.0, 50.0)     # seconds
MS_RRT = (1000.0, 500.0)


def std(seconds, ms):
    return (torch.as_tensor(seconds, dtype=torch.float64) - ms[0]) / ms[1]


def sec(standardized, ms):
    return (standardized.double() * ms[1] + ms[0]).clamp(min=0)


print("predicted suffix")
acts = torch.tensor([[1, 2, END, 1],      # END at position 2
                     [1, 2, 1, 2],        # no END: whole window
                     [END, 1, 1, 1]])     # END at once
mask = predicted_suffix_mask(acts, END)
check("up to and including the first END, or the whole window",
      mask.tolist() == [[True, True, True, False], [True] * 4, [True, False, False, False]])

print("\ntime: RRT kept")
ttne_s = torch.tensor([[100., 300., 0., 999.],
                       [-200., 50., 50., 100.],   # a negative step clamps to 0
                       [0., 400., 400., 400.]])   # one step of zero duration
ttne = std(ttne_s, MS_TTNE).float()
rrt = std([800., 400., 600.], MS_RRT).float()
ttne_new, rrt_new = ttne_to_rrt(ttne, rrt, acts, END, MS_TTNE, MS_RRT)
steps = sec(ttne_new, MS_TTNE) * mask
check("summed TTNE equals the RRT", torch.allclose(steps.sum(1), sec(rrt_new, MS_RRT), atol=1e-2),
      f"{steps.sum(1).tolist()} vs {sec(rrt_new, MS_RRT).tolist()}")
check("RRT unchanged", torch.allclose(sec(rrt_new, MS_RRT), torch.tensor([800., 400., 600.]).double(), atol=1e-2))
check("proportional: relative timing kept",
      torch.allclose(steps[0, :3], torch.tensor([200., 600., 0.]).double(), atol=1e-2),
      str(steps[0, :3].tolist()))
check("clamped step stays zero", abs(steps[1, 0].item()) < 1e-2, str(steps[1].tolist()))
check("zero total spreads the RRT evenly", abs(steps[2, 0].item() - 600.) < 1e-2, str(steps[2].tolist()))
check("positions after the predicted END unchanged", ttne_new[0, 3] == ttne[0, 3])

rrt_neg = std([-300., 400., 600.], MS_RRT).float()
ttne_neg, rrt_neg_new = ttne_to_rrt(ttne, rrt_neg, acts, END, MS_TTNE, MS_RRT)
check("negative RRT clamps to zero, steps become zero",
      abs(sec(rrt_neg_new, MS_RRT)[0].item()) < 1e-6
      and (sec(ttne_neg, MS_TTNE)[0, :3].abs() < 1e-2).all().item())

print("\ntime: TTNE kept")
rrt_bu = rrt_to_ttne(ttne, acts, END, MS_TTNE, MS_RRT)
check("RRT equals the summed TTNE",
      torch.allclose(sec(rrt_bu, MS_RRT), torch.tensor([400., 200., 0.]).double(), atol=1e-2),
      str(sec(rrt_bu, MS_RRT).tolist()))

print("\noutcome: suffix kept")
sfx = torch.tensor([[1, 3, 4, END],      # last determining 4 -> class 1
                    [1, 5, END, 0],      # class 2
                    [1, 2, END, 0],      # no determining activity
                    [3, END, 0, 0],      # class 0, not evaluated
                    [4, 1, END, 0]])     # class 1, head agrees
logits = torch.tensor([[2.0, 1.0, 0.5],
                       [0.3, 0.2, 0.1],
                       [3.0, 0.0, 0.0],
                       [0.0, 5.0, 0.0],
                       [0.0, 4.0, 1.0]])
eligible = torch.tensor([True, True, True, False, True])
out_class, changed = outcome_to_suffix(logits, sfx, DET, END, eligible)
check("predicted outcome follows the suffix where it implies one, head elsewhere",
      out_class.tolist() == [1, 2, 0, 1, 1], str(out_class.tolist()))
check("changed only the eligible conflicting instances", changed.tolist() == [True, True, False, False, False])

print("\noutcome: outcome kept")
act_new, changed = suffix_to_outcome(sfx, logits, DET, END, eligible)
check("last determining activity replaced by the predicted class's",
      act_new[0].tolist() == [1, 3, 3, END] and act_new[1].tolist() == [1, 3, END, 0],
      str(act_new[:2].tolist()))
check("suffix without a determining activity, ineligible or agreeing rows unchanged",
      torch.equal(act_new[2:], sfx[2:]))
check("changed flags", changed.tolist() == [True, True, False, False, False])
implied, _ = implied_outcome_from_suffix(act_new, DET, END)
check("no conflict left on eligible rows with a determining activity",
      implied[:2].tolist() == logits[:2].argmax(-1).tolist())

try:
    suffix_to_outcome(sfx, logits, {0: 3, 1: [4, 5]}, END, eligible)
    check("several determining activities per class are refused", False)
except ValueError:
    check("several determining activities per class are refused", True)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name in FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
print("ALL TESTS PASSED")
