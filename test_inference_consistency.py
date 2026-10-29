"""Axiom-1 diagnostics of `inference_loop`, on a stub model. CPU only, no data.

    python test_inference_consistency.py

The summed TTNE suffix must run up to the predicted END (the last window
position when the decoder emits none), independent of the true suffix length;
the capped variant stops at the earlier of the predicted and the true END.
"""

import os
import pickle
import sys
import tempfile

import torch

from SuTraN.inference_procedure import inference_loop

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'OK  ' if condition else 'FAIL'} {name}{('  -- ' + detail) if detail else ''}")


END, W = 3, 4   # activities 1, 2 and END; window of four positions

# Three instances: the decoder stops early, never stops, and stops before the
# true END. TTNE and RRT are standardized with mean 0 and std 1, so the
# standardized values are seconds.
true_acts = torch.tensor([[1, END, 0, 0],
                          [1, END, 0, 0],
                          [1, 2, 1, END]])
pred_acts = torch.tensor([[1, END, 0, 0],
                          [1, 2, 2, 2],
                          [1, END, 0, 0]])
pred_ttne = torch.tensor([[10., 0., 99., 99.],
                          [10., 5., 5., 5.],
                          [7., 1., 50., 50.]])
pred_rrt = torch.tensor([10., 15., 8.])

expected_sum = torch.tensor([10., 25., 8.])          # up to the predicted END
expected_capped = torch.tensor([10., 15., 8.])       # min(predicted, true END)


class StubModel(torch.nn.Module):
    """Returns the fixed decode above, whatever the inputs."""

    def forward(self, inputs, window_size, *_):
        dev = inputs[0].device
        return pred_acts.to(dev), pred_ttne.to(dev), pred_rrt.to(dev)


n = true_acts.shape[0]
pad_mask = torch.ones(n, W, dtype=torch.bool)
pad_mask[:, 0] = False                               # prefixes of length 1
ttne_labels = torch.zeros(n, W, 1)
rrt_labels = torch.zeros(n, W, 1)
# Inputs (the stub ignores them; index num_categoricals_pref + 1 is the padding
# mask) followed by the labels: TTNE, RRT, activities.
dataset = (torch.ones(n, W, dtype=torch.long), torch.zeros(n, W, 1), pad_mask,
           ttne_labels, rrt_labels, true_acts)

with tempfile.TemporaryDirectory() as results_path:
    inference_loop(model=StubModel(), inference_dataset=dataset,
                   remaining_runtime_head=True, outcome_bool=False, out_mask=False,
                   out_type=None, num_outclasses=None, num_categoricals_pref=1,
                   mean_std_ttne=[0.0, 1.0], mean_std_tsp=[0.0, 1.0],
                   mean_std_tss=[0.0, 1.0], mean_std_rrt=[0.0, 1.0],
                   og_caseint=torch.arange(n), instance_mask_out=None,
                   results_path=results_path, store_preds=True)
    with open(os.path.join(results_path, "consistency_diagnostics.pkl"), "rb") as f:
        diag = pickle.load(f)
    per_instance = torch.load(os.path.join(results_path, "consistency_per_instance.pt"), weights_only=False)
    stored_ttne = torch.load(os.path.join(results_path, "suffix_ttne_preds.pt"), weights_only=False)

print("\nsummed TTNE suffix")
check("sums up to the predicted END",
      torch.allclose(per_instance["ax1_sum_ts_pred"], expected_sum),
      str(per_instance["ax1_sum_ts_pred"].tolist()))
check("capped variant stops at the earlier END",
      torch.allclose(per_instance["ax1_sum_ts_pred_capped"], expected_capped),
      str(per_instance["ax1_sum_ts_pred_capped"].tolist()))
check("predicted lengths include END; whole window without one",
      per_instance["pred_len"].tolist() == [2, 4, 2], str(per_instance["pred_len"].tolist()))

print("\naggregates")
gap = (expected_sum - pred_rrt).abs()
check("mean |gap| uses the predicted END",
      abs(diag["mean_abs_gap_CB"] - gap.mean().item()) < 1e-4,
      f"{diag['mean_abs_gap_CB']:.3f} vs {gap.mean().item():.3f}")
check("capped |gap| is zero here", abs(diag["mean_abs_gap_capped_CB"]) < 1e-4)
check("fraction without a predicted END", abs(diag["frac_no_pred_end_CB"] - 1 / 3) < 1e-6)

print("\nstored predictions")
check("stored TTNE are the raw predictions (not padded after END)",
      torch.allclose(stored_ttne, pred_ttne))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name in FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
print("ALL TESTS PASSED")
