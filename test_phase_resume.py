"""Tests for two-phase training support (phase_resume.py + naming).

    python test_phase_resume.py

The direction rule and the folder naming are pure. The log-sigma recovery is
checked against a finished BPIC_12 baseline when one is present locally, and
skipped otherwise.
"""

import glob
import os
import sys

import torch

from phase_resume import (direction_from_diagnostics, phase1_checkpoint_epoch,
                          phase1_log_sigmas)
from TRAIN_EVAL_FUNCTIONALITY.naming import build_model_string

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'OK  ' if condition else 'FAIL'} {name}{('  -- ' + detail) if detail else ''}")


# ------------------------------------------------------------ direction rule --
print("direction rule: freeze the better prediction")
d = direction_from_diagnostics({"mae_rrt_IB": 4.0, "mae_ttne_sum_IB": 5.0})
check("time: RRT head better -> freeze rrt", d["axiom1"]["detach_mode"] == "rrt"
      and d["axiom1"]["better"] == "rrt_head")
d = direction_from_diagnostics({"mae_rrt_IB": 6.0, "mae_ttne_sum_IB": 5.0})
check("time: ttne sum better -> freeze ttne", d["axiom1"]["detach_mode"] == "ttne")
d = direction_from_diagnostics({"mae_rrt_IB": 5.0, "mae_ttne_sum_IB": 5.0})
check("time: exact tie -> freeze neither", d["axiom1"]["detach_mode"] == "none"
      and d["axiom1"]["better"] is None)
d = direction_from_diagnostics({"outcome_head_macro_f1_IB": 0.70,
                                "outcome_suffix_macro_f1_IB": 0.65})
check("outcome: head better -> freeze outcome head",
      d["axiom2"]["detach_mode_outcome"] == "outcome")
d = direction_from_diagnostics({"outcome_head_macro_f1_IB": 0.62,
                                "outcome_suffix_macro_f1_IB": 0.64})
check("outcome: suffix better -> freeze suffix (act)",
      d["axiom2"]["detach_mode_outcome"] == "act")
d = direction_from_diagnostics({"mae_rrt_IB": 4.0, "mae_ttne_sum_IB": 5.0,
                                "outcome_head_accuracy_IB": 0.9})
check("no macro-F1 diagnostics -> no outcome decision", "axiom2" not in d)
check("empty diagnostics -> no decision", direction_from_diagnostics({}) == {})
# The outcome rule uses macro-F1, not accuracy: accuracy favouring the head must
# not override macro-F1 favouring the suffix.
d = direction_from_diagnostics({"outcome_head_macro_f1_IB": 0.62,
                                "outcome_suffix_macro_f1_IB": 0.64,
                                "outcome_head_accuracy_IB": 0.75,
                                "outcome_suffix_accuracy_IB": 0.73})
check("outcome rule follows macro-F1, not accuracy", d["axiom2"]["better"] == "suffix_implied")

# ------------------------------------------------------------------- naming --
print("\nfolder naming")
base = dict(seed=101, batch_size=512, out_type="multiclass_outcome")
baseline = build_model_string(**base)
control = build_model_string(**base, phase1_epoch="best")
treated = build_model_string(**base, phase1_epoch="best", lambda_ltn_outcome=25,
                             detach_mode_outcome="auto", axiom2_impl="native_last")
check("baseline name unchanged", baseline == "SUTRAN_DA_results_bs_512_multiclass_outcome_seed_101",
      baseline)
check("lambda=0 phase-2 control differs from its baseline", control != baseline, control)
check("phase-2 tag present", "_p2from_best" in treated, treated)
check("auto direction visible in the name", "_detachout_auto" in treated)
check("restart lr is untagged", "_lr" not in control.replace("_ltn", ""))
check("continue lr is tagged",
      "_lrcontinue" in build_model_string(**base, phase1_epoch="best", phase2_lr="continue"))
check("integer start epoch is tagged",
      "_p2from_40" in build_model_string(**base, phase1_epoch=40))

# ---------------------------------------- log-sigma recovery (real baseline) --
print("\nphase-1 checkpoint + log-sigmas from a finished BPIC_12 baseline")
hits = glob.glob("BPIC_12/SUTRAN_DA_results_bs_512_multiclass_outcome_seed_101/**/"
                 "backup_results.csv", recursive=True)
if not hits:
    print("  (skipped: no local BPIC_12 seed-101 baseline)")
else:
    bdir = os.path.dirname(hits[0])
    tasks = ["activity_suffix", "timestamp_suffix", "remaining_runtime", "multiclass_outcome"]
    epoch = phase1_checkpoint_epoch(bdir, "best", tasks)
    ckpt = torch.load(os.path.join(bdir, f"model_epoch_{epoch}.pt"), map_location="cpu")
    comps = ["ActivitySuffix", "TimestampSuffix", "RemainingRuntime", "Outcome"]
    values, source = phase1_log_sigmas(bdir, epoch, comps, ckpt)
    check("best epoch has a checkpoint", os.path.exists(os.path.join(bdir, f"model_epoch_{epoch}.pt")))
    check("one log-sigma per task", len(values) == 4, str(values))
    check("recovered from TensorBoard (old checkpoint)", source == "tensorboard")
    check("learned, not the -1.5 init", all(abs(v + 1.5) > 1e-3 for v in values))
    # A non-validated epoch has no log entry: fail loudly instead of guessing.
    try:
        phase1_log_sigmas(bdir, epoch + 1, comps, {})
        check("non-validated epoch rejected", False)
    except ValueError:
        check("non-validated epoch rejected", True)
    # A checkpoint that stores them takes precedence.
    v2, s2 = phase1_log_sigmas(bdir, epoch, comps, {"log_sigmas": torch.tensor([0.1, 0.2, 0.3, 0.4])})
    check("checkpoint values preferred", s2 == "checkpoint" and abs(v2[2] - 0.3) < 1e-6)

# ------------------------------------------------------------------ summary --
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name in FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
print("ALL TESTS PASSED")
