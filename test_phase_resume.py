"""Tests for two-phase training support (phase_resume.py + naming).

    python test_phase_resume.py

The direction rule and the folder naming are pure. The log-sigma recovery is
checked against a finished BPIC_12 baseline when one is present locally, and
skipped otherwise.
"""

import glob
import os
import sys

import pandas as pd
import torch

from phase_resume import (direction_from_diagnostics, phase1_checkpoint_epoch,
                          phase1_log_sigmas, plateau_epoch, stopping_counter)
from TRAIN_EVAL_FUNCTIONALITY.naming import build_model_string

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'OK  ' if condition else 'FAIL'} {name}{('  -- ' + detail) if detail else ''}")


# ------------------------------------------------------------ direction rule --
print("direction rule: freeze the better prediction")
d = direction_from_diagnostics({"mae_rrt_CB": 4.0, "mae_ttne_sum_CB": 5.0})
check("time: RRT head better -> freeze rrt", d["axiom1"]["detach_mode"] == "rrt"
      and d["axiom1"]["better"] == "rrt_head")
d = direction_from_diagnostics({"mae_rrt_CB": 6.0, "mae_ttne_sum_CB": 5.0})
check("time: ttne sum better -> freeze ttne", d["axiom1"]["detach_mode"] == "ttne")
d = direction_from_diagnostics({"mae_rrt_CB": 5.0, "mae_ttne_sum_CB": 5.0})
check("time: exact tie -> freeze neither", d["axiom1"]["detach_mode"] == "none"
      and d["axiom1"]["better"] is None)
d = direction_from_diagnostics({"outcome_head_macro_f1_CB": 0.70,
                                "outcome_suffix_macro_f1_CB": 0.65})
check("outcome: head better -> freeze outcome head",
      d["axiom2"]["detach_mode_outcome"] == "outcome")
d = direction_from_diagnostics({"outcome_head_macro_f1_CB": 0.62,
                                "outcome_suffix_macro_f1_CB": 0.64})
check("outcome: suffix better -> freeze suffix (act)",
      d["axiom2"]["detach_mode_outcome"] == "act")
d = direction_from_diagnostics({"mae_rrt_CB": 4.0, "mae_ttne_sum_CB": 5.0,
                                "outcome_head_accuracy_CB": 0.9})
check("no macro-F1 diagnostics -> no outcome decision", "axiom2" not in d)
check("empty diagnostics -> no decision", direction_from_diagnostics({}) == {})
# The outcome rule uses macro-F1, not accuracy: accuracy favouring the head must
# not override macro-F1 favouring the suffix.
d = direction_from_diagnostics({"outcome_head_macro_f1_CB": 0.62,
                                "outcome_suffix_macro_f1_CB": 0.64,
                                "outcome_head_accuracy_CB": 0.75,
                                "outcome_suffix_accuracy_CB": 0.73})
check("outcome rule follows macro-F1, not accuracy", d["axiom2"]["better"] == "suffix_implied")

print("\ndirection rule: basis and tie margin")
mixed = {"mae_rrt_CB": 4.0, "mae_ttne_sum_CB": 5.0, "mae_rrt_IB": 6.0, "mae_ttne_sum_IB": 5.0}
check("case-based by default", direction_from_diagnostics(mixed)["axiom1"]["better"] == "rrt_head")
check("instance-based on request",
      direction_from_diagnostics(mixed, basis="IB")["axiom1"]["better"] == "ttne_sum")
check("IB keys alone give no CB decision",
      direction_from_diagnostics({"mae_rrt_IB": 4.0, "mae_ttne_sum_IB": 5.0}) == {})
d = direction_from_diagnostics({"outcome_head_macro_f1_CB": 0.643,
                                "outcome_suffix_macro_f1_CB": 0.660},
                               margins={"axiom2": 0.02})
check("difference within the margin -> freeze neither",
      d["axiom2"]["detach_mode_outcome"] == "none" and d["axiom2"]["margin"] == 0.02)
d = direction_from_diagnostics({"mae_rrt_CB": 4.0, "mae_ttne_sum_CB": 5.0},
                               margins={"axiom1": 0.5})
check("difference beyond the margin -> freeze the better side",
      d["axiom1"]["detach_mode"] == "rrt")
try:
    direction_from_diagnostics({}, basis="cb")
    check("unknown basis rejected", False)
except ValueError:
    check("unknown basis rejected", True)

# ------------------------------------------------------- stopping counter --
print("\nstopping counter replay")
cols = {"Activity suffix: 1-DL (validation)": [0.5, 0.6, 0.6, 0.55, 0.61, 0.6, 0.6],
        "TTNE - standardized MAE validation": [1.0, 0.9, 0.9, 0.95, 0.9, 0.9, 0.9],
        "RRT - standardized MAE validation": [1.0, 0.8, 0.85, 0.8, 0.9, 0.8, 0.9]}
curve = pd.DataFrame({"epoch": [3, 7, 11, 15, 19, 23, 27], **cols})
tasks3 = ["activity_suffix", "timestamp_suffix", "remaining_runtime"]
counts = stopping_counter(curve, tasks3).tolist()
check("resets on any strict improvement, ties do not count",
      counts == [0, 0, 1, 2, 0, 1, 2], str(counts))
check("plateau epoch is the first validation reaching the count",
      plateau_epoch(curve, tasks3, 2) == 15 and plateau_epoch(curve, tasks3, 1) == 11)
check("never reached -> None", plateau_epoch(curve, tasks3, 3) is None)

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
check("plateau start is tagged",
      "_p2from_plateau" in build_model_string(**base, phase1_epoch="plateau"))
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
    # The replayed counter reaches the patience (6) exactly where training stopped.
    curve = pd.read_csv(os.path.join(bdir, "backup_results.csv"))
    check("replay stops where the run stopped",
          plateau_epoch(curve, tasks, 6) == int(curve["epoch"].iloc[-1]))
    k = phase1_checkpoint_epoch(bdir, "plateau", tasks)
    check("'plateau' resolves to the counter-2 epoch with a checkpoint",
          k == plateau_epoch(curve, tasks, 2)
          and os.path.exists(os.path.join(bdir, f"model_epoch_{k}.pt")), f"epoch {k}")
    _, src = phase1_log_sigmas(bdir, k, comps, {})
    check("log-sigmas available at the plateau epoch", src == "tensorboard")

# ------------------------------------------------------------------ summary --
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name in FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
print("ALL TESTS PASSED")
