"""Tests for the epoch-dependent axiom schedules (pure, no torch, no data).

    python test_axiom_schedules.py

Covers the lambda multiplier for every schedule, the single-switch detach
schedule, and the fail-fast validators.
"""

import sys

from axiom_schedules import (
    LAMBDA_SCHEDULES,
    detach_for_epoch,
    lambda_multiplier,
    validate_detach_schedule,
    validate_lambda_schedule,
)

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'OK  ' if condition else 'FAIL'} {name}{('  -- ' + detail) if detail else ''}")


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ------------------------------------------------------ constant is always 1 --
print("constant schedule")
check("constant == 1 at epoch 0", lambda_multiplier("constant", 0, 10) == 1.0)
check("constant == 1 ignores warmup_epochs",
      lambda_multiplier("constant", 5, 0) == 1.0)

# ------------------------------------------------------------- warmup (hard) --
print("\nwarmup schedule (hard switch at warmup_epochs)")
check("warmup off before K", lambda_multiplier("warmup", 9, 10) == 0.0)
check("warmup on at K", lambda_multiplier("warmup", 10, 10) == 1.0)
check("warmup on after K", lambda_multiplier("warmup", 50, 10) == 1.0)

# -------------------------------------------------------------- ramp (linear)--
print("\nramp schedule (linear 0 -> 1 over [0, K])")
check("ramp 0 at epoch 0", lambda_multiplier("ramp", 0, 10) == 0.0)
check("ramp 0.5 at K/2", approx(lambda_multiplier("ramp", 5, 10), 0.5))
check("ramp 1 at K", lambda_multiplier("ramp", 10, 10) == 1.0)
check("ramp clamps at 1 past K", lambda_multiplier("ramp", 100, 10) == 1.0)

# ------------------------------------------------------------ cosine (smooth)--
print("\ncosine schedule (smooth 0 -> 1 over [0, K])")
check("cosine 0 at epoch 0", approx(lambda_multiplier("cosine", 0, 10), 0.0))
check("cosine 0.5 at K/2", approx(lambda_multiplier("cosine", 5, 10), 0.5))
check("cosine 1 at K", approx(lambda_multiplier("cosine", 10, 10), 1.0))
check("cosine clamps at 1 past K", approx(lambda_multiplier("cosine", 100, 10), 1.0))
check("cosine below ramp before midpoint",
      lambda_multiplier("cosine", 2, 10) < lambda_multiplier("ramp", 2, 10))

# --------------------------------------------------------- reverse_warmup -----
print("\nreverse_warmup schedule (on then off at K)")
check("reverse on before K", lambda_multiplier("reverse_warmup", 9, 10) == 1.0)
check("reverse off at K", lambda_multiplier("reverse_warmup", 10, 10) == 0.0)

# ------------------------------------------------- degenerate warmup_epochs ---
print("\ndegenerate warmup_epochs <= 0")
for sched in ("warmup", "ramp", "cosine"):
    check(f"{sched} with K=0 is full on", lambda_multiplier(sched, 0, 0) == 1.0)
check("reverse_warmup with K=0 is off", lambda_multiplier("reverse_warmup", 0, 0) == 0.0)

# ------------------------------------------------------- bad schedule name ----
print("\nbad schedule name is rejected")
try:
    lambda_multiplier("bogus", 0, 10)
    check("bad schedule raises", False)
except ValueError:
    check("bad schedule raises", True)

# -------------------------------------------------------- detach schedule -----
print("\ndetach schedule (single switch)")
check("late=None keeps base (early)", detach_for_epoch("act", None, 20, 0) == "act")
check("late=None keeps base (late)", detach_for_epoch("act", None, 20, 99) == "act")
check("base before switch", detach_for_epoch("act", "none", 20, 19) == "act")
check("late at switch", detach_for_epoch("act", "none", 20, 20) == "none")
check("late after switch", detach_for_epoch("act", "none", 20, 50) == "none")

# ---------------------------------------------------------- validators --------
print("\nvalidators")
# lambda: constant needs no warmup; others need warmup_epochs > 0.
try:
    validate_lambda_schedule("ax", "constant", 0)
    check("constant valid with K=0", True)
except ValueError:
    check("constant valid with K=0", False)
try:
    validate_lambda_schedule("ax", "warmup", 0)
    check("warmup K=0 rejected", False)
except ValueError:
    check("warmup K=0 rejected", True)
try:
    validate_lambda_schedule("ax", "bogus", 10)
    check("bad lambda schedule rejected", False)
except ValueError:
    check("bad lambda schedule rejected", True)

AX2_MODES = ("none", "act", "outcome")
# detach: base must be valid; late None is fine; late set needs switch > 0.
try:
    validate_detach_schedule("ax", "act", None, 0, AX2_MODES)
    check("no-switch detach valid", True)
except ValueError:
    check("no-switch detach valid", False)
try:
    validate_detach_schedule("ax", "act", "none", 20, AX2_MODES)
    check("act->none@20 valid", True)
except ValueError:
    check("act->none@20 valid", False)
try:
    validate_detach_schedule("ax", "act", "none", 0, AX2_MODES)
    check("switch epoch 0 rejected", False)
except ValueError:
    check("switch epoch 0 rejected", True)
try:
    validate_detach_schedule("ax", "bogus", None, 0, AX2_MODES)
    check("bad base mode rejected", False)
except ValueError:
    check("bad base mode rejected", True)
try:
    validate_detach_schedule("ax", "act", "bogus", 20, AX2_MODES)
    check("bad late mode rejected", False)
except ValueError:
    check("bad late mode rejected", True)

# ------------------------------------------ all detach directions, both axioms --
# The switch is a plain string handoff, so it supports every detach mode either
# axiom defines (ax1: none/ttne/rrt; ax2: none/act/outcome) in any base->late
# combination. The module-side effect of each mode is covered by the per-axiom
# loss tests; here we only confirm the schedule layer accepts and routes them.
print("\nall detach directions (both axioms)")
AX1_MODES = ("none", "ttne", "rrt")
for base, late in (("ttne", "none"), ("rrt", "none"), ("none", "ttne"), ("none", "rrt")):
    check(f"ax1 {base}->{late} routes at switch",
          detach_for_epoch(base, late, 10, 5) == base
          and detach_for_epoch(base, late, 10, 10) == late)
    try:
        validate_detach_schedule("ax1", base, late, 10, AX1_MODES)
        check(f"ax1 {base}->{late} validates", True)
    except ValueError:
        check(f"ax1 {base}->{late} validates", False)
for base, late in (("act", "none"), ("outcome", "none"), ("none", "act"), ("none", "outcome")):
    check(f"ax2 {base}->{late} routes at switch",
          detach_for_epoch(base, late, 10, 5) == base
          and detach_for_epoch(base, late, 10, 10) == late)
    try:
        validate_detach_schedule("ax2", base, late, 10, AX2_MODES)
        check(f"ax2 {base}->{late} validates", True)
    except ValueError:
        check(f"ax2 {base}->{late} validates", False)
# A mode from the wrong axiom is rejected (ttne is not an ax2 mode).
try:
    validate_detach_schedule("ax2", "none", "ttne", 10, AX2_MODES)
    check("cross-axiom mode rejected", False)
except ValueError:
    check("cross-axiom mode rejected", True)

# -------------------------------------------------- schedule set is complete --
check("LAMBDA_SCHEDULES has the 5 expected",
      set(LAMBDA_SCHEDULES) == {"constant", "warmup", "ramp", "cosine", "reverse_warmup"})

# ------------------------------------------------------------------ summary --
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name in FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
print("ALL TESTS PASSED")
