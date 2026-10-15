"""Epoch-dependent schedules for the axiom-consistency loss terms.

Each axiom term is added to the composite loss as ``lambda * (1 - sat)`` with a
per-term detach mode (see `axiom_builders.py`). These helpers let a run vary both
of those with the training epoch:

- **lambda schedule** scales the target lambda by a factor in ``[0, 1]`` that
  depends on the (absolute, 0-indexed) epoch. The scalar ``lambda`` stays the
  target; the schedule only shapes the path to it (e.g. keep it off for the
  first K epochs, or ramp it in).
- **detach schedule** switches once from a base detach mode to a ``late`` detach
  mode at a given epoch. ``late is None`` keeps the base mode for the whole run.

Everything here is pure (no torch, no state), so it is unit-tested without a GPU
and called once per epoch from the train loop. The defaults reproduce constant
lambda and a fixed detach mode exactly.
"""

from __future__ import annotations

import math

LAMBDA_SCHEDULES = ("constant", "warmup", "ramp", "cosine", "reverse_warmup")


def lambda_multiplier(schedule, epoch, warmup_epochs):
    """Fraction of the target lambda in effect at ``epoch``.

    Parameters
    ----------
    schedule : str
        One of ``LAMBDA_SCHEDULES``:
          - ``constant``       : always 1.0 (``warmup_epochs`` ignored).
          - ``warmup``         : 0.0 while ``epoch < warmup_epochs``, then 1.0
                                 (a hard switch).
          - ``ramp``           : linear 0 -> 1 over ``[0, warmup_epochs]``, then 1.
          - ``cosine``         : smooth 0 -> 1 over ``[0, warmup_epochs]`` (half a
                                 cosine), then 1.
          - ``reverse_warmup`` : 1.0 while ``epoch < warmup_epochs``, then 0.0.
    epoch : int
        Absolute, 0-indexed training epoch. This counts real epochs, so the
        schedule is unaffected by how often validation runs.
    warmup_epochs : int
        Epoch at which the ramp/warmup completes (or at which ``reverse_warmup``
        switches off). Ignored for ``constant``.

    Returns
    -------
    float
        A multiplier in ``[0, 1]`` applied to the target lambda.
    """
    if schedule not in LAMBDA_SCHEDULES:
        raise ValueError(
            f"lambda schedule must be one of {LAMBDA_SCHEDULES}, got {schedule!r}"
        )
    if schedule == "constant":
        return 1.0

    we = int(warmup_epochs)
    if we <= 0:
        # Nothing to warm up over: ramp/warmup/cosine are immediately full on;
        # reverse_warmup is immediately off.
        return 0.0 if schedule == "reverse_warmup" else 1.0

    if schedule == "warmup":
        return 0.0 if epoch < we else 1.0
    if schedule == "reverse_warmup":
        return 1.0 if epoch < we else 0.0

    frac = min(max(epoch / we, 0.0), 1.0)
    if schedule == "ramp":
        return frac
    if schedule == "cosine":
        return 0.5 * (1.0 - math.cos(math.pi * frac))
    raise AssertionError(f"unhandled schedule {schedule!r}")  # pragma: no cover


def detach_for_epoch(base_mode, late_mode, switch_epoch, epoch):
    """The detach mode in effect at ``epoch``.

    Returns ``base_mode`` while ``epoch < switch_epoch`` and ``late_mode`` from
    then on. ``late_mode is None`` disables the switch (``base_mode`` throughout).

    Parameters
    ----------
    base_mode : str
        The detach mode used before the switch.
    late_mode : str or None
        The detach mode used from ``switch_epoch`` onward. ``None`` = no switch.
    switch_epoch : int
        Absolute, 0-indexed epoch at which the mode changes.
    epoch : int
        Absolute, 0-indexed training epoch.
    """
    if late_mode is None:
        return base_mode
    return base_mode if epoch < int(switch_epoch) else late_mode


def validate_lambda_schedule(name, schedule, warmup_epochs):
    """Raise if a lambda-schedule spec is malformed (call once at run setup).

    ``name`` only appears in the error message (e.g. the axiom it belongs to).
    """
    if schedule not in LAMBDA_SCHEDULES:
        raise ValueError(
            f"{name}: lambda schedule must be one of {LAMBDA_SCHEDULES}, "
            f"got {schedule!r}"
        )
    if schedule != "constant" and int(warmup_epochs) <= 0:
        raise ValueError(
            f"{name}: schedule {schedule!r} needs warmup_epochs > 0, "
            f"got {warmup_epochs!r}"
        )


def validate_detach_schedule(name, base_mode, late_mode, switch_epoch, valid_modes):
    """Raise if a detach-schedule spec is malformed (call once at run setup).

    ``valid_modes`` is the tuple of detach modes the axiom term accepts.
    """
    if base_mode not in valid_modes:
        raise ValueError(
            f"{name}: detach base mode must be one of {valid_modes}, got {base_mode!r}"
        )
    if late_mode is None:
        return
    if late_mode not in valid_modes:
        raise ValueError(
            f"{name}: detach late mode must be one of {valid_modes} or None, "
            f"got {late_mode!r}"
        )
    if int(switch_epoch) <= 0:
        raise ValueError(
            f"{name}: a detach switch needs switch_epoch > 0, got {switch_epoch!r}"
        )
