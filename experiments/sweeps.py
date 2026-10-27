"""The sweep registry -- the ONE file you edit to define a new experiment.

Each entry is a grid spec (swept params as lists, fixed params as scalars) plus
the SLURM resources one array task needs. `experiments/submit.py` reads the grid
to compute the array size, so there is no hand-synced `--array` range.

Add an experiment by adding one `Sweep(...)` to `SWEEPS`. Grid keys are keyword
arguments of `run_mto_experiment.run_mto_experiment`; omitted keys use its
defaults.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Sweep:
    grid: dict            # {param: value | [values]} -- see experiments/grid.py
    partition: str        # SLURM partition, e.g. "tue.gpu1.q"
    gres: str             # generic resource, e.g. "gpu:1g.6gb:1"
    time: str             # walltime PER array task (one training run)
    concurrency: int = 6  # max array tasks running at once (%N); set to free slices
    cpus_per_task: int = 4
    mem: str = "8G"


# Seeds of the full-data baselines: five per log, as in the SuTraN+ protocol,
# including the seeds 103-105 used by the axiom sweeps so runs can be paired.
_BASELINE_SEEDS = [101, 102, 103, 104, 105]


def _uw_baseline(log_name, time, mem, seeds=_BASELINE_SEEDS, batch_size=512):
    """Full-data SuTraN+ baseline (UW, consistency axioms off) for one log.

    All baseline sweeps share this grid, so they differ only in the log, the
    SLURM resources that log needs, and (for controls) the seeds and batch size.
    """
    return Sweep(
        grid=dict(
            log_name=log_name,
            mto_technique="uw",
            seed=seeds,
            lambda_ltn=0.0,
            lambda_ltn_outcome=0.0,
            subset_fraction=1.0,
            val_subset_fraction=1.0,
            num_epochs=200,
            # patience counts validations: 6 * validate_every = 24 epochs
            # without improvement, the SuTraN+ early-stopping window.
            validate_every=4,
            patience=6,
            batch_size=batch_size,
        ),
        partition="mcs.gpu.q",
        gres="gpu:nvidia_geforce_rtx_2080_ti:1",
        time=time,
        concurrency=5,
        mem=mem,
    )


SWEEPS = {
    # =====================================================================
    # HISTORICAL / SUPERSEDED sweeps. Kept for provenance (what each was for and
    # why it was retired is noted per entry below). Do not delete -- old run-dir
    # names and analyses reference these grids.
    #   - "verify"              : EW equivalence smoke (still valid).
    #   - "axiom1_uw_bpic17dr"  : pre-impl-comparison lambda/detach sweep,
    #                             default impl only -> superseded by
    #                             "axiom1_impl_uw".
    #   - "axiom2_uw_bpic17dr"  : ditto -> superseded by "axiom2_impl_uw".
    # =====================================================================

    # Small end-to-end run to confirm the new cleaned setup
    # reproduces the old one. Equal-weighting BASELINE (all axioms off) on a
    # small case subset for a few epochs, with early stopping effectively
    # disabled (patience > num_epochs) so every epoch runs deterministically.
    #
    # Equivalence check: run the IDENTICAL command in the OLD preliminary repo,
    # on the SAME GPU type, and diff backup_results.csv (per-epoch train/val
    # metrics should match if the refactor is behaviour-preserving):
    #   python -m TRAIN_EVAL_FUNCTIONALITY.run_mto_experiment \
    #     --log_name BPIC_17_DR --MTO_technique equal_weighting --seed 42 \
    #     --subset_fraction 0.1 --val_subset_fraction 0.1 --num_epochs 5 --patience 99
    "verify": Sweep(
        grid=dict(
            log_name="BPIC_17_DR",
            mto_technique="equal_weighting",
            seed=42,
            subset_fraction=0.1,
            val_subset_fraction=0.1,
            num_epochs=5,
            patience=99,
        ),
        partition="tue.gpu1.q",
        gres="gpu:1g.6gb:1",
        time="02:00:00",
        concurrency=1,
    ),

    # Axiom-1 (time consistency) sweep on BPIC_17_DR, UW technique. The
    # lambda_ltn=0.0 baseline is produced once per seed (detach modes collapse
    # at lambda 0).
    # CAVEAT: lambda was calibrated on equal weighting; under UW the composite
    # loss has a different magnitude, so lambda does NOT transfer -- read
    # ltn_ax1_contrib in backup_results.csv after the first runs and rescale the
    # grid if the axiom's share of the objective is off.
    "axiom1_uw_bpic17dr": Sweep(
        grid=dict(
            log_name="BPIC_17_DR",
            mto_technique="uw",
            seed=[103, 104, 105],
            lambda_ltn=[0.0, 0.1, 0.25, 0.5, 1.0],
            detach_mode=["none", "ttne", "rrt"],
            subset_fraction=0.5,
            val_subset_fraction=0.5,
            num_epochs=40,
            patience=8,
        ),
        partition="tue.gpu1.q",
        gres="gpu:1g.6gb:1",
        time="08:00:00",
        concurrency=6,
    ),

    # Axiom-2 (outcome consistency) sweep on BPIC_17_DR, UW technique. Axiom 1
    # held OFF (lambda_ltn=0.0). Same lambda-transfer caveat as axiom1_uw: check
    # ltn_ax2_contrib and rescale for UW. All three detach modes swept (neither
    # outcome route dominates, so no gradient direction is justified in advance).
    "axiom2_uw_bpic17dr": Sweep(
        grid=dict(
            log_name="BPIC_17_DR",
            mto_technique="uw",
            lambda_ltn=0.0,
            seed=[103, 104, 105],
            lambda_ltn_outcome=[0.0, 0.25, 0.5, 1.0, 2.0],
            detach_mode_outcome=["none", "act", "outcome"],
            subset_fraction=0.5,
            val_subset_fraction=0.5,
            num_epochs=40,
            patience=8,
        ),
        partition="tue.gpu1.q",
        gres="gpu:1g.6gb:1",
        time="08:00:00",
        concurrency=6,
    ),

    # =====================================================================
    # AXIOM-IMPLEMENTATION COMPARISON (UW only).
    # Protocol: (1) prove the UW cluster path + a no-axiom baseline
    # with `uw_baseline_smoke`; (2) calibrate lambda PER IMPL with the pilots
    # (read the contribution share -- lambda does NOT transfer across impls);
    # (3) run the full comparisons. detach modes are HELD FIXED at the default
    # ("none") so the comparison isolates the impl structure, not the detach
    # axis (a separate study if wanted).
    # =====================================================================

    # (1) UW baseline smoke: axioms OFF, tiny subset, few epochs, one seed.
    # Proves the UW submit -> array -> run path end-to-end on the cluster and
    # yields a no-axiom UW reference. Also the first run that exercises the
    # compute_cost.csv logging.
    "uw_baseline_smoke": Sweep(
        grid=dict(
            log_name="BPIC_17_DR",
            mto_technique="uw",
            seed=103,
            lambda_ltn=0.0,
            lambda_ltn_outcome=0.0,
            subset_fraction=0.1,
            val_subset_fraction=0.1,
            num_epochs=6,
            patience=99,
        ),
        partition="tue.gpu1.q",
        gres="gpu:1g.6gb:1",
        time="02:00:00",
        concurrency=1,
    ),

    # (1b) Impls smoke: turn each axiom-2 impl ON briefly to confirm it BUILDS,
    # runs on the GPU, and writes plausible diagnostics (consistency_diagnostics
    # + compute_cost). uw_baseline_smoke has axioms OFF, so it never exercises an
    # impl -- this does. 4 fast runs (2 epochs, tiny subset). The axiom-1 impls
    # are low-risk (they are exercised by axiom1_impl_pilot); this smokes the
    # novel axiom-2 LTN machinery (product survival, Reichenbach/OR, pi_0 p-mean).
    "uw_impls_smoke": Sweep(
        grid=dict(
            log_name="BPIC_17_DR",
            mto_technique="uw",
            seed=103,
            lambda_ltn=0.0,
            axiom2_impl=["collapsed_q", "native_last", "native_exists", "brier_mean"],
            lambda_ltn_outcome=1.0,
            detach_mode_outcome="none",
            subset_fraction=0.05,
            val_subset_fraction=0.05,
            num_epochs=2,
            patience=99,
        ),
        partition="tue.gpu1.q",
        gres="gpu:1g.6gb:1",
        time="01:00:00",
        concurrency=4,
    ),

    # (1c) Batch-size probe: axioms OFF, vary ONLY batch_size, 2 epochs. Compare
    # `train_seconds` (and `peak_gpu_mem_mb`) across the runs via collect_results:
    # if epoch time keeps dropping as batch_size rises you were GPU-starved -> raise
    # it; where it flattens is the sweet spot. batch_size is in the run-dir name
    # (so these do NOT dedup), tagged only when != 128, so default runs elsewhere
    # keep byte-identical names. Pick a value and then FIX it across the real
    # sweeps (the compute-cost comparison needs a constant batch size).
    "batch_size_probe": Sweep(
        grid=dict(
            log_name="BPIC_17_DR",
            mto_technique="uw",
            seed=103,
            lambda_ltn=0.0,
            lambda_ltn_outcome=0.0,
            subset_fraction=0.1,
            val_subset_fraction=0.1,
            num_epochs=2,
            patience=99,
            batch_size=[128, 256, 512, 1024],
        ),
        partition="tue.gpu1.q",
        gres="gpu:1g.6gb:1",
        time="01:00:00",
        concurrency=4,
    ),

    # (2a) Axiom-1 lambda-calibration pilot: both impls x a coarse lambda grid,
    # one seed, tiny subset, all epochs (patience>epochs). Read ltn_ax1_contrib
    # vs the composite loss in backup_results.csv at the best epoch to pick a
    # per-impl lambda that hits a common contribution share; refine the grid in
    # "axiom1_impl_uw" accordingly. (naive_penalty's loss scale differs from the
    # LTN 1-sat, so its lambda will differ.)
    "axiom1_impl_pilot": Sweep(
        grid=dict(
            log_name="BPIC_17_DR",
            mto_technique="uw",
            seed=103,
            axiom1_impl=["ltn_smooth_eq", "naive_penalty"],
            lambda_ltn=[0.1, 0.25, 0.5, 1.0],
            detach_mode="none",
            subset_fraction=0.1,
            val_subset_fraction=0.1,
            num_epochs=8,
            patience=99,
            # Left at the default batch size (128): the pilot only reads the
            # contribution SHARE for lambda calibration, which is regime-robust,
            # and this keeps the already-run (bs=128) pilot results collectable.
        ),
        partition="tue.gpu1.q",
        gres="gpu:1g.6gb:1",
        time="02:00:00",
        concurrency=8,
    ),

    # (2b) Axiom-2 lambda-calibration pilot: all four impls x a coarse lambda
    # grid, one seed, tiny subset, all epochs. Same read-the-share procedure.
    # The per-position natives (native_last/native_exists) dilute satisfaction,
    # so expect them to need a LARGER lambda than collapsed_q/brier_mean for the
    # same contribution share -- that is the point of calibrating per impl.
    "axiom2_impl_pilot": Sweep(
        grid=dict(
            log_name="BPIC_17_DR",
            mto_technique="uw",
            seed=103,
            lambda_ltn=0.0,
            axiom2_impl=["collapsed_q", "native_last", "native_exists", "brier_mean"],
            lambda_ltn_outcome=[0.1, 1.0, 5, 10, 25, 50, 100],
            detach_mode_outcome="none",
            subset_fraction=0.1,
            val_subset_fraction=0.1,
            num_epochs=8,
            patience=99,
            # Default batch size (128); see axiom1_impl_pilot note.
        ),
        partition="tue.gpu1.q",
        gres="gpu:1g.6gb:1",
        time="02:00:00",
        concurrency=8,
    ),

    # (3a) Axiom-1 full comparison. lambda=0.0 gives the shared no-axiom baseline
    # (collapses across impls -> one run per seed). The lambda>0 grid below is a
    # PLACEHOLDER -- REFINE it per impl from axiom1_impl_pilot before running, and
    # compare best-of-lambda per impl (never at a shared lambda).
    "axiom1_impl_uw": Sweep(
        grid=dict(
            log_name="BPIC_17_DR",
            mto_technique="uw",
            seed=[103, 104, 105],
            axiom1_impl=["ltn_smooth_eq", "naive_penalty"],
            lambda_ltn=[0.0, 0.25, 0.5, 1.0, 2.0],   # REFINE from the pilot
            detach_mode="none",
            subset_fraction=0.5,
            val_subset_fraction=0.5,
            # EW never triggered early stopping at 40 epochs, so give more
            # headroom (60) while validating every 2nd epoch to keep the
            # (expensive, autoregressive) validation cost ~flat. NB patience
            # counts VALIDATION EVENTS: 8 * validate_every = ~16 epochs of no
            # improvement. Confirm the curve plateaus in the pilot; raise if not.
            num_epochs=60,
            validate_every=2,
            patience=8,
            batch_size=512,     # GPU-starved at 128; 512 is the throughput knee (probe)
        ),
        partition="mcs.gpu.q",
        # Whole 2080 Ti (11 GB; no MIG on these nodes). Pin the TYPE so every run
        # shares identical hardware -> the compute-cost comparison stays valid
        # (bare gpu:1 would scatter across the V100 node too). 6 of these on
        # mcs-gpub001; concurrency 4 leaves headroom for other M&CS users.
        gres="gpu:nvidia_geforce_rtx_2080_ti:1",
        time="08:00:00",
        concurrency=4,
    ),

    # (3b) Axiom-2 full comparison. lambda_ltn_outcome=0.0 gives the shared
    # no-axiom baseline (collapses across impls). The lambda>0 grid is a
    # PLACEHOLDER -- REFINE per impl from axiom2_impl_pilot, best-of per impl.
    "axiom2_impl_uw": Sweep(
        grid=dict(
            log_name="BPIC_17_DR",
            mto_technique="uw",
            lambda_ltn=0.0,
            seed=[103, 104, 105],
            axiom2_impl=["collapsed_q", "native_last", "native_exists", "brier_mean"],
            lambda_ltn_outcome=[0.0, 0.5, 1, 2.5, 5],
            detach_mode_outcome="none",
            subset_fraction=0.5,
            val_subset_fraction=0.5,
            # See axiom1_impl_uw: 60 epochs, validate every 2nd (patience counts
            # validation events -> ~16 epochs). Confirm plateau in the pilot.
            num_epochs=60,
            validate_every=2,
            patience=8,
            batch_size=512,     # GPU-starved at 128; 512 is the throughput knee (probe)
        ),
        partition="mcs.gpu.q",
        # Whole 2080 Ti (11 GB; no MIG on these nodes). Pin the TYPE so every run
        # shares identical hardware -> the compute-cost comparison stays valid
        # (bare gpu:1 would scatter across the V100 node too). 6 of these on
        # mcs-gpub001; concurrency 4 leaves headroom for other M&CS users.
        gres="gpu:nvidia_geforce_rtx_2080_ti:1",
        time="08:00:00",
        concurrency=4,
    ),

    # (3c) native_last follow-up. In axiom2_impl_uw, native_last drove its own
    # term -> ~0.01 yet the independent head-vs-suffix disagreement did not move:
    # the Reichenbach implication Last -> Eq is satisfiable by weakening the
    # antecedent (suppressing determining-activity mass) instead of aligning the
    # head -- a degenerate escape the collapsed/naive forms don't have (they score
    # an unconditional, renormalised equality). Two things to establish:
    #   (i)  detach_mode_outcome="none" at HIGH lambda -> the escape persists
    #        (more lambda rewards antecedent-weakening; disagreement stays flat).
    #   (ii) detach_mode_outcome="act" (freeze the suffix side) closes the escape
    #        -- gradient can no longer lower the antecedent, so the head is pulled
    #        onto the suffix-implied outcome. Same FOL formula; a gradient-routing
    #        choice ("treat the decoded suffix as given"). The principled rescue of
    #        the faithful LTN impl, and it feeds the SQ3 detach story.
    # lambda={0,0.5,1,2.5,5} with detach=none already ran in axiom2_impl_uw (and
    # lambda=0 collapses across detach too), so they are skipped on resume -- they
    # are listed only so one `collect_results` call yields the whole native_last
    # curve + baseline in a single CSV. New runs: none@{10,25,50,100} and
    # act@{0.5..100}. Test-set yardstick only; per-split (train/val) baseline gaps
    # come from the one-off eval-only pass, not from bloating every run here.
    "axiom2_native_last_highlambda": Sweep(
        grid=dict(
            log_name="BPIC_17_DR",
            mto_technique="uw",
            lambda_ltn=0.0,
            seed=[103, 104, 105],
            axiom2_impl="native_last",
            detach_mode_outcome=["none", "act"],
            lambda_ltn_outcome=[0, 0.5, 1, 2.5, 5, 10, 25, 50, 100],
            subset_fraction=0.5,
            val_subset_fraction=0.5,
            num_epochs=60,
            validate_every=2,
            patience=8,
            batch_size=512,
        ),
        partition="mcs.gpu.q",
        gres="gpu:nvidia_geforce_rtx_2080_ti:1",
        time="08:00:00",
        concurrency=4,
    ),

    # =====================================================================
    # EPOCH-DEPENDENT AXIOM SCHEDULES (UW only).
    # The axiom term's weight and detach mode can vary with the epoch (see
    # `axiom_schedules`). Two schedule families are swept against the fixed
    # (constant-lambda, fixed-detach) points already produced by
    # axiom2_native_last_highlambda:
    #   - lambda schedule : keep the term off or ramp it in before reaching the
    #                       target weight (warm-start / ramp / reverse control).
    #   - detach schedule : switch the detach mode once mid-training (e.g. move
    #                       the gradient from the outcome head only to both heads).
    # native_last is used throughout so these overlay directly on its fixed-point
    # curves; the target weights are the operating points read off that sweep
    # (lambda_outcome=25 reaches the strong-consistency endpoint under
    # detach="none"; lambda_outcome in 1..10 under detach="act" is the
    # outcome-head accuracy operating range).
    # =====================================================================

    # (S0) Schedule smoke: one tiny run exercising BOTH a non-constant lambda
    # schedule AND a detach-mode switch, to confirm the schedule code path builds,
    # runs on the GPU, writes the scheduled run-dir tags, and logs a per-epoch
    # lambda contribution that tracks the schedule. Tiny subset, 4 epochs, one
    # seed; not a result, just a build check for the new machinery.
    "axiom2_schedule_smoke": Sweep(
        grid=dict(
            log_name="BPIC_17_DR",
            mto_technique="uw",
            seed=103,
            lambda_ltn=0.0,
            axiom2_impl="native_last",
            lambda_ltn_outcome=5,
            lambda_ltn_outcome_schedule="ramp",
            lambda_ltn_outcome_warmup_epochs=3,
            detach_mode_outcome="act",
            detach_mode_outcome_late="none",
            detach_switch_outcome_epoch=2,
            subset_fraction=0.05,
            val_subset_fraction=0.05,
            num_epochs=4,
            patience=99,
        ),
        partition="mcs.gpu.q",
        gres="gpu:nvidia_geforce_rtx_2080_ti:1",
        time="01:00:00",
        concurrency=1,
    ),

    # (S1) Naive-control frontier (brier_mean). The collapsed Brier equality
    # aggregated by a plain mean instead of the LTN p-mean-error quantifier.
    # Draws the same detach {none, act} x lambda curves native_last has in
    # axiom2_native_last_highlambda, so the faithful and naive impls can be
    # overlaid. lambda_outcome=0 collapses onto the shared no-axiom baseline (and
    # across detach), so it is a single reference run. brier_mean scores an
    # unconditional equality (no antecedent to weaken), so it reaches a given
    # consistency level at a lower weight than native_last -- the grid stops at 10.
    # Needs no schedule machinery, so it can run before the schedule sweeps.
    "axiom2_brier_detach_uw": Sweep(
        grid=dict(
            log_name="BPIC_17_DR",
            mto_technique="uw",
            lambda_ltn=0.0,
            seed=[103, 104, 105],
            axiom2_impl="brier_mean",
            detach_mode_outcome=["none", "act"],
            lambda_ltn_outcome=[0, 0.5, 1, 2.5, 5, 10],
            subset_fraction=0.5,
            val_subset_fraction=0.5,
            num_epochs=60,
            validate_every=2,
            patience=8,
            batch_size=512,
        ),
        partition="mcs.gpu.q",
        gres="gpu:nvidia_geforce_rtx_2080_ti:1",
        time="08:00:00",
        concurrency=4,
    ),

    # (S2) Lambda-schedule sweep. native_last at the strong-consistency target
    # (detach="none", target lambda_outcome=25), reaching that target along
    # different paths: "warmup" (off, then full on), "ramp" (linear in),
    # "reverse_warmup" (full on early, off late -- the control, expected to lose).
    # warmup_epochs is the epoch the schedule completes/releases. Compared against
    # the constant-lambda=25 point from axiom2_native_last_highlambda: does
    # learning the marginals first retain more outcome-head accuracy / suffix
    # quality at the same end consistency?
    "axiom2_lambda_schedule_uw": Sweep(
        grid=dict(
            log_name="BPIC_17_DR",
            mto_technique="uw",
            lambda_ltn=0.0,
            seed=[103, 104, 105],
            axiom2_impl="native_last",
            detach_mode_outcome="none",
            lambda_ltn_outcome=25,
            lambda_ltn_outcome_schedule=["warmup", "ramp", "reverse_warmup"],
            lambda_ltn_outcome_warmup_epochs=[15, 30],
            subset_fraction=0.5,
            val_subset_fraction=0.5,
            num_epochs=60,
            validate_every=2,
            patience=8,
            batch_size=512,
        ),
        partition="mcs.gpu.q",
        gres="gpu:nvidia_geforce_rtx_2080_ti:1",
        time="08:00:00",
        concurrency=4,
    ),

    # Axiom 2 has two non-trivial detach directions (which head's gradient is
    # frozen): "act" freezes the suffix side (only the outcome head moves) and
    # "outcome" freezes the outcome head (only the activity side moves). The three
    # sweeps below cover both directions, as fixed baselines and as scheduled
    # switches. The static "none"/"act" baselines are already in
    # axiom2_native_last_highlambda; the schedule machinery is axiom-agnostic, so
    # the axiom-1 directions ("ttne"/"rrt") follow the same pattern when needed.

    # (S3a) Static "outcome"-detach baseline (the third fixed point, alongside the
    # existing none/act curves). Freezing the outcome head routes the whole axiom
    # gradient to the activity side, pulling the suffix toward the outcome head.
    # Needed so the scheduled "outcome" directions have a fixed-point to compare
    # against. Short weight grid; weight 0 collapses onto the shared baseline.
    "axiom2_detach_outcome_static_uw": Sweep(
        grid=dict(
            log_name="BPIC_17_DR",
            mto_technique="uw",
            lambda_ltn=0.0,
            seed=[103, 104, 105],
            axiom2_impl="native_last",
            detach_mode_outcome="outcome",
            lambda_ltn_outcome=[2.5, 10, 25],
            subset_fraction=0.5,
            val_subset_fraction=0.5,
            num_epochs=60,
            validate_every=2,
            patience=8,
            batch_size=512,
        ),
        partition="mcs.gpu.q",
        gres="gpu:nvidia_geforce_rtx_2080_ti:1",
        time="08:00:00",
        concurrency=4,
    ),

    # (S3b) Release schedule: freeze one head for the first phase, then release to
    # both heads ("none") at the switch epoch. Sweeps the frozen head (act or
    # outcome) over the switch epoch and target weight. The question is whether a
    # run can keep the single-head-phase behaviour (e.g. the outcome-head accuracy
    # under "act") while gaining the mutual-agreement consistency of the "none"
    # phase, which no fixed detach mode reaches alone.
    "axiom2_detach_release_uw": Sweep(
        grid=dict(
            log_name="BPIC_17_DR",
            mto_technique="uw",
            lambda_ltn=0.0,
            seed=[103, 104, 105],
            axiom2_impl="native_last",
            detach_mode_outcome=["act", "outcome"],
            detach_mode_outcome_late="none",
            detach_switch_outcome_epoch=[15, 30],
            lambda_ltn_outcome=[10, 25],
            subset_fraction=0.5,
            val_subset_fraction=0.5,
            num_epochs=60,
            validate_every=2,
            patience=8,
            batch_size=512,
        ),
        partition="mcs.gpu.q",
        gres="gpu:nvidia_geforce_rtx_2080_ti:1",
        time="08:00:00",
        concurrency=4,
    ),

    # (S3c) Engage schedule: move both heads together first ("none"), then freeze
    # one head at the switch epoch. The mirror image of the release schedule --
    # establish mutual agreement, then let a single head specialise. Single
    # operating point (switch epoch, target weight) across the two frozen-head
    # choices to keep it small.
    "axiom2_detach_engage_uw": Sweep(
        grid=dict(
            log_name="BPIC_17_DR",
            mto_technique="uw",
            lambda_ltn=0.0,
            seed=[103, 104, 105],
            axiom2_impl="native_last",
            detach_mode_outcome="none",
            detach_mode_outcome_late=["act", "outcome"],
            detach_switch_outcome_epoch=30,
            lambda_ltn_outcome=25,
            subset_fraction=0.5,
            val_subset_fraction=0.5,
            num_epochs=60,
            validate_every=2,
            patience=8,
            batch_size=512,
        ),
        partition="mcs.gpu.q",
        gres="gpu:nvidia_geforce_rtx_2080_ti:1",
        time="08:00:00",
        concurrency=4,
    ),

    # =====================================================================
    # FULL-DATA BASELINES: SuTraN+ with UW and the consistency axioms off, on
    # BPIC_12, BPIC_19, BPIC_17_DR and BPIC_17 (grid in `_uw_baseline`).
    #
    # Protocol follows SuTraN+ (Wuyts, 2026, Sect. 6.4.4): up to 200 epochs,
    # early stopping after 24 epochs without validation improvement, the best
    # validated checkpoint is evaluated, five seeds per log. Two settings are
    # shared with the axiom sweeps instead of the original protocol: batch size
    # 512 (the throughput knee of batch_size_probe) and validation every 4th
    # epoch. A validation decodes the full validation set autoregressively, so
    # on the logs with long suffixes it costs several training epochs.
    #
    # Per-run costs measured by `baseline_timing_pilot` (RTX 2080 Ti, batch 512,
    # full data):
    #
    #   log          train/epoch  validation  final val + test   worst case
    #   BPIC_12          6.9 s        15 s         0.9 min          0.6 h
    #   BPIC_19          105 s        23 s         1.4 min          6.2 h
    #   BPIC_17_DR        69 s       189 s         9.6 min          6.6 h
    #   BPIC_17          203 s       866 s          45 min         23.8 h
    #
    # Worst case = 200 training epochs + 50 validations + final evaluation.
    # Walltimes are ~3x the worst case: runs share the node, and training under
    # contention has been measured up to ~2.8x slower. Memory covers the three
    # splits held in RAM (BPIC_17: ~12 GB of tensors; 32 GB verified).
    #
    # A run stopped by its walltime leaves a result folder, which
    # `experiments.run` treats as finished. Delete incomplete folders (no
    # TEST_SET_RESULTS) before resubmitting.
    # =====================================================================
    "baseline_uw_bpic12": _uw_baseline("BPIC_12", time="02:00:00", mem="8G"),
    "baseline_uw_bpic19": _uw_baseline("BPIC_19", time="18:00:00", mem="16G"),
    "baseline_uw_bpic17dr": _uw_baseline("BPIC_17_DR", time="20:00:00", mem="16G"),
    "baseline_uw_bpic17": _uw_baseline("BPIC_17", time="3-00:00:00", mem="32G"),

    # Batch-size control for BPIC_17. At batch 512 the BPIC_17 baseline matches
    # the published SuTraN+ UW results on time and outcome but not on the
    # activity suffix (case-based DL similarity 0.418 vs 0.486); BPIC_17_DR and
    # BPIC_19 match on every metric at the same settings. This repeats the
    # BPIC_17 baseline at batch 128 (the original SuTraN+ batch size) with
    # everything else unchanged. Batch 128 is the default, so these runs get
    # untagged folder names next to the `_bs_512` baseline folders. Two seeds
    # suffice: the gap is about ten times the seed standard deviation.
    # batch_size_probe measured 128 at ~1.33x the epoch time of 512, and the
    # validation decode does not depend on the training batch size, so the
    # worst case (~28 h) stays inside the 3-day walltime of the baseline.
    "baseline_uw_bpic17_bs128": _uw_baseline(
        "BPIC_17", time="3-00:00:00", mem="32G", seeds=[101, 102], batch_size=128),

    # Timing pilot for the baselines: one short full-data run per log, with
    # validations after epochs 1 and 3. compute_cost.csv gives the training
    # time per epoch; the gap between the checkpoints of epochs 1 and 2 minus a
    # training epoch gives the validation time. Seed 99 keeps the result
    # folders distinct from the baseline runs, which share every other
    # folder-name key.
    "baseline_timing_pilot": Sweep(
        grid=dict(
            log_name=["BPIC_12", "BPIC_19", "BPIC_17_DR", "BPIC_17"],
            mto_technique="uw",
            seed=99,
            lambda_ltn=0.0,
            lambda_ltn_outcome=0.0,
            num_epochs=4,
            validate_every=2,
            patience=99,
            batch_size=512,
        ),
        partition="mcs.gpu.q",
        gres="gpu:nvidia_geforce_rtx_2080_ti:1",
        time="04:00:00",
        concurrency=4,
        mem="32G",
    ),
}
