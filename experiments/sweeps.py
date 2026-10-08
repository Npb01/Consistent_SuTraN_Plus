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
}
