"""Re-evaluate the UW no-axiom baselines (lambda=0) to add the train/val/test
consistency diagnostics, without retraining.

    python -m experiments.eval_baselines [SEED ...]        # default: 103 104 105

Each baseline's best checkpoint already exists (from axiom1_impl_uw /
axiom2_impl_uw); `eval_only=True` loads it and runs inference on all three
splits, writing TRAIN_SET_RESULTS/ and VAL_SET_RESULTS/ beside the existing
TEST_SET_RESULTS/. Used for RQ1: is the model more self-contradictory on the data
it trained on than on held-out data? The config below matches the baseline
run-dir exactly, so results land in the existing folders (no new runs).

This is a thin convenience wrapper over the `--eval_only` CLI flag of
`run_mto_experiment`; a single baseline can equally be done with, e.g.:

    python -m TRAIN_EVAL_FUNCTIONALITY.run_mto_experiment \
        --log_name BPIC_17_DR --MTO_technique uw --seed 103 \
        --subset_fraction 0.5 --val_subset_fraction 0.5 \
        --num_epochs 60 --validate_every 2 --patience 8 --batch_size 512 \
        --eval_only

Autoregressive inference over three splits is ~1-1.5 h per seed on one GPU.
"""

from __future__ import annotations

import sys

from TRAIN_EVAL_FUNCTIONALITY.run_mto_experiment import run_mto_experiment

# Must mirror the axiom*_impl_uw baseline config so resolve_run_dir points at the
# already-trained baseline folder (no new run is created).
_BASELINE = dict(
    log_name="BPIC_17_DR",
    mto_technique="uw",
    lambda_ltn=0.0,
    lambda_ltn_outcome=0.0,
    subset_fraction=0.5,
    val_subset_fraction=0.5,
    num_epochs=60,
    validate_every=2,
    patience=8,
    batch_size=512,
)


def main(seeds):
    for seed in seeds:
        print(f"\n==== eval_only baseline seed {seed} ====", flush=True)
        run_mto_experiment(seed=seed, eval_only=True, **_BASELINE)
        print(f"==== done seed {seed} ====", flush=True)


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [103, 104, 105]
    main(seeds)
