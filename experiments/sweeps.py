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
    # Formalised smoke run: tiny, fast, on the L4 (which can report GPU util, so
    # this is where you read mean-util / epoch-time before sizing a real sweep).
    "profile": Sweep(
        grid=dict(
            log_name="BPIC_17_DR",
            mto_technique="equal_weighting",
            seed=99,
            subset_fraction=0.5,
            val_subset_fraction=0.5,
            num_epochs=2,
            patience=8,
        ),
        partition="tue.gpu3.q",
        gres="gpu:l4.22gb:1",
        time="01:00:00",
        concurrency=1,
    ),

    # Axiom-1 lambda x detach sweep on BPIC_17_DR. The lambda_ltn=0.0 baseline is
    # produced once per seed automatically: at lambda 0 the three detach_mode
    # values yield the same folder name and dedup collapses them.
    "axiom1_bpic17dr": Sweep(
        grid=dict(
            log_name="BPIC_17_DR",
            mto_technique="equal_weighting",
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
}
