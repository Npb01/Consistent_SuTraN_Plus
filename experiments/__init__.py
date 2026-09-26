"""Experiment orchestration: define sweeps as data, run one config per SLURM
array task, and build the sbatch command from the sweep's declared resources.

Entry points:
    python -m experiments.run <sweep> [index]     # list configs, or run one
    python -m experiments.submit <sweep> [--dry-run]   # submit the array (on the server)
"""
