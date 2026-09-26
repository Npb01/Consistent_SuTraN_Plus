"""Run ONE config from a sweep, selected by index. One SLURM array task = one
call of this module.

    python -m experiments.run                     # list all sweeps
    python -m experiments.run <sweep>             # list that sweep's configs
    python -m experiments.run <sweep> <index>     # run config <index>
"""
import sys

from experiments import grid
from experiments.sweeps import SWEEPS
from TRAIN_EVAL_FUNCTIONALITY.run_mto_experiment import run_mto_experiment


def _list_sweeps():
    print("Available sweeps:")
    for name, sweep in SWEEPS.items():
        n = len(grid.configs_for(sweep))
        print(f"  {name:<20} {n:>3} configs   ({sweep.partition}, {sweep.gres})")


def main(argv):
    if not argv:
        _list_sweeps()
        return 0

    name = argv[0]
    if name not in SWEEPS:
        print(f"unknown sweep {name!r}\n")
        _list_sweeps()
        return 2

    configs = grid.configs_for(SWEEPS[name])

    if len(argv) == 1:
        for i, cfg in enumerate(configs):
            d = grid.resolve_run_dir(cfg)
            print(f"{i:>3}  {d.name}{'   [exists]' if d.exists() else ''}")
        print(f"\n{len(configs)} configs")
        return 0

    idx = int(argv[1])
    if not 0 <= idx < len(configs):
        print(f"index {idx} out of range 0..{len(configs) - 1}")
        return 2

    cfg = configs[idx]
    d = grid.resolve_run_dir(cfg)
    if d.exists():
        # Re-submitting a sweep then naturally resumes: finished configs are
        # skipped. To rerun one, delete its folder first.
        print(f"SKIP [{idx}] already exists: {d}", flush=True)
        return 0

    print(f"RUN [{idx}] -> {d}\n     {cfg}", flush=True)
    run_mto_experiment(**cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
