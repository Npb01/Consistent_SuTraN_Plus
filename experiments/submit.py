"""Build and run the sbatch command for a sweep, on the server.

The array size and concurrency come from the sweep's config count and declared
resources -- never hand-typed -- so `--array` cannot desync from the grid.

    python -m experiments.submit <sweep>             # submit
    python -m experiments.submit <sweep> --dry-run   # print the command only
"""
import subprocess
import sys
from pathlib import Path

from experiments import grid
from experiments.sweeps import SWEEPS

# Relative to the repo root (== /scratch-shared/$USER/thesis on the server,
# which is the cwd when you run this).
ARRAY_SCRIPT = "jobs/array.sh"
LOG_DIR = "logs"


def build_sbatch(name):
    sweep = SWEEPS[name]
    n = len(grid.configs_for(sweep))
    if n == 0:
        raise SystemExit(f"sweep {name!r} expands to 0 configs")
    array = f"0-{n - 1}%{sweep.concurrency}"
    return [
        "sbatch",
        f"--job-name={name}",
        f"--partition={sweep.partition}",
        f"--gres={sweep.gres}",
        f"--cpus-per-task={sweep.cpus_per_task}",
        f"--mem={sweep.mem}",
        f"--time={sweep.time}",
        f"--array={array}",
        ARRAY_SCRIPT,
        name,
    ], n


def main(argv):
    if not argv or argv[0] not in SWEEPS:
        print("usage: python -m experiments.submit <sweep> [--dry-run]")
        print("sweeps:", ", ".join(SWEEPS))
        return 2

    name = argv[0]
    dry = "--dry-run" in argv[1:]
    cmd, n = build_sbatch(name)

    print(f"{name}: {n} configs")
    print(" ".join(cmd))
    if dry:
        return 0

    # Slurm opens the --output file before array.sh runs, so logs/ must exist.
    Path(LOG_DIR).mkdir(exist_ok=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
