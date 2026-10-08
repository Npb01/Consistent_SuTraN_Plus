"""Build and run the sbatch command for a sweep, on the server.

The array size and concurrency come from the sweep's config count and declared
resources -- never hand-typed -- so `--array` cannot desync from the grid.

    python -m experiments.submit <sweep>             # submit
    python -m experiments.submit <sweep> --dry-run   # print the command only
"""
import os
import subprocess
import sys
from pathlib import Path

from experiments import grid
from experiments.sweeps import SWEEPS

# The array script is cluster-agnostic; submit.py passes it the repo root
# (--chdir) and a relative log path (--output), so no site path is baked in.
ARRAY_SCRIPT = "jobs/array.sh"
LOG_DIR = "logs"


def build_sbatch(name):
    sweep = SWEEPS[name]
    n = len(grid.configs_for(sweep))
    if n == 0:
        raise SystemExit(f"sweep {name!r} expands to 0 configs")
    array = f"0-{n - 1}%{sweep.concurrency}"
    # Run from (and log under) the repo root -- the cwd when you invoke submit
    # on the server. SLURM resolves the relative --output against --chdir.
    repo = str(Path.cwd())
    # Site overrides: env vars win over the sweep's declared defaults, so a new
    # cluster needs no edits to sweeps.py. SLURM_ACCOUNT is added only if set.
    partition = os.environ.get("SLURM_PARTITION", sweep.partition)
    gres = os.environ.get("SLURM_GRES", sweep.gres)
    cmd = [
        "sbatch",
        f"--job-name={name}",
        f"--chdir={repo}",
        f"--output={LOG_DIR}/%x-%A_%a.out",
        f"--partition={partition}",
        f"--gres={gres}",
        f"--cpus-per-task={sweep.cpus_per_task}",
        f"--mem={sweep.mem}",
        f"--time={sweep.time}",
        f"--array={array}",
    ]
    account = os.environ.get("SLURM_ACCOUNT")
    if account:
        cmd.append(f"--account={account}")
    cmd += [ARRAY_SCRIPT, name]
    return cmd, n


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
