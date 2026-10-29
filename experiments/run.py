"""Run ONE config from a sweep, selected by index. One SLURM array task = one
call of this module.

    python -m experiments.run                     # list all sweeps
    python -m experiments.run <sweep>             # list that sweep's configs
    python -m experiments.run <sweep> <index>     # run config <index>
"""
import datetime
import os
import socket
import subprocess
import sys
from pathlib import Path

from experiments import grid
from experiments.sweeps import SWEEPS
from TRAIN_EVAL_FUNCTIONALITY.run_mto_experiment import run_mto_experiment

REPO_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_FILE = "provenance.txt"


def _code_version(repo_root, env):
    """Which code is running: the deploy stamp on the cluster, git locally.

    ``scripts/deploy.sh`` writes ``DEPLOYED_VERSION.txt`` next to the code it
    uploads, so a SLURM job reads it there. A local run has no deploy of its own
    (a stamp left behind by an earlier deploy describes the server, not this
    checkout), so git is asked instead.
    """
    if "SLURM_JOB_ID" in env:
        stamp = Path(repo_root) / "DEPLOYED_VERSION.txt"
        if stamp.exists():
            return stamp.read_text().strip()
        return "unknown (not deployed via deploy.sh)"
    try:
        def git(*args):
            return subprocess.run(["git", *args], cwd=repo_root, capture_output=True,
                                  text=True, check=True).stdout
        commit = git("rev-parse", "HEAD").strip()
        dirty = len(git("status", "--porcelain").splitlines())
        return f"commit:    {commit}\ndirty:     {dirty} uncommitted files\nlocal run"
    except (OSError, subprocess.CalledProcessError):
        return "unknown (no deploy stamp, no git)"


def _gpu_name():
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"


def write_provenance(run_dir, sweep, idx, cfg, repo_root=REPO_ROOT, env=None):
    """Append which code, job and machine started this run to the run folder.

    Appending keeps the history of a folder that was rerun in place or
    re-evaluated: every start leaves one block, the last block is the latest.
    """
    env = os.environ if env is None else env
    job = env.get("SLURM_ARRAY_JOB_ID")
    job = f"{job}_{env.get('SLURM_ARRAY_TASK_ID')}" if job else env.get("SLURM_JOB_ID", "none")
    block = "\n".join([
        f"started:   {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"sweep:     {sweep} [{idx}]",
        f"slurm_job: {job}",
        f"host:      {socket.gethostname()}",
        f"gpu:       {_gpu_name()}",
        _code_version(repo_root, env),
        f"config:    {cfg}",
    ])
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(run_dir) / PROVENANCE_FILE, "a") as f:
        f.write(block + "\n\n")


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
            status = ("   [done]" if grid.is_complete(d)
                      else "   [incomplete]" if d.exists() else "")
            print(f"{i:>3}  {d.name}{status}")
        print(f"\n{len(configs)} configs")
        return 0

    idx = int(argv[1])
    if not 0 <= idx < len(configs):
        print(f"index {idx} out of range 0..{len(configs) - 1}")
        return 2

    cfg = configs[idx]
    d = grid.resolve_run_dir(cfg)
    if cfg.get("eval_only"):
        # Re-evaluates the finished run in this folder; safe to repeat.
        if not d.exists():
            print(f"FAIL [{idx}] eval_only needs a trained run at: {d}", flush=True)
            return 1
    elif grid.is_complete(d):
        # Re-submitting a sweep then naturally resumes: finished configs are
        # skipped. To rerun one, delete its folder first.
        print(f"SKIP [{idx}] already finished: {d}", flush=True)
        return 0
    elif d.exists():
        # A folder without test results was stopped before the end (e.g. by
        # its walltime). The run starts over in the same folder; submit.py
        # refuses to start a sweep that still has jobs queued or running, so
        # no live run writes here.
        print(f"RERUN [{idx}] incomplete folder: {d}", flush=True)

    print(f"RUN [{idx}] -> {d}\n     {cfg}", flush=True)
    write_provenance(d, name, idx, cfg)
    run_mto_experiment(**cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
