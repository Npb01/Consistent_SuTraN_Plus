"""Tests for the provenance record written into each run folder.

    python test_run_provenance.py
"""

import sys
import tempfile
from pathlib import Path

from experiments.run import PROVENANCE_FILE, write_provenance

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'OK  ' if condition else 'FAIL'} {name}{('  -- ' + detail) if detail else ''}")


cfg = {"log_name": "bpic_12", "seed": 101}
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    code = tmp / "code"
    code.mkdir()
    (code / "DEPLOYED_VERSION.txt").write_text("commit:    abc123\ndirty:     0 uncommitted files\n")
    slurm = {"SLURM_JOB_ID": "7", "SLURM_ARRAY_JOB_ID": "500", "SLURM_ARRAY_TASK_ID": "3"}

    print("cluster job")
    run_dir = tmp / "BPIC_12" / "run"
    write_provenance(run_dir, "ft_time_bpic12", 3, cfg, repo_root=code, env=slurm)
    text = (run_dir / PROVENANCE_FILE).read_text()
    check("run folder created", run_dir.is_dir())
    check("deploy stamp copied", "commit:    abc123" in text and "dirty:     0" in text)
    check("array job id recorded", "slurm_job: 500_3" in text)
    check("sweep and index recorded", "sweep:     ft_time_bpic12 [3]" in text)
    check("config recorded", str(cfg) in text)

    print("\nrerun in place")
    write_provenance(run_dir, "ft_time_bpic12", 3, cfg, repo_root=code, env=slurm)
    text = (run_dir / PROVENANCE_FILE).read_text()
    check("second start appended, first kept", text.count("started:") == 2)

    print("\nmissing stamp on the cluster")
    run_dir = tmp / "BPIC_12" / "nostamp"
    write_provenance(run_dir, "s", 0, cfg, repo_root=tmp / "empty", env=slurm)
    check("marked unknown", "unknown (not deployed via deploy.sh)"
          in (run_dir / PROVENANCE_FILE).read_text())

    print("\nlocal run")
    run_dir = tmp / "BPIC_12" / "local"
    write_provenance(run_dir, "s", 0, cfg, repo_root=code, env={})
    text = (run_dir / PROVENANCE_FILE).read_text()
    check("stale deploy stamp ignored", "abc123" not in text)
    check("no slurm job", "slurm_job: none" in text)
    check("version from git or marked unknown",
          "local run" in text or "unknown (no deploy stamp, no git)" in text)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name in FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
print("ALL TESTS PASSED")
