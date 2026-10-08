#!/bin/bash
# Generic SLURM array script. All site-specific settings -- resources (--array,
# --partition, --gres, --time, --job-name), the working directory (--chdir) and
# the log path (--output) -- are supplied on the command line by
# experiments/submit.py, so nothing here is cluster-specific. Do NOT sbatch this
# directly, or the array size and paths will not match the sweep.
#
#     python -m experiments.submit <sweep>
#
# The only positional argument is the sweep name; the array task index selects
# which config to run.

set -euo pipefail
# submit.py passes --chdir=<repo root>; SLURM_SUBMIT_DIR is that same dir and is
# the robust fallback if the script is ever launched without --chdir.
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
source .venv/bin/activate

SWEEP="${1:?array.sh needs a sweep name (submit via: python -m experiments.submit <sweep>)}"

# Provenance: which code and which GPU produced this run. DEPLOYED_VERSION.txt is
# written by scripts/deploy.sh, so the commit is recorded without git on the server.
{
  echo "sweep:     ${SWEEP}"
  echo "deployed:  $(cat DEPLOYED_VERSION.txt 2>/dev/null || echo 'unknown (not deployed via deploy.sh)')"
  echo "host:      $(hostname)"
  echo "gpu:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  echo "slurm_job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  echo "started:   $(date -Is)"
}

python -c "import torch; print('cuda available:', torch.cuda.is_available())"

python -u -m experiments.run "${SWEEP}" "${SLURM_ARRAY_TASK_ID}"
