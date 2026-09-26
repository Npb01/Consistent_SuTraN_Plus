#!/bin/bash
# Generic SLURM array script. Resources (--array, --partition, --gres, --time,
# --job-name) are supplied on the command line by experiments/submit.py -- do
# NOT sbatch this directly, or the array size will not match the sweep.
#
#     python -m experiments.submit <sweep>
#
# The only positional argument is the sweep name; the array task index selects
# which config to run.
#SBATCH --output=/scratch-shared/%u/thesis/logs/%x-%A_%a.out

set -euo pipefail
cd "/scratch-shared/${USER}/thesis"
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
