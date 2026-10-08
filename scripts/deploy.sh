#!/usr/bin/env bash
# Upload CODE to the cluster via rsync (no git needed on the server).
#
#   scripts/deploy.sh <host-or-ssh-alias> [remote_dir]
#
# remote_dir is where the code lives on the cluster (the SLURM jobs run there).
# Resolution order: the 2nd argument, else $REMOTE_DIR from the environment
# (set it once in a sourced cluster.env), else "thesis" under the remote home.
#
# Data, results, checkpoints, the venv and logs are excluded (.deployignore),
# so this is fast and safe to run often. Data is uploaded separately, once
# (see experiments/README.md). A DEPLOYED_VERSION.txt stamp records the commit
# so the job provenance knows exactly what ran.
set -euo pipefail

REMOTE="${1:?usage: scripts/deploy.sh <host-or-ssh-alias> [remote_dir]}"
REMOTE_DIR="${2:-${REMOTE_DIR:-thesis}}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"

# Commit stamp (works even with a dirty tree; records how dirty).
{
  echo "commit:    $(git rev-parse HEAD 2>/dev/null || echo 'no-git')"
  echo "dirty:     $(git status --porcelain 2>/dev/null | wc -l) uncommitted files"
  echo "deployed:  $(date -Is)  from $(hostname)"
} > DEPLOYED_VERSION.txt

# -z (compression) is intentionally omitted: it stalled rsync over this SSH
# path, and code is tiny while checkpoints are incompressible. RSYNC_RSH adds
# keepalives so a dropped connection aborts instead of hanging forever.
export RSYNC_RSH="ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
echo "Deploying code -> ${REMOTE}:${REMOTE_DIR}"
rsync -av --delete \
  --exclude-from="${HERE}/.deployignore" \
  "${HERE}/" "${REMOTE}:${REMOTE_DIR}/"

echo
echo "Done. On the server:"
echo "  cd ${REMOTE_DIR} && uv sync                          # first time / after dependency changes"
echo "  uv run python -m experiments.submit <sweep>         # add --dry-run to preview"
