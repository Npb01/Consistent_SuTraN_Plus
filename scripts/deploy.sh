#!/usr/bin/env bash
# Upload CODE to the cluster via rsync (no git needed on the server).
#
#   scripts/deploy.sh <user@login-host> [remote_dir]
#
# Data, results, checkpoints, the venv and logs are excluded (.deployignore),
# so this is fast and safe to run often. Data is uploaded separately, once
# (see experiments/README.md). A DEPLOYED_VERSION.txt stamp records the commit
# so the job provenance knows exactly what ran.
set -euo pipefail

REMOTE="${1:?usage: scripts/deploy.sh <user@host> [remote_dir]}"
REMOTE_USER="${REMOTE%%@*}"
REMOTE_DIR="${2:-/scratch-shared/${REMOTE_USER}/thesis}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"

# Commit stamp (works even with a dirty tree; records how dirty).
{
  echo "commit:    $(git rev-parse HEAD 2>/dev/null || echo 'no-git')"
  echo "dirty:     $(git status --porcelain 2>/dev/null | wc -l) uncommitted files"
  echo "deployed:  $(date -Is)  from $(hostname)"
} > DEPLOYED_VERSION.txt

echo "Deploying code -> ${REMOTE}:${REMOTE_DIR}"
rsync -avz --delete \
  --exclude-from="${HERE}/.deployignore" \
  "${HERE}/" "${REMOTE}:${REMOTE_DIR}/"

echo
echo "Done. On the server:"
echo "  cd ${REMOTE_DIR} && uv sync         # first time / after dependency changes"
echo "  python -m experiments.submit <sweep>"
