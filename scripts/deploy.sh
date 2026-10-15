#!/usr/bin/env bash
# Upload CODE to the cluster via rsync (no git needed on the server).
#
#   scripts/deploy.sh <host-or-ssh-alias> [remote_dir]
#
# remote_dir is where the code lives on the cluster (the SLURM jobs run there).
# Resolution order (see scripts/_cluster_env.sh): the 2nd argument, else a
# REMOTE_DIR already in the environment, else experiments/cluster.env (auto-
# loaded when present -- no manual `source` needed), else a home-dir default that
# is warned about loudly, since jobs usually run from a scratch dir and a default
# deploy would create a stray copy that `submit` never sees.
#
# Data, results, checkpoints, the venv and logs are excluded (.deployignore),
# so this is fast and safe to run often. Data is uploaded separately, once
# (see experiments/README.md). A DEPLOYED_VERSION.txt stamp records the commit
# AND the exact remote target, so the job provenance knows what ran and where.
set -euo pipefail

REMOTE="${1:?usage: scripts/deploy.sh <host-or-ssh-alias> [remote_dir]}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"

# Resolve REMOTE_DIR + REMOTE_DIR_SRC (auto-loads experiments/cluster.env).
source "${HERE}/scripts/_cluster_env.sh"
_resolve_remote_dir "${HERE}" "${2:-}"

# Commit stamp (works even with a dirty tree; records how dirty) plus the exact
# remote target and how it was resolved, so provenance shows where this landed.
{
  echo "commit:    $(git rev-parse HEAD 2>/dev/null || echo 'no-git')"
  echo "dirty:     $(git status --porcelain 2>/dev/null | wc -l) uncommitted files"
  echo "deployed:  $(date -Is)  from $(hostname)"
  echo "remote:    ${REMOTE}:${REMOTE_DIR}  (via ${REMOTE_DIR_SRC})"
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
