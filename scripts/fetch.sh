#!/usr/bin/env bash
# Download results (and ALL checkpoints) from the cluster to the laptop.
#
#   scripts/fetch.sh <user@login-host> [LOG_NAME] [remote_dir]
#
# With a LOG_NAME (e.g. BPIC_17_DR) only that log's result tree is pulled;
# without one, all result folders plus logs/ are pulled. Checkpoints are large
# and are included by design -- narrow with LOG_NAME to keep a pull small.
#
# The input tensors live in the SAME <LOG>/ folder as the results, but you
# already have them locally (they are the upload source), so top-level data
# files are excluded from the pull. The anchored "/" patterns match only the
# top of <LOG>/, so result-dir checkpoints (<LOG>/SUTRAN_DA_results_*/...*.pt)
# are still fetched.
set -euo pipefail

REMOTE="${1:?usage: scripts/fetch.sh <host-or-ssh-alias> [LOG_NAME] [remote_dir]}"
LOG="${2:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"

# Resolve REMOTE_DIR (3rd arg > env > auto-loaded cluster.env > default); shared
# with scripts/deploy.sh so both always agree on where the code lives.
source "${HERE}/scripts/_cluster_env.sh"
_resolve_remote_dir "${HERE}" "${3:-}"

# -z omitted (stalls over SSH here; checkpoints are incompressible). Keepalives
# so a dropped transfer aborts instead of hanging.
export RSYNC_RSH="ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
RSYNC="rsync -av --info=progress2"
# Skip the top-level input tensors (you already have them); keep result dirs
# and their checkpoints. Anchored to the top of <LOG>/ so subdir files survive.
NODATA="--exclude=/*.pt --exclude=/*.pkl --exclude=/*.csv --exclude=/*.md5"

if [[ -n "${LOG}" ]]; then
  echo "Fetching ${LOG} results (incl. checkpoints, excl. input tensors) <- ${REMOTE}"
  ${RSYNC} ${NODATA} "${REMOTE}:${REMOTE_DIR}/${LOG}/" "${HERE}/${LOG}/"
else
  echo "Fetching all result logs + job logs <- ${REMOTE}"
  for L in BPIC_17_DR BPIC_17 BPIC_19; do
    ${RSYNC} ${NODATA} "${REMOTE}:${REMOTE_DIR}/${L}/" "${HERE}/${L}/" 2>/dev/null || \
      echo "  (skipped ${L}: not present on server)"
  done
  ${RSYNC} "${REMOTE}:${REMOTE_DIR}/logs/" "${HERE}/logs/"
fi
