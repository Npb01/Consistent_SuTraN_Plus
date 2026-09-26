#!/usr/bin/env bash
# Download results (and ALL checkpoints) from the cluster to the laptop.
#
#   scripts/fetch.sh <user@login-host> [LOG_NAME] [remote_dir]
#
# With a LOG_NAME (e.g. BPIC_17_DR) only that log's result tree is pulled;
# without one, all result folders plus logs/ are pulled. Checkpoints are large
# and are included by design -- narrow with LOG_NAME to keep a pull small.
set -euo pipefail

REMOTE="${1:?usage: scripts/fetch.sh <user@host> [LOG_NAME] [remote_dir]}"
LOG="${2:-}"
REMOTE_USER="${REMOTE%%@*}"
REMOTE_DIR="${3:-/scratch-shared/${REMOTE_USER}/thesis}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"

RSYNC="rsync -avz --info=progress2"

if [[ -n "${LOG}" ]]; then
  echo "Fetching ${LOG} results (incl. checkpoints) <- ${REMOTE}"
  ${RSYNC} "${REMOTE}:${REMOTE_DIR}/${LOG}/" "${HERE}/${LOG}/"
else
  echo "Fetching all result logs + job logs <- ${REMOTE}"
  for L in BPIC_17_DR BPIC_17 BPIC_19; do
    ${RSYNC} "${REMOTE}:${REMOTE_DIR}/${L}/" "${HERE}/${L}/" 2>/dev/null || \
      echo "  (skipped ${L}: not present on server)"
  done
  ${RSYNC} "${REMOTE}:${REMOTE_DIR}/logs/" "${HERE}/logs/"
fi
