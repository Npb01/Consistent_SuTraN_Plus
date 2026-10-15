#!/usr/bin/env bash
# Shared REMOTE_DIR resolution for scripts/deploy.sh and scripts/fetch.sh.
#
# Source this, then call `_resolve_remote_dir <repo_root> [explicit_target]`.
# It sets two variables in the caller:
#   REMOTE_DIR      where the code lives on the cluster
#   REMOTE_DIR_SRC  how that was resolved (for logging / the deploy stamp)
#
# experiments/cluster.env (git-ignored, see cluster.example.env) is auto-loaded
# when present, so REMOTE_DIR does NOT need to be sourced by hand each session.
# Precedence: an explicit target argument, then a REMOTE_DIR already in the
# environment, then the value from cluster.env, then a home-dir default (warned).

# Fallback target under the remote home when nothing else resolves. A
# project-specific name so a stray default copy is at least obvious.
_DEFAULT_REMOTE_DIR="consistent_sutran_plus"

_resolve_remote_dir() {
  local here="$1"
  local explicit="${2:-}"
  local env_remote_dir="${REMOTE_DIR:-}"      # whatever was already in the env
  local cluster_env="${here}/experiments/cluster.env"
  local file_remote_dir=""

  if [ -f "${cluster_env}" ]; then
    # Load the file's exports (allexport so bare assignments export too).
    set -a; . "${cluster_env}"; set +a
    file_remote_dir="${REMOTE_DIR:-}"
    echo "Loaded config from experiments/cluster.env" >&2
  fi

  if   [ -n "${explicit}" ];        then REMOTE_DIR="${explicit}";        REMOTE_DIR_SRC="argument"
  elif [ -n "${env_remote_dir}" ];  then REMOTE_DIR="${env_remote_dir}";  REMOTE_DIR_SRC="REMOTE_DIR env"
  elif [ -n "${file_remote_dir}" ]; then REMOTE_DIR="${file_remote_dir}"; REMOTE_DIR_SRC="cluster.env"
  else REMOTE_DIR="${_DEFAULT_REMOTE_DIR}"; REMOTE_DIR_SRC="default"; fi

  if [ "${REMOTE_DIR_SRC}" = "default" ]; then
    echo "WARNING: REMOTE_DIR is not set and no target was given -- using the home-dir" >&2
    echo "         default '~/${REMOTE_DIR}'. Jobs usually run from a scratch dir, so this" >&2
    echo "         can create a stray copy that 'submit' never sees." >&2
    if [ ! -f "${cluster_env}" ]; then
      echo "         Create it once so this is automatic:" >&2
      echo "           cp experiments/cluster.example.env experiments/cluster.env" >&2
      echo "         then set REMOTE_DIR inside it." >&2
    else
      echo "         Set REMOTE_DIR inside experiments/cluster.env." >&2
    fi
  fi
}
