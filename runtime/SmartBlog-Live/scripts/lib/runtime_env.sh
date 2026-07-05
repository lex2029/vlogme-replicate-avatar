#!/usr/bin/env bash

worker_root_dir() {
  local script_path="${1:-${BASH_SOURCE[0]}}"
  cd "$(dirname "$script_path")/../.." && pwd
}

source_conf() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  set -a
  # shellcheck disable=SC1090
  source "$file"
  set +a
}

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required parameter: $name"
    exit 1
  fi
}

require_vars() {
  local key
  for key in "$@"; do
    require_var "$key"
  done
}

is_true() {
  local raw="${1:-}"
  raw="${raw,,}"
  [[ "$raw" == "1" || "$raw" == "true" || "$raw" == "yes" || "$raw" == "on" ]]
}

boot_log() {
  if is_true "${WORKER_BOOT_LOG:-0}"; then
    echo "$@"
  fi
}

load_worker_env() {
  local locked_conf="${1:-}"
  local runtime_conf="${2:-}"
  local profile_conf="${3:-}"
  local secrets_env="${4:-}"
  local runtime_local_conf="${WORKER_RUNTIME_LOCAL_CONF:-}"
  local secrets_local_env="${WORKER_SECRETS_LOCAL_CONF:-}"
  if [[ -z "$runtime_local_conf" && -n "$runtime_conf" ]]; then
    runtime_local_conf="${runtime_conf%.conf}.local.conf"
  fi
  if [[ -z "$secrets_local_env" && -n "$secrets_env" ]]; then
    secrets_local_env="${secrets_env%.conf}.local.conf"
  fi
  source_conf "$locked_conf"
  source_conf "$secrets_env"
  source_conf "$secrets_local_env"
  source_conf "$runtime_conf"
  source_conf "$runtime_local_conf"
  source_conf "$profile_conf"
  source_conf "$secrets_env"
  source_conf "$secrets_local_env"
}
