#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WORKER_ASSET_ROOT="${WORKER_ASSET_ROOT:-/workspace/smartblog-assets}"
source "$ROOT_DIR/scripts/lib/runtime_env.sh"

LOCKED_CONF="${WORKER_LOCKED_CONF:-$ROOT_DIR/config/worker_locked.conf}"
RUNTIME_CONF="${WORKER_RUNTIME_CONF:-$ROOT_DIR/config/worker_runtime.conf}"
PROFILE_CONF="${WORKER_PROFILE_CONF:-$ROOT_DIR/config/worker_profile.local.conf}"
if [[ ! -f "$PROFILE_CONF" ]]; then
  PROFILE_CONF="$ROOT_DIR/config/worker_profile.render_allinone.conf"
fi
UPSCALE_CONF="${SMARTBLOG_UPSCALE_CONF:-$ROOT_DIR/config/upscale_service.conf}"
SECRETS_ENV="$ROOT_DIR/config/worker_secrets.conf"
RUN_STATE_DIR="${WORKER_RUN_STATE_DIR:-$ROOT_DIR/runtime}"
PID_FILE="${VLOGME_RENDER_WORKER_PID_FILE:-$RUN_STATE_DIR/vlogme_render_worker.pid}"
LOG_DIR="${VLOGME_RENDER_WORKER_LOG_DIR:-$ROOT_DIR/logs/vlogme_render_worker}"
CURRENT_LOG_FILE="${VLOGME_RENDER_WORKER_CURRENT_LOG_FILE:-$RUN_STATE_DIR/current_vlogme_render_worker_log.txt}"

load_worker_env "$LOCKED_CONF" "$RUNTIME_CONF" "$PROFILE_CONF" "$SECRETS_ENV"
source_conf "$UPSCALE_CONF"
source_conf "$SECRETS_ENV"
source_conf "${SECRETS_ENV%.conf}.local.conf"

ENABLED="${VLOGME_RENDER_WORKER_ENABLED:-0}"
PYTHON="${VLOGME_RENDER_WORKER_PYTHON:-$ROOT_DIR/.venv/bin/python}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

is_true_local() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

pid_alive() {
  local pid="${1:-}"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

current_pid() {
  if [[ -f "$PID_FILE" ]]; then
    head -n1 "$PID_FILE" 2>/dev/null | xargs || true
  fi
}

start() {
  if ! is_true_local "$ENABLED"; then
    log "VlogMe render worker disabled"
    return 0
  fi
  if [[ -z "${VLOGME_WORKER_API_KEY:-${WORKER_API_KEY:-${POSTPROCESSING_WORKER_TOKEN:-${VLOGME_RENDER_WORKER_TOKEN:-}}}}" ]]; then
    log "VlogMe render worker enabled but VLOGME_WORKER_API_KEY/WORKER_API_KEY is missing"
    return 1
  fi
  local pid
  pid="$(current_pid)"
  if pid_alive "$pid"; then
    log "VlogMe render worker already running pid=$pid"
    return 0
  fi
  if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3)"
  fi
  mkdir -p "$RUN_STATE_DIR" "$LOG_DIR"
  local log_file="$LOG_DIR/vlogme_render_worker_$(date '+%Y%m%d_%H%M%S').log"
  printf '%s\n' "$log_file" >"$CURRENT_LOG_FILE"
  log "starting VlogMe render worker python=$PYTHON log=$log_file"
  (
    cd "$ROOT_DIR"
    export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
    exec "$PYTHON" -m avalife.vlogme_render_worker
  ) >"$log_file" 2>&1 &
  pid="$!"
  printf '%s\n' "$pid" >"$PID_FILE"
  log "VlogMe render worker pid=$pid"
}

wait_ready() {
  if ! is_true_local "$ENABLED"; then
    return 0
  fi
  local pid
  for _ in $(seq 1 10); do
    pid="$(current_pid)"
    if pid_alive "$pid"; then
      log "VlogMe render worker running"
      return 0
    fi
    sleep 1
  done
  log "VlogMe render worker did not stay alive"
  tail -n 160 "$(cat "$CURRENT_LOG_FILE" 2>/dev/null || true)" 2>/dev/null || true
  return 1
}

stop() {
  local pid
  pid="$(current_pid)"
  if pid_alive "$pid"; then
    log "stopping VlogMe render worker pid=$pid"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      pid_alive "$pid" || break
      sleep 1
    done
    pid_alive "$pid" && kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
}

status() {
  local pid
  pid="$(current_pid)"
  local mode="${VLOGME_RENDER_API_MODE:-worker_api}"
  local api="${VLOGME_WORKER_API_URL:-${VLOGME_RENDER_WORKER_API_URL:-https://vlogme.ai/api/public/v1/worker-api}}"
  if [[ "$mode" == "legacy" || "$mode" == "legacy_render" || "$mode" == "render" ]]; then
    api="${VLOGME_RENDER_LEGACY_API_BASE:-${VLOGME_RENDER_API_BASE:-https://vlogme.ai/api/public/v1/render}}"
  fi
  log "enabled=$ENABLED pid=${pid:-none} mode=$mode api=$api job_types=${VLOGME_RENDER_JOB_TYPES:-visual_scene}"
  if pid_alive "$pid"; then
    log "process=alive"
  else
    log "process=absent"
  fi
}

case "${1:-status}" in
  start)
    start
    ;;
  start-wait)
    start
    wait_ready
    ;;
  wait)
    wait_ready
    ;;
  stop)
    stop
    ;;
  restart)
    stop
    start
    ;;
  restart-wait)
    stop
    start
    wait_ready
    ;;
  status)
    status
    ;;
  *)
    echo "Usage: $0 start|start-wait|wait|stop|restart|restart-wait|status" >&2
    exit 2
    ;;
esac
