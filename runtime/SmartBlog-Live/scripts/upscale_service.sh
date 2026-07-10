#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/lib/runtime_env.sh"

UPSCALE_CONF="${SMARTBLOG_UPSCALE_CONF:-$ROOT_DIR/config/upscale_service.conf}"
SECRETS_ENV="$ROOT_DIR/config/worker_secrets.conf"
RUN_STATE_DIR="${WORKER_RUN_STATE_DIR:-$ROOT_DIR/runtime}"
PID_FILE="${SMARTBLOG_UPSCALE_SERVICE_PID_FILE:-$RUN_STATE_DIR/upscale_service.pid}"
LOG_DIR="${SMARTBLOG_UPSCALE_SERVICE_LOG_DIR:-$ROOT_DIR/logs/upscale_service}"
CURRENT_LOG_FILE="${SMARTBLOG_UPSCALE_SERVICE_CURRENT_LOG_FILE:-$RUN_STATE_DIR/current_upscale_service_log.txt}"

load_worker_env "" "$UPSCALE_CONF" "" "$SECRETS_ENV"

HOST="${SMARTBLOG_UPSCALE_HOST:-0.0.0.0}"
PORT="${SMARTBLOG_UPSCALE_PORT:-8888}"
PYTHON="${SMARTBLOG_UPSCALE_PYTHON:-$ROOT_DIR/.venv/bin/python}"
TIMEOUT_SEC="${SMARTBLOG_UPSCALE_SERVICE_READY_TIMEOUT_SEC:-300}"

if [[ -z "${SMARTBLOG_UPSCALE_SHARED_SECRET:-}" && -n "${REMOTE_EDGE_FILE_UPSCALE_SHARED_SECRET:-}" ]]; then
  export SMARTBLOG_UPSCALE_SHARED_SECRET="$REMOTE_EDGE_FILE_UPSCALE_SHARED_SECRET"
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

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

health() {
  "$PYTHON" - "$HOST" "$PORT" <<'PY'
import json, sys, urllib.request
host, port = sys.argv[1], int(sys.argv[2])
host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
try:
    with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as r:
        data = json.loads(r.read().decode("utf-8"))
    if data.get("ok"):
        print(json.dumps(data, ensure_ascii=False))
        raise SystemExit(0)
except Exception as e:
    print(str(e), file=sys.stderr)
raise SystemExit(1)
PY
}

start() {
  local pid
  pid="$(current_pid)"
  if pid_alive "$pid"; then
    if health >/dev/null 2>&1; then
      log "upscale service already ready pid=$pid"
    else
      log "upscale service already running but not ready yet pid=$pid"
    fi
    return 0
  fi
  if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3)"
  fi
  mkdir -p "$RUN_STATE_DIR" "$LOG_DIR"
  local log_file="$LOG_DIR/upscale_service_$(date '+%Y%m%d_%H%M%S').log"
  printf '%s\n' "$log_file" >"$CURRENT_LOG_FILE"
  log "starting upscale service host=$HOST port=$PORT python=$PYTHON log=$log_file"
  (
    cd "$ROOT_DIR"
    export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES="${SMARTBLOG_UPSCALE_CUDA_VISIBLE_DEVICES:-${SMARTBLOG_UPSCALE_GPU:-0}}"
    exec "$PYTHON" -m avalife.upscale_service
  ) >"$log_file" 2>&1 &
  pid="$!"
  printf '%s\n' "$pid" >"$PID_FILE"
  log "upscale service pid=$pid"
}

wait_ready() {
  local deadline=$(( $(date +%s) + TIMEOUT_SEC ))
  while (( $(date +%s) < deadline )); do
    if health >/dev/null 2>&1; then
      log "upscale service ready"
      return 0
    fi
    local pid
    pid="$(current_pid)"
    if [[ -n "$pid" ]] && ! pid_alive "$pid"; then
      log "upscale service exited before ready pid=$pid"
      tail -n 160 "$(cat "$CURRENT_LOG_FILE" 2>/dev/null || true)" 2>/dev/null || true
      return 1
    fi
    sleep 2
  done
  log "upscale service not ready after ${TIMEOUT_SEC}s"
  tail -n 160 "$(cat "$CURRENT_LOG_FILE" 2>/dev/null || true)" 2>/dev/null || true
  return 1
}

stop() {
  local pid
  pid="$(current_pid)"
  if pid_alive "$pid"; then
    log "stopping upscale service pid=$pid"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      pid_alive "$pid" || break
      sleep 1
    done
    pid_alive "$pid" && kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
}

case "${1:-start}" in
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
    pid="$(current_pid)"
    if pid_alive "$pid"; then
      echo "running pid=$pid"
      health || true
    else
      echo "stopped"
    fi
    ;;
  *)
    echo "Usage: $0 {start|start-wait|wait|stop|restart|restart-wait|status}" >&2
    exit 2
    ;;
esac
