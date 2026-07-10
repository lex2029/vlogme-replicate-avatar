#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SMARTBLOG_APP_DIR:-/workspace/SmartBlog-Live-media}"
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

load_worker_env "$LOCKED_CONF" "$RUNTIME_CONF" "$PROFILE_CONF" "$SECRETS_ENV"
source_conf "$UPSCALE_CONF"
source_conf "$SECRETS_ENV"
source_conf "${SECRETS_ENV%.conf}.local.conf"

"${SMARTBLOG_UPSCALE_PYTHON:-$ROOT_DIR/.venv/bin/python}" - <<'PY'
import json
import sys
import urllib.request

for port in (8798, 8799, 8888):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise SystemExit(f"service {port} health request failed: {e}")
    if not data.get("ok") or (port in (8798, 8799) and not data.get("ready")):
        raise SystemExit(f"service {port} unhealthy: {data}")
PY

if is_true "${SMARTBLOG_MUSETALK_SERVICE_ENABLED:-0}"; then
  "${SMARTBLOG_UPSCALE_PYTHON:-$ROOT_DIR/.venv/bin/python}" - <<'PY'
import json
import os
import sys
import urllib.request

host = os.getenv("SMARTBLOG_MUSETALK_SERVICE_HOST", "127.0.0.1")
if host in {"0.0.0.0", "::"}:
    host = "127.0.0.1"
port = int(os.getenv("SMARTBLOG_MUSETALK_SERVICE_PORT", "8800"))
try:
    with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3) as resp:
        data = json.loads(resp.read().decode("utf-8"))
except Exception as e:
    raise SystemExit(f"MuseTalk health request failed: {e}")
if not data.get("ok") or not data.get("ready"):
    raise SystemExit(f"MuseTalk unhealthy: {data}")
PY
fi

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

if is_true_local "${VLOGME_RENDER_WORKER_ENABLED:-0}"; then
  pid_file="${VLOGME_RENDER_WORKER_PID_FILE:-$RUN_STATE_DIR/vlogme_render_worker.pid}"
  pid=""
  if [[ -f "$pid_file" ]]; then
    pid="$(head -n1 "$pid_file" 2>/dev/null | xargs || true)"
  fi
  if ! pid_alive "$pid"; then
    echo "VlogMe render worker is enabled but not alive pid=${pid:-none}" >&2
    exit 1
  fi
fi
