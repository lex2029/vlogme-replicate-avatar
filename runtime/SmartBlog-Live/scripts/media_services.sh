#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${SMARTBLOG_MEDIA_PROFILE:-rtxpro6000-media}"

usage() {
  cat <<'EOF'
Usage:
  scripts/media_services.sh start|start-wait|stop|restart|restart-wait|status

Starts/stops the render media host services only:
  - upscale_service
  - Hunyuan service
  - MMAudio service
  - optional MuseTalk lip-sync service
  - optional VlogMe render API poller

This entrypoint does not start the SmartBlog job worker, modeld, or local edge.
EOF
}

apply_profile() {
  "$ROOT_DIR/scripts/profile.sh" "$PROFILE" >/dev/null
}

start_services() {
  local wait="${1:-0}"
  apply_profile
  if [[ "$wait" == "1" ]]; then
    "$ROOT_DIR/scripts/upscale_service.sh" start-wait
    "$ROOT_DIR/scripts/hunyuan_service.sh" start-wait
    "$ROOT_DIR/scripts/mmaudio_service.sh" start-wait
    "$ROOT_DIR/scripts/musetalk_service.sh" start-wait
    "$ROOT_DIR/scripts/vlogme_render_worker.sh" start-wait
  else
    "$ROOT_DIR/scripts/upscale_service.sh" start
    "$ROOT_DIR/scripts/hunyuan_service.sh" start
    "$ROOT_DIR/scripts/mmaudio_service.sh" start
    "$ROOT_DIR/scripts/musetalk_service.sh" start
    "$ROOT_DIR/scripts/vlogme_render_worker.sh" start
  fi
}

stop_services() {
  "$ROOT_DIR/scripts/vlogme_render_worker.sh" stop || true
  "$ROOT_DIR/scripts/musetalk_service.sh" stop || true
  "$ROOT_DIR/scripts/mmaudio_service.sh" stop || true
  "$ROOT_DIR/scripts/hunyuan_service.sh" stop || true
  "$ROOT_DIR/scripts/upscale_service.sh" stop || true
}

status_services() {
  apply_profile
  "$ROOT_DIR/scripts/upscale_service.sh" status || true
  "$ROOT_DIR/scripts/hunyuan_service.sh" status || true
  "$ROOT_DIR/scripts/mmaudio_service.sh" status || true
  "$ROOT_DIR/scripts/musetalk_service.sh" status || true
  "$ROOT_DIR/scripts/vlogme_render_worker.sh" status || true
}

case "${1:-start-wait}" in
  start)
    start_services 0
    ;;
  start-wait)
    start_services 1
    ;;
  stop)
    stop_services
    ;;
  restart)
    stop_services
    start_services 0
    ;;
  restart-wait)
    stop_services
    start_services 1
    ;;
  status)
    status_services
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
