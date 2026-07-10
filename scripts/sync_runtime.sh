#!/usr/bin/env bash
set -euo pipefail

SRC="${1:-/Users/alekseibabkin/.cache/codex/vlogme-runtime/SmartBlog-Live-b200-render-media-supervisor}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="$ROOT_DIR/runtime/SmartBlog-Live"

rm -rf "$DST"
mkdir -p "$DST/scripts/lib" "$DST/config" "$DST/assets"

rsync -a "$SRC/avalife" "$DST/"
rsync -a "$SRC/liveavatar" "$DST/"
rsync -a \
  "$SRC/.dockerignore" \
  "$SRC/Dockerfile.rtx6000-media" \
  "$SRC/LICENSE" \
  "$SRC/README.md" \
  "$SRC/requirements-b200-avatar.txt" \
  "$SRC/requirements-rtx6000-media.txt" \
  "$DST/"
rsync -a "$SRC/assets/ref_user_photo.jpg" "$DST/assets/"
rsync -a \
  "$SRC/config/upscale_service.conf" \
  "$SRC/config/worker_locked.conf" \
  "$SRC/config/worker_runtime.conf" \
  "$SRC/config/worker_profile.render_allinone.conf" \
  "$SRC/config/worker_profile.render_edge.conf" \
  "$SRC/config/worker_profile.render_overrides.conf" \
  "$DST/config/"
rsync -a "$SRC/scripts/lib/runtime_env.sh" "$DST/scripts/lib/"
rsync -a \
  "$SRC/scripts/hunyuan_service.sh" \
  "$SRC/scripts/install_musetalk_service.sh" \
  "$SRC/scripts/media_services.sh" \
  "$SRC/scripts/media_worker_registry.py" \
  "$SRC/scripts/mmaudio_service.sh" \
  "$SRC/scripts/modeld.sh" \
  "$SRC/scripts/musetalk_service.sh" \
  "$SRC/scripts/patch_musetalk_no_mmpose.py" \
  "$SRC/scripts/profile.sh" \
  "$SRC/scripts/preseed_b200_avatar_assets.sh" \
  "$SRC/scripts/preseed_rtx_media_assets.sh" \
  "$SRC/scripts/rtx6000_media_healthcheck.sh" \
  "$SRC/scripts/runpod_rtx6000_media_start.sh" \
  "$SRC/scripts/upscale_service.sh" \
  "$SRC/scripts/vlogme_render_worker.sh" \
  "$SRC/scripts/download_worker_weights.sh" \
  "$SRC/scripts/verify_worker_weights_hf.py" \
  "$DST/scripts/"

if [[ -f "$DST/config/worker_secrets.conf" ]]; then
  echo "Refusing to keep copied worker_secrets.conf" >&2
  rm -f "$DST/config/worker_secrets.conf"
  exit 1
fi

echo "Synced runtime into $DST"
