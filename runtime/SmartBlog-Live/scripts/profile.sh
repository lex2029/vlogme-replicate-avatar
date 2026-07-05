#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_DIR="$ROOT_DIR/config"
ACTIVE_PROFILE="$PROFILE_DIR/worker_profile.local.conf"
EDGE_ACTIVE_PROFILE="$PROFILE_DIR/worker_profile.edge.local.conf"
RENDER_PROFILE="$PROFILE_DIR/worker_profile.render_allinone.conf"
RENDER_EDGE_PROFILE="$PROFILE_DIR/worker_profile.render_edge.conf"
RENDER_OVERRIDES="$PROFILE_DIR/worker_profile.render_overrides.conf"

usage() {
  cat <<'EOF'
Usage:
  scripts/profile.sh show
  scripts/profile.sh b300-avatar-commander [--restart]
  scripts/profile.sh b300-hunyuan-allinone [--restart]
  scripts/profile.sh b200-avatar-commander [--restart]
  scripts/profile.sh rtxpro6000-media [--restart]
EOF
}

show_profile() {
  local src="$ACTIVE_PROFILE"
  local name
  if [[ ! -f "$src" ]]; then
    src="$RENDER_PROFILE"
  fi
  name="$(sed -n 's/^WORKER_PROFILE_NAME=//p' "$src" 2>/dev/null | tail -n1 | xargs || true)"
  echo "profile=${name:-unknown}"
  echo "config=$src"
}

append_override_section() {
  local section="$1"
  local target="$2"
  local keys tmp
  [[ -f "$RENDER_OVERRIDES" ]] || return 0
  keys="$(awk -v wanted="$section" '
    BEGIN { in_section = 0 }
    /^[[:space:]]*\[[^]]+\][[:space:]]*$/ {
      name = $0
      sub(/^[[:space:]]*\[/, "", name)
      sub(/\][[:space:]]*$/, "", name)
      in_section = (name == wanted)
      next
    }
    in_section && /^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=/ {
      line = $0
      sub(/^[[:space:]]*/, "", line)
      sub(/[[:space:]]*=.*/, "", line)
      print line
    }
  ' "$RENDER_OVERRIDES" | sort -u | tr '\n' ' ')"
  if [[ -n "$keys" ]]; then
    tmp="$(mktemp)"
    awk -v keys="$keys" '
      BEGIN {
        split(keys, parts, /[[:space:]]+/)
        for (i in parts) {
          if (parts[i] != "") delete_key[parts[i]] = 1
        }
      }
      /^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=/ {
        key = $0
        sub(/^[[:space:]]*/, "", key)
        sub(/[[:space:]]*=.*/, "", key)
        if (key in delete_key) next
      }
      { print }
    ' "$target" >"$tmp"
    mv "$tmp" "$target"
  fi
  awk -v wanted="$section" '
    BEGIN { in_section = 0; found = 0 }
    /^[[:space:]]*\[[^]]+\][[:space:]]*$/ {
      name = $0
      sub(/^[[:space:]]*\[/, "", name)
      sub(/\][[:space:]]*$/, "", name)
      in_section = (name == wanted)
      if (in_section) { found = 1 }
      next
    }
    in_section { print }
    END { exit found ? 0 : 0 }
  ' "$RENDER_OVERRIDES" >>"$target"
}

apply_render_profile() {
  local name="$1"

  [[ -f "$RENDER_PROFILE" ]] || { echo "Missing render profile: $RENDER_PROFILE" >&2; exit 1; }
  [[ -f "$RENDER_EDGE_PROFILE" ]] || { echo "Missing render edge profile: $RENDER_EDGE_PROFILE" >&2; exit 1; }

  cp "$RENDER_PROFILE" "$ACTIVE_PROFILE"
  append_override_section "$name" "$ACTIVE_PROFILE"

  cp "$RENDER_EDGE_PROFILE" "$EDGE_ACTIVE_PROFILE"
  append_override_section "$name:edge" "$EDGE_ACTIVE_PROFILE"

  echo "Applied render profile: $name"
  echo "  base=$(basename "$RENDER_PROFILE")"
  echo "  edge=$(basename "$RENDER_EDGE_PROFILE")"
  echo "  overrides=$(basename "$RENDER_OVERRIDES") section=$name"
}

cmd="${1:-show}"
restart="${2:-}"

case "$cmd" in
  show)
    show_profile
    ;;
    b300-avatar-commander|b300-hunyuan-allinone|b200-avatar-commander|rtxpro6000-media)
    apply_render_profile "$cmd"
    if [[ "$restart" == "--restart" ]]; then
      exec "$ROOT_DIR/scripts/control.sh" restart-hard
    fi
    ;;
  *)
    usage
    exit 1
    ;;
esac
