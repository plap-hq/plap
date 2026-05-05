#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
state_file="$repo_root/.dev/dev.env"
default_log_file="$repo_root/.dev/plap.log.jsonl"

logdy_version="${PLAP_LOGDY_VERSION:-v0.17.1}"
logdy_port="${PLAP_LOGDY_PORT:-8080}"
logdy_ui_ip="${PLAP_LOGDY_UI_IP:-127.0.0.1}"
logdy_port_explicit="${PLAP_LOGDY_PORT:+1}"
logdy_root="$repo_root/.dev/tools/logdy/$logdy_version"
logdy_binary="$logdy_root/logdy"

usage() {
  cat <<'EOF'
Usage: scripts/log-explorer.sh

Run Logdy against plap's local JSON log file.

Behavior:
  - downloads a pinned Logdy release binary into .dev/tools/logdy/
  - never installs globally
  - never edits PATH
  - runs in the foreground; stop it with Ctrl-C

Environment overrides:
  PLAP_LOGDY_VERSION   Default: v0.17.1
  PLAP_LOGDY_PORT      Default: 8080
  PLAP_LOGDY_UI_IP     Default: 127.0.0.1

If 8080 is already in use and you did not explicitly set PLAP_LOGDY_PORT,
the script will auto-pick a free local port.

Log file resolution:
  1. PLAP_LOG_FILE from .dev/dev.env (when available)
  2. .dev/plap.log.jsonl fallback
EOF
}

load_state() {
  if [[ -f "$state_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$state_file"
    set +a
  fi
}

resolve_log_file() {
  if [[ -n "${PLAP_LOG_FILE:-}" ]]; then
    printf '%s\n' "$PLAP_LOG_FILE"
    return
  fi
  printf '%s\n' "$default_log_file"
}

port_in_use() {
  local port="$1"
  python3 - "$logdy_ui_ip" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

pick_free_port() {
  python3 - "$logdy_ui_ip" <<'PY'
import socket
import sys

host = sys.argv[1]
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind((host, 0))
    print(sock.getsockname()[1])
PY
}

resolve_port() {
  local preferred_port="$1"

  if [[ -n "$logdy_port_explicit" ]]; then
    if port_in_use "$preferred_port"; then
      printf 'Requested Logdy port %s is already in use on %s.\n' "$preferred_port" "$logdy_ui_ip" >&2
      exit 1
    fi
    printf '%s\n' "$preferred_port"
    return
  fi

  if port_in_use "$preferred_port"; then
    local picked_port
    picked_port="$(pick_free_port)"
    printf 'Auto-picked Logdy port %s because %s was already in use on %s.\n' "$picked_port" "$preferred_port" "$logdy_ui_ip" >&2
    printf '%s\n' "$picked_port"
    return
  fi

  printf '%s\n' "$preferred_port"
}

detect_asset_name() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"

  case "$os" in
    Linux) os="linux" ;;
    Darwin) os="darwin" ;;
    *)
      printf 'Unsupported OS for Logdy wrapper: %s\n' "$os" >&2
      exit 1
      ;;
  esac

  case "$arch" in
    x86_64|amd64) arch="amd64" ;;
    aarch64|arm64) arch="arm64" ;;
    *)
      printf 'Unsupported architecture for Logdy wrapper: %s\n' "$arch" >&2
      exit 1
      ;;
  esac

  printf 'logdy_%s_%s\n' "$os" "$arch"
}

download_logdy() {
  local asset_name url tmp_file
  asset_name="$(detect_asset_name)"
  url="https://github.com/logdyhq/logdy-core/releases/download/${logdy_version}/${asset_name}"
  tmp_file="$logdy_binary.tmp"

  mkdir -p "$logdy_root"

  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --output "$tmp_file" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$tmp_file" "$url"
  else
    printf 'Need curl or wget to download Logdy locally.\n' >&2
    exit 1
  fi

  mv "$tmp_file" "$logdy_binary"
  chmod +x "$logdy_binary"
}

ensure_logdy_binary() {
  if [[ -x "$logdy_binary" ]]; then
    printf '%s\n' "$logdy_binary"
    return
  fi

  download_logdy
  printf '%s\n' "$logdy_binary"
}

main() {
  if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
  fi

  if [[ $# -gt 0 ]]; then
    printf 'scripts/log-explorer.sh does not take commands; just run it and Ctrl-C to stop.\n\n' >&2
    usage >&2
    exit 1
  fi

  load_state

  local log_file binary resolved_port
  log_file="$(resolve_log_file)"
  mkdir -p "$(dirname -- "$log_file")"
  touch "$log_file"

  binary="$(ensure_logdy_binary)"
  resolved_port="$(resolve_port "$logdy_port")"

  printf 'Using Logdy binary: %s\n' "$binary"
  printf 'Reading log file: %s\n' "$log_file"
  printf 'Web UI: http://%s:%s\n' "$logdy_ui_ip" "$resolved_port"
  printf 'Stop with Ctrl-C.\n\n'

  exec "$binary" follow --full-read --no-analytics --no-updates --ui-ip "$logdy_ui_ip" --port "$resolved_port" "$log_file"
}

main "$@"
