#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
state_file="$repo_root/.dev/.env"
default_log_file="$repo_root/.dev/plap.log.jsonl"

seq_image="${PLAP_SEQ_IMAGE:-datalust/seq:2025.2}"
seq_port="${PLAP_SEQ_PORT:-5341}"
seq_ui_ip="${PLAP_SEQ_UI_IP:-127.0.0.1}"
seq_port_explicit="${PLAP_SEQ_PORT:+1}"
seq_startup_timeout_seconds="${PLAP_SEQ_STARTUP_TIMEOUT_SECONDS:-30}"
seq_batch_size="${PLAP_SEQ_BATCH_SIZE:-100}"
seq_poll_interval_seconds="${PLAP_SEQ_POLL_INTERVAL_SECONDS:-0.25}"
seq_raw_event_limit_bytes="${PLAP_SEQ_RAW_EVENT_LIMIT_BYTES:-134217728}"
seq_raw_payload_limit_bytes="${PLAP_SEQ_RAW_PAYLOAD_LIMIT_BYTES:-$((seq_raw_event_limit_bytes * 2))}"
seq_container_name=""
seq_data_volume=""
cleanup_done=0

usage() {
  cat <<'EOF'
Usage: scripts/log-explorer.sh

Run an ephemeral Seq instance against plap's local JSON log file.

Behavior:
  - starts a temporary local Seq Docker container
  - imports the full current log file on launch
  - follows appended log lines live until you stop it
  - removes the Seq container and its data when you exit

Environment overrides:
  PLAP_SEQ_IMAGE                    Default: datalust/seq:2025.2
  PLAP_SEQ_PORT                     Default: 5341
  PLAP_SEQ_UI_IP                    Default: 127.0.0.1
  PLAP_SEQ_STARTUP_TIMEOUT_SECONDS  Default: 30
  PLAP_SEQ_BATCH_SIZE               Default: 100
  PLAP_SEQ_POLL_INTERVAL_SECONDS    Default: 0.25
  PLAP_SEQ_RAW_EVENT_LIMIT_BYTES    Default: 134217728 (128 MiB)
  PLAP_SEQ_RAW_PAYLOAD_LIMIT_BYTES  Default: 268435456 (2x event limit)

If 5341 is already in use and you did not explicitly set PLAP_SEQ_PORT,
the script will auto-pick a free local port.

Log file resolution:
  1. Explicit PLAP_LOG_FILE
  2. PLAP_DEV_LOG_FILE from .dev/.env
  3. .dev/plap.log.jsonl fallback
EOF
}

load_state() {
  local explicit_log_file="${PLAP_LOG_FILE-}"
  local has_explicit_log_file="${PLAP_LOG_FILE+1}"
  if [[ -f "$state_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$state_file"
    set +a
  fi
  if [[ -n "$has_explicit_log_file" ]]; then
    export PLAP_LOG_FILE="$explicit_log_file"
  fi
}

resolve_log_file() {
  if [[ -n "${PLAP_LOG_FILE:-}" ]]; then
    printf '%s\n' "$PLAP_LOG_FILE"
    return
  fi
  if [[ -n "${PLAP_DEV_LOG_FILE:-}" ]]; then
    printf '%s\n' "$PLAP_DEV_LOG_FILE"
    return
  fi
  printf '%s\n' "$default_log_file"
}

port_in_use() {
  local port="$1"
  python3 - "$seq_ui_ip" "$port" <<'PY'
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
  python3 - "$seq_ui_ip" <<'PY'
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

  if [[ -n "$seq_port_explicit" ]]; then
    if port_in_use "$preferred_port"; then
      printf 'Requested Seq port %s is already in use on %s.\n' "$preferred_port" "$seq_ui_ip" >&2
      exit 1
    fi
    printf '%s\n' "$preferred_port"
    return
  fi

  if port_in_use "$preferred_port"; then
    local picked_port
    picked_port="$(pick_free_port)"
    printf 'Auto-picked Seq port %s because %s was already in use on %s.\n' "$picked_port" "$preferred_port" "$seq_ui_ip" >&2
    printf '%s\n' "$picked_port"
    return
  fi

  printf '%s\n' "$preferred_port"
}

ensure_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    printf 'docker is required for the Seq log explorer.\n' >&2
    exit 1
  fi
}

ensure_seq_image() {
  if docker image inspect "$seq_image" >/dev/null 2>&1; then
    return
  fi
  printf 'Pulling Seq image: %s\n' "$seq_image"
  docker pull "$seq_image" >/dev/null
}

create_seq_data_volume() {
  printf 'plap-log-explorer-seq-data-%s-%s\n' "$$" "$RANDOM"
}

start_seq_container() {
  local port="$1"
  seq_container_name="plap-log-explorer-seq-$$-$RANDOM"
  seq_data_volume="$(create_seq_data_volume)"
  docker run \
    --detach \
    --rm \
    --name "$seq_container_name" \
    --env ACCEPT_EULA=Y \
    --env SEQ_FIRSTRUN_NOAUTHENTICATION=true \
    --volume "$seq_data_volume:/data" \
    --publish "$port:80" \
    "$seq_image" >/dev/null
}

wait_for_seq() {
  local port="$1"
  local timeout_seconds="$2"
  python3 - "$seq_ui_ip" "$port" "$timeout_seconds" <<'PY'
from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request

host = sys.argv[1]
port = int(sys.argv[2])
deadline = time.monotonic() + float(sys.argv[3])
url = f"http://{host}:{port}/health"

while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            if 200 <= response.status < 300:
                raise SystemExit(0)
    except Exception:
        time.sleep(0.25)

raise SystemExit(1)
PY
}

configure_seq_limits() {
  local seq_url="$1"
  local event_limit_bytes="$2"
  local payload_limit_bytes="$3"
  python3 - "$seq_url" "$event_limit_bytes" "$payload_limit_bytes" <<'PY'
from __future__ import annotations

import json
import sys
import urllib.request

seq_url = sys.argv[1].rstrip("/")

try:
    event_limit = int(sys.argv[2])
    payload_limit = int(sys.argv[3])
except ValueError as exc:
    raise SystemExit(f"Seq limits must be integers: {exc}") from exc

if event_limit <= 0:
    raise SystemExit("PLAP_SEQ_RAW_EVENT_LIMIT_BYTES must be positive")
if payload_limit <= 0:
    raise SystemExit("PLAP_SEQ_RAW_PAYLOAD_LIMIT_BYTES must be positive")
if payload_limit < event_limit:
    raise SystemExit("PLAP_SEQ_RAW_PAYLOAD_LIMIT_BYTES must be >= PLAP_SEQ_RAW_EVENT_LIMIT_BYTES")


def update(setting_id: str, name: str, value: int) -> None:
    request = urllib.request.Request(
        f"{seq_url}/api/settings/{setting_id}",
        data=json.dumps({"Name": name, "Value": value, "Id": setting_id}).encode(),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with urllib.request.urlopen(request, timeout=10.0) as response:
        payload = json.loads(response.read().decode())
    if payload.get("Value") != value:
        raise SystemExit(
            f"Seq setting {setting_id} did not persist expected value {value}: {payload!r}"
        )


update(
    "setting-raweventmaximumcontentlength",
    "raweventmaximumcontentlength",
    event_limit,
)
update(
    "setting-rawpayloadmaximumcontentlength",
    "rawpayloadmaximumcontentlength",
    payload_limit,
)
PY
}

print_seq_logs_on_failure() {
  if [[ -z "$seq_container_name" ]]; then
    return
  fi
  if docker container inspect "$seq_container_name" >/dev/null 2>&1; then
    printf '\nSeq container logs:\n' >&2
    docker logs "$seq_container_name" >&2 || true
  fi
}

cleanup() {
  local exit_code="${1:-0}"
  if [[ "$cleanup_done" -eq 1 ]]; then
    return
  fi
  cleanup_done=1
  trap - EXIT INT TERM
  set +e
  if [[ -n "$seq_container_name" ]] && docker container inspect "$seq_container_name" >/dev/null 2>&1; then
    docker rm -f "$seq_container_name" >/dev/null 2>&1 || true
  fi
  if [[ -n "$seq_data_volume" ]] && docker volume inspect "$seq_data_volume" >/dev/null 2>&1; then
    docker volume rm -f "$seq_data_volume" >/dev/null 2>&1 || true
  fi
  return
}

handle_interrupt() {
  cleanup 130
  exit 130
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

  trap 'exit_code=$?; cleanup "$exit_code"; exit "$exit_code"' EXIT
  trap handle_interrupt INT TERM

  load_state
  ensure_docker
  ensure_seq_image

  local log_file resolved_port seq_url
  log_file="$(resolve_log_file)"
  mkdir -p "$(dirname -- "$log_file")"
  touch "$log_file"

  resolved_port="$(resolve_port "$seq_port")"
  start_seq_container "$resolved_port"

  if ! wait_for_seq "$resolved_port" "$seq_startup_timeout_seconds"; then
    printf 'Timed out waiting for Seq to start on http://%s:%s.\n' "$seq_ui_ip" "$resolved_port" >&2
    print_seq_logs_on_failure
    exit 1
  fi

  seq_url="http://$seq_ui_ip:$resolved_port"
  configure_seq_limits "$seq_url" "$seq_raw_event_limit_bytes" "$seq_raw_payload_limit_bytes"

  printf 'Using Seq image: %s\n' "$seq_image"
  printf 'Reading log file: %s\n' "$log_file"
  printf 'Web UI: %s\n' "$seq_url"
  printf 'Seq data volume: %s\n' "$seq_data_volume"
  printf 'Seq raw event limit: %s bytes\n' "$seq_raw_event_limit_bytes"
  printf 'Seq raw payload limit: %s bytes\n' "$seq_raw_payload_limit_bytes"
  printf 'Behavior: full import on launch, live tail while running, full cleanup on exit.\n'
  printf 'Stop with Ctrl-C.\n\n'

  python3 "$script_dir/seq_log_forwarder.py" \
    --batch-size "$seq_batch_size" \
    --log-file "$log_file" \
    --poll-interval-seconds "$seq_poll_interval_seconds" \
    --seq-url "$seq_url"
}

main "$@"
