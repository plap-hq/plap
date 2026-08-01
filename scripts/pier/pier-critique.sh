#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
state_file="$repo_root/.dev/.env"
default_agent="mini-swe-agent"
default_model="openai/plap-ai/wisp"
default_prompt="$script_dir/critique.md"
custom_environment_import_path="scripts.pier:PlapPierDockerEnvironment"
custom_pi_agent_import_path="scripts.pier.pier_pi:PiAgent"

usage() {
  cat <<'EOF'
Usage: scripts/pier/pier-critique.sh JOB_DIR [PIER_CRITIQUE_ARGS...]

Run `pier critique run` against a local job using local plap as the critique model.

Defaults applied by this wrapper:
  --agent mini-swe-agent
  --model openai/plap-ai/wisp
  --prompt scripts/pier/critique.md
  --environment-import-path scripts.pier:PlapPierDockerEnvironment

Examples:
  ./scripts/pier/pier-critique.sh jobs/2026-06-05__12-59-06
  ./scripts/pier/pier-critique.sh jobs/2026-06-05__12-59-06 --trial-name some_trial
  ./scripts/pier/pier-critique.sh jobs/2026-06-05__12-59-06 --agent opencode --model openai/plap-ai/wisp
EOF
}

die() {
  printf '%s\n' "$*" >&2
  exit 1
}

url_port() {
  local url="$1"
  local scheme remainder authority
  scheme="${url%%://*}"
  remainder="${url#*://}"
  authority="${remainder%%/*}"

  if [[ "$authority" == \[*\]:* ]]; then
    printf '%s\n' "${authority##*:}"
    return
  fi

  if [[ "$authority" == *:* ]]; then
    printf '%s\n' "${authority##*:}"
    return
  fi

  if [[ "$scheme" == "https" ]]; then
    printf '443\n'
    return
  fi
  printf '80\n'
}

has_option() {
  local long_flag="$1"
  local short_flag="$2"
  shift 2
  local arg
  for arg in "$@"; do
    case "$arg" in
      "$long_flag"|"$short_flag"|"$long_flag="*|"$short_flag="*)
        return 0
        ;;
    esac
  done
  return 1
}

has_agent_selection() {
  has_option --agent -a "$@" || has_option --agent-import-path '' "$@"
}

has_environment_kwarg() {
  local key="$1"
  shift
  local previous=""
  local arg
  for arg in "$@"; do
    case "$arg" in
      --environment-kwarg="$key"=*|--ek="$key"=*)
        return 0
        ;;
    esac
    if [[ "$previous" == "--environment-kwarg" || "$previous" == "--ek" ]]; then
      if [[ "$arg" == "$key"=* ]]; then
        return 0
      fi
    fi
    previous="$arg"
  done
  return 1
}

normalize_agent_args() {
  normalized_args=()
  while (($#)); do
    case "$1" in
      --agent=pi)
        normalized_args+=(--agent-import-path "$custom_pi_agent_import_path")
        ;;
      --agent|-a)
        if (($# >= 2)) && [[ "$2" == "pi" ]]; then
          normalized_args+=(--agent-import-path "$custom_pi_agent_import_path")
          shift
        else
          normalized_args+=("$1")
        fi
        ;;
      *)
        normalized_args+=("$1")
        ;;
    esac
    shift
  done
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -eq 0 ]]; then
  usage
  exit 0
fi

if [[ ! -f "$state_file" ]]; then
  die "Missing $state_file. Start the dev server first with 'pixi run dev --host 0.0.0.0'."
fi

set -a
# shellcheck disable=SC1090
source "$state_file"
set +a

: "${PLAP_DEV_BASE_URL:?PLAP_DEV_BASE_URL is missing from .dev/.env}"
: "${PLAP_DEV_API_KEY:?PLAP_DEV_API_KEY is missing from .dev/.env}"

normalize_agent_args "$@"
set -- "${normalized_args[@]}"

effective_base_url="$PLAP_DEV_BASE_URL"
scheme="${PLAP_DEV_BASE_URL%%://*}"
remainder="${PLAP_DEV_BASE_URL#"$scheme://"}"
host_and_port="${remainder%%/*}"
path=""
if [[ "$remainder" == */* ]]; then
  path="/${remainder#*/}"
fi

host="$host_and_port"
port=""
if [[ "$host_and_port" == *:* ]]; then
  host="${host_and_port%:*}"
  port="${host_and_port##*:}"
fi

if [[ "$host" == "127.0.0.1" || "$host" == "localhost" || "$host" == "::1" ]]; then
  if [[ -n "$port" ]]; then
    effective_base_url="$scheme://host.docker.internal:$port$path"
  else
    effective_base_url="$scheme://host.docker.internal$path"
  fi
fi

effective_base_port="$(url_port "$effective_base_url")"
[[ -n "$effective_base_port" ]] || die "Could not determine the port from $effective_base_url"

printf 'Running Pier critique via %s\n' "$effective_base_url" >&2

cmd=(uvx --from datacurve-pier pier critique run)

if ! has_option --prompt -p "$@"; then
  if [[ ! -f "$default_prompt" ]]; then
    die "Missing critique prompt: $default_prompt"
  fi
  cmd+=(--prompt "$default_prompt")
fi
if ! has_agent_selection "$@"; then
  cmd+=(--agent "$default_agent")
fi
if ! has_option --model -m "$@"; then
  cmd+=(--model "$default_model")
fi
if ! has_option --environment-import-path '' "$@"; then
  cmd+=(--environment-import-path "$custom_environment_import_path")
fi
if ! has_environment_kwarg plap_safe_port "$@"; then
  cmd+=(--environment-kwarg "plap_safe_port=$effective_base_port")
fi

cmd+=(
  --agent-env "OPENAI_API_KEY=$PLAP_DEV_API_KEY"
  --agent-env "OPENAI_API_BASE=$effective_base_url"
  --agent-env "OPENAI_BASE_URL=$effective_base_url"
  "$@"
)

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
exec "${cmd[@]}"
