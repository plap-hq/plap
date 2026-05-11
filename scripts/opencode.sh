#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
state_file="$repo_root/.dev/dev.env"
provider_package_dir="$repo_root/node/ai-sdk"
provider_module="file://$provider_package_dir/index.mjs"

if [[ ! -f "$state_file" ]]; then
  printf 'Missing %s. Start the dev server first with scripts/dev.sh or pixi run dev.\n' "$state_file" >&2
  exit 1
fi

set -a
source "$state_file"
set +a

: "${PLAP_DEV_BASE_URL:?PLAP_DEV_BASE_URL is missing from .dev/dev.env}"
: "${PLAP_DEV_API_KEY:?PLAP_DEV_API_KEY is missing from .dev/dev.env}"

ensure_provider_deps() {
  if [[ -d "$provider_package_dir/node_modules/@ai-sdk/openai" ]]; then
    return
  fi
  if ! command -v npm >/dev/null 2>&1; then
    printf 'npm is required to install local deps for %s\n' "$provider_package_dir" >&2
    exit 1
  fi

  printf 'Installing local OpenCode provider deps into %s...\n' "$provider_package_dir" >&2
  NPM_CONFIG_UPDATE_NOTIFIER=false npm install --prefix "$provider_package_dir" --no-package-lock --no-fund --no-audit --silent
}

ensure_provider_deps

export OPENCODE_DISABLE_GLOBAL_CONFIG=1
export OPENCODE_CONFIG="$repo_root/scripts/opencode.json"
# Temporary workaround: archive/opencode globally clamps max output tokens to 32k
# unless this flag is set. The correct fix is external to opencode source: load a
# local server plugin with a `chat.params` hook that sets `maxOutputTokens = undefined`
# for the PLAP/opencode provider so PLAP can own the request budget.
export OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=1000000
export OPENCODE_CONFIG_CONTENT="$(PLAP_DEV_BASE_URL="$PLAP_DEV_BASE_URL" PLAP_DEV_API_KEY="$PLAP_DEV_API_KEY" PLAP_PROVIDER_MODULE="$provider_module" python3 - <<'PY'
import json
import os

print(
    json.dumps(
        {
            "provider": {
                "plap": {
                    "npm": os.environ["PLAP_PROVIDER_MODULE"],
                    "options": {
                        "baseURL": os.environ["PLAP_DEV_BASE_URL"],
                        "apiKey": os.environ["PLAP_DEV_API_KEY"],
                    },
                }
            }
        }
    )
)
PY
)"

exec opencode "$@"
