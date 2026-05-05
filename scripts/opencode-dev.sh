#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
state_file="$repo_root/.dev/dev.env"

if [[ ! -f "$state_file" ]]; then
  printf 'Missing %s. Start the dev server first with scripts/dev.sh or pixi run dev.\n' "$state_file" >&2
  exit 1
fi

set -a
source "$state_file"
set +a

: "${PLAP_DEV_BASE_URL:?PLAP_DEV_BASE_URL is missing from .dev/dev.env}"
: "${PLAP_DEV_API_KEY:?PLAP_DEV_API_KEY is missing from .dev/dev.env}"

export OPENCODE_DISABLE_GLOBAL_CONFIG=1
export OPENCODE_CONFIG="$repo_root/scripts/opencode.json"
export OPENCODE_CONFIG_CONTENT="$(PLAP_DEV_BASE_URL="$PLAP_DEV_BASE_URL" PLAP_DEV_API_KEY="$PLAP_DEV_API_KEY" python3 - <<'PY'
import json
import os

print(
    json.dumps(
        {
            "provider": {
                "plap": {
                    "options": {
                        "baseURL": os.environ["PLAP_DEV_BASE_URL"],
                        "apiKey": os.environ["PLAP_DEV_API_KEY"],
                    }
                }
            }
        }
    )
)
PY
)"

cd "$repo_root"
exec opencode "$@"
