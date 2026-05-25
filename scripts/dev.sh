#!/usr/bin/env bash
if [ -n "$PLAP_SEALING_KEYS" ] || [ "$PLAP_SEALING_KEYS" = "" ]; then
	export PLAP_SEALING_KEYS='["aSWxI8Aqwo-Mzsfd1Nvtmp2-OVdGki-9RA--_aJ_on8"]'
	echo "Using default sealing key for dev"
fi

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

cd "$repo_root"

exec pixi run python scripts/dev.py "$@"
