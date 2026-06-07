#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/harbor-downloader.sh DATASET_REF OUTPUT_DIR [--overwrite]

Download a Harbor dataset into OUTPUT_DIR using Harbor export mode.

Examples:
  ./scripts/harbor-downloader.sh "swe-bench/swe-bench-verified" datasets
  ./scripts/harbor-downloader.sh "scale-ai/swe-bench-pro" datasets --overwrite

After download, inspect OUTPUT_DIR and run:
  ./scripts/pier.sh -p <downloaded-dataset-dir>
EOF
}

die() {
  printf '%s\n' "$*" >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

dataset_ref="${1:-}"
output_dir="${2:-}"

if [[ -z "$dataset_ref" || -z "$output_dir" ]]; then
  usage >&2
  exit 1
fi

shift 2

overwrite_args=()
while (($#)); do
  case "$1" in
    --overwrite)
      overwrite_args+=("--overwrite")
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
  shift
done

printf 'Downloading Harbor dataset %s into %s\n' "$dataset_ref" "$output_dir" >&2
uvx --from harbor harbor datasets download "$dataset_ref" -o "$output_dir" --export "${overwrite_args[@]}"
printf 'Download finished. Inspect %s and run ./scripts/pier.sh -p <downloaded-dataset-dir>\n' "$output_dir" >&2
