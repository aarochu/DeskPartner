#!/usr/bin/env bash
# Single-episode (or N) lerobot-record from config/recording.yaml.
# Usage:
#   ./p4_data_collection/scripts/record_episode.sh
#   ./p4_data_collection/scripts/record_episode.sh --task "Pick crumpled paper and drop in trash" --num 1
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

exec python -m p4_data_collection.record_episode "$@"
