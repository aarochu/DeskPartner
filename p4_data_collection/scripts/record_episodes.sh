#!/usr/bin/env bash
# Back-compat name → new record_episode entrypoint
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec python -m p4_data_collection.record_episode "$@"
