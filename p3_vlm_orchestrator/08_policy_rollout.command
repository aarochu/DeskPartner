#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_ROOT/rebot_setup/vendor/rebot_lerobot}"

if [[ -x "$RUNTIME_ROOT/reBotArm_control_py/.venv/bin/python" ]]; then
  PYTHON_BIN="$RUNTIME_ROOT/reBotArm_control_py/.venv/bin/python"
elif [[ -x "$RUNTIME_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$RUNTIME_ROOT/.venv/bin/python"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

typeset -a rollout_python_paths
rollout_python_paths=(
  "$REPO_ROOT"
  "$RUNTIME_ROOT/lerobot/src"
  "$RUNTIME_ROOT/lerobot-robot-seeed-b601"
)
export RUNTIME_ROOT
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$RUNTIME_ROOT/lerobot-home}"
export PYTHONPATH="${(j/:/)rollout_python_paths}${PYTHONPATH:+:$PYTHONPATH}"

print -- "Policy rollout safety reminder"
print -- "- Software q/x/Escape and signals are not a physical e-stop."
print -- "- Keep a dedicated operator on the physical e-stop/power cut for live mode."
print -- "- Stage offline, shadow, one-cycle live, then bounded multi-cycle live."
print -- "- Stopping/disconnecting releases follower torque; support the arm."
print -- ""

exec "$PYTHON_BIN" -m p3_vlm_orchestrator.policy_rollout.cli "$@"
