# Person 4 policy rollout runbook

This runbook covers Stage 1 only: `Pick up one can and place it in the taped sorting zone`.
It does not authorize multi-can clearing or unattended operation.

## Locked policy contract

- Actions and observations use seven physical follower-space joint positions, in
  this exact order: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll, gripper`.
- The image order is `front`, then `side`: `observation.images.front` is the
  fixed Logitech overhead camera and `observation.images.side` is the Innomaker
  wrist/claw camera.
- The adapter restores the checkpoint's saved LeRobot preprocessor and
  postprocessor. Do not recreate normalization by hand.
- Each control cycle uses only the first predicted action, then obtains a fresh
  observation and predicts again.

Run every command from the DeskPartner repository root. Set the checkpoint and
dataset to absolute paths so the handoff is auditable:

```zsh
CHECKPOINT=/absolute/path/to/checkpoint
DATASET=/absolute/path/to/finalized/lerobot-dataset
```

## Checkpoint handoff and inspection

Person 3 must provide the complete immutable checkpoint directory, including
`model.safetensors`, `config.json`, `preprocessor_config.json`,
`postprocessor_config.json`, and `rebot_training_profile.json`. Record the
absolute checkpoint path and a SHA-256 identity for its weights in each trial
manifest:

```zsh
shasum -a 256 "$CHECKPOINT/model.safetensors"
./p3_vlm_orchestrator/08_policy_rollout.command inspect --checkpoint "$CHECKPOINT"
```

Inspection must show the exact task, seven-joint order, `front,side` image
order, profile digest, and both saved processor configs. `inspect` validates
metadata and does not load policy weights or touch hardware.

## Four mandatory gates

Do not skip a gate. A fault or unexpected clamp returns the checkpoint to the
previous gate.

### Gate A: offline recorded frames

```zsh
./p3_vlm_orchestrator/08_policy_rollout.command offline \
  --checkpoint "$CHECKPOINT" --dataset "$DATASET" --episodes 2 --device cpu
```

Confirm finite `(10, 7)` action chunks on two distinct finalized episodes and
review the printed first-action deltas. This gate opens no serial port.

### Gate B: shadow mode

With the follower supported, the workspace clear, and a dedicated operator at
the physical e-stop/power cut, run prediction without sending actions:

```zsh
./p3_vlm_orchestrator/08_policy_rollout.command shadow \
  --checkpoint "$CHECKPOINT" --cycles 20 --speed-scale 0.10
```

Shadow mode still runs the calibrated workspace guard. It must send no action.

### Gate C: one live cycle in an empty workspace

Only after Gate B has no faults or clamps, use 10% speed:

```zsh
./p3_vlm_orchestrator/08_policy_rollout.command live \
  --checkpoint "$CHECKPOINT" --cycles 1 --speed-scale 0.10 --live
```

Type both exact live acknowledgements when prompted. The independent e-stop
operator watches the arm, not the terminal.

### Gate D: five live cycles in an empty workspace

Only after a clean Gate C:

```zsh
./p3_vlm_orchestrator/08_policy_rollout.command live \
  --checkpoint "$CHECKPOINT" --cycles 5 --speed-scale 0.10 --live
```

The first physical runs stay in an empty workspace at 10-20% speed. Do not put
a can in the workspace until all four gates have passed.

## Held-out placement evaluation

Reserve one fixed set of 10-15 distinct can placements that was not used for
training. Every checkpoint comparison must use the same placement IDs. A trial
is one placement after its final allowed outcome, not each attempt.

For each placement, keep the physical e-stop operator present and run:

```zsh
LOG_PATH="$PWD/runs/policy/checkpoint-a-held-out-01.jsonl"
./p3_vlm_orchestrator/08_policy_rollout.command live \
  --checkpoint "$CHECKPOINT" --speed-scale 0.10 --live \
  --episode --retry-on-failure --log-path "$LOG_PATH"
```

The keyboard controls are `s` for operator success, `f` for operator failure,
and `q`, `x`, or Escape to stop. An operator failure may be retried once only
after the program has disconnected, a person has manually reset the can and
cleared the workspace, and the exact reset acknowledgement is entered. There
is no automatic reset or motion. Safety faults, timeout, and stop are never
retried.

Create one read-only JSON manifest per checkpoint with exactly this envelope
and trial schema. Include 10-15 trial objects; the abbreviated example shows
one:

```json
{
  "schema_version": 1,
  "checkpoint": "/absolute/path/to/checkpoint",
  "checkpoint_digest": "64-lowercase-hex-weight-digest",
  "trials": [
    {
      "checkpoint": "/absolute/path/to/checkpoint",
      "checkpoint_digest": "64-lowercase-hex-weight-digest",
      "placement_id": "held-out-01",
      "attempts_used": 1,
      "grasp_success": true,
      "placement_success": true,
      "terminal_reason": "operator_success",
      "safety_faults": 0,
      "clamps": 0,
      "completion_s": 12.5,
      "source_jsonl_paths": [
        "/absolute/path/to/runs/policy/checkpoint-a-held-out-01.jsonl"
      ]
    }
  ]
}
```

Allowed terminal reasons are `operator_success`, `operator_failure`, `stopped`,
`timeout`, and `safety_fault`. Human reviewers label grasp and placement from
both camera views. `completion_s` is the sum of policy-running elapsed seconds
across the final trial's one or two attempts; exclude manual-reset downtime.
This same definition is used for every checkpoint. Sum clamp counts across both
attempts. `completion_s` must be a JSON number, not a quoted string.
`source_jsonl_paths` must identify absolute, existing, regular, non-symlink JSONL files.
The current retry loop appends both attempts to one file: List the shared JSONL path once, never duplicate it.
Reporting reads the files without modifying them and requires exactly the terminal or
`terminal_fallback` rows for attempts `1..attempts_used`. Their final reason,
total clamps, safety-fault count, and summed elapsed seconds must match the
manifest. A safety fault is a failed trial and is never retried.

Generate the per-checkpoint report:

```zsh
./p3_vlm_orchestrator/08_policy_rollout.command report \
  --manifest /absolute/path/to/checkpoint-a-trials.json \
  --output-name checkpoint-a-held-out
```

Compare two or more checkpoints:

```zsh
./p3_vlm_orchestrator/08_policy_rollout.command compare \
  --manifest /absolute/path/to/checkpoint-a-trials.json \
             /absolute/path/to/checkpoint-b-trials.json \
  --output-name stage1-checkpoint-comparison
```

JSON and CSV are written under `runs/policy/reports/`; rollout audit JSONL is
written under `runs/policy/`. Existing reports are never overwritten.
Comparison ranks higher placement success first, then fewer safety faults,
then fewer clamps, then lower unrounded mean completion time. Reports round the
displayed mean to six decimal places only after ranking. Exact remaining ties
use checkpoint identity only for deterministic output. Training loss is never
a selection criterion.

## Current fail-closed limitation

The tracked `data/calibration/calibration.json` has null `plane_to_arm.A` and
`plane_to_arm.b` values, and the workspace corners in `config/workspace.yaml`
are also null. Shadow and live rollout therefore fail closed until a reviewed
physical workspace calibration for this rig is supplied. Never invent values,
bypass the guard, or weaken the check to make a run start.

The adapter is generic across policies registered in the active LeRobot
runtime and always restores their saved processors. The bundled LeRobot 0.4.4
tree can describe SmolVLA, but this machine does not have its optional
Transformers dependencies or the newer MolmoAct2 plugin. A real MolmoAct2
checkpoint requires the official current LeRobot Python 3.12 runtime with its
MolmoAct2 dependencies, normally on the GPU host. Generic loading does not mean
missing local policy dependencies are supported.

## Stop, rollback, and shutdown

Software stop is secondary to the physical e-stop. If motion is unsafe, the
dedicated operator immediately uses the physical e-stop or power cut; the
terminal operator also presses `q`, `x`, or Escape. Support the follower before
disconnect because the driver releases torque. After any stop or fault there
is no automatic home: do not command a return pose, restart, or continue with a
can present. Preserve the JSONL, inspect the terminal and fault rows, correct
the cause, clear the workspace, and restart at the last previously passed gate.

No automated test performs physical motion. Automated verification uses pure
reporting data, fake adapters, and help/import checks only.
