# DeskPartner documentation

All project docs live here. Code stays under `p1_*` … `p5_*`, `rebot_operator_kit/`, and `rebot_setup/`; each package has a short stub pointing back to this tree.

Vendor trees under `rebot_setup/vendor/` keep their upstream READMEs in place (not duplicated here).

## Start here

| Doc | What it is |
|-----|------------|
| [GROUND_TRUTH.md](./GROUND_TRUTH.md) | SOW / scope source of truth |
| [TEAM.md](./TEAM.md) | Seat assignments + arm contention |
| [REBOT_ARM_COMMANDS.md](./REBOT_ARM_COMMANDS.md) | Common arm / teleop commands |
| [Rerun_bounty_progress.md](./Rerun_bounty_progress.md) | Non-SO-101 Rerun bounty progress |
| [p5_rerun_port/](./p5_rerun_port/#how-it-connects-flowchart) | Rerun bounty pipeline + flowchart |
| [p5_rerun_port/QUERY_API.md](./p5_rerun_port/QUERY_API.md) | Rerun Query API refine step |
| [rebot_setup/](./rebot_setup/) | Portable SDK + LeRobot machine setup |
| [../README.md](../README.md) | Operator bring-up (root) |

## Track A

| Area | Docs |
|------|------|
| P1 arm motion | [p1_arm_motion/](./p1_arm_motion/) |
| P2 vision / calibration | [p2_vision_calibration/](./p2_vision_calibration/) |
| P3 VLM orchestrator | [p3_vlm_orchestrator/](./p3_vlm_orchestrator/) |

## Track B

| Area | Docs |
|------|------|
| P4 data collection | [p4_data_collection/](./p4_data_collection/) |
| P5 training | [p5_training/](./p5_training/) |
| P5 Rerun port (bounty) | [p5_rerun_port/](./p5_rerun_port/) |
| Operator kit (macOS GUI) | [rebot_operator_kit/](./rebot_operator_kit/) |

## Secrets

Never commit `config/.env` or real API tokens. Use `config/.env.example` placeholders only.
