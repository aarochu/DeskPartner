"""CLI: run the desk cleaner closed loop."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="DeskPartner closed-loop cleaner")
    parser.add_argument("--config", default="config/workspace.yaml")
    parser.add_argument("--arm-config", default="config/arm.yaml")
    parser.add_argument("--fixtures", action="store_true", help="Use fake plan (no camera/VLM)")
    parser.add_argument("--max-cycles", type=int, default=20)
    parser.add_argument("--perception", choices=["vlm", "cv"], default=None)
    args = parser.parse_args()

    # Load .env if present
    env_path = Path("config/.env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

    if args.perception:
        os.environ["DESKPARTNER_PERCEPTION"] = args.perception

    from p1_arm_motion.arm_client import ArmConfig, DryRunArmClient
    from p3_vlm_orchestrator.orchestrator import Orchestrator

    arm_cfg = ArmConfig.from_yaml(args.arm_config)
    arm = DryRunArmClient(arm_cfg)
    orch = Orchestrator(
        arm=arm,
        arm_cfg=arm_cfg,
        workspace_path=args.config,
        use_fixtures=args.fixtures,
    )
    orch.run(max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()
