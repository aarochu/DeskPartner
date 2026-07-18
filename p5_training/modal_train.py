"""Modal MolmoAct 2 fine-tune stub (LoRA / action-expert-only).

Friday night: confirm with organizers —
  (a) newt platform path, or
  (b) Ai2 MolmoAct 2 / LeRobot scripts on our Modal GPUs.

Saturday PM: kick off after P4 hands off a verified single-arm dataset.
"""

from __future__ import annotations

NOTES = """
Expected shape (adjust to newt OR Ai2 entrypoint):

  modal run p5_training/modal_train.py \\
    --dataset-repo-id deskpartner/crumpled_paper_molmoact2 \\
    --config p5_training/configs/molmoact2_single_arm.yaml

Inside the Modal image:
  - install MolmoAct 2 + LeRobot policy deps (no robot serial ports needed)
  - mount / download P4 dataset
  - run LoRA or action-expert-only fine-tune from organizer-approved checkpoint
  - write checkpoints to a Modal Volume
  - expose inference endpoint that returns action chunks

Pitch line if B wins:
  Fine-tuned a foundation VLA on data collected here, on hardware nobody else ran.

Cut rule: one bad Sat-night run → one retry → then cut B.
"""


def main() -> None:
    print(NOTES)


if __name__ == "__main__":
    main()
