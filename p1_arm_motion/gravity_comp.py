"""Party trick launcher notes for gravity compensation."""

from __future__ import annotations

NOTES = """
Weightless mode (demo act 2):

  cd ~/reBotArm_control_py
  uv run python example/9_gravity_compensation.py

Control law (from deck): τ = g(q), soft kp/kd — arm feels nearly weightless.
Hand the arm to the judge. Ctrl+C to stop.

Safety: clear the zone of people/objects first; keep e-stop reachable.
"""


def main() -> None:
    print(NOTES)


if __name__ == "__main__":
    main()
