"""Best-effort URDF logging into Rerun."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from p5_rerun_port.config import resolve_urdf_path
from p5_rerun_port.constants import FOLLOWER


def log_urdf(rec: Any, urdf_path: Path | None = None, entity: str = f"{FOLLOWER}/urdf") -> Path | None:
    """Log the reBot URDF file into the recording (static). Returns path used or None."""
    path = resolve_urdf_path(urdf_path)
    if path is None:
        print("WARN: reBot URDF not found - set path or clone reBotArm_control_py", flush=True)
        return None
    try:
        import rerun as rr

        # Path annotation only: mesh import often fails when STL siblings are
        # missing/case-mismatched. Joint scalars remain the training signal.
        rec.log(entity, rr.TextDocument(f"URDF path: {path}"), static=True)
        print(f"urdf: referenced {path} under {entity}", flush=True)
        return path
    except Exception as exc:
        print(f"WARN: URDF log failed ({exc}); continuing with joint scalars only", flush=True)
        return path
