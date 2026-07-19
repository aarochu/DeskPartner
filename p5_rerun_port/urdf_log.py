"""Best-effort URDF logging into Rerun."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from p5_rerun_port.config import resolve_urdf_path
from p5_rerun_port.constants import FOLLOWER


def log_urdf(rec: Any, urdf_path: Path | None = None, entity: str = f"{FOLLOWER}/urdf") -> Path | None:
    """Log the reBot URDF meshes/transforms into the recording when available."""
    path = resolve_urdf_path(urdf_path)
    if path is None:
        print("WARN: reBot URDF not found - set path or clone reBotArm_control_py", flush=True)
        return None
    try:
        import rerun as rr

        from rerun_loader_urdf import URDFLogger

        logger = URDFLogger(str(path), entity_path_prefix=entity)
        # The vendored SolidWorks export keeps ``meshes/`` beside ``urdf/``
        # while its XML uses package-root-relative paths.
        package_root = path.parent.parent
        if (package_root / "meshes").is_dir():
            logger.root_filepath = package_root
        logger.log(recording=rec)
        rec.log(f"{entity}/source", rr.TextDocument(f"URDF path: {path}"), static=True)
        print(f"urdf: logged meshes and transforms from {path} under {entity}", flush=True)
        return path
    except Exception as exc:
        try:
            import rerun as rr

            rec.log(entity, rr.TextDocument(f"URDF path: {path}"), static=True)
        except Exception:
            pass
        print(
            f"WARN: URDF mesh log failed ({exc}); recorded the source path and continued with joint scalars",
            flush=True,
        )
        return path
