"""Purpose-built Rerun Viewer layout for collecting and evaluating reBot takes."""

from __future__ import annotations

from typing import Any

from p5_rerun_port.constants import CAMERA_TO_RERUN, FOLLOWER


def build_hackathon_blueprint() -> Any:
    """Return the central operator layout used by interactive Rerun recordings.

    Imports stay lazy so ``--no-viewer`` recording and non-Rerun utilities do not
    need to initialize the Viewer/blueprint stack.
    """
    import rerun.blueprint as rrb

    cameras = rrb.Vertical(
        rrb.Spatial2DView(
            origin=f"/{CAMERA_TO_RERUN['front']}",
            name="Front camera",
        ),
        rrb.Spatial2DView(
            origin=f"/{CAMERA_TO_RERUN['side']}",
            name="Side camera",
        ),
        row_shares=[1, 1],
        name="Synchronized cameras",
    )
    robot_and_metrics = rrb.Vertical(
        rrb.Spatial3DView(
            origin=f"/{FOLLOWER}/urdf",
            name="reBot URDF",
            line_grid=True,
        ),
        rrb.TimeSeriesView(
            origin=f"/{FOLLOWER}",
            contents=[
                f"/{FOLLOWER}/position",
                f"/{FOLLOWER}/goal",
            ],
            name="Goal vs position",
        ),
        row_shares=[3, 2],
        name="Embodiment and tracking",
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            cameras,
            robot_and_metrics,
            column_shares=[2, 3],
            name="Can-sorting operator view",
        ),
        rrb.SelectionPanel(expanded=False),
        rrb.BlueprintPanel(expanded=False),
        rrb.TimePanel(expanded=True, timeline="time"),
        auto_views=False,
        auto_layout=False,
    )
