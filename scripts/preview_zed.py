"""Live ZED Mini preview (left eye + optional side-by-side). Press q to quit, s to save."""

from __future__ import annotations

import sys

import cv2
import numpy as np
import pyzed.sl as sl


def main() -> int:
    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.NONE  # RGB only — faster for a quick look

    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED: {err}")
        print("Check: USB3 port, camera powered/plugged, close ZED Explorer if it's holding the device.")
        return 1

    info = cam.get_camera_information()
    print(
        f"Opened {info.camera_model} serial={info.serial_number} "
        f"fw={info.camera_configuration.firmware_version}"
    )

    runtime = sl.RuntimeParameters()
    left = sl.Mat()
    win = "ZED left (q=quit, s=save, b=side-by-side toggle)"
    sbs = False
    print(win)

    while True:
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            print("grab failed")
            break

        view = sl.VIEW.SIDE_BY_SIDE if sbs else sl.VIEW.LEFT
        cam.retrieve_image(left, view)
        frame = left.get_data()
        # ZED returns BGRA
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            f"ZED {info.camera_model}  {w}x{h}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            sbs = not sbs
        if key == ord("s"):
            out = "zed_preview.jpg"
            cv2.imwrite(out, frame)
            print(f"Saved {out}")

    cam.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
