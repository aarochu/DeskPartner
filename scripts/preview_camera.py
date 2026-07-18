"""Live camera preview. Press q to quit, n for next camera index."""

from __future__ import annotations

import argparse
import sys

import cv2


def open_cap(index: int) -> cv2.VideoCapture:
    # CAP_DSHOW is more reliable on Windows USB cams
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview a USB camera")
    parser.add_argument("--index", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    index = args.index
    cap = open_cap(index)
    if not cap.isOpened():
        print(f"Could not open camera index {index}. Try --index 1 or 2.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    win = "DeskPartner camera (q=quit, n=next index, s=save frame)"
    print(f"Showing camera index {index}. Keys: q quit | n next camera | s save frame")

    while True:
        ok, frame = cap.read()
        if not ok:
            print(f"Failed to read from index {index}")
            break

        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            f"index={index}  {w}x{h}",
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
        if key == ord("s"):
            out = f"camera_preview_{index}.jpg"
            cv2.imwrite(out, frame)
            print(f"Saved {out}")
        if key == ord("n"):
            cap.release()
            index += 1
            print(f"Trying camera index {index}...")
            cap = open_cap(index)
            if not cap.isOpened():
                print(f"No camera at index {index}; wrapping to 0")
                index = 0
                cap = open_cap(index)
                if not cap.isOpened():
                    print("No cameras found.")
                    break
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
