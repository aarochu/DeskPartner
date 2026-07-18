"""Preflight: compare live camera props to episode-1 reference. Fail-loud on drift.

  # After cameras locked for episode 1:
  python -m p4_data_collection.check_camera_lock --save-ref

  # Before every later session:
  python -m p4_data_collection.check_camera_lock
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from p4_data_collection.config_loader import load_recording_config


def _open(index: int | str) -> cv2.VideoCapture:
    idx = int(index) if str(index).isdigit() else index
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(idx)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    return cap


def snapshot_cameras(cfg: dict) -> dict:
    out = {"cameras": {}}
    for name, cam in (cfg.get("cameras") or {}).items():
        idx = cam["index_or_path"]
        cap = _open(idx)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera '{name}' index={idx}")
        for _ in range(5):
            cap.read()
        ok, frame = cap.read()
        props = {
            "index_or_path": idx,
            "width_set": cam["width"],
            "height_set": cam["height"],
            "fps_set": cam["fps"],
            "fourcc_set": cam.get("fourcc"),
            "width_actual": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height_actual": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps_actual": float(cap.get(cv2.CAP_PROP_FPS)),
            "exposure": float(cap.get(cv2.CAP_PROP_EXPOSURE)),
            "wb": float(cap.get(cv2.CAP_PROP_WB_TEMPERATURE)),
            "brightness": float(cap.get(cv2.CAP_PROP_BRIGHTNESS)),
            "frame_ok": bool(ok),
            "frame_shape": list(frame.shape) if ok else None,
        }
        cap.release()
        out["cameras"][name] = props
    return out


def compare(ref: dict, cur: dict, tol_res: int = 0) -> list[str]:
    fails = []
    for name, rcam in (ref.get("cameras") or {}).items():
        ccam = (cur.get("cameras") or {}).get(name)
        if not ccam:
            fails.append(f"missing camera '{name}' now")
            continue
        for key in ("width_actual", "height_actual"):
            if abs(int(ccam[key]) - int(rcam[key])) > tol_res:
                fails.append(f"{name}.{key}: was {rcam[key]}, now {ccam[key]}")
        # exposure/wb can be noisy; warn if large delta
        for key, thresh in (("exposure", 0.5), ("wb", 200), ("brightness", 10)):
            try:
                if abs(float(ccam[key]) - float(rcam[key])) > thresh:
                    fails.append(
                        f"{name}.{key} drifted: was {rcam[key]}, now {ccam[key]}"
                    )
            except (TypeError, ValueError):
                pass
        if not ccam.get("frame_ok"):
            fails.append(f"{name}: cannot grab frame")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--save-ref", action="store_true")
    ap.add_argument("--ref", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_recording_config(args.config)
    ref_path = args.ref or Path(cfg.get("camera_lock_reference") or "data/recording/camera_lock_ref.json")

    try:
        cur = snapshot_cameras(cfg)
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 2

    if args.save_ref:
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(json.dumps(cur, indent=2))
        print(f"PASS: saved camera lock reference -> {ref_path}")
        for name, c in cur["cameras"].items():
            print(f"  {name}: {c['width_actual']}x{c['height_actual']} exp={c['exposure']}")
        return 0

    if not ref_path.exists():
        print(f"FAIL: no reference at {ref_path}. Run with --save-ref after locking episode-1 cams.")
        return 2

    ref = json.loads(ref_path.read_text())
    fails = compare(ref, cur)
    if fails:
        print("FAIL: camera lock drift detected — DO NOT RECORD until fixed:")
        for f in fails:
            print(f"  * {f}")
        return 2
    print("PASS: cameras match episode-1 lock reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
