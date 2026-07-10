#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REPLACEMENT = r'''import sys
from os import path
import numpy as np
import cv2
import torch
from tqdm import tqdm

from face_detection import FaceAlignment, LandmarksType


device = "cuda" if torch.cuda.is_available() else "cpu"
fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)
coord_placeholder = (0.0, 0.0, 0.0, 0.0)


def read_imgs(img_list):
    frames = []
    print("reading images...")
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def _bbox_from_detector(face_box, frame_shape, upperbondrange=0):
    if face_box is None:
        return coord_placeholder
    h, w = int(frame_shape[0]), int(frame_shape[1])
    x1, y1, x2, y2 = [int(v) for v in face_box]
    x1 = max(0, min(w - 1, x1))
    x2 = max(x1 + 1, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(y1 + 1, min(h, y2))

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    # MuseTalk needs a stable lower-face crop. Without MMPose landmarks we use
    # the SFD face box and bias the top toward the upper-middle of the face.
    # Positive bbox_shift moves the crop lower; negative moves it higher.
    pad_x = int(round(bw * 0.04))
    top = int(round(y1 + bh * 0.20 + int(upperbondrange)))
    bottom = int(round(y2 + bh * 0.04))
    left = int(round(x1 - pad_x))
    right = int(round(x2 + pad_x))

    left = max(0, min(w - 1, left))
    right = max(left + 1, min(w, right))
    top = max(0, min(h - 1, top))
    bottom = max(top + 1, min(h, bottom))
    return (left, top, right, bottom)


def get_bbox_range(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    return f"Total frame:「{len(frames)}」 SFD bbox fallback active, current value: {upperbondrange}"


def get_landmark_and_bbox(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    coords_list = []
    if upperbondrange != 0:
        print("get face bounding boxes with fallback bbox_shift:", upperbondrange)
    else:
        print("get face bounding boxes with fallback default value")
    for idx in tqdm(range(0, len(frames), batch_size_fa)):
        batch = [x for x in frames[idx:idx + batch_size_fa] if x is not None]
        if not batch:
            coords_list.append(coord_placeholder)
            continue
        detections = fa.get_detections_for_batch(np.asarray(batch))
        for frame, det in zip(batch, detections):
            coords_list.append(_bbox_from_detector(det, frame.shape, upperbondrange=upperbondrange))
    print("********************************************bbox_shift parameter adjustment**********************************************************")
    print(f"Total frame:「{len(frames)}」 SFD bbox fallback active, current value: {upperbondrange}")
    print("*************************************************************************************************************************************")
    return coords_list, frames


if __name__ == "__main__":
    img_list = ["./results/lyria/00000.png", "./results/lyria/00001.png"]
    coords_list, full_frames = get_landmark_and_bbox(img_list)
    print(coords_list)
'''


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: patch_musetalk_no_mmpose.py /opt/MuseTalk")
    root = Path(sys.argv[1]).resolve()
    target = root / "musetalk/utils/preprocessing.py"
    if not target.exists():
        raise SystemExit(f"missing MuseTalk preprocessing.py: {target}")
    backup = target.with_suffix(".py.smartblog-original")
    current = target.read_text(encoding="utf-8")
    if "SFD bbox fallback active" in current:
        print(f"already patched: {target}")
        patch_torch_load_compat(root)
        return
    if not backup.exists():
        backup.write_text(current, encoding="utf-8")
    target.write_text(REPLACEMENT, encoding="utf-8")
    print(f"patched: {target}")
    patch_torch_load_compat(root)


def patch_file(path: Path, replacements: dict[str, str]) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    original = text
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    if text != original:
        backup = path.with_suffix(path.suffix + ".smartblog-torchload-original")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        path.write_text(text, encoding="utf-8")
        print(f"patched torch.load compat: {path}")


def patch_torch_load_compat(root: Path) -> None:
    patch_file(
        root / "musetalk/utils/face_parsing/resnet.py",
        {
            "state_dict = torch.load(model_path) #modelzoo.load_url(resnet18_url)": (
                "state_dict = torch.load(model_path, weights_only=False) #modelzoo.load_url(resnet18_url)"
            ),
        },
    )
    patch_file(
        root / "musetalk/utils/face_parsing/__init__.py",
        {
            "net.load_state_dict(torch.load(model_pth))": (
                "net.load_state_dict(torch.load(model_pth, weights_only=False))"
            ),
            "net.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu')))": (
                "net.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu'), weights_only=False))"
            ),
        },
    )
    patch_file(
        root / "musetalk/utils/face_detection/detection/sfd/sfd_detector.py",
        {
            "model_weights = torch.load(path_to_detector)": (
                "model_weights = torch.load(path_to_detector, weights_only=False)"
            ),
        },
    )
    patch_file(
        root / "musetalk/models/unet.py",
        {
            "weights = torch.load(model_path) if torch.cuda.is_available() else torch.load(model_path, map_location=self.device)": (
                "weights = torch.load(model_path, weights_only=False) if torch.cuda.is_available() else torch.load(model_path, map_location=self.device, weights_only=False)"
            ),
        },
    )


if __name__ == "__main__":
    main()
