"""Return axis-aligned boxes from a segmentation map.

Accepts an ID map (``*_seg.npy``, shape ``H x W``) or a colorized mask
(``*_seg.png``, shape ``H x W x 3``). Background is id 0 or RGB black.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _label_map(seg: np.ndarray) -> np.ndarray:
    seg = np.ascontiguousarray(seg)
    if seg.ndim == 3 and seg.shape[-1] in (3, 4):
        rgb = seg[..., :3].astype(np.uint32, copy=False)
        return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    if seg.ndim == 2:
        return seg
    raise ValueError(f"expected HxW or HxWx3 segmentation, got shape {seg.shape}")


def bboxes_from_seg(seg: np.ndarray, ignore_ids=(0,), min_area: int = 1) -> list[dict]:
    labels = _label_map(seg)
    ignore = {int(i) for i in ignore_ids}
    boxes = []
    for label in np.unique(labels):
        label = int(label)
        if label in ignore:
            continue
        ys, xs = np.nonzero(labels == label)
        if ys.size < min_area:
            continue
        x_min = int(xs.min())
        y_min = int(ys.min())
        x_max = int(xs.max())
        y_max = int(ys.max())
        boxes.append(
            {
                "id": label,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "width": x_max - x_min + 1,
                "height": y_max - y_min + 1,
                "area": int(ys.size),
            }
        )
    boxes.sort(key=lambda b: b["area"], reverse=True)
    return boxes


def load_seg(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    import imageio.v3 as iio

    return iio.imread(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seg", type=Path, help="segmentation .npy ID map or color .png")
    parser.add_argument("--min-area", type=int, default=1)
    args = parser.parse_args()
    boxes = bboxes_from_seg(load_seg(args.seg), min_area=args.min_area)
    print(json.dumps(boxes, indent=2))


if __name__ == "__main__":
    main()
