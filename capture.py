import argparse
from datetime import datetime
from pathlib import Path

import genesis as gs
import imageio.v3 as iio
import numpy as np

from envs.test_env import TestEnv


def parse_res(value: str) -> tuple[int, int]:
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def capture_stem(seed: int, res: tuple[int, int]) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"head_s{seed}_{res[0]}x{res[1]}_{stamp}"


def as_image(arr):
    image = np.ascontiguousarray(arr)
    if image.ndim == 4:
        image = image[0]
    return image


def depth_preview(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if finite.any():
        lo = float(depth[finite].min())
        hi = float(depth[finite].max())
        if hi > lo:
            preview[finite] = ((depth[finite] - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return preview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--res", type=parse_res, default=parse_res("1920x1080"))
    parser.add_argument("--out-dir", type=Path, default=Path("captures"))
    args = parser.parse_args()

    gs.init(backend=gs.gpu)
    env = TestEnv(seed=args.seed, res=args.res)
    rgb, depth, seg, normal = env.render()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / capture_stem(args.seed, args.res)

    rgb = as_image(rgb)
    iio.imwrite(f"{stem}.png", rgb)
    print(f"saved {stem}.png")

    if seg is not None:
        seg = as_image(seg)
        np.save(f"{stem}_seg.npy", seg)
        print(f"saved {stem}_seg.npy")
        if seg.ndim == 2:
            ids = seg.astype(np.int64, copy=False)
            max_id = int(ids.max()) if ids.size else 0
            palette = np.random.default_rng(0).integers(0, 256, size=(max_id + 1, 3), dtype=np.uint8)
            palette[0] = 0
            iio.imwrite(f"{stem}_seg.png", palette[ids])
        else:
            iio.imwrite(f"{stem}_seg.png", seg)
        print(f"saved {stem}_seg.png")

    if depth is not None:
        depth = as_image(depth)
        np.save(f"{stem}_depth.npy", depth)
        iio.imwrite(f"{stem}_depth.png", depth_preview(depth))
        print(f"saved {stem}_depth.png")

    if normal is not None:
        normal = as_image(normal)
        if normal.dtype != np.uint8:
            normal = np.clip((normal * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        iio.imwrite(f"{stem}_normal.png", normal)
        print(f"saved {stem}_normal.png")


if __name__ == "__main__":
    main()
