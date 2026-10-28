"""Rubber playground ball for a Genesis world.

The sphere, the rubber material, and the ground contact are fixed. The whole
ball is one flat color, yellow unless ``--color`` is set.

The mesh is authored Z-up and written as a Y-up GLB, which Genesis loads
without ``file_meshes_are_zup``. The bottom sits on z = 0. Load it with
``align=False`` so the principal-axis frame does not tip it over:

    scene.add_entity(
        gs.morphs.Mesh(file="assets/balls/ball_red.glb", align=False),
    )

Write a named copy with ``python obj_gen/ball_gen.py toy --color #c81018``.
Write every palette color with ``python obj_gen/ball_gen.py --all``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh

try:
    from obj_gen.arch_gen import PALETTE, parse_color, to_y_up
except ImportError:
    from arch_gen import PALETTE, parse_color, to_y_up

RADIUS = 0.11
U_STEPS = 32
V_STEPS = 64
ROUGHNESS = 0.65
METALLIC = 0.0
BODY_COLOR = PALETTE["yellow"]

BALLS_DIR = Path(__file__).resolve().parents[1] / "assets" / "balls"


def _rubber_material(color: tuple[int, int, int]) -> trimesh.visual.material.PBRMaterial:
    return trimesh.visual.material.PBRMaterial(
        name="rubber",
        baseColorFactor=[*color, 255],
        metallicFactor=METALLIC,
        roughnessFactor=ROUGHNESS,
        doubleSided=False,
    )


def build_ball(color: tuple[int, int, int] = BODY_COLOR) -> trimesh.Trimesh:
    """Z-up sphere whose lowest point is z = 0."""
    mesh = trimesh.creation.uv_sphere(radius=RADIUS, count=(U_STEPS, V_STEPS))
    mesh.apply_translation((0.0, 0.0, RADIUS))
    mesh.visual = trimesh.visual.TextureVisuals(material=_rubber_material(color))
    if mesh.bounds[0, 2] < -1e-6 or not mesh.is_watertight:
        raise RuntimeError("ball mesh is not a closed shell on the ground")
    return mesh


def output_path(name: str) -> Path:
    """``name`` is the file stem inside ``assets/balls``. A trailing ``.glb`` is accepted."""
    stem = name[:-4] if name.lower().endswith(".glb") else name
    if stem in {"", ".", ".."} or Path(stem).name != stem:
        raise ValueError(f"model name must be a file name, got {name!r}")
    BALLS_DIR.mkdir(parents=True, exist_ok=True)
    return BALLS_DIR / f"{stem}.glb"


def main(name: str = "ball", color: tuple[int, int, int] = BODY_COLOR) -> Path:
    path = output_path(name)
    path.write_bytes(to_y_up(build_ball(color)).export(file_type="glb"))
    return path


def write_palette() -> list[Path]:
    return [main(f"ball_{name}", color) for name, color in PALETTE.items()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write a rubber ball GLB into assets/balls.")
    parser.add_argument("name", nargs="?", default="ball", help="File name. '.glb' is optional.")
    parser.add_argument(
        "--color",
        type=parse_color,
        default=BODY_COLOR,
        help="RGB as #rrggbb or r,g,b. Defaults to 255,196,0.",
    )
    parser.add_argument("--all", action="store_true", help="Write ball_<name>.glb for every palette color.")
    args = parser.parse_args()
    if args.all:
        for path in write_palette():
            print(path)
    else:
        print(main(args.name, args.color))
