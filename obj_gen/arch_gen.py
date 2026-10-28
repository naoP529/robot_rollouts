"""Inflatable festival arch for a Genesis world.

The tube, the foot flare, and the vinyl material are fixed. The whole arch is
one flat color, yellow unless ``--color`` is set. Passing a Genesis surface
color replaces that color.

The mesh is authored Z-up and written as a Y-up GLB, which Genesis loads
without ``file_meshes_are_zup``. Feet sit on z = 0. Load it with
``align=False`` so the principal-axis frame does not tip it over:

    scene.add_entity(
        gs.morphs.Mesh(file="assets/arches/arch.glb", fixed=True, align=False),
    )

Write a named copy with ``python obj_gen/arch_gen.py gate --color #c81018``.
Write every palette color with ``python obj_gen/arch_gen.py --all``.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh

# Semicircular tube, meters. Feet are the end caps on z = 0.
CENTERLINE_RADIUS = 4.5
TUBE_RADIUS = 0.62
FOOT_FLARE = 0.18
FLARE_HEIGHT = 0.95
N_ALONG = 160
N_AROUND = 64

# Smooth vinyl. Roughness stays put when the color changes.
ROUGHNESS = 0.22
METALLIC = 0.0
BODY_COLOR = (255, 196, 0)
PALETTE = {
    "yellow": (255, 196, 0),
    "red": (224, 49, 49),
    "orange": (247, 103, 7),
    "pink": (230, 73, 128),
    "purple": (112, 72, 232),
    "blue": (28, 126, 214),
    "cyan": (18, 184, 134),
    "green": (47, 158, 68),
    "white": (241, 243, 245),
    "black": (33, 37, 41),
}

ARCHES_DIR = Path(__file__).resolve().parents[1] / "assets" / "arches"


def _phi_flare() -> float:
    return float(np.arcsin(np.clip(FLARE_HEIGHT / CENTERLINE_RADIUS, 0.0, 1.0)))


def _section_radius(phi: float, phi_flare: float) -> float:
    end = min(phi, np.pi - phi)
    if end >= phi_flare:
        return TUBE_RADIUS
    blend = 1.0 - end / phi_flare
    return TUBE_RADIUS * (1.0 + FOOT_FLARE * blend * blend)


def _centerline(phi: float) -> np.ndarray:
    return np.array(
        [-CENTERLINE_RADIUS * np.cos(phi), 0.0, CENTERLINE_RADIUS * np.sin(phi)],
        dtype=np.float64,
    )


def _frame(phi: float) -> tuple[np.ndarray, np.ndarray]:
    tangent = np.array([np.sin(phi), 0.0, np.cos(phi)], dtype=np.float64)
    outward = np.array([-np.cos(phi), 0.0, np.sin(phi)], dtype=np.float64)
    return outward, np.cross(tangent, outward)


def build_arch(color: tuple[int, int, int] = BODY_COLOR) -> trimesh.Trimesh:
    """Z-up tube with capped feet."""
    phi_flare = _phi_flare()
    n_rings = N_ALONG + 1
    phi = np.linspace(0.0, np.pi, n_rings)
    alpha = np.linspace(0.0, 2.0 * np.pi, N_AROUND, endpoint=False)

    vertices = np.zeros((n_rings * N_AROUND + 2, 3), dtype=np.float64)
    for i, p in enumerate(phi):
        outward, side = _frame(float(p))
        center = _centerline(float(p))
        radius = _section_radius(float(p), phi_flare)
        ring = center + radius * (
            np.cos(alpha)[:, None] * outward + np.sin(alpha)[:, None] * side
        )
        vertices[i * N_AROUND : (i + 1) * N_AROUND] = ring

    left_center = n_rings * N_AROUND
    right_center = left_center + 1
    vertices[left_center] = _centerline(0.0)
    vertices[right_center] = _centerline(np.pi)

    faces = []
    for i in range(N_ALONG):
        for j in range(N_AROUND):
            jn = (j + 1) % N_AROUND
            a = i * N_AROUND + j
            b = i * N_AROUND + jn
            c = (i + 1) * N_AROUND + j
            d = (i + 1) * N_AROUND + jn
            faces.append((a, d, b))
            faces.append((a, c, d))
    for j in range(N_AROUND):
        jn = (j + 1) % N_AROUND
        faces.append((left_center, j, jn))
        c = N_ALONG * N_AROUND + j
        d = N_ALONG * N_AROUND + jn
        faces.append((right_center, d, c))

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        visual=trimesh.visual.TextureVisuals(material=_vinyl_material(color)),
        process=False,
    )
    _orient_outward(mesh)
    return mesh


def _vinyl_material(color: tuple[int, int, int]) -> trimesh.visual.material.PBRMaterial:
    return trimesh.visual.material.PBRMaterial(
        name="vinyl",
        baseColorFactor=[*color, 255],
        metallicFactor=METALLIC,
        roughnessFactor=ROUGHNESS,
        doubleSided=False,
    )


def _orient_outward(mesh: trimesh.Trimesh) -> None:
    """Flip any face whose normal points into the tube, then check the shell encloses volume."""
    n_tube = N_ALONG * N_AROUND * 2
    tube = mesh.faces[:n_tube]
    phi = np.linspace(0.0, np.pi, N_ALONG + 1)[:-1]
    stations = np.stack([_centerline(float(p)) for p in phi], axis=0)
    centers = np.repeat(stations, N_AROUND * 2, axis=0)
    mids = mesh.vertices[tube].mean(axis=1)
    normals = np.cross(
        mesh.vertices[tube[:, 1]] - mesh.vertices[tube[:, 0]],
        mesh.vertices[tube[:, 2]] - mesh.vertices[tube[:, 0]],
    )
    if np.einsum("ij,ij->i", normals, mids - centers).mean() < 0.0:
        tube[:] = tube[:, ::-1]

    # Both end caps lie on the ground, so their outward normals point down.
    for cap in (mesh.faces[n_tube : n_tube + N_AROUND], mesh.faces[n_tube + N_AROUND :]):
        normals = np.cross(
            mesh.vertices[cap[:, 1]] - mesh.vertices[cap[:, 0]],
            mesh.vertices[cap[:, 2]] - mesh.vertices[cap[:, 0]],
        )
        if normals[:, 2].mean() > 0.0:
            cap[:] = cap[:, ::-1]

    signed = _signed_volume(mesh.vertices, mesh.faces)
    if signed <= 0.0 or mesh.bounds[0, 2] < -1e-6 or not mesh.is_watertight:
        raise RuntimeError(f"arch mesh is not a closed shell on the ground (volume={signed})")


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    tris = (vertices - vertices.mean(axis=0))[faces]
    return float(np.einsum("ij,ij->", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])) / 6.0)


def to_y_up(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """GLB is Y-up. Genesis converts (x, y, z) to (x, -z, y), which restores this Z-up arch."""
    turned = mesh.copy()
    turned.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1.0, 0.0, 0.0]))
    return turned


def output_path(name: str) -> Path:
    """``name`` is the file stem inside ``assets/arches``. A trailing ``.glb`` is accepted."""
    stem = name[:-4] if name.lower().endswith(".glb") else name
    if stem in {"", ".", ".."} or Path(stem).name != stem:
        raise ValueError(f"model name must be a file name, got {name!r}")
    ARCHES_DIR.mkdir(parents=True, exist_ok=True)
    return ARCHES_DIR / f"{stem}.glb"


def parse_color(text: str) -> tuple[int, int, int]:
    """``#rrggbb`` or ``r,g,b`` with each channel in 0–255."""
    raw = text.strip()
    if raw.startswith("#"):
        raw = raw[1:]
    if "," in raw:
        parts = raw.split(",")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(f"color must be r,g,b, got {text!r}")
        try:
            channels = tuple(int(part) for part in parts)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"color channels must be integers, got {text!r}") from exc
    elif len(raw) == 6:
        try:
            channels = tuple(int(raw[i : i + 2], 16) for i in (0, 2, 4))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"color must be #rrggbb, got {text!r}") from exc
    else:
        raise argparse.ArgumentTypeError(f"color must be #rrggbb or r,g,b, got {text!r}")
    if any(channel < 0 or channel > 255 for channel in channels):
        raise argparse.ArgumentTypeError(f"color channels must be in 0–255, got {text!r}")
    return channels


def main(name: str = "arch", color: tuple[int, int, int] = BODY_COLOR) -> Path:
    path = output_path(name)
    path.write_bytes(to_y_up(build_arch(color)).export(file_type="glb"))
    return path


def write_palette() -> list[Path]:
    return [main(f"arch_{name}", color) for name, color in PALETTE.items()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write an inflatable arch GLB into assets/arches.")
    parser.add_argument("name", nargs="?", default="arch", help="File name. '.glb' is optional.")
    parser.add_argument(
        "--color",
        type=parse_color,
        default=BODY_COLOR,
        help="RGB as #rrggbb or r,g,b. Defaults to 255,196,0.",
    )
    parser.add_argument("--all", action="store_true", help="Write arch_<name>.glb for every palette color.")
    args = parser.parse_args()
    if args.all:
        for path in write_palette():
            print(path)
    else:
        print(main(args.name, args.color))
