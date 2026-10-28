import math
import random
from pathlib import Path

import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import pos_lookat_up_to_T

BALLS = [
    {"name": "ball_yellow", "query": "a yellow ball", "path": r"assets/balls/ball_yellow.glb"},
    {"name": "ball_red", "query": "a red ball", "path": r"assets/balls/ball_red.glb"},
    {"name": "ball_orange", "query": "an orange ball", "path": r"assets/balls/ball_orange.glb"},
    {"name": "ball_pink", "query": "a pink ball", "path": r"assets/balls/ball_pink.glb"},
    {"name": "ball_purple", "query": "a purple ball", "path": r"assets/balls/ball_purple.glb"},
    {"name": "ball_blue", "query": "a blue ball", "path": r"assets/balls/ball_blue.glb"},
    {"name": "ball_cyan", "query": "a cyan ball", "path": r"assets/balls/ball_cyan.glb"},
    {"name": "ball_green", "query": "a green ball", "path": r"assets/balls/ball_green.glb"},
    {"name": "ball_white", "query": "a white ball", "path": r"assets/balls/ball_white.glb"},
    {"name": "ball_black", "query": "a black ball", "path": r"assets/balls/ball_black.glb"},
]
ARCHES = [
    {"name": "arch_yellow", "query": "a yellow inflatable arch", "path": r"assets/arches/arch_yellow.glb"},
    {"name": "arch_red", "query": "a red inflatable arch", "path": r"assets/arches/arch_red.glb"},
    {"name": "arch_orange", "query": "an orange inflatable arch", "path": r"assets/arches/arch_orange.glb"},
    {"name": "arch_pink", "query": "a pink inflatable arch", "path": r"assets/arches/arch_pink.glb"},
    {"name": "arch_purple", "query": "a purple inflatable arch", "path": r"assets/arches/arch_purple.glb"},
    {"name": "arch_blue", "query": "a blue inflatable arch", "path": r"assets/arches/arch_blue.glb"},
    {"name": "arch_cyan", "query": "a cyan inflatable arch", "path": r"assets/arches/arch_cyan.glb"},
    {"name": "arch_green", "query": "a green inflatable arch", "path": r"assets/arches/arch_green.glb"},
    {"name": "arch_white", "query": "a white inflatable arch", "path": r"assets/arches/arch_white.glb"},
    {"name": "arch_black", "query": "a black inflatable arch", "path": r"assets/arches/arch_black.glb"},
]
OBSTACLE_QUERY = "a tree, rock, plant, crate, or other structure"
OBSTACLES = [
    {
        "name": "soccer_ball",
        "query": OBSTACLE_QUERY,
        "path": r"assets/obstacles/dirty_football_1k.gltf/dirty_football_1k.gltf",
        "scale": 1.0,
    },
    {
        "name": "american_football",
        "query": OBSTACLE_QUERY,
        "path": r"assets/obstacles/american_football_1k.gltf/american_football_1k.gltf",
        "scale": 1.0,
    },
    {
        "name": "moon_rock",
        "query": OBSTACLE_QUERY,
        "path": r"assets/obstacles/moon_rock_03_1k.gltf/moon_rock_03_1k.gltf",
        "scale": 8.0,
    },
    {
        "name": "celandine_plant",
        "query": OBSTACLE_QUERY,
        "path": r"assets/obstacles/celandine_01_1k.gltf/celandine_01_1k.gltf",
        "scale": 1.0,
    },
    {
        "name": "island_tree",
        "query": OBSTACLE_QUERY,
        "path": r"assets/obstacles/island_tree_02_1k.gltf/island_tree_02_1k.gltf",
        "scale": 1.0,
    },
    {
        "name": "low_boulder",
        "query": OBSTACLE_QUERY,
        "path": r"assets/obstacles/namaqualand_boulder_02_1k.gltf/namaqualand_boulder_02_1k.gltf",
        "scale": 1.0,
    },
    {
        "name": "tall_boulder",
        "query": OBSTACLE_QUERY,
        "path": r"assets/obstacles/namaqualand_boulder_03_1k.gltf/namaqualand_boulder_03_1k.gltf",
        "scale": 1.0,
    },
]

BASE_INIT_POS = (0.0, 0.0, 0.42)
BASE_INIT_QUAT = (1.0, 0.0, 0.0, 0.0)
DEFAULT_JOINT_ANGLES = {
    "FL_hip_joint": 0.0,
    "FR_hip_joint": 0.0,
    "RL_hip_joint": 0.0,
    "RR_hip_joint": 0.0,
    "FL_thigh_joint": 0.8,
    "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RL_calf_joint": -1.5,
    "RR_calf_joint": -1.5,
}

N_ARCHES = 3
N_BALLS = 8
# Authored ball diameter is 0.22 m. Scale 3 → 0.66 m, a bit larger than Go2.
BALL_SCALE = 3.0
# Authored inner width is 7.76 m. Scale 0.19 → 1.47 m, just enough for two balls.
ARCH_SCALE = 0.19
MIN_DIST_ROBOT = 1.5
MIN_DIST_ARCH = 2.0
MIN_DIST_BALL = 1.0
PLACE_X = (2.0, 12.0)
PLACE_Y = (-6.0, 6.0)
ARCH_X = (3.0, 10.0)
ARCH_YAW_DEG = (65.0, 115.0)
MAX_PLACE_TRIES = 400

HEADLIGHT = {"type": "directional", "dir": (-0.13, -0.22, -0.97), "color": (1.00, 0.96, 0.90), "intensity": 6.0}
AMBIENT_LIGHT = (0.35, 0.42, 0.55)
CAM_POS = (0.335, 0.0, 0.03)
CAM_LOOKAT = (2.285, 0.0, 0.01)
CAM_UP = (0.0, 0.0, 1.0)
CAM_FOV = 70.0

class TestEnv:
    def __init__(self, seed=0, res=(1920, 1080), texture_dir=None, lighting=None, follow_robot=True):
        self.dt = 0.02
        self.device = gs.device
        self.seed = seed
        self.res = res
        self.follow_robot = follow_robot
        self.rng = random.Random(seed)
        lighting = lighting or {"ambient_light": AMBIENT_LIGHT, "lights": [HEADLIGHT]}
        texture_dir = Path(texture_dir) if texture_dir else Path("assets/texture/leafy_glass")
        self.scene = gs.Scene(
            show_viewer=False,
            vis_options=gs.options.VisOptions(
                ambient_light=lighting["ambient_light"],
                lights=list(lighting["lights"]),
                segmentation_level="entity",
            ),
        )
        self._name_counts = {}
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=BASE_INIT_POS,
                quat=BASE_INIT_QUAT,
            ),
            name="go2_robot",
        )
        surface = gs.surfaces.Rough(
            roughness=None,
            diffuse_texture=gs.textures.ImageTexture(image_path=str(texture_dir / "diffuse.jpg")),
            roughness_texture=gs.textures.ImageTexture(
                image_path=str(texture_dir / "roughness.jpg"), encoding="linear"
            ),
            normal_texture=gs.textures.ImageTexture(
                image_path=str(texture_dir / "normal.jpg"), encoding="linear"
            ),
        )
        self.scene.add_entity(
            morph=gs.morphs.Plane(pos=(0, 0, 0), normal=(0, 0, 1)),
            surface=surface,
            name="ground",
        )

        self.placements = []
        self._place_meshes()

        # Head_upper is a fixed link and is collapsed into base on load.
        self.camera = self.scene.add_camera(
            res=self.res,
            pos=CAM_POS,
            lookat=CAM_LOOKAT,
            up=CAM_UP,
            fov=CAM_FOV,
            GUI=False,
        )

        self.scene.build()
        self._set_standing_pose()
        if self.follow_robot:
            offset_T = pos_lookat_up_to_T(
                np.asarray(CAM_POS, dtype=np.float64),
                np.asarray(CAM_LOOKAT, dtype=np.float64),
                np.asarray(CAM_UP, dtype=np.float64),
            )
            self.camera.attach(self.robot.get_link("base"), offset_T)
            self.camera.move_to_attach()

    def set_camera(self, pos, lookat, up=CAM_UP):
        self.camera.set_pose(pos=pos, lookat=lookat, up=up)

    def render(self, rgb=True, depth=True, segmentation=True, normal=True, colorize_seg=False):
        if self.follow_robot:
            self.camera.move_to_attach()
        return self.camera.render(
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
            colorize_seg=colorize_seg,
            normal=normal,
        )

    def _sample_xy(self, x_range, y_range, min_sep):
        for _ in range(MAX_PLACE_TRIES):
            x = self.rng.uniform(*x_range)
            y = self.rng.uniform(*y_range)
            if math.hypot(x, y) < MIN_DIST_ROBOT:
                continue
            if all(math.hypot(x - px, y - py) >= max(min_sep, psep) for px, py, psep in self.placements):
                self.placements.append((x, y, min_sep))
                return x, y
        raise RuntimeError("failed to place object without overlap")

    def _unique_name(self, base: str) -> str:
        n = self._name_counts.get(base, 0)
        self._name_counts[base] = n + 1
        return f"{base}_{n}"

    def _add_mesh(self, spec, pos, euler, scale=1.0):
        self.scene.add_entity(
            gs.morphs.Mesh(
                file=spec["path"],
                pos=pos,
                euler=euler,
                scale=scale,
                align=False,
                fixed=True,
                collision=False,
            ),
            name=self._unique_name(spec["name"]),
        )

    def _place_meshes(self):
        for _ in range(N_ARCHES):
            spec = self.rng.choice(ARCHES)
            x, y = self._sample_xy(ARCH_X, PLACE_Y, MIN_DIST_ARCH)
            yaw = self.rng.uniform(*ARCH_YAW_DEG)
            self._add_mesh(spec, pos=(x, y, 0.0), euler=(0.0, 0.0, yaw), scale=ARCH_SCALE)

        for _ in range(N_BALLS):
            spec = self.rng.choice(BALLS)
            x, y = self._sample_xy(PLACE_X, PLACE_Y, MIN_DIST_BALL)
            yaw = self.rng.uniform(0.0, 360.0)
            self._add_mesh(spec, pos=(x, y, 0.0), euler=(0.0, 0.0, yaw), scale=BALL_SCALE)

    def _set_standing_pose(self):
        dof_pos = [DEFAULT_JOINT_ANGLES[joint.name] for joint in self.robot.joints[1:]]
        qpos = torch.tensor(
            [*BASE_INIT_POS, *BASE_INIT_QUAT, *dof_pos],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.robot.set_qpos(qpos, zero_velocity=True)
