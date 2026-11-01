"""Arcade course: start area, start line, scattered balls, then a row of arches."""

from __future__ import annotations

import math
import random
from pathlib import Path

import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import inv_quat, pos_lookat_up_to_T, quat_to_xyz, transform_by_quat, transform_quat_by_quat

HEAD_CAM_POS = (0.335, 0.0, 0.03)
HEAD_CAM_LOOKAT = (2.285, 0.0, 0.01)
HEAD_CAM_UP = (0.0, 0.0, 1.0)
HEAD_CAM_FOV = 70.0
HEAD_CAM_RES = (1920, 1080)
OVERVIEW_FOV = 45.0

BALL_SCALE = 3.0
ARCH_SCALE = 0.19
# Genesis's default rigid density (600 kg/m^3) would make this solid-sphere mesh weigh ~90kg
# at BALL_SCALE, far too heavy for the robot to push. This keeps it near a real inflatable
# playground ball's mass (~0.75kg).
BALL_RHO = 5.0
BALL_RADIUS = 0.11 * BALL_SCALE
BALL_DIAMETER = 2.0 * BALL_RADIUS
BALL_MIN_SEP = 0.8 * BALL_DIAMETER
ARCH_TUBE_RADIUS = 0.62 * ARCH_SCALE

START_X0 = 0.0
START_X1 = 1.0  # keep >= 2*ROBOT_MARGIN so the robot's spawn range stays valid
LINE_DEPTH = 0.08
LINE_HEIGHT = 0.02
BALL_GAP = 0.2
BALL_DEPTH = 4.0
ARCH_GAP = 2.3
ARCH_X_JITTER = 1.5
ARCH_SPACING = 2.4
ARCH_MARGIN = 1.2
MIN_WIDTH = 6.0
ROBOT_MARGIN = 0.45
MAX_PLACE_TRIES = 400
BASE_Z = 0.42
IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
TEXTURE_MAPS = ("diffuse.jpg", "roughness.jpg", "normal.jpg")


def _yaw_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw_rad
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _course_width(n_arches: int) -> float:
    span = max(n_arches - 1, 0) * ARCH_SPACING
    return max(MIN_WIDTH, span + 2.0 * ARCH_MARGIN)


def _arch_ys(n_arches: int) -> list[float]:
    if n_arches <= 0:
        return []
    span = (n_arches - 1) * ARCH_SPACING
    y0 = -0.5 * span
    return [y0 + i * ARCH_SPACING for i in range(n_arches)]


def _sample_xy(rng: random.Random, x_range, y_range, min_sep, placed: list[tuple[float, float]]) -> tuple[float, float]:
    for _ in range(MAX_PLACE_TRIES):
        x = rng.uniform(*x_range)
        y = rng.uniform(*y_range)
        if all(math.hypot(x - px, y - py) >= min_sep for px, py in placed):
            placed.append((x, y))
            return x, y
    raise RuntimeError("failed to place object without overlap")


def _sample_centered(rng: random.Random, half_range: float, sigma_frac: float = 0.4) -> float:
    """Normal distribution centered on 0 (course midline), clamped to [-half_range, half_range]."""
    sigma = max(half_range * sigma_frac, 1e-6)
    for _ in range(MAX_PLACE_TRIES):
        y = rng.gauss(0.0, sigma)
        if -half_range <= y <= half_range:
            return y
    return max(-half_range, min(half_range, rng.gauss(0.0, sigma)))


class ArcadeEnv:
    """Go2 locomotion on a left-to-right arcade course.

    Velocity commands are ``(lin_vel_x, lin_vel_y, ang_vel_yaw)`` via
    ``set_commands``. ``step`` applies one control step. ``reset`` restores
    robots only; balls and arches stay put.
    """

    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        balls,
        arches,
        lighting,
        texture_dir,
        show_viewer=False,
        rendered_envs_idx=None,
        follow_robot=True,
        seed=0,
        res=(1920, 1080),
    ):
        self.num_envs: int = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = 3
        self.cfg = env_cfg
        self.device = gs.device
        self.seed = seed
        self.res = tuple(res)
        self.balls_spec = list(balls)
        self.arches_spec = list(arches)

        self.simulate_action_latency = True
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.obs_scales: dict[str, float] = obs_cfg["obs_scales"]

        n_arches = len(self.arches_spec)
        n_balls = len(self.balls_spec)
        self.width = _course_width(n_arches)
        self.line_x0 = START_X1
        self.line_x1 = START_X1 + LINE_DEPTH
        self.ball_x0 = self.line_x1 + BALL_GAP
        self.ball_x1 = self.ball_x0 + BALL_DEPTH
        self.arch_ys = _arch_ys(n_arches)
        self.arch_x = self.ball_x1 + ARCH_GAP if n_arches else self.ball_x1
        # Arches sit near arch_x but are jittered along x so they don't form one flat row.
        arch_rng = random.Random(seed)
        self.arch_xs = [self.arch_x + arch_rng.uniform(-ARCH_X_JITTER, ARCH_X_JITTER) for _ in range(n_arches)]
        farthest_arch_x = max(self.arch_xs) if self.arch_xs else self.arch_x
        self.course_end = farthest_arch_x + ARCH_TUBE_RADIUS + 0.4 if n_arches else self.ball_x1
        self.course_center = (0.5 * (START_X0 + self.course_end), 0.0, 0.0)
        half_w = 0.5 * self.width
        base_z = float(env_cfg.get("base_init_pos", (0.0, 0.0, BASE_Z))[2])

        self._robot_inits = []
        self._ball_inits = []
        for env_idx in range(num_envs):
            rng = random.Random(seed + env_idx)
            rx = rng.uniform(START_X0 + ROBOT_MARGIN, START_X1 - ROBOT_MARGIN)
            ry = _sample_centered(rng, half_w - ROBOT_MARGIN)
            self._robot_inits.append((rx, ry, base_z))
            placed = []
            env_balls = []
            x_range = (self.ball_x0 + BALL_RADIUS, self.ball_x1 - BALL_RADIUS)
            y_range = (-half_w + BALL_RADIUS, half_w - BALL_RADIUS)
            for _ in range(n_balls):
                x, y = _sample_xy(rng, x_range, y_range, BALL_MIN_SEP, placed)
                yaw = rng.uniform(0.0, math.tau)
                env_balls.append((x, y, yaw))
            self._ball_inits.append(env_balls)

        if show_viewer and rendered_envs_idx is None:
            rendered_envs_idx = (0,)
        if rendered_envs_idx is not None:
            rendered_envs_idx = tuple(rendered_envs_idx)
            if len(rendered_envs_idx) == 0 or any(idx < 0 or idx >= num_envs for idx in rendered_envs_idx):
                raise ValueError(f"rendered_envs_idx must be in [0, {num_envs}), got {rendered_envs_idx}")
        follow_env0 = bool(show_viewer and follow_robot and rendered_envs_idx == (0,))
        origin = self._robot_inits[0]
        if follow_env0:
            # follow_entity adds this to the robot pose, so it is an offset, not a world point.
            camera_pos = (1.5, -1.5, 0.9)
            camera_lookat = origin
            camera_up = (0.0, 0.0, 1.0)
            camera_fov = 40.0
        else:
            _, camera_pos, camera_lookat, camera_up, camera_fov = next(
                cam for cam in self._overview_cameras() if cam[0] == "front_right"
            )

        texture_dir = Path(texture_dir)
        for name in TEXTURE_MAPS:
            if not (texture_dir / name).is_file():
                raise FileNotFoundError(f"missing {name} under {texture_dir}")

        max_pairs = 64 + 8 * (n_balls + n_arches)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                max_collision_pairs=max_pairs,
                tolerance=1e-5,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=camera_pos,
                camera_lookat=camera_lookat,
                camera_up=camera_up,
                camera_fov=camera_fov,
                refresh_rate=60,
                realtime_factor=1.0,
            ),
            vis_options=gs.options.VisOptions(
                ambient_light=lighting["ambient_light"],
                lights=list(lighting["lights"]),
                background_color=lighting.get("background_color", (0.04, 0.08, 0.12)),
                show_world_frame=True,
                rendered_envs_idx=rendered_envs_idx,
                segmentation_level="entity",
            ),
            show_viewer=show_viewer,
        )

        self._name_counts = {}
        self.scene.add_entity(
            morph=gs.morphs.Plane(pos=(0, 0, 0), normal=(0, 0, 1)),
            surface=gs.surfaces.Rough(
                roughness=None,
                diffuse_texture=gs.textures.ImageTexture(image_path=str(texture_dir / "diffuse.jpg")),
                roughness_texture=gs.textures.ImageTexture(
                    image_path=str(texture_dir / "roughness.jpg"), encoding="linear"
                ),
                normal_texture=gs.textures.ImageTexture(
                    image_path=str(texture_dir / "normal.jpg"), encoding="linear"
                ),
            ),
            name="ground",
        )
        self.scene.add_entity(
            gs.morphs.Box(
                pos=(0.5 * (self.line_x0 + self.line_x1), 0.0, 0.5 * LINE_HEIGHT),
                size=(LINE_DEPTH, self.width, LINE_HEIGHT),
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Plastic(color=(1.0, 1.0, 1.0)),
            name="start_line",
        )
        for spec, x, y in zip(self.arches_spec, self.arch_xs, self.arch_ys):
            self._add_mesh(
                spec,
                pos=(x, y, 0.0),
                euler=(0.0, 0.0, 90.0),
                scale=ARCH_SCALE,
                collision=True,
                convexify=False,
                decimate=True,
            )

        self._ball_entities = []
        env0_balls = self._ball_inits[0] if n_balls else []
        for spec, (x, y, yaw) in zip(self.balls_spec, env0_balls):
            entity = self._add_mesh(
                spec,
                pos=(x, y, 0.0),
                euler=(0.0, 0.0, math.degrees(yaw)),
                scale=BALL_SCALE,
                collision=True,
                fixed=False,
                material=gs.materials.Rigid(rho=BALL_RHO),
            )
            self._ball_entities.append(entity)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=origin,
                quat=IDENTITY_QUAT,
            ),
            name="go2_robot",
        )

        self.head_cam_res = tuple(self.env_cfg.get("head_cam_res", self.res))
        self.head_cam_fov = float(self.env_cfg.get("head_cam_fov", HEAD_CAM_FOV))
        self.cameras = {}
        self.cameras["head"] = self.scene.add_camera(
            res=self.head_cam_res,
            pos=HEAD_CAM_POS,
            lookat=HEAD_CAM_LOOKAT,
            up=HEAD_CAM_UP,
            fov=self.head_cam_fov,
            GUI=False,
        )
        self.camera = self.cameras["head"]
        for name, pos, lookat, up, fov in self._overview_cameras():
            self.cameras[name] = self.scene.add_camera(
                res=self.res,
                pos=pos,
                lookat=lookat,
                up=up,
                fov=fov,
                GUI=False,
            )

        spacing = (0.0, 0.0)
        if show_viewer and not follow_env0:
            spacing = (self.course_end + 2.0, self.width + 2.0)
        self.scene.build(
            n_envs=num_envs,
            env_spacing=spacing,
            center_envs_at_origin=not follow_env0,
        )
        if follow_env0:
            self.scene.viewer.follow_entity(self.robot)
        self._apply_ball_poses()
        offset_T = pos_lookat_up_to_T(
            np.asarray(HEAD_CAM_POS, dtype=np.float64),
            np.asarray(HEAD_CAM_LOOKAT, dtype=np.float64),
            np.asarray(HEAD_CAM_UP, dtype=np.float64),
        )
        self.camera.attach(self.robot.get_link("base"), offset_T)
        self.camera.move_to_attach()

        # dofs_idx_local for set_dofs_kp/kv and get_dofs_position/velocity below is entity-local, not
        # scene-global, so the robot's own dof_start must be subtracted. Free (non-fixed) rigid bodies
        # added before the robot occupy global dof slots too, which shifts the robot's dof_start.
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start - self.robot.dof_start for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)

        self.init_base_pos = torch.tensor(self._robot_inits, dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(IDENTITY_QUAT, dtype=gs.tc_float, device=gs.device).repeat(self.num_envs, 1)
        self.inv_base_init_quat = inv_quat(self.init_base_quat[0])
        self.init_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.init_qpos = torch.concatenate(
            (
                self.init_base_pos,
                self.init_base_quat,
                self.init_dof_pos.repeat(self.num_envs, 1),
            ),
            dim=-1,
        )
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        self.base_lin_vel = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = self.init_projected_gravity.repeat(self.num_envs, 1)
        self.episode_length_buf = torch.zeros((self.num_envs,), dtype=gs.tc_int, device=gs.device)
        self.commands = torch.zeros((self.num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = self.init_dof_pos.repeat(self.num_envs, 1)
        self.dof_vel = torch.zeros_like(self.actions)
        self.base_pos = self.init_base_pos.clone()
        self.base_quat = self.init_base_quat.clone()
        self.base_euler = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.obs_buf = torch.empty((self.num_envs, 0), dtype=gs.tc_float, device=gs.device)

        self.reset()

    def _unique_name(self, base: str) -> str:
        n = self._name_counts.get(base, 0)
        self._name_counts[base] = n + 1
        return f"{base}_{n}"

    def _add_mesh(
        self,
        spec,
        pos,
        euler,
        scale=1.0,
        collision=True,
        convexify=None,
        batch_fixed_verts=False,
        decimate=None,
        fixed=True,
        material=None,
    ):
        kwargs = dict(
            file=spec["path"],
            pos=pos,
            euler=euler,
            scale=scale,
            align=False,
            fixed=fixed,
            collision=collision,
            batch_fixed_verts=batch_fixed_verts,
        )
        if convexify is not None:
            kwargs["convexify"] = convexify
        if decimate is not None:
            kwargs["decimate"] = decimate
        add_kwargs = {"name": self._unique_name(spec["name"])}
        if material is not None:
            add_kwargs["material"] = material
        return self.scene.add_entity(gs.morphs.Mesh(**kwargs), **add_kwargs)

    def _overview_cameras(self):
        cx, cy, cz = self.course_center
        diag = math.hypot(self.course_end - START_X0, self.width)
        dist = 1.25 * 0.5 * diag / math.tan(math.radians(OVERVIEW_FOV / 2.0))
        top_h = dist
        elev = math.radians(35.0)
        horiz = dist * math.cos(elev)
        height = dist * math.sin(elev)
        side = horiz / math.sqrt(2.0)
        lookat = (cx, cy, cz)
        return (
            ("top", (cx, cy, top_h), lookat, (1.0, 0.0, 0.0), OVERVIEW_FOV),
            ("front_left", (cx - side, cy - side, height), lookat, (0.0, 0.0, 1.0), OVERVIEW_FOV),
            ("front_right", (cx - side, cy + side, height), lookat, (0.0, 0.0, 1.0), OVERVIEW_FOV),
            ("back_left", (cx + side, cy - side, height), lookat, (0.0, 0.0, 1.0), OVERVIEW_FOV),
            ("back_right", (cx + side, cy + side, height), lookat, (0.0, 0.0, 1.0), OVERVIEW_FOV),
        )

    def _apply_ball_poses(self, envs_idx=None):
        if not self._ball_entities:
            return
        n_balls = len(self._ball_entities)
        pos = torch.zeros((self.num_envs, n_balls, 3), dtype=gs.tc_float, device=gs.device)
        quat = torch.zeros((self.num_envs, n_balls, 4), dtype=gs.tc_float, device=gs.device)
        for env_idx, env_balls in enumerate(self._ball_inits):
            for ball_idx, (x, y, yaw) in enumerate(env_balls):
                pos[env_idx, ball_idx] = torch.tensor((x, y, 0.0), dtype=gs.tc_float, device=gs.device)
                quat[env_idx, ball_idx] = torch.tensor(_yaw_quat(yaw), dtype=gs.tc_float, device=gs.device)
        if envs_idx is not None:
            pos = pos[envs_idx]
            quat = quat[envs_idx]
        for ball_idx, entity in enumerate(self._ball_entities):
            entity.set_pos(pos[:, ball_idx], envs_idx=envs_idx, zero_velocity=True)
            entity.set_quat(quat[:, ball_idx], envs_idx=envs_idx, zero_velocity=True)

    def set_commands(self, commands):
        """Set velocity commands. Shape ``(3,)`` is broadcast to every env, or pass ``(num_envs, 3)``."""
        commands = torch.as_tensor(commands, dtype=gs.tc_float, device=self.device)
        if commands.ndim == 1:
            if commands.shape != (self.num_commands,):
                raise ValueError(f"expected command shape {(self.num_commands,)}, got {tuple(commands.shape)}")
            commands = commands.unsqueeze(0).expand(self.num_envs, -1)
        elif commands.shape != (self.num_envs, self.num_commands):
            raise ValueError(
                f"expected command shape {(self.num_envs, self.num_commands)}, got {tuple(commands.shape)}"
            )
        self.commands.copy_(commands)
        self._update_observation()

    def step(self, actions):
        actions = torch.as_tensor(actions, dtype=gs.tc_float, device=self.device)
        self.actions.copy_(actions.clamp(-self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]))
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))
        self.scene.step()

        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        done = self.episode_length_buf > self.max_episode_length
        done |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        done |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        done |= self.scene.rigid_solver.get_error_envs_mask()

        self._update_observation()
        self.last_actions.copy_(self.actions)

        return self.obs_buf.clone(), done

    def get_observations(self):
        return self.obs_buf.clone()

    def render(self, rgb=True, depth=False, segmentation=False, normal=False, colorize_seg=False):
        return self.render_view(
            "head",
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
            normal=normal,
            colorize_seg=colorize_seg,
        )

    def render_view(self, name, rgb=True, depth=False, segmentation=False, normal=False, colorize_seg=False):
        camera = self.cameras.get(name)
        if camera is None:
            names = ", ".join(sorted(self.cameras))
            raise KeyError(f"unknown camera {name!r}; expected one of: {names}")
        if name == "head":
            camera.move_to_attach()
        return camera.render(
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
            colorize_seg=colorize_seg,
            normal=normal,
        )

    def _reset_idx(self, envs_idx=None):
        if envs_idx is None:
            self.robot.set_qpos(self.init_qpos, zero_velocity=True, skip_forward=True)
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity.expand_as(self.projected_gravity))
            self.dof_pos.copy_(self.init_dof_pos.expand_as(self.dof_pos))
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.episode_length_buf.zero_()
            return

        self.robot.set_qpos(self.init_qpos[envs_idx], envs_idx=envs_idx, zero_velocity=True, skip_forward=True)
        torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
        torch.where(envs_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
        torch.where(
            envs_idx[:, None],
            self.init_projected_gravity.expand_as(self.projected_gravity),
            self.projected_gravity,
            out=self.projected_gravity,
        )
        torch.where(envs_idx[:, None], self.init_dof_pos.expand_as(self.dof_pos), self.dof_pos, out=self.dof_pos)
        self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
        self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
        self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
        self.actions.masked_fill_(envs_idx[:, None], 0.0)
        self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
        self.episode_length_buf.masked_fill_(envs_idx, 0)

    def _update_observation(self):
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],
                self.projected_gravity,
                self.commands * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                self.dof_vel * self.obs_scales["dof_vel"],
                self.actions,
            ],
            dim=-1,
        )

    def reset(self, envs_idx=None):
        """Reset robot state. ``envs_idx`` is a bool mask of shape ``(num_envs,)``, or ``None`` for every env."""
        self._reset_idx(envs_idx)
        self._apply_ball_poses(envs_idx)
        self._update_observation()
        return self.obs_buf.clone()
