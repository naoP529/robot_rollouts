import math

import torch

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


class Go2Env:
    """Go2 locomotion environment for rollouts.

    Velocity commands are supplied by the caller through ``set_commands``, as
    ``(lin_vel_x, lin_vel_y, ang_vel_yaw)``. ``step`` applies one control step
    and does not reset. Call ``reset`` to put robots back at the initial pose.
    Commands are kept across resets.

    ``show_viewer=True`` opens the interactive viewer on environment 0 and follows
    that robot. ``rendered_envs_idx`` draws those environments instead, spaced on
    a grid under a fixed camera. With the viewer open, ``step`` refreshes the
    window and paces the simulation to real time.
    """

    def __init__(self, num_envs, env_cfg, obs_cfg, show_viewer=False, rendered_envs_idx=None):
        self.num_envs: int = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = 3
        self.cfg = env_cfg
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.obs_scales: dict[str, float] = obs_cfg["obs_scales"]

        if show_viewer and rendered_envs_idx is None:
            rendered_envs_idx = (0,)
        if rendered_envs_idx is not None:
            rendered_envs_idx = tuple(rendered_envs_idx)
            if len(rendered_envs_idx) == 0 or any(idx < 0 or idx >= num_envs for idx in rendered_envs_idx):
                raise ValueError(f"rendered_envs_idx must be in [0, {num_envs}), got {rendered_envs_idx}")
        # follow_entity tracks env 0's simulated pose, which matches the drawn pose only when
        # that env is unshifted. A grid of envs is centered at the origin instead.
        follow_env0 = show_viewer and rendered_envs_idx == (0,)
        base_pos = env_cfg["base_init_pos"]
        if follow_env0:
            # Used as an offset from the robot once the camera starts following.
            camera_pos = (1.5, -1.5, 0.9)
            camera_lookat = (base_pos[0], base_pos[1], base_pos[2])
        else:
            n_shown = len(rendered_envs_idx) if rendered_envs_idx is not None else max(num_envs, 1)
            cols = math.ceil(math.sqrt(n_shown))
            rows = math.ceil(n_shown / cols)
            span = max(rows - 1, cols - 1, 1) * 2.0
            dist = max(3.0, span)
            camera_pos = (dist, -dist, max(1.5, dist * 0.45))
            camera_lookat = (0.0, 0.0, base_pos[2])

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                # For this locomotion policy, there are usually no more than 20 collision pairs. Setting a low value
                # can save memory. Violating this condition will raise an exception.
                max_collision_pairs=20,
                tolerance=1e-5,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=camera_pos,
                camera_lookat=camera_lookat,
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=True,
                rendered_envs_idx=rendered_envs_idx,
            )
            if show_viewer
            else None,
            show_viewer=show_viewer,
        )

        self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                fixed=True,
            )
        )

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
            ),
        )

        self.scene.build(
            n_envs=num_envs,
            env_spacing=(0.0, 0.0) if follow_env0 or not show_viewer else (2.0, 2.0),
            center_envs_at_origin=not follow_env0,
        )
        if follow_env0:
            self.scene.viewer.follow_entity(self.robot)

        # names to indices
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # Define global gravity direction vector
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)

        # Initial state
        self.init_base_pos = torch.tensor(self.env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
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
        self.base_pos = self.init_base_pos.repeat(self.num_envs, 1)
        self.base_quat = self.init_base_quat.repeat(self.num_envs, 1)
        self.base_euler = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.obs_buf = torch.empty((self.num_envs, 0), dtype=gs.tc_float, device=gs.device)

        self.reset()

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
        # Refreshes the viewer. With it open, steps are paced to scene.viewer.realtime_factor (1.0).
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

    def _reset_idx(self, envs_idx=None):
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.episode_length_buf.zero_()
        else:
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(envs_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(
                envs_idx[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity
            )
            torch.where(envs_idx[:, None], self.init_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)

    def _update_observation(self):
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            dim=-1,
        )

    def reset(self, envs_idx=None):
        """Reset robot state. ``envs_idx`` is a bool mask of shape ``(num_envs,)``, or ``None`` for every env."""
        self._reset_idx(envs_idx)
        self._update_observation()
        return self.obs_buf.clone()
