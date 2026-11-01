"""Run a walk command for a duration and update the robot pose in state."""

from __future__ import annotations

import math

import torch

from cap.state import Pose, RolloutState

VX_RANGE = (-0.5, 1.0)
VY_RANGE = (-0.5, 0.5)
VR_RANGE = (-1.5, 1.5)
DURATION_RANGE = (0.1, 6.0)


def wrap_angle(yaw: float) -> float:
    """Wrap a yaw into (-pi, pi]."""
    return (yaw + math.pi) % (2 * math.pi) - math.pi


def clip_command(vx: float, vy: float, vr: float) -> tuple[float, float, float]:
    """Clip a velocity command to the walker's limits."""
    return (
        min(max(vx, VX_RANGE[0]), VX_RANGE[1]),
        min(max(vy, VY_RANGE[0]), VY_RANGE[1]),
        min(max(vr, VR_RANGE[0]), VR_RANGE[1]),
    )


def pose_from_env(env, env_idx: int = 0) -> Pose:
    """Read the robot pose from the environment."""
    pos = env.base_pos[env_idx]
    euler = env.base_euler[env_idx]
    return Pose(
        float(pos[0].detach().cpu()),
        float(pos[1].detach().cpu()),
        wrap_angle(math.radians(float(euler[2].detach().cpu()))),
    )


def body_from_world(pose: Pose, x: float, y: float, yaw: float | None = None) -> tuple[float, float, float]:
    """Convert a world target into a body-frame delta."""
    if pose.yaw is None:
        raise ValueError("robot pose requires yaw")
    dx = x - pose.x
    dy = y - pose.y
    c = math.cos(pose.yaw)
    s = math.sin(pose.yaw)
    body_dx = c * dx + s * dy
    body_dy = -s * dx + c * dy
    if yaw is None:
        body_dr = 0.0
    else:
        body_dr = wrap_angle(yaw - pose.yaw)
    return body_dx, body_dy, body_dr


def apply_body_delta(pose: Pose, dx: float, dy: float, dr: float) -> Pose:
    """Apply a body-frame delta to a pose."""
    if pose.yaw is None:
        raise ValueError("robot pose requires yaw")
    c = math.cos(pose.yaw)
    s = math.sin(pose.yaw)
    return Pose(
        pose.x + c * dx - s * dy,
        pose.y + s * dx + c * dy,
        wrap_angle(pose.yaw + dr),
    )


def predict_body_delta(command2loc, duration_s: float, command: tuple[float, float, float]) -> tuple[float, float, float]:
    """Predict the body-frame motion of a walk command."""
    device = next(command2loc.parameters()).device
    x = torch.tensor([[duration_s, *command]], dtype=torch.float32, device=device)
    with torch.no_grad():
        dx, dy, dr = command2loc(x)[0].tolist()
    return float(dx), float(dy), float(dr)


def predict_command(loc2command, duration_s: float, dx: float, dy: float, dr: float) -> tuple[float, float, float]:
    """Predict the walk command for a body-frame delta."""
    device = next(loc2command.parameters()).device
    x = torch.tensor([[duration_s, dx, dy, dr]], dtype=torch.float32, device=device)
    with torch.no_grad():
        vx, vy, vr = loc2command(x)[0].tolist()
    return clip_command(float(vx), float(vy), float(vr))


def run_command(env, walker, state: RolloutState, command, duration_s: float, update_state: bool = True) -> Pose:
    """Run a walk command for a duration and update the robot pose in state."""
    vx, vy, vr = clip_command(*command)
    env.set_commands([vx, vy, vr])
    obs = env.get_observations()
    n_steps = max(1, int(round(duration_s / env.dt)))
    for _ in range(n_steps):
        with torch.no_grad():
            action = walker(obs).mean
        obs, _done = env.step(action)
    pose = pose_from_env(env)
    if update_state:
        state.robot.pose = pose
        state.robot.path.append(pose)
    return pose
