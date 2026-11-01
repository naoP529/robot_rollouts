from __future__ import annotations

import math

from cap.state import Pose, RolloutState, require_class_name, strip_instance_suffix
from obj_gen.arch_gen import CENTERLINE_RADIUS, TUBE_RADIUS

# Opening span is (centerline - tube) * ARCH_SCALE. Through-axis thickness is tube * ARCH_SCALE.
# ARCH_SCALE mirrors envs/arcade.py; not imported here to keep this module free of the genesis dependency.
ARCH_SCALE = 0.19
OPENING_HALF = (CENTERLINE_RADIUS - TUBE_RADIUS) * ARCH_SCALE
PASS_CLEAR = TUBE_RADIUS * ARCH_SCALE


def is_finished(state: RolloutState, ball: str, arch: str, env=None, env_idx: int = 0) -> bool:
    require_class_name(ball, "is_finished(ball=...)")
    require_class_name(arch, "is_finished(arch=...)")
    ball_pose = _ball_pose(state, ball, env, env_idx)
    arch_pose = _arch_pose(state, arch, env)
    if ball_pose is None or arch_pose is None:
        return False
    along, lateral = _plane_coords(ball_pose, arch_pose)
    return along >= PASS_CLEAR and abs(lateral) <= OPENING_HALF


def _ball_pose(state: RolloutState, key: str, env, env_idx: int) -> Pose | None:
    pose = _env_ball_pose(env, key, env_idx)
    if pose is not None:
        return pose
    return _observed_pose(state, key, "ball")


def _arch_pose(state: RolloutState, key: str, env) -> Pose | None:
    pose = _env_arch_pose(env, key)
    if pose is not None:
        return pose
    return _observed_pose(state, key, "arch")


def _observed_pose(state: RolloutState, key: str, kind: str) -> Pose | None:
    item = state.find_observed(key)
    if item is None:
        name = _asset_name(key, kind)
        item = next((obs for obs in state.observed if obs.name == name), None)
    return item.pose if item is not None else None


def _env_arch_pose(env, key: str) -> Pose | None:
    specs = getattr(env, "arches_spec", None)
    ys = getattr(env, "arch_ys", None)
    xs = getattr(env, "arch_xs", None)
    if specs is None or ys is None or xs is None:
        return None
    name = _asset_name(key, "arch")
    for spec, x, y in zip(specs, xs, ys):
        if spec.get("name") == name:
            return Pose(float(x), float(y))
    return None


def _env_ball_pose(env, key: str, env_idx: int) -> Pose | None:
    specs = getattr(env, "balls_spec", None)
    entities = getattr(env, "_ball_entities", None)
    if not specs or not entities:
        return None
    name = _asset_name(key, "ball")
    for spec, entity in zip(specs, entities):
        if spec.get("name") != name:
            continue
        pos = entity.get_pos()
        if hasattr(pos, "detach"):
            pos = pos.detach().cpu().numpy()
        row = pos.reshape(-1, 3)
        index = min(max(env_idx, 0), len(row) - 1)
        return Pose(float(row[index, 0]), float(row[index, 1]))
    return None


def _plane_coords(ball: Pose, arch: Pose) -> tuple[float, float]:
    dx = ball.x - arch.x
    dy = ball.y - arch.y
    if arch.yaw is None:
        return dx, dy
    c = math.cos(arch.yaw)
    s = math.sin(arch.yaw)
    return c * dx + s * dy, -s * dx + c * dy


def _asset_name(key: str, kind: str) -> str:
    prefix = f"{kind}_"
    if key.startswith(prefix):
        base = strip_instance_suffix(key)
        return base if base.startswith(prefix) else key
    return f"{prefix}{key}"
