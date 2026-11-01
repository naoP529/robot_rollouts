from __future__ import annotations

import math

from cap_easymode.blocks.get_loc import get_loc
from cap_easymode.state import Pose, RolloutState
from obj_gen.arch_gen import CENTERLINE_RADIUS, TUBE_RADIUS

# Opening span is (centerline - tube) * ARCH_SCALE. Through-axis thickness is tube * ARCH_SCALE.
# ARCH_SCALE mirrors envs/arcade.py; not imported here to keep this module free of the genesis dependency.
ARCH_SCALE = 0.19
OPENING_HALF = (CENTERLINE_RADIUS - TUBE_RADIUS) * ARCH_SCALE
PASS_CLEAR = TUBE_RADIUS * ARCH_SCALE


def is_finished(env, state: RolloutState, ball: str, arch: str, env_idx: int = 0) -> bool:
    """Return whether the ball has passed through the arch, using exact positions read
    straight from the simulator."""
    ball_pose = get_loc(env, state, ball, env_idx)
    arch_pose = get_loc(env, state, arch)
    if ball_pose is None or arch_pose is None:
        return False
    along, lateral = _plane_coords(ball_pose, arch_pose)
    return along >= PASS_CLEAR and abs(lateral) <= OPENING_HALF


def _plane_coords(ball: Pose, arch: Pose) -> tuple[float, float]:
    dx = ball.x - arch.x
    dy = ball.y - arch.y
    if arch.yaw is None:
        return dx, dy
    c = math.cos(arch.yaw)
    s = math.sin(arch.yaw)
    return c * dx + s * dy, -s * dx + c * dy
