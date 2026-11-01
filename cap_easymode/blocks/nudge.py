from __future__ import annotations

import math

from cap_easymode.blocks.drive import run_command
from cap_easymode.blocks.get_loc import get_loc
from cap_easymode.blocks.look_at import look_at
from cap_easymode.state import Pose, RolloutState


def nudge(
    env,
    walker,
    state: RolloutState,
    name: str,
    xy: tuple[float, float],
    speed: float = 1.0,
    duration_s: float = 0.5,
) -> Pose | None:
    """Push the named ball by walking toward xy, then return its exact post-push pose
    read straight from the simulator. No predicted pose, no re-detection."""
    robot = state.robot.pose
    if robot is None or robot.yaw is None:
        raise ValueError("robot pose is missing")
    look_at(env, walker, state, math.atan2(xy[1] - robot.y, xy[0] - robot.x))
    command = (speed, 0.0, 0.0)
    run_command(env, walker, state, command, duration_s)
    return get_loc(env, state, name)
