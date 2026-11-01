from __future__ import annotations

import math

from cap_easymode.blocks.drive import (
    DURATION_RANGE,
    VX_RANGE,
    body_from_world,
    predict_command,
    run_command,
)
from cap_easymode.state import Pose, RolloutState

XY_TOL = 0.15
YAW_TOL = 0.1
MAX_SEGMENTS = 40
XY_SPEED = 0.8
YAW_SPEED = 1.2


def goto(
    env,
    walker,
    loc2command,
    state: RolloutState,
    x: float,
    y: float,
    yaw: float | None = None,
) -> Pose:
    for _ in range(MAX_SEGMENTS):
        robot = state.robot.pose
        if robot is None or robot.yaw is None:
            raise ValueError("robot pose is missing")
        remaining_xy = math.hypot(x - robot.x, y - robot.y)
        if yaw is None:
            target_yaw = math.atan2(y - robot.y, x - robot.x) if remaining_xy > XY_TOL else robot.yaw
        else:
            target_yaw = yaw
        body_dx, body_dy, body_dr = body_from_world(robot, x, y, target_yaw)
        if remaining_xy <= XY_TOL and (yaw is None or abs(body_dr) <= YAW_TOL):
            return robot

        duration, body_dx, body_dy, body_dr = _segment(body_dx, body_dy, body_dr)
        command = predict_command(loc2command, duration, body_dx, body_dy, body_dr)
        run_command(env, walker, state, command, duration)
    return state.robot.pose


def _segment(dx: float, dy: float, dr: float) -> tuple[float, float, float, float]:
    dist = math.hypot(dx, dy)
    t_xy = dist / XY_SPEED if XY_SPEED > 0 else 0.0
    t_yaw = abs(dr) / YAW_SPEED if YAW_SPEED > 0 else 0.0
    duration = max(t_xy, t_yaw, DURATION_RANGE[0])
    if duration > DURATION_RANGE[1]:
        scale = DURATION_RANGE[1] / duration
        dx *= scale
        dy *= scale
        dr *= scale
        duration = DURATION_RANGE[1]
    dist = math.hypot(dx, dy)
    max_forward = VX_RANGE[1] * duration
    if dist > max_forward > 0:
        scale = max_forward / dist
        dx *= scale
        dy *= scale
    return duration, dx, dy, dr
