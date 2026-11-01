from __future__ import annotations

from cap.blocks.drive import (
    DURATION_RANGE,
    VR_RANGE,
    apply_body_delta,
    predict_body_delta,
    run_command,
    wrap_angle,
)
from cap.blocks.get_loc import get_loc
from cap.state import Pose, RolloutState

YAW_TOL = 0.05
MAX_TURNS = 4


def look_at(env, walker, command2loc, state: RolloutState, yaw: float) -> Pose:
    for _ in range(MAX_TURNS):
        robot = get_loc(state, "robot")
        if robot is None or robot.yaw is None:
            raise ValueError("robot pose is missing")
        delta = wrap_angle(yaw - robot.yaw)
        if abs(delta) <= YAW_TOL:
            return robot
        vr_limit = VR_RANGE[1]
        duration = min(max(abs(delta) / vr_limit, DURATION_RANGE[0]), DURATION_RANGE[1])
        vr = max(-vr_limit, min(vr_limit, delta / duration))
        command = (0.0, 0.0, vr)
        run_command(env, walker, state, command, duration, update_state=False)
        dx, dy, dr = predict_body_delta(command2loc, duration, command)
        pose = apply_body_delta(robot, dx, dy, dr)
        state.robot.pose = pose
        state.robot.path.append(pose)
    robot = get_loc(state, "robot")
    if robot is None:
        raise ValueError("robot pose is missing")
    return robot
