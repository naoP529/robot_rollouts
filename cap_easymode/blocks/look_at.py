from __future__ import annotations

from cap_easymode.blocks.drive import DURATION_RANGE, VR_RANGE, run_command, wrap_angle
from cap_easymode.state import Pose, RolloutState

YAW_TOL = 0.05
MAX_TURNS = 4


def look_at(env, walker, state: RolloutState, yaw: float) -> Pose:
    for _ in range(MAX_TURNS):
        robot = state.robot.pose
        if robot is None or robot.yaw is None:
            raise ValueError("robot pose is missing")
        delta = wrap_angle(yaw - robot.yaw)
        if abs(delta) <= YAW_TOL:
            return robot
        vr_limit = VR_RANGE[1]
        duration = min(max(abs(delta) / vr_limit, DURATION_RANGE[0]), DURATION_RANGE[1])
        vr = max(-vr_limit, min(vr_limit, delta / duration))
        command = (0.0, 0.0, vr)
        run_command(env, walker, state, command, duration)
    return state.robot.pose
