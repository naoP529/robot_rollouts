from __future__ import annotations

import math

from cap.blocks.detect import detect_from_capture, rgb_of
from cap.blocks.drive import DURATION_RANGE, apply_body_delta, predict_body_delta, run_command
from cap.blocks.get_loc import get_loc
from cap.state import RolloutState, require_class_name

SPIN_VR = 0.6  # rad/s. Moderate fraction of VR_RANGE[1]=1.5 for a steady, continuous spin.


def search(env, walker, command2loc, owl, state: RolloutState, name: str, step_yaw: float = 0.5) -> list[str]:
    require_class_name(name, "search(name=...)")
    if state.query_of(name) is None:
        return []

    robot = get_loc(state, "robot")
    if robot is None or robot.yaw is None:
        raise ValueError("robot pose is missing")

    n_segments = max(1, round(2 * math.pi / step_yaw))
    duration = min(max(step_yaw / SPIN_VR, DURATION_RANGE[0]), DURATION_RANGE[1])
    command = (0.0, 0.0, SPIN_VR)

    captures = [(robot, rgb_of(env))]
    for _ in range(n_segments):
        run_command(env, walker, state, command, duration, update_state=False)
        dx, dy, dr = predict_body_delta(command2loc, duration, command)
        robot = apply_body_delta(robot, dx, dy, dr)
        state.robot.pose = robot
        state.robot.path.append(robot)
        captures.append((robot, rgb_of(env)))

    return detect_from_capture(env, owl, state, name, captures)
