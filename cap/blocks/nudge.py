from __future__ import annotations

import math

from cap.blocks.detect import detect
from cap.blocks.drive import apply_body_delta, predict_body_delta, run_command
from cap.blocks.get_loc import get_loc
from cap.blocks.look_at import look_at
from cap.state import Observed, Pose, RolloutState, require_instance_id, strip_instance_suffix


def nudge(
    env,
    walker,
    command2loc,
    owl,
    state: RolloutState,
    obj_id: str,
    xy: tuple[float, float],
    speed: float = 1.0,
    duration_s: float = 0.5,
) -> Pose | None:
    require_instance_id(obj_id, "nudge(obj_id=...)")
    robot = get_loc(state, "robot")
    if robot is None or robot.yaw is None:
        raise ValueError("robot pose is missing")
    look_at(env, walker, command2loc, state, math.atan2(xy[1] - robot.y, xy[0] - robot.x))

    start = get_loc(state, "robot")
    if start is None or start.yaw is None:
        raise ValueError("robot pose is missing")
    command = (speed, 0.0, 0.0)
    run_command(env, walker, state, command, duration_s)
    dx, dy, dr = predict_body_delta(command2loc, duration_s, command)
    predicted = apply_body_delta(Pose(xy[0], xy[1], start.yaw), dx, dy, dr)
    object_pose = Pose(predicted.x, predicted.y)
    _write_object(state, obj_id, object_pose)

    now = get_loc(state, "robot")
    if now is None:
        raise ValueError("robot pose is missing")
    look_at(env, walker, command2loc, state, math.atan2(object_pose.y - now.y, object_pose.x - now.x))
    detect(env, owl, state, _name_of(state, obj_id), prefer_id=obj_id)
    return get_loc(state, obj_id)


def _name_of(state: RolloutState, obj_id: str) -> str:
    item = state.find_observed(obj_id)
    if item is not None:
        return item.name
    return strip_instance_suffix(obj_id)


def _write_object(state: RolloutState, obj_id: str, pose: Pose) -> None:
    item = state.find_observed(obj_id)
    if item is not None:
        item.pose = pose
        return
    name = _name_of(state, obj_id)
    query = state.query_of(name) or name
    state.observed.append(Observed(id=obj_id, name=name, query=query, pose=pose))
    state.pop_unseen(name)
