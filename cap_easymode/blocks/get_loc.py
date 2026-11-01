from __future__ import annotations

from cap_easymode.state import Pose, RolloutState


def get_loc(env, state: RolloutState, name: str, env_idx: int = 0) -> Pose | None:
    """Return the exact pose of the robot (name="robot") or a named ball/arch, read
    directly from the simulator. Unlike cap's detect/search, this never fails to find
    an object that exists in the scene and never returns a stale or noisy position."""
    if name == "robot":
        return state.robot.pose
    pose = _ball_pose(env, name, env_idx)
    if pose is not None:
        return pose
    return _arch_pose(env, name)


def _ball_pose(env, name: str, env_idx: int) -> Pose | None:
    specs = getattr(env, "balls_spec", None) or []
    entities = getattr(env, "_ball_entities", None) or []
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


def _arch_pose(env, name: str) -> Pose | None:
    specs = getattr(env, "arches_spec", None)
    xs = getattr(env, "arch_xs", None)
    ys = getattr(env, "arch_ys", None)
    if not specs or xs is None or ys is None:
        return None
    for spec, x, y in zip(specs, xs, ys):
        if spec.get("name") == name:
            return Pose(float(x), float(y))
    return None
