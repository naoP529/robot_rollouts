from cap.state import Pose, RolloutState, require_instance_id


def get_loc(state: RolloutState, obj_id: str) -> Pose | None:
    if obj_id == "robot":
        return state.robot.pose
    require_instance_id(obj_id, "get_loc(obj_id=...)")
    item = state.find_observed(obj_id)
    if item is None:
        return None
    return item.pose
