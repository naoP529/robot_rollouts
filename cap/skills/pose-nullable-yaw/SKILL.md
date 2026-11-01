---
name: pose-nullable-yaw
description: "get_loc's yaw is None for non-oriented objects; never pass it to float() unchecked."
---

# Handling Nullable Yaw in Poses

`get_loc(obj_id)` returns an object or dictionary with `x`, `y`, and `yaw`, where `yaw` is `None`
for non-oriented objects (such as balls or arches) and a float only for `"robot"`.

Never pass `pose.yaw` or `pose.get("yaw", 0.0)` directly to `float()` without handling `None`
(e.g., use `float(pose.yaw or 0.0)` or `float(pose.get("yaw") or 0.0)`).
