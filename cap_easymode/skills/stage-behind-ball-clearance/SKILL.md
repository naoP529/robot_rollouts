---
name: stage-behind-ball-clearance
description: "Keep >=0.9m standoff and route detour waypoints behind the ball's rear line so goto does not clip the ball and chase it away."
---

## Problem

When the robot needs to move from the target side of a ball to a staging point behind it (along `-u`, where `u` is the unit vector from the ball to the target), using a small standoff (`0.55 m`) or a single shallow detour point (`bx + 0.65 * p - 0.2 * u`) causes the robot to clip the ball during `goto`.

In the `(u, p)` frame centered on the ball, the straight segment from `(-0.2, 0.65)` to `(-0.55, 0)` passes within `0.48 m` of the ball center. Because the combined robot and ball collision radius is larger than `0.6 m`, the robot strikes the ball while trying to get behind it, knocking the ball further away along `-u` and triggering a runaway chase loop.

## Technique

1. **Use sufficient clearance for staging, but do not re-trigger detours when already behind the ball:**
   - Set the staging standoff behind the ball to `0.9 m - 1.0 m` (`stage = ball - u * standoff`).
   - If the robot is already aligned behind the ball (`-1.30 <= dot_u <= -0.30` and `abs(dot_p) <= 0.30`), only call `look_at(target_yaw)` and return without calling `goto`. Including `-0.45 < dot_u <= -0.30` and `abs(dot_p) <= 0.30` is critical immediately after `nudge`, when the robot finishes `0.30 m - 0.45 m` behind the ball: calling `look_at` keeps the robot in place while any residual ball motion settles, instead of triggering a backward multi-waypoint detour around an in-flight ball.
   - Do **not** use `seg_point_dist(robot, stage, ball) < 0.90`, `dot_u > -standoff`, or an unconditional `dot_u > -0.55` check to trigger a lateral detour when the robot is already behind the ball (`dot_u <= -0.30` and `abs(dot_p) <= 0.40`). Moving from `(-0.35, 0.35)` directly to `(-0.95, 0)` moves away from the ball in the `-u` half-plane; forcing a `1.0 m` lateral detour to `w2` wastes two `goto` calls on every re-stage.

2. **Place the detour waypoint behind the ball's lateral line only when approaching from the front or side:**
   - Choose the lateral unit vector `p = (-uy, ux)` on the side closer to the robot: if `(robot - ball) · p < 0`, negate `p`.
   - If the robot starts on the `+u` side of the ball (`dot_u > 0`), first move to a wide lateral waypoint `ball + p * 1.0 + u * 0.2`, then to a rear-side waypoint `ball + p * 1.0 - u * standoff`, and only then to `stage = ball - u * standoff`.
   - If the robot is laterally offset (`abs(dot_p) > 0.40`, or `-0.55 < dot_u <= 0` and `abs(dot_p) > 0.30`), use the rear-side waypoint `ball + p * 1.0 - u * standoff` before moving to `stage`.
   - Otherwise (`dot_u <= -0.30` and `abs(dot_p) <= 0.40`), move directly to `stage = ball - u * standoff` without detouring out to `w2`.
