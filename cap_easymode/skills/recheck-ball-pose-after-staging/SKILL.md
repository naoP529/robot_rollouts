---
name: recheck-ball-pose-after-staging
description: "Re-read the ball pose after staging and skip nudge to re-stage if the ball rolled away while the robot was repositioning."
---

## Problem

Immediately after `nudge` returns, the ball often still has momentum and continues rolling for `1.5 m - 2.5 m` while the robot executes `goto` waypoints to stage for the next push.

If the policy reads the ball pose once after `nudge`, plans and executes a multi-waypoint staging sequence around that in-flight position, and then immediately calls `nudge` without checking whether the ball moved during staging:
- The ball may have rolled several meters away from the staged position, so `nudge` drives the robot through empty space and stops short of the ball.
- Or the ball may have rolled past the robot's staged position, so `nudge` strikes the ball from the wrong side or along an obsolete push vector.

## Technique

1. **Re-read the ball pose after staging and abort `nudge` if the ball drifted:**
   - Record `(bx_stage, by_stage)` before calling the staging `goto` sequence.
   - Once staging completes, re-read `(bx, by)` and `(rx, ry)`.
   - If the ball moved significantly during staging (`math.hypot(bx - bx_stage, by - by_stage) > 0.25`), do not execute `nudge`; `continue` the control loop to recompute the subgoal and re-stage behind the ball's settled position.

2. **Avoid infinite re-staging loops or mode flips when the ball did not move:**
   - If the ball did **not** move significantly during staging (`math.hypot(bx - bx_stage, by - by_stage) <= 0.25`), keep the staged subgoal `(tx, ty, mode)` rather than re-running discrete subgoal mode selection on `(bx, by)`. Re-evaluating a hard mode boundary (such as `abs(lat) <= 0.20` between `"through"` and `"align"`) after staging allows millimeter-level ball drift to switch modes, rotate `u` by `90°`, and trigger a cross-ball `goto` right after staging.
   - Recompute the unit vector `u` from the current ball position `(bx, by)` to the preserved target `(tx, ty)` and check `rel = (rx - bx, ry - by)`:
     - Longitudinal projection: `dot_u = rel_x * ux + rel_y * uy`
     - Lateral offset: `dot_p = rel_x * (-uy) + rel_y * ux`
   - Do **not** require a tight `abs(dot_p) <= 0.22` gate that skips `nudge` and repeats the exact same `0.95 m` staging call. `goto` stopping tolerance or an obstacle near the `0.95 m` staging point can leave the robot at `abs(dot_p) ≈ 0.25 m` on every attempt, causing an infinite `goto` loop with zero nudges until timeout.
   - Instead, when the ball did not move:
     - If `-1.45 <= dot_u <= -0.30` and `abs(dot_p) <= 0.35`, proceed directly with `nudge`.
     - If `abs(dot_p) > 0.35` or `dot_u > -0.30` after `0.95 m` staging (e.g., because the `0.95 m` staging point was blocked by an arch post or another obstacle), step directly to a closer standoff (`bx - ux * 0.60, by - uy * 0.60`) and then execute `nudge` rather than `continue`-ing to repeat the `0.95 m` staging sequence.
