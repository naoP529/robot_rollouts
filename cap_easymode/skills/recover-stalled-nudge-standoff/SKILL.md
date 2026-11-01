---
name: recover-stalled-nudge-standoff
description: "When a nudge produces zero ball displacement because standoff left the robot short of contact, step closer or increase push duration instead of re-staging to the same standoff."
---

## Problem

When `nudge` is called after `goto(sx, sy, target_yaw)` with a `0.95 m` standoff behind the ball, `goto` stops within its goal tolerance (`~0.15 m`), leaving the robot center `0.50 m - 0.65 m` behind the ball. A short push (`speed=1.0`, `duration_s=0.65`) from that distance often stops right at the ball's edge without making contact, leaving the ball at the exact same coordinates.

If the policy then re-runs its staging logic with a fixed rear threshold (e.g. triggering a detour whenever `dot_u > -0.55`), it backs up to the detour waypoint, returns to `stage`, and repeats the exact same short `nudge` in an infinite no-contact loop.

## Technique

1. **Match push travel to the actual robot-to-ball gap:**
   - Before calling `nudge`, measure the current distance `gap = hypot(bx - rx, by - ry)`.
   - Ensure `speed * duration_s` exceeds `gap + 0.35 m` so the robot both closes the standoff gap and drives through the ball.

2. **Detect zero-displacement nudges and step closer:**
   - Record `(bx_prev, by_prev)` before `nudge` and compare with `(bx_now, by_now)` afterward.
   - If `hypot(bx_now - bx_prev, by_now - by_prev) < 0.03`:
     - The robot is already aligned behind the ball (`abs(dot_p) < 0.15`, `dot_u < 0`), so **do not** retreat to a detour waypoint or back out to `0.95 m` standoff.
     - Instead, either immediately issue a longer/faster `nudge` from the current pose, or `goto` a closer staging point (`standoff = 0.55 m - 0.65 m`) directly along `-u` before nudging.
