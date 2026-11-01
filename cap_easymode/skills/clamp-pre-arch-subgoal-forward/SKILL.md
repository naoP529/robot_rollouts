---
name: clamp-pre-arch-subgoal-forward
description: "When a laterally offset ball is already in front of the arch, clamp the pre-arch alignment subgoal distance to -lon so the push vector never points backward and stages the robot into the arch posts."
---

## Problem

When a ball is in front of an arch line (`lon = (ball - arch) · pass_u < -0.45 m`, e.g. `lon = -0.55 m`) but laterally offset from the opening (`abs(lat) > 0.20 m`), targeting a fixed pre-arch alignment distance `d_entry = 1.0 m` (`subgoal = arch - 1.0 * pass_u`, `lon_target = -1.0 m`) places the subgoal **behind** the ball along `pass_u` (`lon_target < lon`).

Because the push vector `u` points backward (`u · pass_u < 0`), the staging point behind the ball (`stage = ball - standoff * u`) shifts forward onto the `+pass_u` side of the ball (`lon_stage = lon - standoff * (u · pass_u) ≈ -0.05 m`). That places `stage` directly on the line of arches (`lon ≈ 0`), where the target arch's posts or an adjacent arch's posts block the robot from reaching the staging pose.

## Technique

1. **Project the ball into the arch frame `(pass_u, perp_u)`:**
   - Compute `lon = (bx - ax) * pass_ux + (by - ay) * pass_uy` and `lat = (bx - ax) * (-pass_uy) + (by - ay) * pass_ux`.

2. **Clamp the pre-arch alignment distance when `lon <= -0.45 m`:**
   - When the ball is already in front of the arch with sufficient room to turn into the opening (`lon <= -0.45 m`) but laterally misaligned (`abs(lat) > 0.20 m`), clamp the alignment distance to `d_align = max(0.50, min(d_entry, -lon))` instead of using a fixed `d_entry = 1.0 m`.
   - Set the alignment subgoal to `arch - d_align * pass_u` so that `lon_target = -d_align >= lon` and `u · pass_u >= 0`.
   - This pushes the ball laterally or diagonally forward onto the arch centerline while keeping the staging point `ball - standoff * u` strictly in front of the ball (`lon_stage <= lon <= -0.45 m`), well clear of the line of arch posts.
   - Reserve backward recovery pushes (`u · pass_u < 0`) only for cases where the ball has drifted too close to or past the arch posts (`lon > -0.45 m` with `abs(lat) > 0.20 m`).
