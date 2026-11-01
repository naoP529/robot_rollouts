---
name: infer-arch-pass-axis-without-yaw
description: "When an arch pose has no yaw, infer its pass-through normal from the line of arches or cardinal axis rather than the initial ball-to-arch diagonal."
---

## Problem

`get_loc(arch_name)` returns `x` and `y` without `yaw` (`yaw` is `None`). If a policy infers the arch's pass-through axis `pass_u` from the vector between the ball's starting position and the arch center (`arch - ball_0`), any initial diagonal offset tilts `pass_u` relative to the true arch opening.

In Attempt 0, the ball started at `(2.08, -1.27)` and the arch was at `(6.78, -3.60)`, giving `pass_u = (0.89, -0.45)`. In Attempt 1, the script re-initialized `ball_0` from the ball's intermediate position `(6.52, -4.25)`, giving `pass_u = (0.37, 0.93)`—nearly perpendicular to the arch opening. Pushing along these oblique vectors repeatedly drove the ball into the arch posts (`x ≈ 6.3–6.6, y ≈ -4.2` and `y ≈ -2.2`) instead of through the opening.

## Technique

1. **Infer the arch opening axis from the scene's arch layout when `yaw` is `None`:**
   - Query the positions of all arches in the scene. Even when individual arches are staggered in depth (e.g. `x` ranging from `6.7` to `8.9` while `y` spans `-3.6` to `3.6`), their openings remain cardinal-aligned. Compare the overall bounding span (`span_y = max(y) - min(y)` vs `span_x = max(x) - min(x)`) to pick the cardinal pass-through normal (`(±1, 0)` when `span_y >= span_x`, else `(0, ±1)`). Avoid using an un-snapped PCA eigenvector of the arch coordinates, as staggered arch depths will tilt the PCA normal away from the true cardinal opening.
   - If only one arch is known, snap the pass-through axis to the cardinal axis (`(±1, 0)` or `(0, ±1)`) that has the larger component from the workspace center toward the arch, rather than using the raw diagonal `arch - ball` vector.

2. **Never recompute `pass_u` from a displaced ball position on retry:**
   - Compute the arch's pass-through normal `pass_u` from the arch geometry or cardinal alignment, independent of where the ball currently sits.
   - Stage the ball at a pre-arch alignment point `arch - pass_u * d_entry` (with `d_entry ≈ 0.8 m - 1.2 m` along the true normal) until lateral error `abs((ball - arch) · pass_perp)` is under `0.2 m`, then push straight along `pass_u` through `arch + pass_u * 0.9`.
