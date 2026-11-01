---
name: noisy-detection-verification
description: "How to confirm a detect/search candidate is real, not a far-field phantom, before acting on it."
---

# Verifying Noisy Multi-Candidate Detections

`detect(name)` and `search(name)` return up to ~14 candidate IDs per call, including far-field
phantoms (30–300+ m away) and nearby random false positives.

Never assume the first returned ID or a single close-range hit is real:

1. **Cluster and sort by hit count before distance:**
   - Ignore obviously out-of-bounds detections (e.g., `> 20 m` from the robot).
   - Merge detections from `search(name)` / `detect(name)` within `~0.6 m` into clusters, updating
     each cluster's `(x, y)` mean and `count`.
   - Sort clusters primarily by hit count (e.g., `count >= 2` before `count == 1`, then higher
     `count` first) and only secondarily by distance to the robot. Sorting mainly by distance
     causes the robot to test many nearby `count == 1` phantoms before checking a real object 5–10 m
     away that was detected multiple times across the sweep.

2. **Require multi-call consistency at close standoff:**
   - Move to a close standoff (`1.2–1.5 m` from cluster `c_xy`) and `look_at` `c_xy`.
   - Call `detect(name, prefer_id=cid)` 3 times (e.g., at `look_yaw` and `look_yaw ± 0.15` rad).
   - For each call, find the returned detection closest to `c_xy` within `0.65 m` of `c_xy` and
     within `standoff + 0.8 m` of the robot.
   - **Confirm only if at least 2 of the 3 calls** return a qualifying detection and those
     detections agree with each other (within `~0.4 m`). Accepting a single hit within `1.0 m` out
     of 3 calls will frequently lock onto random noise.

3. **Expand patrol coverage if initial search finds nothing verified:**
   - Objects (especially arches) can be 8–10 m from the origin. If searching near the start pose
     yields no verified cluster, include mid-field patrol points (e.g., `x = 4..7 m`, `y = -3..3 m`)
     rather than patrolling only around `(±2, ±2)`.
