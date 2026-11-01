---
name: nearest-match-tracking
description: "How to keep tracking one physical object across re-detections when detect/search mint new, non-persistent IDs."
---

# Tracking an Object Across Re-Detections

Detection IDs (e.g., `ball_yellow_12`, `ball_yellow_76`) are not persistent across calls — new
calls often mint new IDs for the same physical object, and every `detect()` call can also return
up to ~14 noisy false positives.

1. **Always `look_at` the expected position before calling `detect()`:**
   - `detect()` is directional. Before refining or checking an object's position (ball or arch),
     orient toward it with `look_at(angle_to(robot_xy, expected_xy))`.

2. **Use a tight spatial gate when looking directly at the tracked object:**
   - Compute `get_loc(id)` for all returned IDs and pick the one closest to `expected_xy`.
   - When staged right behind the ball (`~0.7 m` away) looking straight at `ball_xy`, require a
     detection within `0.65 m` of `ball_xy` (and `< 1.4 m` from the robot) before nudging.
   - Immediately after a nudge, set `expected_xy` along the push direction (`~0.4–0.6 m` forward),
     `look_at(expected_xy)`, and match within `~1.0 m`.

3. **Never chain unverified wide-gate re-detections (avoid phantom random walks):**
   - If the ball is missing when looking directly at `ball_xy` or `expected_xy`, do **not** sweep
     multiple camera angles with a wide `1.5–2.2 m` gate and accept a single unverified hit. Across
     6–7 `detect()` calls and a `2 m` radius, a random false positive will almost always appear,
     causing `ball_xy` to random-walk across empty space on phantoms.
   - Instead, if an angular sweep finds a candidate at `cand_xy`, `look_at(cand_xy)` (moving closer
     if `> 1.8 m` away) and require **at least 2 consistent detections** within `0.5 m` of
     `cand_xy` across 2–3 `detect()` calls before accepting it.
   - If no candidate passes verification (or pre-nudge confirmation fails twice in a row), treat the
     track as lost and run full multi-hit search and verification. Likewise, if a target arch is not
     consistently detected when within `3 m` and looking straight at `arch_xy`, invalidate `arch_xy`
     and re-find the arch.
