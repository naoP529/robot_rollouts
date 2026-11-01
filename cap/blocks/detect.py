from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image

from cap.blocks.get_loc import get_loc
from cap.state import Observed, Pose, RolloutState, require_class_name, require_instance_id
from envs.go2env import HEAD_CAM_FOV, HEAD_CAM_LOOKAT, HEAD_CAM_POS, HEAD_CAM_UP

MATCH_DIST = 0.75
DETECT_THRESHOLD = 0.1
NMS_IOU_THRESHOLD = 0.3


def detect(env, owl, state: RolloutState, name: str, prefer_id: str | None = None) -> list[str]:
    require_class_name(name, "detect(name=...)")
    if prefer_id is not None:
        require_instance_id(prefer_id, "detect(prefer_id=...)")
    robot = get_loc(state, "robot")
    if robot is None or robot.yaw is None:
        raise ValueError("robot pose is missing")
    rgb = rgb_of(env)
    return detect_from_capture(env, owl, state, name, [(robot, rgb)], prefer_id)


def detect_from_capture(
    env,
    owl,
    state: RolloutState,
    name: str,
    captures: list[tuple[Pose, np.ndarray]],
    prefer_id: str | None = None,
) -> list[str]:
    """Like detect(), but runs OWLv2 over pre-captured (pose, rgb) frames and merges all hits
    into one associate() call. Used by search() to detect once after a continuous spin instead
    of stopping to detect at every step."""
    query = state.query_of(name)
    if query is None:
        return []
    hits: list[tuple[float, float]] = []
    for robot, rgb in captures:
        if robot is None or robot.yaw is None:
            raise ValueError("robot pose is missing")
        boxes = owl_boxes(owl, rgb, query)
        for box in boxes:
            xy = ground_hit(robot, box, rgb.shape[1], rgb.shape[0], env)
            if xy is not None:
                hits.append(xy)
    return associate(state, name, query, hits, prefer_id)


def _owl_parts(owl):
    if isinstance(owl, tuple):
        return owl
    return owl.processor, owl.model


def rgb_of(env) -> np.ndarray:
    out = env.render(rgb=True, depth=False, segmentation=False, normal=False)
    rgb = out[0] if isinstance(out, tuple) else out
    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.ascontiguousarray(rgb)
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0.0, 1.0)
        if rgb.max() <= 1.0:
            rgb = rgb * 255.0
        rgb = rgb.astype(np.uint8)
    return rgb


def owl_boxes(owl, rgb: np.ndarray, query: str) -> list[list[float]]:
    processor, model = _owl_parts(owl)
    image = Image.fromarray(rgb)
    text_labels = [[query]]
    inputs = processor(text=text_labels, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    target_sizes = torch.tensor([(image.height, image.width)], device=model.device)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=DETECT_THRESHOLD, text_labels=text_labels
    )
    boxes = results[0]["boxes"].detach().cpu()
    scores = results[0]["scores"].detach().cpu()
    keep = _nms(boxes, scores, NMS_IOU_THRESHOLD)
    return boxes[keep].tolist()


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Greedy IoU-based NMS. OWLv2 returns many overlapping boxes per object with no NMS of its own."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    x0, y0, x1, y1 = boxes.unbind(-1)
    areas = (x1 - x0).clamp(min=0) * (y1 - y0).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break
        rest = order[1:]
        xx0 = torch.maximum(x0[i], x0[rest])
        yy0 = torch.maximum(y0[i], y0[rest])
        xx1 = torch.minimum(x1[i], x1[rest])
        yy1 = torch.minimum(y1[i], y1[rest])
        inter = (xx1 - xx0).clamp(min=0) * (yy1 - yy0).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter).clamp(min=1e-8)
        order = rest[iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long)


def _base_z(env) -> float:
    if hasattr(env, "base_pos"):
        return float(env.base_pos[0, 2].detach().cpu())
    pos = env.robot.get_pos()
    if hasattr(pos, "detach"):
        pos = pos.detach().cpu().numpy()
    return float(np.asarray(pos).reshape(-1, 3)[0, 2])


def _head_camera_world(robot: Pose, base_z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yaw = robot.yaw
    c = math.cos(yaw)
    s = math.sin(yaw)

    def to_world(local):
        return np.array(
            [
                robot.x + c * local[0] - s * local[1],
                robot.y + s * local[0] + c * local[1],
                base_z + local[2],
            ],
            dtype=np.float64,
        )

    origin = to_world(HEAD_CAM_POS)
    lookat = to_world(HEAD_CAM_LOOKAT)
    up_world = np.array(
        [
            c * HEAD_CAM_UP[0] - s * HEAD_CAM_UP[1],
            s * HEAD_CAM_UP[0] + c * HEAD_CAM_UP[1],
            HEAD_CAM_UP[2],
        ],
        dtype=np.float64,
    )
    if np.linalg.norm(up_world) < 1e-8:
        up_world = np.array(HEAD_CAM_UP, dtype=np.float64)
    forward = lookat - origin
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up_world)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    return origin, right, down, forward


def ground_hit(robot: Pose, box, width: int, height: int, env) -> tuple[float, float] | None:
    origin, right, down, forward = _head_camera_world(robot, _base_z(env))
    fov = float(getattr(env, "head_cam_fov", getattr(getattr(env, "camera", None), "fov", HEAD_CAM_FOV)))
    fy = (height * 0.5) / math.tan(math.radians(fov) * 0.5)
    fx = fy
    cx = width * 0.5
    cy = height * 0.5
    u = 0.5 * (box[0] + box[2])
    v = box[3]
    direction = (u + 0.5 - cx) / fx * right + (v + 0.5 - cy) / fy * down + forward
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return None
    direction = direction / norm
    if abs(direction[2]) < 1e-8:
        return None
    t = -origin[2] / direction[2]
    if t <= 0:
        return None
    hit = origin + t * direction
    return float(hit[0]), float(hit[1])


def associate(
    state: RolloutState,
    name: str,
    query: str,
    hits: list[tuple[float, float]],
    prefer_id: str | None,
) -> list[str]:
    updated: list[str] = []
    remaining = list(enumerate(hits))
    used_obs: set[str] = set()
    used_det: set[int] = set()

    if prefer_id is not None and remaining:
        target = state.find_observed(prefer_id)
        if target is not None:
            det_i, xy = min(
                remaining,
                key=lambda item: math.hypot(item[1][0] - target.pose.x, item[1][1] - target.pose.y),
            )
            if math.hypot(xy[0] - target.pose.x, xy[1] - target.pose.y) <= MATCH_DIST:
                target.pose = Pose(xy[0], xy[1])
                updated.append(target.id)
                used_obs.add(target.id)
                used_det.add(det_i)

    pairs = []
    for det_i, xy in remaining:
        if det_i in used_det:
            continue
        for item in state.observed:
            if item.name != name or item.id in used_obs:
                continue
            dist = math.hypot(xy[0] - item.pose.x, xy[1] - item.pose.y)
            if dist <= MATCH_DIST:
                pairs.append((dist, det_i, item))
    pairs.sort(key=lambda row: row[0])
    for _dist, det_i, item in pairs:
        if det_i in used_det or item.id in used_obs:
            continue
        xy = hits[det_i]
        item.pose = Pose(xy[0], xy[1])
        updated.append(item.id)
        used_obs.add(item.id)
        used_det.add(det_i)

    for det_i, xy in remaining:
        if det_i in used_det:
            continue
        new_id = state.next_instance_id(name)
        state.observed.append(Observed(id=new_id, name=name, query=query, pose=Pose(xy[0], xy[1])))
        state.pop_unseen(name)
        updated.append(new_id)
        used_det.add(det_i)
        # OWLv2 post-processing applies no NMS, so one real object can produce many
        # overlapping boxes. Fold any other still-unclaimed hit near this new object
        # into it instead of minting a separate instance per box.
        for other_i, other_xy in remaining:
            if other_i in used_det:
                continue
            if math.hypot(other_xy[0] - xy[0], other_xy[1] - xy[1]) <= MATCH_DIST:
                used_det.add(other_i)
    return updated
