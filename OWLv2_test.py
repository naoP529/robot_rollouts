import argparse
from collections import defaultdict
from pathlib import Path

import genesis as gs
import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from envs.cfg.model_data import ARCHES, BALLS
from envs.test_env import TestEnv
from utils.seg_bbox import bboxes_from_seg

# name, query, path. nameはscene.entitiesのnameと前方一致
ALL_OBJ = BALLS + ARCHES
IOU_THRESH = 0.5


def parse_res(value: str) -> tuple[int, int]:
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def as_image(arr):
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 4:
        arr = arr[0]
    return arr


def iou(box_a, box_b) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(gt_boxes, pred_boxes, iou_thresh=IOU_THRESH) -> int:
    pairs = []
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            pairs.append((iou(gt, pred), i, j))
    pairs.sort(reverse=True)
    used_gt = set()
    used_pred = set()
    matched = 0
    for score, i, j in pairs:
        if score < iou_thresh:
            break
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i)
        used_pred.add(j)
        matched += 1
    return matched


def gt_from_seg(env, seg) -> list[dict]:
    query_list = []
    entities_list = env.scene.entities
    # bboxのid == scene.entitiesのidx+1
    for b in bboxes_from_seg(as_image(seg)):
        idx = b["id"] - 1
        entity = next((e for e in entities_list if e.idx == idx), None)
        if entity is None:
            continue
        e_name = entity.name
        obj_data = next((o for o in ALL_OBJ if e_name.startswith(o["name"])), None)
        if obj_data is None:
            continue
        query_list.append(
            {
                "idx": idx,
                "name": obj_data["name"],
                "entity_name": e_name,
                "query": obj_data["query"],
                "correct_bbox": [b["x_min"], b["y_min"], b["x_max"], b["y_max"]],
            }
        )
    return query_list


def detect(processor, model, image, queries):
    if not queries:
        return [], []
    text_labels = [queries]
    inputs = processor(text=text_labels, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    target_sizes = torch.tensor([(image.height, image.width)], device=model.device)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=text_labels
    )
    result = results[0]
    pred_boxes = result["boxes"].detach().cpu().tolist()
    pred_labels = result["text_labels"]
    return pred_boxes, pred_labels


def score_detections(query_list, pred_boxes, pred_labels):
    gt_by_query = defaultdict(list)
    for q in query_list:
        gt_by_query[q["query"]].append(q["correct_bbox"])

    pred_by_query = defaultdict(list)
    for box, label in zip(pred_boxes, pred_labels):
        pred_by_query[label].append(box)

    n_gt = 0
    n_pred = 0
    n_matched = 0
    per_query = []
    for query in sorted(set(gt_by_query) | set(pred_by_query)):
        gts = gt_by_query[query]
        preds = pred_by_query[query]
        matched = match_boxes(gts, preds)
        n_gt += len(gts)
        n_pred += len(preds)
        n_matched += matched
        per_query.append((query, len(gts), len(preds), matched))

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_matched": n_matched,
        "recall": n_matched / n_gt if n_gt else 0.0,
        "precision": n_matched / n_pred if n_pred else 0.0,
        "per_query": per_query,
    }


def print_score(prefix, result):
    print(
        f"{prefix}recall={result['recall']:.3f}  precision={result['precision']:.3f}  "
        f"matched={result['n_matched']}  n_gt={result['n_gt']}  n_pred={result['n_pred']}"
    )
    for query, q_gt, q_pred, q_matched in result["per_query"]:
        print(f"  {query}  n_gt={q_gt}  n_pred={q_pred}  matched={q_matched}")


def evaluate_seed(seed, res, processor, model):
    gs.init(backend=gs.gpu)
    try:
        env = TestEnv(seed=seed, res=res)
        rgb, _, seg, _ = env.render(depth=False, normal=False)
        query_list = gt_from_seg(env, seg)
        image = Image.fromarray(as_image(rgb))
        queries = sorted({q["query"] for q in query_list})
        pred_boxes, pred_labels = detect(processor, model, image, queries)
        result = score_detections(query_list, pred_boxes, pred_labels)
        result["seed"] = seed
        return result
    finally:
        gs.destroy()


def aggregate(results):
    n_gt = sum(r["n_gt"] for r in results)
    n_pred = sum(r["n_pred"] for r in results)
    n_matched = sum(r["n_matched"] for r in results)
    by_query = defaultdict(lambda: [0, 0, 0])
    for r in results:
        for query, q_gt, q_pred, q_matched in r["per_query"]:
            by_query[query][0] += q_gt
            by_query[query][1] += q_pred
            by_query[query][2] += q_matched
    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_matched": n_matched,
        "recall": n_matched / n_gt if n_gt else 0.0,
        "precision": n_matched / n_pred if n_pred else 0.0,
        "per_query": [(q, *by_query[q]) for q in sorted(by_query)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    parser.add_argument("--n", type=int, default=8, help="number of scenes")
    parser.add_argument("--res", type=parse_res, default=parse_res("1920x1080"))
    parser.add_argument("--out-dir", type=Path, default=Path("captures"))
    args = parser.parse_args()
    seeds = list(range(args.seed, args.seed + args.n))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    model.to(device)
    model.eval()

    results = []
    for seed in seeds:
        print(f"=== seed {seed} ===")
        result = evaluate_seed(seed, args.res, processor, model)
        print_score("", result)
        results.append(result)

    print("=== aggregate ===")
    print_score("", aggregate(results))


if __name__ == "__main__":
    main()
