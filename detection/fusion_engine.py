"""
IAMARS — FusionEngine
Weighted Box Fusion (WBF) across the 3 detection models.

TUNING FIX:
  - iou_thr lowered from 0.45 to 0.35: boxes from 3 models were not
    overlapping enough at 0.45, causing 1 drone to appear as 2 fused boxes.
  - skip_box_thr lowered from 0.05 to 0.01: keep all weak detections
    so ByteTrack has consistent input every frame.
"""

import numpy as np


class FusionEngine:
    def __init__(self, iou_thr: float = 0.35, skip_box_thr: float = 0.01):
        self.iou_thr      = iou_thr
        self.skip_box_thr = skip_box_thr

    def fuse(self, model_results: list[dict], frame_shape: tuple) -> dict:
        H, W = frame_shape[:2]
        if H == 0 or W == 0:
            return self._empty()

        all_boxes_norm = []
        all_scores     = []
        all_labels     = []
        all_weights    = []

        for r in model_results:
            boxes  = r["boxes"]
            scores = r["scores"]
            labels = r["labels"]
            weight = float(r["weight"])

            if len(boxes) == 0:
                all_boxes_norm.append(np.empty((0, 4), dtype=np.float32))
                all_scores.append(np.empty((0,),   dtype=np.float32))
                all_labels.append(np.empty((0,),   dtype=np.int32))
                all_weights.append(weight)
                continue

            norm = boxes.copy().astype(np.float32)
            norm[:, [0, 2]] /= W
            norm[:, [1, 3]] /= H
            norm = np.clip(norm, 0.0, 1.0)

            all_boxes_norm.append(norm)
            all_scores.append(scores.astype(np.float32))
            all_labels.append(labels.astype(np.int32))
            all_weights.append(weight)

        fused_norm, fused_scores, fused_labels = self._wbf(
            all_boxes_norm, all_scores, all_labels, all_weights
        )

        if len(fused_norm) == 0:
            return self._empty()

        fused_boxes = fused_norm.copy()
        fused_boxes[:, [0, 2]] *= W
        fused_boxes[:, [1, 3]] *= H

        return {
            "boxes":  fused_boxes.astype(np.float32),
            "scores": fused_scores.astype(np.float32),
            "labels": fused_labels.astype(np.int32),
        }

    def _wbf(self, boxes_list, scores_list, labels_list, weights):
        pool_boxes   = []
        pool_scores  = []
        pool_labels  = []

        for boxes, scores, labels, w in zip(
            boxes_list, scores_list, labels_list, weights
        ):
            for b, s, l in zip(boxes, scores, labels):
                pool_boxes.append(b)
                pool_scores.append(float(s) * w)
                pool_labels.append(int(l))

        if not pool_boxes:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,))

        pool_boxes  = np.array(pool_boxes,  dtype=np.float32)
        pool_scores = np.array(pool_scores, dtype=np.float32)
        pool_labels = np.array(pool_labels, dtype=np.int32)

        order = np.argsort(-pool_scores)
        pool_boxes  = pool_boxes[order]
        pool_scores = pool_scores[order]
        pool_labels = pool_labels[order]

        used     = np.zeros(len(pool_boxes), dtype=bool)
        clusters = []

        for i in range(len(pool_boxes)):
            if used[i]:
                continue
            cluster = [i]
            used[i] = True
            for j in range(i + 1, len(pool_boxes)):
                if not used[j]:
                    if self._iou(pool_boxes[i], pool_boxes[j]) >= self.iou_thr:
                        cluster.append(j)
                        used[j] = True
            clusters.append(cluster)

        fused_boxes  = []
        fused_scores = []
        fused_labels = []

        for cluster in clusters:
            idxs     = np.array(cluster)
            c_boxes  = pool_boxes[idxs]
            c_scores = pool_scores[idxs]
            c_labels = pool_labels[idxs]

            total_weight = c_scores.sum()
            if total_weight == 0:
                continue

            w_norm  = c_scores / total_weight
            fused_b = (c_boxes * w_norm[:, None]).sum(axis=0)
            fused_s = c_scores.sum() / len(weights)
            fused_l = int(np.bincount(c_labels).argmax())

            if fused_s >= self.skip_box_thr:
                fused_boxes.append(fused_b)
                fused_scores.append(fused_s)
                fused_labels.append(fused_l)

        if not fused_boxes:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,))

        return (
            np.array(fused_boxes,  dtype=np.float32),
            np.array(fused_scores, dtype=np.float32),
            np.array(fused_labels, dtype=np.int32),
        )

    @staticmethod
    def _iou(a, b):
        x1 = max(a[0], b[0]);  y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]);  y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union  = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _empty():
        return {
            "boxes":  np.empty((0, 4), dtype=np.float32),
            "scores": np.empty((0,),   dtype=np.float32),
            "labels": np.empty((0,),   dtype=np.int32),
        }