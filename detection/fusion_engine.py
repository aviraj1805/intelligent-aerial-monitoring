"""
IAMARS — FusionEngine
Weighted Box Fusion (WBF) across the 3 detection models.
Audio model output is NOT passed here — it is a flag only.

WBF algorithm (Solovyev et al., 2021):
  1. Normalise all boxes to [0,1] using frame dimensions.
  2. Sort every box by score descending.
  3. Cluster boxes by IoU >= iou_thr.
  4. Fuse each cluster into one box: weighted average of coords & scores.
  5. Denormalise back to pixel coords.
"""

import numpy as np


class FusionEngine:
    """
    Parameters
    ----------
    iou_thr   : float  — IoU threshold for clustering boxes (default 0.45)
    skip_box_thr : float — discard fused boxes with score < this (default 0.05)
    """

    def __init__(self, iou_thr: float = 0.45, skip_box_thr: float = 0.05):
        self.iou_thr      = iou_thr
        self.skip_box_thr = skip_box_thr

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────
    def fuse(
        self,
        model_results: list[dict],
        frame_shape: tuple,          # (H, W, C) or (H, W)
    ) -> dict:
        """
        Parameters
        ----------
        model_results : list of dicts from MultiModelDetector.run_parallel()
            Each dict: {name, boxes (N,4) xyxy, scores (N,), labels (N,), weight}
        frame_shape   : tuple — used to normalise box coordinates

        Returns
        -------
        dict:
            boxes  (M, 4)  — fused xyxy boxes in pixel coords
            scores (M,)    — fused confidence scores
            labels (M,)    — majority-vote class label per fused box
        """
        H, W = frame_shape[:2]

        if H == 0 or W == 0:
            return self._empty()

        # ── Collect all boxes across models ──────────────────────────────────
        all_boxes_norm  = []   # list of (N_i, 4) normalised
        all_scores      = []   # list of (N_i,)
        all_labels      = []   # list of (N_i,)
        all_weights     = []   # list of scalar weights (one per model)

        for r in model_results:
            boxes  = r["boxes"]    # (N, 4) xyxy pixels
            scores = r["scores"]   # (N,)
            labels = r["labels"]   # (N,)
            weight = float(r["weight"])

            if len(boxes) == 0:
                all_boxes_norm.append(np.empty((0, 4), dtype=np.float32))
                all_scores.append(np.empty((0,),   dtype=np.float32))
                all_labels.append(np.empty((0,),   dtype=np.int32))
                all_weights.append(weight)
                continue

            # Normalise to [0, 1]
            norm = boxes.copy().astype(np.float32)
            norm[:, [0, 2]] /= W
            norm[:, [1, 3]] /= H
            norm = np.clip(norm, 0.0, 1.0)

            all_boxes_norm.append(norm)
            all_scores.append(scores.astype(np.float32))
            all_labels.append(labels.astype(np.int32))
            all_weights.append(weight)

        # ── Run WBF ──────────────────────────────────────────────────────────
        fused_norm, fused_scores, fused_labels = self._wbf(
            all_boxes_norm, all_scores, all_labels, all_weights
        )

        if len(fused_norm) == 0:
            return self._empty()

        # ── Denormalise back to pixel coords ─────────────────────────────────
        fused_boxes = fused_norm.copy()
        fused_boxes[:, [0, 2]] *= W
        fused_boxes[:, [1, 3]] *= H

        return {
            "boxes":  fused_boxes.astype(np.float32),
            "scores": fused_scores.astype(np.float32),
            "labels": fused_labels.astype(np.int32),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # WBF core
    # ─────────────────────────────────────────────────────────────────────────
    def _wbf(
        self,
        boxes_list:  list[np.ndarray],   # each (N_i, 4) normalised
        scores_list: list[np.ndarray],   # each (N_i,)
        labels_list: list[np.ndarray],   # each (N_i,)
        weights:     list[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns fused (boxes, scores, labels) — all normalised.
        """
        # Flatten everything into one pool, tagging each box with model weight
        pool_boxes   = []
        pool_scores  = []
        pool_labels  = []
        pool_weights = []

        for boxes, scores, labels, w in zip(
            boxes_list, scores_list, labels_list, weights
        ):
            for b, s, l in zip(boxes, scores, labels):
                pool_boxes.append(b)
                pool_scores.append(float(s) * w)   # weighted score
                pool_labels.append(int(l))
                pool_weights.append(w)

        if not pool_boxes:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,))

        pool_boxes   = np.array(pool_boxes,   dtype=np.float32)   # (T, 4)
        pool_scores  = np.array(pool_scores,  dtype=np.float32)   # (T,)
        pool_labels  = np.array(pool_labels,  dtype=np.int32)     # (T,)
        pool_weights = np.array(pool_weights, dtype=np.float32)   # (T,)

        # Sort by weighted score descending
        order = np.argsort(-pool_scores)
        pool_boxes   = pool_boxes[order]
        pool_scores  = pool_scores[order]
        pool_labels  = pool_labels[order]
        pool_weights = pool_weights[order]

        # Cluster by IoU
        used    = np.zeros(len(pool_boxes), dtype=bool)
        clusters: list[list[int]] = []

        for i in range(len(pool_boxes)):
            if used[i]:
                continue
            cluster = [i]
            used[i] = True
            for j in range(i + 1, len(pool_boxes)):
                if not used[j]:
                    iou = self._iou(pool_boxes[i], pool_boxes[j])
                    if iou >= self.iou_thr:
                        cluster.append(j)
                        used[j] = True
            clusters.append(cluster)

        # Fuse each cluster
        fused_boxes  = []
        fused_scores = []
        fused_labels = []

        for cluster in clusters:
            idxs    = np.array(cluster)
            c_boxes  = pool_boxes[idxs]    # (K, 4)
            c_scores = pool_scores[idxs]   # (K,)
            c_labels = pool_labels[idxs]   # (K,)

            total_weight = c_scores.sum()
            if total_weight == 0:
                continue

            # Weighted average of box coordinates
            w_norm  = c_scores / total_weight
            fused_b = (c_boxes * w_norm[:, None]).sum(axis=0)

            # Fused score: mean of cluster scores (normalised by num models)
            fused_s = c_scores.sum() / len(weights)

            # Majority-vote label
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

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        """IoU of two xyxy boxes."""
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
    def _empty() -> dict:
        return {
            "boxes":  np.empty((0, 4), dtype=np.float32),
            "scores": np.empty((0,),   dtype=np.float32),
            "labels": np.empty((0,),   dtype=np.int32),
        }