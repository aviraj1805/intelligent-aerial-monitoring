"""Weighted box fusion (WBF) for combining several detectors.

Uses the reference implementation from the ``ensemble-boxes`` package
(Solovyev et al., 2021) instead of the earlier hand-written version.

How it works, in short: boxes from all models that overlap by more than
``iou_thr`` are grouped. Each group becomes one box whose corners are the
confidence-weighted average of the group. The fused confidence is the
average confidence, scaled down when fewer models agree. So a box that all
models find scores higher than a box only one model finds.
"""

from __future__ import annotations

import numpy as np


def weighted_box_fusion(
    boxes_per_model: list[np.ndarray],
    scores_per_model: list[np.ndarray],
    model_weights: list[float],
    image_wh: tuple[int, int],
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse boxes from several models.

    boxes_per_model  : list of (N_i, 4) arrays, pixel xyxy
    scores_per_model : list of (N_i,) arrays
    Returns fused (M, 4) pixel boxes and (M,) scores.
    """
    from ensemble_boxes import weighted_boxes_fusion

    w, h = image_wh
    scale = np.array([w, h, w, h], dtype=np.float32)

    norm_boxes, scores, labels = [], [], []
    for b, s in zip(boxes_per_model, scores_per_model):
        b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
        norm_boxes.append(np.clip(b / scale, 0.0, 1.0).tolist())
        scores.append(np.asarray(s, dtype=np.float32).tolist())
        labels.append([0] * len(b))

    if sum(len(s) for s in scores) == 0:
        return np.empty((0, 4), np.float32), np.empty((0,), np.float32)

    fused, fused_scores, _ = weighted_boxes_fusion(
        norm_boxes,
        scores,
        labels,
        weights=list(model_weights),
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )
    return (np.asarray(fused, np.float32) * scale).astype(np.float32), np.asarray(
        fused_scores, np.float32
    )
