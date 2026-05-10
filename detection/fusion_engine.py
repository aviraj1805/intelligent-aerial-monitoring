import numpy as np
from ensemble_boxes import weighted_boxes_fusion


class FusionEngine:
    def __init__(self, config_path="model_configs.yaml"):
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.iou_thr = config["fusion"]["iou_threshold"]
        self.skip_thr = config["fusion"]["skip_box_threshold"]

    def fuse(self, model_results, frame_shape):
        H, W = frame_shape[:2]
        boxes_list, scores_list, labels_list, weights = [], [], [], []

        for r in model_results:
            if r is None or len(r["boxes"]) == 0:
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])
            else:
                b = r["boxes"].copy().astype(float)
                b[:, [0, 2]] /= W
                b[:, [1, 3]] /= H
                b = np.clip(b, 0, 1)
                boxes_list.append(b.tolist())
                scores_list.append(r["scores"].tolist())
                labels_list.append([0] * len(b))

            weights.append(r["weight"])

        # No detections from any model
        if all(len(b) == 0 for b in boxes_list):
            return np.array([]), np.array([])

        boxes, scores, _ = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights,
            iou_thr=self.iou_thr,
            skip_box_thr=self.skip_thr
        )

        # Denormalize back to pixel coordinates
        boxes[:, [0, 2]] *= W
        boxes[:, [1, 3]] *= H

        return boxes, scores