"""
IAMARS — ByteTracker
Wraps supervision's ByteTrack for stable IDs across frames.

TUNING FIX for continuous tracking:
  - lost_track_buffer raised to 90 frames (4.5s at 20fps): gives
    drone time to reappear after brief occlusion without losing ID.
  - minimum_matching_threshold lowered to 0.2: less strict IoU
    matching so a slightly repositioned box still matches the track.
  - Score boosting retained: WBF scores (~0.07-0.15) are scaled to
    [0.6, 0.95] so ByteTrack never rejects them.
"""

import numpy as np
import warnings
import supervision as sv

BOOST_MIN = 0.60
BOOST_MAX = 0.95


class ByteTracker:
    def __init__(
        self,
        minimum_matching_threshold: float = 0.2,
        lost_track_buffer: int            = 90,
        minimum_consecutive_frames: int   = 1,
        frame_rate: int                   = 20,
        track_activation_threshold: float = 0.25,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.tracker = sv.ByteTrack(
                track_activation_threshold = track_activation_threshold,
                lost_track_buffer          = lost_track_buffer,
                minimum_matching_threshold = minimum_matching_threshold,
                frame_rate                 = float(frame_rate),
                minimum_consecutive_frames = minimum_consecutive_frames,
            )

    def update(self, fused: dict, frame_shape: tuple) -> dict:
        boxes  = fused["boxes"]
        scores = fused["scores"]
        labels = fused["labels"]

        if len(boxes) == 0:
            detections = sv.Detections.empty()
        else:
            boosted = self._boost_scores(scores)
            detections = sv.Detections(
                xyxy       = boxes.astype(np.float32),
                confidence = boosted.astype(np.float32),
                class_id   = labels.astype(np.int32),
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                tracked = self.tracker.update_with_detections(detections)
        except Exception as exc:
            print(f"[WARN] ByteTrack update error: {exc}")
            return self._empty()

        if tracked is None or len(tracked) == 0:
            return self._empty()

        track_ids = (
            tracked.tracker_id.astype(np.int32)
            if tracked.tracker_id is not None
            else np.zeros(len(tracked), dtype=np.int32)
        )

        restored_scores = self._restore_scores(tracked.xyxy, boxes, scores)

        return {
            "boxes":     tracked.xyxy.astype(np.float32),
            "scores":    restored_scores.astype(np.float32),
            "labels":    tracked.class_id.astype(np.int32)
                         if tracked.class_id is not None
                         else np.zeros(len(tracked), dtype=np.int32),
            "track_ids": track_ids,
        }

    def reset(self):
        self.tracker.reset()

    @staticmethod
    def _boost_scores(scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores
        s_min = scores.min()
        s_max = scores.max()
        if s_max - s_min < 1e-6:
            return np.full_like(scores, (BOOST_MIN + BOOST_MAX) / 2)
        normed  = (scores - s_min) / (s_max - s_min)
        boosted = BOOST_MIN + normed * (BOOST_MAX - BOOST_MIN)
        return boosted.astype(np.float32)

    @staticmethod
    def _restore_scores(tracked_boxes, orig_boxes, orig_scores):
        if len(orig_boxes) == 0 or len(tracked_boxes) == 0:
            return np.full(len(tracked_boxes), 0.5, dtype=np.float32)
        restored = np.full(len(tracked_boxes), 0.5, dtype=np.float32)
        for i, tb in enumerate(tracked_boxes):
            best_iou = -1.0
            best_idx = 0
            for j, ob in enumerate(orig_boxes):
                x1 = max(tb[0], ob[0]);  y1 = max(tb[1], ob[1])
                x2 = min(tb[2], ob[2]);  y2 = min(tb[3], ob[3])
                inter = max(0, x2-x1) * max(0, y2-y1)
                if inter == 0:
                    continue
                a1 = (tb[2]-tb[0]) * (tb[3]-tb[1])
                a2 = (ob[2]-ob[0]) * (ob[3]-ob[1])
                iou = inter / (a1 + a2 - inter + 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_iou > 0.05:
                restored[i] = orig_scores[best_idx]
        return restored

    @staticmethod
    def _empty():
        return {
            "boxes":     np.empty((0, 4), dtype=np.float32),
            "scores":    np.empty((0,),   dtype=np.float32),
            "labels":    np.empty((0,),   dtype=np.int32),
            "track_ids": np.empty((0,),   dtype=np.int32),
        }