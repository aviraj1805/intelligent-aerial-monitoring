"""
IAMARS — ByteTracker
Wraps supervision's ByteTrack to assign stable IDs across frames.

ByteTrack parameters (per spec):
    minimum_matching_threshold  = 0.3
    lost_track_buffer           = 60
    minimum_consecutive_frames  = 1
"""

import numpy as np
import supervision as sv


class ByteTracker:
    """
    Thin wrapper around supervision.ByteTrack.

    Usage
    -----
    tracker = ByteTracker()
    tracked = tracker.update(fused_result, frame_shape)
    # tracked dict: {boxes (M,4), scores (M,), labels (M,), track_ids (M,)}
    """

    def __init__(
        self,
        minimum_matching_threshold: float = 0.3,
        lost_track_buffer: int            = 60,
        minimum_consecutive_frames: int   = 1,
        frame_rate: int                   = 20,
    ):
        self.tracker = sv.ByteTrack(
            minimum_matching_threshold  = minimum_matching_threshold,
            lost_track_buffer           = lost_track_buffer,
            minimum_consecutive_frames  = minimum_consecutive_frames,
            frame_rate                  = frame_rate,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────────────────────────────────────
    def update(self, fused: dict, frame_shape: tuple) -> dict:
        """
        Parameters
        ----------
        fused       : dict from FusionEngine.fuse()
                      {boxes (M,4) xyxy pixels, scores (M,), labels (M,)}
        frame_shape : (H, W, ...) of the current frame

        Returns
        -------
        dict:
            boxes     (K, 4)  — tracked boxes xyxy pixels
            scores    (K,)    — confidence scores
            labels    (K,)    — class labels
            track_ids (K,)    — stable integer track IDs
        """
        boxes  = fused["boxes"]    # (M, 4)
        scores = fused["scores"]   # (M,)
        labels = fused["labels"]   # (M,)

        # ── Build supervision Detections object ──────────────────────────────
        if len(boxes) == 0:
            detections = sv.Detections.empty()
        else:
            detections = sv.Detections(
                xyxy       = boxes.astype(np.float32),
                confidence = scores.astype(np.float32),
                class_id   = labels.astype(np.int32),
            )

        # ── Run ByteTrack update ─────────────────────────────────────────────
        try:
            tracked: sv.Detections = self.tracker.update_with_detections(detections)
        except Exception as exc:
            print(f"[WARN] ByteTrack update error: {exc}")
            return self._empty()

        # ── Extract results ───────────────────────────────────────────────────
        if tracked is None or len(tracked) == 0:
            return self._empty()

        track_ids = (
            tracked.tracker_id.astype(np.int32)
            if tracked.tracker_id is not None
            else np.zeros(len(tracked), dtype=np.int32)
        )

        return {
            "boxes":     tracked.xyxy.astype(np.float32),
            "scores":    tracked.confidence.astype(np.float32)
                         if tracked.confidence is not None
                         else np.zeros(len(tracked), dtype=np.float32),
            "labels":    tracked.class_id.astype(np.int32)
                         if tracked.class_id is not None
                         else np.zeros(len(tracked), dtype=np.int32),
            "track_ids": track_ids,
        }

    def reset(self) -> None:
        """Reset tracker state (e.g. between video clips)."""
        self.tracker.reset()

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _empty() -> dict:
        return {
            "boxes":     np.empty((0, 4), dtype=np.float32),
            "scores":    np.empty((0,),   dtype=np.float32),
            "labels":    np.empty((0,),   dtype=np.int32),
            "track_ids": np.empty((0,),   dtype=np.int32),
        }