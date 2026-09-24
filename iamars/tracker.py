"""Stage 2: multi-object tracking with ByteTrack (supervision implementation).

ByteTrack in one paragraph: each existing track has a Kalman-predicted box
for the current frame. Detections are split into high-confidence
(> ``high_thresh``) and low-confidence (0.1 to ``high_thresh``). First, tracks
are matched to high-confidence boxes by IoU using the Hungarian algorithm.
Then tracks still unmatched get a second chance against the low-confidence
boxes, which recovers drones that are blurred or partly hidden. A new track
starts only from a high-confidence box. A track that goes unmatched is
kept "lost" for ``track_buffer`` frames before it is deleted.

Fixes compared with the previous version (see git history):
* Real detector confidences are passed through. The old code rescaled every
  frame's scores to [0.60, 0.95], so even the weakest false positive looked
  confident and the high/low split above never happened.
* ``match_thresh`` is a *distance* (1 - IoU) in supervision, so a larger
  value is more permissive. The old comment claimed the opposite and set
  0.2 (IoU >= 0.8 required), which caused ID switches.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from iamars import config
from iamars.detector import Detections


@dataclass
class Tracks:
    """Tracked boxes in one frame. ``ids`` are stable across frames."""

    xyxy: np.ndarray
    conf: np.ndarray
    ids: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)


class ByteTracker:
    def __init__(
        self,
        frame_rate: float = 30.0,
        high_thresh: float = config.TRACK_HIGH_THRESH,
        track_buffer: int = config.TRACK_BUFFER,
        match_thresh: float = config.TRACK_MATCH_THRESH,
    ):
        import supervision as sv

        self._sv = sv
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tracker = sv.ByteTrack(
                track_activation_threshold=high_thresh,
                lost_track_buffer=track_buffer,
                minimum_matching_threshold=match_thresh,
                frame_rate=frame_rate,
                minimum_consecutive_frames=1,
            )

    def update(self, dets: Detections) -> Tracks:
        sv = self._sv
        if len(dets) == 0:
            sv_dets = sv.Detections.empty()
        else:
            sv_dets = sv.Detections(
                xyxy=dets.xyxy.astype(np.float32),
                confidence=dets.conf.astype(np.float32),
                class_id=np.zeros(len(dets), dtype=int),
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self.tracker.update_with_detections(sv_dets)

        if len(out) == 0 or out.tracker_id is None:
            return Tracks(
                np.empty((0, 4), np.float32), np.empty((0,), np.float32), np.empty((0,), int)
            )
        return Tracks(
            xyxy=out.xyxy.astype(np.float32),
            conf=(out.confidence if out.confidence is not None else np.ones(len(out))).astype(
                np.float32
            ),
            ids=out.tracker_id.astype(int),
        )

    def reset(self) -> None:
        self.tracker.reset()
