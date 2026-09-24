"""End-to-end video pipeline: detect -> track -> estimate -> predict.

``Pipeline.process(frame)`` runs all stages on one frame and returns a
:class:`FrameResult` with the tracks and how long each stage took.

The intercept stage is *not* run on video: a single camera gives
directions but not distances, so metric intercept points would be made up.
It is evaluated in 3D simulation instead (scripts/eval_intercept_sim.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from iamars import config
from iamars.detector import Detections
from iamars.estimator import BoxKalmanManager
from iamars.predictor import predict_constant_velocity
from iamars.tracker import ByteTracker


@dataclass
class TrackState:
    track_id: int
    box: np.ndarray            # (4,) Kalman-smoothed xyxy, pixels
    conf: float
    velocity: np.ndarray       # (2,) pixels / second
    trajectory: np.ndarray     # (H, 2) predicted future centres, pixels
    age: int                   # frames since the track started


@dataclass
class FrameResult:
    frame_index: int
    detections: Detections
    tracks: list[TrackState] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)


class Pipeline:
    def __init__(
        self,
        detector,
        fps: float = 30.0,
        horizon_frames: int = config.PREDICT_HORIZON,
        tracker_kwargs: dict | None = None,
    ):
        self.detector = detector
        self.fps = fps
        self.dt = 1.0 / fps
        self.horizon_frames = horizon_frames
        self.tracker = ByteTracker(frame_rate=fps, **(tracker_kwargs or {}))
        self.estimator = BoxKalmanManager(fps=fps)
        self.frame_index = 0

    def process(self, frame: np.ndarray) -> FrameResult:
        t = {}
        t0 = time.perf_counter()
        dets = self.detector.detect(frame)
        t1 = time.perf_counter()
        tracks = self.tracker.update(dets)
        t2 = time.perf_counter()
        states = self.estimator.update(tracks.ids, tracks.xyxy)
        t3 = time.perf_counter()

        out = []
        for tid, conf in zip(tracks.ids, tracks.conf):
            x = states[int(tid)]  # [cx, cy, w, h, vx, vy]
            cx, cy, w, h, vx, vy = x
            traj = predict_constant_velocity(
                [cx, cy], [vx, vy], self.horizon_frames * self.dt, self.dt
            )
            out.append(TrackState(
                track_id=int(tid),
                box=np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32),
                conf=float(conf),
                velocity=np.array([vx, vy], np.float32),
                trajectory=traj.astype(np.float32),
                age=self.estimator.age[int(tid)],
            ))
        t4 = time.perf_counter()

        t["detect"] = (t1 - t0) * 1e3
        t["track"] = (t2 - t1) * 1e3
        t["estimate"] = (t3 - t2) * 1e3
        t["predict"] = (t4 - t3) * 1e3
        t["total"] = (t4 - t0) * 1e3
        res = FrameResult(self.frame_index, dets, out, t)
        self.frame_index += 1
        return res

    def reset(self) -> None:
        self.tracker.reset()
        self.estimator.reset()
        self.frame_index = 0
