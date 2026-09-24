"""Fusion, ByteTrack wrapper, full pipeline and visualisation (no model weights needed)."""

import numpy as np

from iamars.detector import Detections
from iamars.fusion import weighted_box_fusion
from iamars.pipeline import Pipeline
from iamars.tracker import ByteTracker
from iamars.visualize import Annotator


def moving_box(k, x0=100, y0=80, vx=4, vy=2, size=30):
    return [x0 + vx * k, y0 + vy * k, x0 + vx * k + size, y0 + vy * k + size]


class FakeDetector:
    """Returns scripted detections, so tests run without YOLO weights."""

    name = "fake"

    def __init__(self, frames):
        self.frames = frames
        self.i = 0

    def detect(self, frame):
        boxes, confs = self.frames[self.i]
        self.i += 1
        return Detections(np.array(boxes, np.float32).reshape(-1, 4), np.array(confs, np.float32))


def test_fusion_merges_agreeing_boxes():
    a = np.array([[100, 100, 140, 140]], np.float32)
    b = np.array([[102, 101, 142, 141]], np.float32)
    boxes, scores = weighted_box_fusion([a, b], [np.array([0.9]), np.array([0.8])], [1, 1], (640, 480))
    assert len(boxes) == 1
    assert np.allclose(boxes[0], [101, 100.5, 141, 140.5], atol=1.5)


def test_fusion_penalises_single_model_boxes():
    a = np.array([[100, 100, 140, 140]], np.float32)
    empty = np.empty((0, 4), np.float32)
    _, both = weighted_box_fusion([a, a], [np.array([0.9]), np.array([0.9])], [1, 1], (640, 480))
    _, one = weighted_box_fusion([a, empty], [np.array([0.9]), np.array([])], [1, 1], (640, 480))
    assert one[0] < both[0]


def test_fusion_handles_no_boxes():
    boxes, scores = weighted_box_fusion([np.empty((0, 4))], [np.empty(0)], [1], (640, 480))
    assert boxes.shape == (0, 4) and scores.shape == (0,)


def test_tracker_keeps_id_for_moving_target():
    tr = ByteTracker(frame_rate=30)
    ids = []
    for k in range(30):
        out = tr.update(Detections(np.array([moving_box(k)], np.float32), np.array([0.8], np.float32)))
        ids.extend(out.ids.tolist())
    assert len(ids) >= 29 and len(set(ids)) == 1


def test_tracker_two_targets_get_two_ids():
    tr = ByteTracker(frame_rate=30)
    seen = set()
    for k in range(20):
        boxes = np.array([moving_box(k), moving_box(k, x0=400, vx=-3)], np.float32)
        seen.update(tr.update(Detections(boxes, np.array([0.8, 0.7], np.float32))).ids.tolist())
    assert len(seen) == 2


def test_tracker_low_confidence_box_does_not_start_track():
    tr = ByteTracker(frame_rate=30)
    for k in range(10):
        out = tr.update(Detections(np.array([moving_box(k)], np.float32), np.array([0.15], np.float32)))
    assert len(out) == 0


def test_tracker_low_confidence_box_continues_existing_track():
    """ByteTrack's key idea: a weak detection can extend an existing track."""
    tr = ByteTracker(frame_rate=30)
    for k in range(10):
        first = tr.update(Detections(np.array([moving_box(k)], np.float32), np.array([0.8], np.float32)))
    tid = first.ids[0]
    out = tr.update(Detections(np.array([moving_box(10)], np.float32), np.array([0.15], np.float32)))
    assert out.ids.tolist() == [tid]


def test_tracker_recovers_id_after_short_gap():
    tr = ByteTracker(frame_rate=30, track_buffer=30)
    for k in range(10):
        tid = tr.update(Detections(np.array([moving_box(k)], np.float32), np.array([0.8], np.float32))).ids[0]
    for _ in range(5):
        tr.update(Detections())
    out = tr.update(Detections(np.array([moving_box(15)], np.float32), np.array([0.8], np.float32)))
    assert out.ids.tolist() == [tid]


def test_pipeline_end_to_end_and_velocity_units():
    frames = [([moving_box(k)], [0.9]) for k in range(40)]
    pipe = Pipeline(FakeDetector(frames), fps=30, horizon_frames=10)
    frame = np.zeros((480, 640, 3), np.uint8)
    for _ in range(40):
        res = pipe.process(frame)
    assert len(res.tracks) == 1
    tr = res.tracks[0]
    assert np.allclose(tr.velocity, [120, 60], rtol=0.25)  # 4 px/frame * 30 fps
    assert tr.trajectory.shape == (10, 2)
    assert set(res.timings_ms) == {"detect", "track", "estimate", "predict", "total"}


def test_annotator_fixed_canvas_size():
    frames = [([moving_box(k)], [0.9]) for k in range(5)]
    pipe = Pipeline(FakeDetector(frames), fps=30)
    ann = Annotator(out_wh=(1280, 720))
    for frame in (np.zeros((1080, 1920, 3), np.uint8), np.zeros((480, 640, 3), np.uint8)):
        canvas = ann.draw(frame, pipe.process(frame), fps=30.0)
        assert canvas.shape == (720, 1280, 3)
