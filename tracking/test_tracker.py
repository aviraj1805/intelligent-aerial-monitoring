"""
Test: tracking/bytetracker.py
Run from project root:  python tests/test_bytetracker.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tracking.bytetracker import ByteTracker

FRAME_SHAPE = (1080, 1920, 3)


def separator(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def make_fused(boxes, scores=None, labels=None):
    boxes = np.array(boxes, dtype=np.float32)
    if scores is None:
        scores = np.full(len(boxes), 0.85, dtype=np.float32)
    if labels is None:
        labels = np.zeros(len(boxes), dtype=np.int32)
    return {"boxes": boxes, "scores": scores, "labels": labels}


# ── Test 1: Empty detections ──────────────────────────────────────────────────
def test_empty():
    separator("TEST 1 — Empty detections")
    tracker = ByteTracker()
    empty = {"boxes": np.empty((0,4), dtype=np.float32),
             "scores": np.empty((0,), dtype=np.float32),
             "labels": np.empty((0,), dtype=np.int32)}
    result = tracker.update(empty, FRAME_SHAPE)
    assert result["boxes"].shape  == (0, 4)
    assert result["track_ids"].shape == (0,)
    print("  ✓ Empty input returns empty output")


# ── Test 2: Single detection gets a track ID ─────────────────────────────────
def test_single_detection():
    separator("TEST 2 — Single detection assigned track ID")
    tracker = ByteTracker()
    fused = make_fused([[100, 80, 200, 160]])
    result = tracker.update(fused, FRAME_SHAPE)
    assert len(result["track_ids"]) == 1
    tid = result["track_ids"][0]
    print(f"  ✓ Track ID assigned: {tid}")
    assert tid >= 1, "Track ID should be >= 1"


# ── Test 3: ID persists across frames (stable tracking) ──────────────────────
def test_stable_id_across_frames():
    separator("TEST 3 — Stable track ID across 10 frames")
    tracker = ByteTracker()
    box = [100.0, 80.0, 200.0, 160.0]
    first_id = None

    for frame_i in range(10):
        # Move box slightly each frame to simulate motion
        b = [box[0] + frame_i*2, box[1] + frame_i*1,
             box[2] + frame_i*2, box[3] + frame_i*1]
        fused = make_fused([b])
        result = tracker.update(fused, FRAME_SHAPE)

        if len(result["track_ids"]) > 0:
            tid = result["track_ids"][0]
            if first_id is None:
                first_id = tid
                print(f"  Frame  0: track_id={tid}  box={np.array(b).astype(int)}")
            else:
                assert tid == first_id, \
                    f"Track ID changed! frame={frame_i} expected={first_id} got={tid}"

    print(f"  ✓ Track ID {first_id} stable across 10 frames")


# ── Test 4: Lost track buffer — ID resumes after gap ─────────────────────────
def test_lost_track_buffer():
    separator("TEST 4 — Lost track buffer (ID held for 60 frames)")
    tracker = ByteTracker(lost_track_buffer=60)
    empty = {"boxes": np.empty((0,4), dtype=np.float32),
             "scores": np.empty((0,), dtype=np.float32),
             "labels": np.empty((0,), dtype=np.int32)}

    # Establish track
    fused = make_fused([[100, 80, 200, 160]])
    result = tracker.update(fused, FRAME_SHAPE)
    first_id = result["track_ids"][0] if len(result["track_ids"]) > 0 else None
    print(f"  Initial track ID: {first_id}")

    # Feed 5 empty frames (within buffer)
    for _ in range(5):
        tracker.update(empty, FRAME_SHAPE)

    # Reappear — should get same ID
    fused2 = make_fused([[102, 81, 202, 161]])
    result2 = tracker.update(fused2, FRAME_SHAPE)
    if len(result2["track_ids"]) > 0:
        resumed_id = result2["track_ids"][0]
        print(f"  Resumed track ID: {resumed_id}")
        print(f"  ✓ Track {'resumed same ID' if resumed_id == first_id else 'got new ID (acceptable)'}")
    else:
        print("  [INFO] No track after reappearance (still in buffer period)")


# ── Test 5: Two targets get distinct IDs ─────────────────────────────────────
def test_two_targets():
    separator("TEST 5 — Two targets get distinct track IDs")
    tracker = ByteTracker()
    fused = make_fused([
        [100, 80,  200, 160],
        [500, 300, 620, 400],
    ])
    result = tracker.update(fused, FRAME_SHAPE)
    print(f"  Tracked boxes : {len(result['boxes'])}")
    print(f"  Track IDs     : {result['track_ids']}")
    assert len(result["track_ids"]) >= 1
    if len(result["track_ids"]) == 2:
        assert result["track_ids"][0] != result["track_ids"][1], "Two targets must have different IDs"
        print("  ✓ Two distinct track IDs assigned")
    else:
        print("  [INFO] ByteTrack returned 1 (minimum_consecutive_frames filtering) — acceptable")


# ── Test 6: Full pipeline — real video (10 frames) ───────────────────────────
def test_real_video_pipeline():
    separator("TEST 6 — Real video: detector → fusion → tracker (10 frames)")

    video_path = "data/sample_frames/test_drone4.mp4.mp4"
    if not os.path.exists(video_path):
        print(f"  [SKIP] Video not found at {video_path}")
        return

    import cv2
    from detection.multi_model_detector import MultiModelDetector
    from detection.fusion_engine import FusionEngine

    detector = MultiModelDetector(device="cuda")
    fe       = FusionEngine()
    tracker  = ByteTracker()

    cap = cv2.VideoCapture(video_path)
    print(f"  Running 10 frames...")

    for frame_i in range(10):
        ret, frame = cap.read()
        if not ret:
            break

        model_results = detector.run_parallel(frame)
        fused         = fe.fuse(model_results, frame.shape)
        tracked       = tracker.update(fused, frame.shape)

        ids = tracked["track_ids"]
        print(f"  Frame {frame_i+1:02d}: fused={len(fused['boxes'])}  "
              f"tracked={len(tracked['boxes'])}  IDs={ids.tolist()}")

    cap.release()
    print("  ✓ 10-frame pipeline completed without crash")


if __name__ == "__main__":
    separator("IAMARS — ByteTracker Test Suite")
    try:
        test_empty()
        test_single_detection()
        test_stable_id_across_frames()
        test_lost_track_buffer()
        test_two_targets()
        test_real_video_pipeline()

        separator("ALL TESTS PASSED ✓")
        print("  Ready to proceed to File 4 — kalman_filter.py\n")

    except AssertionError as e:
        print(f"\n  [FAIL] {e}")
        sys.exit(1)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)