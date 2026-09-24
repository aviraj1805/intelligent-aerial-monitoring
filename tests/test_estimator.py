"""
Test: estimation/kalman_filter.py
Run from project root:  python tests/test_kalman_filter.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from estimation.kalman_filter import KalmanFilterManager


def separator(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def make_tracked(boxes, track_ids, scores=None, labels=None):
    boxes     = np.array(boxes,     dtype=np.float32)
    track_ids = np.array(track_ids, dtype=np.int32)
    if scores is None:
        scores = np.full(len(boxes), 0.85, dtype=np.float32)
    if labels is None:
        labels = np.zeros(len(boxes), dtype=np.int32)
    return {"boxes": boxes, "scores": scores,
            "labels": labels, "track_ids": track_ids}


# ── Test 1: Empty input ───────────────────────────────────────────────────────
def test_empty():
    separator("TEST 1 — Empty input")
    kf = KalmanFilterManager()
    empty = {"boxes": np.empty((0,4), dtype=np.float32),
             "scores": np.empty((0,), dtype=np.float32),
             "labels": np.empty((0,), dtype=np.int32),
             "track_ids": np.empty((0,), dtype=np.int32)}
    result = kf.update(empty)
    assert result["boxes"].shape     == (0, 4)
    assert result["velocities"].shape == (0, 2)
    print("  ✓ Empty input handled correctly")


# ── Test 2: Single track initialised ─────────────────────────────────────────
def test_single_track_init():
    separator("TEST 2 — Single track initialised")
    kf = KalmanFilterManager()
    tracked = make_tracked([[100, 80, 200, 160]], [1])
    result  = kf.update(tracked)
    assert result["boxes"].shape      == (1, 4)
    assert result["velocities"].shape == (1, 2)
    vx, vy = result["velocities"][0]
    print(f"  ✓ Box: {result['boxes'][0].astype(int)}  velocity=({vx:.2f}, {vy:.2f})")
    # First frame velocity should be zero (just initialised)
    assert abs(vx) < 1e-3 and abs(vy) < 1e-3, "Initial velocity should be ~0"
    print("  ✓ Initial velocity is zero")


# ── Test 3: Velocity builds up over linear motion ────────────────────────────
def test_velocity_converges():
    separator("TEST 3 — Velocity converges on linear motion (dx=5, dy=3 per frame)")
    kf  = KalmanFilterManager()
    box = np.array([100.0, 80.0, 200.0, 160.0])
    dx, dy = 5.0, 3.0

    for i in range(20):
        b = box + np.array([dx*i, dy*i, dx*i, dy*i])
        tracked = make_tracked([b], [1])
        result  = kf.update(tracked)

    vx, vy = result["velocities"][0]
    print(f"  After 20 frames: vx={vx:.2f}  vy={vy:.2f}  (expected ~{dx}, ~{dy})")
    assert abs(vx - dx) < 2.0, f"vx {vx:.2f} too far from expected {dx}"
    assert abs(vy - dy) < 2.0, f"vy {vy:.2f} too far from expected {dy}"
    print("  ✓ Velocity converged correctly")


# ── Test 4: Smoothing — noisy box positions ───────────────────────────────────
def test_smoothing():
    separator("TEST 4 — Kalman smoothing reduces positional noise")
    kf  = KalmanFilterManager()
    box = np.array([200.0, 150.0, 300.0, 250.0])
    rng = np.random.default_rng(42)
    raw_errors, smooth_errors = [], []
    true_cx = (box[0] + box[2]) / 2

    for i in range(30):
        noise = rng.normal(0, 8, 4)
        noisy_box = box + noise
        tracked = make_tracked([noisy_box], [1])
        result  = kf.update(tracked)

        raw_cx    = (noisy_box[0] + noisy_box[2]) / 2
        smooth_cx = (result["boxes"][0][0] + result["boxes"][0][2]) / 2
        raw_errors.append(abs(raw_cx - true_cx))
        smooth_errors.append(abs(smooth_cx - true_cx))

    mean_raw    = np.mean(raw_errors)
    mean_smooth = np.mean(smooth_errors)
    print(f"  Mean raw error    : {mean_raw:.2f} px")
    print(f"  Mean smooth error : {mean_smooth:.2f} px")
    assert mean_smooth <= mean_raw, "Kalman smoother should reduce error!"
    print("  ✓ Kalman filter reduces positional noise")


# ── Test 5: Stale filter cleanup ──────────────────────────────────────────────
def test_stale_cleanup():
    separator("TEST 5 — Stale filters are removed when track disappears")
    kf = KalmanFilterManager()

    # Establish two tracks
    tracked = make_tracked([[100,80,200,160],[400,300,500,400]], [1, 2])
    kf.update(tracked)
    assert len(kf._filters) == 2

    # Next frame — only track 1 visible
    tracked2 = make_tracked([[102,81,202,161]], [1])
    kf.update(tracked2)
    assert 2 not in kf._filters, "Track 2 filter should be removed"
    assert 1 in kf._filters,     "Track 1 filter should remain"
    print(f"  ✓ Stale filter removed, active filters: {list(kf._filters.keys())}")


# ── Test 6: Full pipeline — real video 10 frames ─────────────────────────────
def test_real_video_pipeline():
    separator("TEST 6 — Full pipeline with Kalman (10 real frames)")

    video_path = "data/sample_frames/test_drone4.mp4.mp4"
    if not os.path.exists(video_path):
        print(f"  [SKIP] Video not found at {video_path}")
        return

    import cv2
    from detection.multi_model_detector import MultiModelDetector
    from detection.fusion_engine import FusionEngine
    from tracking.bytetracker import ByteTracker

    detector = MultiModelDetector(device="cuda")
    fe       = FusionEngine()
    tracker  = ByteTracker()
    kf_mgr   = KalmanFilterManager()

    cap = cv2.VideoCapture(video_path)
    print("  Frame  | tracked | smoothed boxes + velocity")
    print("  " + "-"*55)

    for frame_i in range(10):
        ret, frame = cap.read()
        if not ret:
            break

        model_results = detector.run_parallel(frame)
        fused         = fe.fuse(model_results, frame.shape)
        tracked       = tracker.update(fused, frame.shape)
        smoothed      = kf_mgr.update(tracked)

        for j in range(len(smoothed["boxes"])):
            b  = smoothed["boxes"][j].astype(int)
            vx = smoothed["velocities"][j][0]
            vy = smoothed["velocities"][j][1]
            tid = smoothed["track_ids"][j]
            print(f"  {frame_i+1:5d}  | ID={tid}  box={b}  v=({vx:.1f},{vy:.1f})")

    cap.release()
    print("  ✓ 10-frame pipeline with Kalman completed without crash")


if __name__ == "__main__":
    separator("IAMARS — KalmanFilter Test Suite")
    try:
        test_empty()
        test_single_track_init()
        test_velocity_converges()
        test_smoothing()
        test_stale_cleanup()
        test_real_video_pipeline()

        separator("ALL TESTS PASSED ✓")
        print("  Ready to proceed to File 5 — trajectory_predictor.py\n")

    except AssertionError as e:
        print(f"\n  [FAIL] {e}")
        sys.exit(1)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)