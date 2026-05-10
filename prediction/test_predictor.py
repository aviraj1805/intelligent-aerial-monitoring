"""
Test: prediction/trajectory_predictor.py
Run from project root:  python tests/test_trajectory_predictor.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prediction.trajectory_predictor import TrajectoryPredictor


def separator(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def make_smoothed(boxes, velocities, track_ids=None, scores=None, labels=None):
    boxes      = np.array(boxes,      dtype=np.float32)
    velocities = np.array(velocities, dtype=np.float32)
    if track_ids is None:
        track_ids = np.arange(1, len(boxes)+1, dtype=np.int32)
    if scores is None:
        scores = np.full(len(boxes), 0.85, dtype=np.float32)
    if labels is None:
        labels = np.zeros(len(boxes), dtype=np.int32)
    return {"boxes": boxes, "scores": scores, "labels": labels,
            "track_ids": track_ids, "velocities": velocities}


# ── Test 1: Empty input ───────────────────────────────────────────────────────
def test_empty():
    separator("TEST 1 — Empty input")
    tp = TrajectoryPredictor(horizon=10)
    smoothed = {"boxes": np.empty((0,4), dtype=np.float32),
                "scores": np.empty((0,), dtype=np.float32),
                "labels": np.empty((0,), dtype=np.int32),
                "track_ids": np.empty((0,), dtype=np.int32),
                "velocities": np.empty((0,2), dtype=np.float32)}
    result = tp.predict(smoothed)
    assert result["trajectories"] == [], f"Expected [], got {result['trajectories']}"
    print("  ✓ Empty input returns empty trajectory list")


# ── Test 2: Correct shape ─────────────────────────────────────────────────────
def test_output_shape():
    separator("TEST 2 — Output shape (horizon=10, 2 tracks)")
    tp = TrajectoryPredictor(horizon=10)
    smoothed = make_smoothed(
        boxes      = [[100,80,200,160], [400,300,500,400]],
        velocities = [[5.0, 3.0],       [-2.0, 1.0]],
        track_ids  = np.array([1, 2], dtype=np.int32),
    )
    result = tp.predict(smoothed)
    trajs  = result["trajectories"]
    assert len(trajs) == 2, f"Expected 2 trajectories, got {len(trajs)}"
    for i, traj in enumerate(trajs):
        assert traj.shape == (10, 2), f"Track {i} shape wrong: {traj.shape}"
    print(f"  ✓ 2 trajectories, each shape (10, 2)")


# ── Test 3: Correct linear extrapolation ─────────────────────────────────────
def test_linear_extrapolation():
    separator("TEST 3 — Linear extrapolation correctness")
    tp = TrajectoryPredictor(horizon=10)

    cx, cy = 150.0, 120.0
    vx, vy =   5.0,   3.0
    box = [cx - 50, cy - 40, cx + 50, cy + 40]   # 100×80 box centred at (cx,cy)

    smoothed = make_smoothed(boxes=[[*box]], velocities=[[vx, vy]])
    result   = tp.predict(smoothed)
    traj     = result["trajectories"][0]   # (10, 2)

    for t in range(1, 11):
        expected_cx = cx + vx * t
        expected_cy = cy + vy * t
        got_cx, got_cy = traj[t-1]
        assert abs(got_cx - expected_cx) < 1e-3, \
            f"t={t}: cx expected {expected_cx:.2f} got {got_cx:.2f}"
        assert abs(got_cy - expected_cy) < 1e-3, \
            f"t={t}: cy expected {expected_cy:.2f} got {got_cy:.2f}"

    print(f"  ✓ All 10 predicted positions match cx+vx*t / cy+vy*t exactly")
    print(f"    t=1  → ({traj[0][0]:.1f}, {traj[0][1]:.1f})")
    print(f"    t=5  → ({traj[4][0]:.1f}, {traj[4][1]:.1f})")
    print(f"    t=10 → ({traj[9][0]:.1f}, {traj[9][1]:.1f})")


# ── Test 4: Zero velocity — stationary target ─────────────────────────────────
def test_stationary_target():
    separator("TEST 4 — Stationary target (vx=vy=0)")
    tp = TrajectoryPredictor(horizon=10)
    smoothed = make_smoothed(
        boxes=[[200, 150, 300, 250]],
        velocities=[[0.0, 0.0]],
    )
    result = tp.predict(smoothed)
    traj   = result["trajectories"][0]
    cx, cy = 250.0, 200.0
    assert np.allclose(traj[:, 0], cx), "cx should stay constant"
    assert np.allclose(traj[:, 1], cy), "cy should stay constant"
    print(f"  ✓ All predicted positions remain at ({cx}, {cy})")


# ── Test 5: predict_single utility ───────────────────────────────────────────
def test_predict_single():
    separator("TEST 5 — predict_single() utility")
    tp   = TrajectoryPredictor(horizon=5)
    traj = tp.predict_single(cx=100, cy=80, vx=10, vy=-5)
    assert traj.shape == (5, 2)
    assert abs(traj[0][0] - 110) < 1e-3   # t=1: 100 + 10*1
    assert abs(traj[4][0] - 150) < 1e-3   # t=5: 100 + 10*5
    print(f"  ✓ predict_single shape=(5,2)")
    print(f"    t=1 → ({traj[0][0]:.1f}, {traj[0][1]:.1f})")
    print(f"    t=5 → ({traj[4][0]:.1f}, {traj[4][1]:.1f})")


# ── Test 6: Full pipeline — real video 10 frames ─────────────────────────────
def test_real_video_pipeline():
    separator("TEST 6 — Full pipeline with trajectory (10 real frames)")

    video_path = "data/sample_frames/test_drone4.mp4.mp4"
    if not os.path.exists(video_path):
        print(f"  [SKIP] Video not found at {video_path}")
        return

    import cv2
    from detection.multi_model_detector import MultiModelDetector
    from detection.fusion_engine import FusionEngine
    from tracking.bytetracker import ByteTracker
    from estimation.kalman_filter import KalmanFilterManager

    detector = MultiModelDetector(device="cuda")
    fe       = FusionEngine()
    tracker  = ByteTracker()
    kf_mgr   = KalmanFilterManager()
    tp       = TrajectoryPredictor(horizon=10)

    cap = cv2.VideoCapture(video_path)
    print("  Frame  | ID | cur_pos       | t+1 pred      | t+10 pred")
    print("  " + "-" * 60)

    for frame_i in range(10):
        ret, frame = cap.read()
        if not ret:
            break

        results  = detector.run_parallel(frame)
        fused    = fe.fuse(results, frame.shape)
        tracked  = tracker.update(fused, frame.shape)
        smoothed = kf_mgr.update(tracked)
        predicted = tp.predict(smoothed)

        for j in range(len(smoothed["boxes"])):
            b   = smoothed["boxes"][j]
            tid = smoothed["track_ids"][j]
            cx  = int((b[0]+b[2])/2);  cy = int((b[1]+b[3])/2)
            traj = predicted["trajectories"][j]
            p1   = traj[0].astype(int)
            p10  = traj[9].astype(int)
            print(f"  {frame_i+1:5d}  | {tid}  | ({cx:4d},{cy:4d})     "
                  f"| ({p1[0]:4d},{p1[1]:4d})     | ({p10[0]:4d},{p10[1]:4d})")

    cap.release()
    print("  ✓ 10-frame trajectory pipeline completed without crash")


if __name__ == "__main__":
    separator("IAMARS — TrajectoryPredictor Test Suite")
    try:
        test_empty()
        test_output_shape()
        test_linear_extrapolation()
        test_stationary_target()
        test_predict_single()
        test_real_video_pipeline()

        separator("ALL TESTS PASSED ✓")
        print("  Ready to proceed to File 6 — fire_solution.py\n")

    except AssertionError as e:
        print(f"\n  [FAIL] {e}")
        sys.exit(1)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)