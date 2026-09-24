"""
Test: visualization/tactical_dashboard.py
Run from project root:  python tests/test_tactical_dashboard.py

Opens a preview window — press Q to quit, S to save a screenshot.
Also runs headless structural tests first.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visualization.tactical_dashboard import TacticalDashboard


def separator(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_empty_pipeline():
    empty_boxes  = np.empty((0, 4), dtype=np.float32)
    empty_scores = np.empty((0,),   dtype=np.float32)
    empty_labels = np.empty((0,),   dtype=np.int32)
    empty_ids    = np.empty((0,),   dtype=np.int32)
    empty_vel    = np.empty((0, 2), dtype=np.float32)

    model_results = [
        {"name": "visiodect", "boxes": empty_boxes.copy(),
         "scores": empty_scores.copy(), "labels": empty_labels.copy(), "weight": 0.5},
        {"name": "uav_ir",    "boxes": empty_boxes.copy(),
         "scores": empty_scores.copy(), "labels": empty_labels.copy(), "weight": 0.3},
        {"name": "uav_rgb",   "boxes": empty_boxes.copy(),
         "scores": empty_scores.copy(), "labels": empty_labels.copy(), "weight": 0.2},
    ]
    fused     = {"boxes": empty_boxes.copy(), "scores": empty_scores.copy(),
                 "labels": empty_labels.copy()}
    tracked   = {"boxes": empty_boxes.copy(), "scores": empty_scores.copy(),
                 "labels": empty_labels.copy(), "track_ids": empty_ids.copy()}
    smoothed  = {**tracked, "velocities": empty_vel.copy()}
    predicted = {**smoothed, "trajectories": []}
    solutions = []
    return model_results, fused, tracked, smoothed, predicted, solutions


def make_mock_detection(cx, cy, tid, vx=3.0, vy=-2.0):
    """Create a mock single-target pipeline state centred at (cx, cy)."""
    box = np.array([[cx-40, cy-30, cx+40, cy+30]], dtype=np.float32)
    sc  = np.array([0.88], dtype=np.float32)
    lb  = np.array([0],   dtype=np.int32)
    ids = np.array([tid], dtype=np.int32)
    vel = np.array([[vx, vy]], dtype=np.float32)

    horizon = 10
    traj_cx = cx + vx * np.arange(1, horizon+1)
    traj_cy = cy + vy * np.arange(1, horizon+1)
    traj    = [np.stack([traj_cx, traj_cy], axis=1).astype(np.float32)]

    model_results = [
        {"name": "visiodect", "boxes": box.copy(), "scores": sc.copy(),
         "labels": lb.copy(), "weight": 0.5},
        {"name": "uav_ir",    "boxes": box.copy(), "scores": sc.copy(),
         "labels": lb.copy(), "weight": 0.3},
        {"name": "uav_rgb",   "boxes": box.copy(), "scores": sc.copy(),
         "labels": lb.copy(), "weight": 0.2},
    ]
    fused    = {"boxes": box.copy(), "scores": sc.copy(), "labels": lb.copy()}
    tracked  = {"boxes": box.copy(), "scores": sc.copy(),
                "labels": lb.copy(), "track_ids": ids.copy()}
    smoothed = {**tracked, "velocities": vel.copy()}
    predicted= {**smoothed, "trajectories": traj}

    dist = np.sqrt((cx - 960)**2 + (cy - 540)**2)
    sol  = [{
        "track_id": tid, "cx": float(cx), "cy": float(cy),
        "azimuth": 45.0, "elevation": 12.5,
        "distance": float(dist),
        "t_intercept": 4.0,
        "intercept_pos": (float(cx+vx*4), float(cy+vy*4)),
        "threat_level": round(max(0, 1 - dist/1000), 3),
        "velocity": (vx, vy),
    }]
    return model_results, fused, tracked, smoothed, predicted, sol


# ── Test 1: Output shape ──────────────────────────────────────────────────────
def test_output_shape():
    separator("TEST 1 — Output canvas shape and dtype")
    dash  = TacticalDashboard(frame_wh=(1920, 1080))
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    mr, fu, tr, sm, pr, sl = make_empty_pipeline()
    out   = dash.render(frame, mr, fu, tr, sm, pr, sl, frame_index=0)
    assert out.shape == (720, 1280, 3), f"Expected (720,1280,3) got {out.shape}"
    assert out.dtype == np.uint8
    print(f"  ✓ Output shape: {out.shape}  dtype: {out.dtype}")


# ── Test 2: No crash on empty pipeline ────────────────────────────────────────
def test_no_crash_empty():
    separator("TEST 2 — No crash on empty detections")
    dash  = TacticalDashboard(frame_wh=(1920, 1080))
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    mr, fu, tr, sm, pr, sl = make_empty_pipeline()
    for i in range(30):
        out = dash.render(frame, mr, fu, tr, sm, pr, sl, frame_index=i)
    assert out.shape == (720, 1280, 3)
    print("  ✓ 30 empty frames rendered without crash")


# ── Test 3: No crash with detections ─────────────────────────────────────────
def test_no_crash_with_detection():
    separator("TEST 3 — No crash with mock detection")
    dash  = TacticalDashboard(frame_wh=(1920, 1080))
    frame = np.random.randint(0, 200, (1080, 1920, 3), dtype=np.uint8)
    for i in range(30):
        cx = 960 + int(3 * i);  cy = 540 + int(-2 * i)
        mr, fu, tr, sm, pr, sl = make_mock_detection(cx, cy, tid=1)
        out = dash.render(frame, mr, fu, tr, sm, pr, sl,
                          frame_index=i, audio_conf=0.75)
    assert out.shape == (720, 1280, 3)
    print("  ✓ 30 detection frames rendered without crash")


# ── Test 4: Log event ─────────────────────────────────────────────────────────
def test_log_event():
    separator("TEST 4 — System log entries")
    dash = TacticalDashboard()
    dash.log_event("TEST EVENT 1")
    dash.log_event("TEST EVENT 2")
    log = list(dash.log)
    assert any("TEST EVENT 1" in l for l in log)
    assert any("TEST EVENT 2" in l for l in log)
    print(f"  ✓ Log has {len(log)} entries")
    for l in log[-3:]:
        print(f"    {l}")


# ── Test 5: Canvas is not all black (content drawn) ───────────────────────────
def test_content_drawn():
    separator("TEST 5 — Canvas has non-background pixels (content drawn)")
    dash  = TacticalDashboard(frame_wh=(1920, 1080))
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    mr, fu, tr, sm, pr, sl = make_empty_pipeline()
    out   = dash.render(frame, mr, fu, tr, sm, pr, sl, frame_index=0)
    # Count pixels that differ from BG colour (10, 12, 14)
    bg    = np.array([14, 12, 10], dtype=np.uint8)
    diff  = np.any(out != bg, axis=2).sum()
    print(f"  Non-background pixels: {diff:,}")
    assert diff > 5000, f"Too few non-BG pixels ({diff}) — dashboard may be blank"
    print("  ✓ Dashboard content is rendered")


# ── Test 6: Live preview on real video ────────────────────────────────────────
def test_live_preview():
    separator("TEST 6 — Live preview on real video (press Q to quit, S to save)")

    video_path = "data/sample_frames/test_drone4.mp4.mp4"
    if not os.path.exists(video_path):
        print(f"  [SKIP] Video not found at {video_path}")
        return

    import cv2
    from detection.multi_model_detector import MultiModelDetector
    from detection.fusion_engine import FusionEngine
    from tracking.bytetracker import ByteTracker
    from estimation.kalman_filter import KalmanFilterManager
    from prediction.trajectory_predictor import TrajectoryPredictor
    from intercept.fire_solution import FireSolution

    detector = MultiModelDetector(device="cuda")
    fe       = FusionEngine()
    tracker  = ByteTracker()
    kf_mgr   = KalmanFilterManager()
    tp       = TrajectoryPredictor(horizon=10)
    fs       = FireSolution(frame_wh=(1920, 1080))
    dash     = TacticalDashboard(frame_wh=(1920, 1080))

    cap      = cv2.VideoCapture(video_path)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {total} frames — showing live dashboard")
    print("  Press Q to stop, S to save screenshot")

    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results   = detector.run_parallel(frame)
        fused     = fe.fuse(results, frame.shape)
        tracked   = tracker.update(fused, frame.shape)
        smoothed  = kf_mgr.update(tracked)
        predicted = tp.predict(smoothed)
        solutions = fs.compute(predicted, frame_index=frame_i)
        audio_c   = 0.0   # no real audio in test

        canvas = dash.render(frame, results, fused, tracked,
                             smoothed, predicted, solutions,
                             frame_index=frame_i, audio_conf=audio_c)

        cv2.imshow("IAMARS — Tactical Dashboard", canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("  Quit by user")
            break
        if key == ord('s'):
            path = f"tests/screenshot_frame{frame_i:05d}.png"
            cv2.imwrite(path, canvas)
            print(f"  Screenshot saved: {path}")

        frame_i += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"  ✓ Preview ran for {frame_i} frames without crash")


if __name__ == "__main__":
    separator("IAMARS — TacticalDashboard Test Suite")
    try:
        test_output_shape()
        test_no_crash_empty()
        test_no_crash_with_detection()
        test_log_event()
        test_content_drawn()
        test_live_preview()

        separator("ALL TESTS PASSED ✓")
        print("  Ready to proceed to File 8 — video_pipeline.py\n")

    except AssertionError as e:
        print(f"\n  [FAIL] {e}")
        sys.exit(1)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)