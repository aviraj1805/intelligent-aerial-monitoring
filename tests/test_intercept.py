"""
Test: intercept/fire_solution.py
Run from project root:  python tests/test_fire_solution.py
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from intercept.fire_solution import FireSolution

FRAME_WH = (1920, 1080)
W, H     = FRAME_WH
OX, OY   = W / 2, H / 2   # origin = image centre


def separator(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def make_predicted(boxes, velocities, trajectories, track_ids=None):
    boxes      = np.array(boxes,      dtype=np.float32)
    velocities = np.array(velocities, dtype=np.float32)
    if track_ids is None:
        track_ids = np.arange(1, len(boxes)+1, dtype=np.int32)
    trajs = []
    for t in trajectories:
        trajs.append(np.array(t, dtype=np.float32))
    return {"boxes": boxes, "scores": np.ones(len(boxes), dtype=np.float32),
            "labels": np.zeros(len(boxes), dtype=np.int32),
            "track_ids": track_ids, "velocities": velocities,
            "trajectories": trajs}


# ── Test 1: Empty input ───────────────────────────────────────────────────────
def test_empty():
    separator("TEST 1 — Empty input")
    fs = FireSolution(frame_wh=FRAME_WH)
    predicted = {"boxes": np.empty((0,4), dtype=np.float32),
                 "scores": np.empty((0,), dtype=np.float32),
                 "labels": np.empty((0,), dtype=np.int32),
                 "track_ids": np.empty((0,), dtype=np.int32),
                 "velocities": np.empty((0,2), dtype=np.float32),
                 "trajectories": []}
    solutions = fs.compute(predicted, frame_index=0)
    assert solutions == [], f"Expected [], got {solutions}"
    print("  ✓ Empty input returns empty solution list")


# ── Test 2: Azimuth correctness ───────────────────────────────────────────────
def test_azimuth():
    separator("TEST 2 — Azimuth correctness for cardinal directions")
    fs = FireSolution(frame_wh=FRAME_WH)

    # Target directly to the right of centre → azimuth ≈ 0°
    # Target directly below centre           → azimuth ≈ 90°
    # Target directly to the left            → azimuth ≈ 180°
    # Target directly above centre           → azimuth ≈ 270°
    cases = [
        ("right",  [OX+200, OY,     OX+250, OY+50],   0.0),
        ("below",  [OX,     OY+200, OX+50,  OY+250], 90.0),
        ("left",   [OX-250, OY,     OX-200, OY+50], 180.0),
        ("above",  [OX,     OY-250, OX+50,  OY-200], 270.0),
    ]

    dummy_traj = [[[OX, OY]] * 10]  # stationary

    for name, box, expected_az in cases:
        pred = make_predicted([box], [[0,0]], dummy_traj)
        sol  = fs.compute(pred, frame_index=0)[0]
        diff = min(abs(sol["azimuth"] - expected_az),
                   360 - abs(sol["azimuth"] - expected_az))
        print(f"  {name:6s}: azimuth={sol['azimuth']:.1f}°  "
              f"(expected≈{expected_az:.0f}°)  diff={diff:.1f}°")
        assert diff < 10.0, f"Azimuth too far off for {name}: got {sol['azimuth']}"

    print("  ✓ All cardinal azimuths within ±10°")


# ── Test 3: Elevation sign ────────────────────────────────────────────────────
def test_elevation_sign():
    separator("TEST 3 — Elevation sign (above centre = positive)")
    fs = FireSolution(frame_wh=FRAME_WH)

    # Target ABOVE image centre → elevation > 0
    box_above = [OX-25, OY-200, OX+25, OY-150]
    # Target BELOW image centre → elevation < 0
    box_below = [OX-25, OY+150, OX+25, OY+200]

    dummy_traj = [[[OX, OY]] * 10]

    sol_above = fs.compute(make_predicted([box_above], [[0,0]], dummy_traj), 0)[0]
    sol_below = fs.compute(make_predicted([box_below], [[0,0]], dummy_traj), 0)[0]

    print(f"  Above centre: elevation={sol_above['elevation']:.2f}°  (expected > 0)")
    print(f"  Below centre: elevation={sol_below['elevation']:.2f}°  (expected < 0)")
    assert sol_above["elevation"] > 0, "Drone above centre should have positive elevation"
    assert sol_below["elevation"] < 0, "Drone below centre should have negative elevation"
    print("  ✓ Elevation sign correct")


# ── Test 4: Time-to-intercept spec compliance ─────────────────────────────────
def test_time_to_intercept():
    separator("TEST 4 — Time-to-intercept (projectile_speed * (frame_index + t))")
    speed    = 300.0
    fs       = FireSolution(projectile_speed=speed, frame_wh=FRAME_WH)

    # Stationary target at distance 900px from origin
    # At frame_index=0: proj_dist = 300*(0+t). Intercept at t=3 (300*3=900 >= 900)
    dist     = 900.0
    cx       = OX + dist   # directly to the right
    cy       = OY
    box      = [cx-25, cy-25, cx+25, cy+25]

    # Stationary trajectory — stays at (cx, cy)
    traj = [[(cx, cy)] * 10]

    pred = make_predicted([box], [[0,0]], traj)
    sol  = fs.compute(pred, frame_index=0)[0]

    print(f"  Distance from origin : {sol['distance']:.1f} px")
    print(f"  t_intercept          : {sol['t_intercept']}")
    print(f"  Expected             : 3 frames  (300 * (0+3) = 900 >= 900)")
    assert sol["t_intercept"] == 3.0, \
        f"Expected t_intercept=3, got {sol['t_intercept']}"
    print("  ✓ t_intercept matches spec formula")


# ── Test 5: Threat level range ────────────────────────────────────────────────
def test_threat_level():
    separator("TEST 5 — Threat level [0, 1]")
    fs = FireSolution(frame_wh=FRAME_WH)
    dummy_traj = [[[OX, OY]] * 10]

    # At origin (distance=0) → threat=1.0
    box_centre = [OX-25, OY-25, OX+25, OY+25]
    # Very far away → threat≈0
    box_far    = [OX+1200, OY, OX+1300, OY+50]

    sol_near = fs.compute(make_predicted([box_centre], [[0,0]], dummy_traj), 0)[0]
    sol_far  = fs.compute(make_predicted([box_far],    [[0,0]], dummy_traj), 0)[0]

    print(f"  At origin   : threat={sol_near['threat_level']:.3f}  (expected 1.0)")
    print(f"  Far away    : threat={sol_far['threat_level']:.3f}   (expected ~0.0)")
    assert sol_near["threat_level"] == 1.0
    assert sol_far["threat_level"]  == 0.0
    print("  ✓ Threat level bounds correct")


# ── Test 6: Output keys ───────────────────────────────────────────────────────
def test_output_keys():
    separator("TEST 6 — All required output keys present")
    fs  = FireSolution(frame_wh=FRAME_WH)
    box = [OX+100, OY-80, OX+200, OY+20]
    dummy_traj = [[[OX+150, OY-30]] * 10]
    pred = make_predicted([box], [[3.0, -2.0]], dummy_traj)
    sol  = fs.compute(pred, frame_index=5)[0]

    required = {"track_id","cx","cy","azimuth","elevation","distance",
                "t_intercept","intercept_pos","threat_level","velocity"}
    missing  = required - sol.keys()
    assert not missing, f"Missing keys: {missing}"
    print(f"  ✓ All keys present: {sorted(sol.keys())}")
    for k, v in sol.items():
        print(f"    {k:16s}: {v}")


# ── Test 7: Full pipeline — real video 10 frames ─────────────────────────────
def test_real_video_pipeline():
    separator("TEST 7 — Full pipeline with FireSolution (10 real frames)")

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

    detector = MultiModelDetector(device="cuda")
    fe       = FusionEngine()
    tracker  = ByteTracker()
    kf_mgr   = KalmanFilterManager()
    tp       = TrajectoryPredictor(horizon=10)
    fs       = FireSolution(frame_wh=(1920, 1080))

    cap = cv2.VideoCapture(video_path)
    print("  Frame | ID | AZ°    | EL°    | dist   | t_int | threat")
    print("  " + "-" * 58)

    for frame_i in range(10):
        ret, frame = cap.read()
        if not ret:
            break

        results   = detector.run_parallel(frame)
        fused     = fe.fuse(results, frame.shape)
        tracked   = tracker.update(fused, frame.shape)
        smoothed  = kf_mgr.update(tracked)
        predicted = tp.predict(smoothed)
        solutions = fs.compute(predicted, frame_index=frame_i)

        for sol in solutions:
            print(f"  {frame_i+1:5d} | {sol['track_id']:2d} | "
                  f"{sol['azimuth']:6.1f} | {sol['elevation']:6.1f} | "
                  f"{sol['distance']:6.1f} | {sol['t_intercept']:5} | "
                  f"{sol['threat_level']:.3f}")

    cap.release()
    print("  ✓ 10-frame FireSolution pipeline completed without crash")


if __name__ == "__main__":
    separator("IAMARS — FireSolution Test Suite")
    try:
        test_empty()
        test_azimuth()
        test_elevation_sign()
        test_time_to_intercept()
        test_threat_level()
        test_output_keys()
        test_real_video_pipeline()

        separator("ALL TESTS PASSED ✓")
        print("  Ready to proceed to File 7 — tactical_dashboard.py\n")

    except AssertionError as e:
        print(f"\n  [FAIL] {e}")
        sys.exit(1)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)