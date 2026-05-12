"""
Test: integration/video_pipeline.py
Final integration test — validates all 8 pipeline files work together.
Run from project root:  python tests/test_video_pipeline.py

This test runs HEADLESS (no display window) for automated validation.
For the live display, use:  python integration/video_pipeline.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def separator(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


# ── Test 1: All imports succeed ───────────────────────────────────────────────
def test_imports():
    separator("TEST 1 — All 7 module imports")
    from detection.multi_model_detector   import MultiModelDetector
    from detection.fusion_engine          import FusionEngine
    from tracking.bytetracker             import ByteTracker
    from estimation.kalman_filter         import KalmanFilterManager
    from prediction.trajectory_predictor  import TrajectoryPredictor
    from intercept.fire_solution          import FireSolution
    from visualization.tactical_dashboard import TacticalDashboard
    print("  ✓ All 7 modules imported successfully")
    return (MultiModelDetector, FusionEngine, ByteTracker,
            KalmanFilterManager, TrajectoryPredictor, FireSolution,
            TacticalDashboard)


# ── Test 2: All models load ───────────────────────────────────────────────────
def test_model_loading(classes):
    separator("TEST 2 — All 4 models load")
    MultiModelDetector = classes[0]
    detector = MultiModelDetector(device="cuda")
    assert len(detector.models) == 3, \
        f"Expected 3 detection models, got {len(detector.models)}"
    assert detector._audio_model is not None, "Audio model failed to load"
    for m in detector.models:
        print(f"  ✓ {m['name']:12s}  weight={m['weight']}  conf={m['conf']}")
    print("  ✓ audio_detection loaded")
    return detector


# ── Test 3: Full pipeline on single dummy frame ───────────────────────────────
def test_single_frame_dummy(classes, detector):
    separator("TEST 3 — Full pipeline on dummy frame (no crash)")
    _, FusionEngine, ByteTracker, KalmanFilterManager, \
        TrajectoryPredictor, FireSolution, TacticalDashboard = classes

    fe      = FusionEngine()
    tracker = ByteTracker()
    kf_mgr  = KalmanFilterManager()
    tp      = TrajectoryPredictor(horizon=10)
    fs      = FireSolution(frame_wh=(1920, 1080))
    dash    = TacticalDashboard(frame_wh=(1920, 1080))

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    results   = detector.run_parallel(frame)
    fused     = fe.fuse(results, frame.shape)
    tracked   = tracker.update(fused, frame.shape)
    smoothed  = kf_mgr.update(tracked)
    predicted = tp.predict(smoothed)
    solutions = fs.compute(predicted, frame_index=0)
    canvas    = dash.render(frame, results, fused, tracked,
                            smoothed, predicted, solutions,
                            frame_index=0)

    assert canvas.shape == (720, 1280, 3)
    print("  ✓ Single dummy frame: all 7 stages completed, canvas=(720,1280,3)")


# ── Test 4: Full pipeline on real video — all 593 frames headless ─────────────
def test_full_video_headless(classes):
    separator("TEST 4 — Full video headless (all frames, no crash)")

    video_path = "data/sample_frames/test_drone4.mp4.mp4"
    if not os.path.exists(video_path):
        print(f"  [SKIP] Video not found at {video_path}")
        return

    import cv2
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    from integration.video_pipeline import build_pipeline, process_frame

    cap         = cv2.VideoCapture(video_path)
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_wh    = (src_w, src_h)

    detector, fe, tracker, kf_mgr, tp, fs, dash = build_pipeline(frame_wh)

    frame_index  = 0
    total_det    = 0
    total_tracks = 0
    crash_frames = 0
    t_start      = time.perf_counter()

    print(f"  Running {total} frames headless...")
    print(f"  {'Frame':>6}  {'FPS':>6}  {'Det':>4}  {'Tracks':>6}  {'Solutions':>9}")
    print("  " + "-" * 46)

    fps_smooth = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        try:
            canvas, n_det, n_fus, n_trk = process_frame(
                frame, frame_index,
                detector, fe, tracker, kf_mgr, tp, fs, dash,
                frame_wh,
            )
            assert canvas.shape == (720, 1280, 3), \
                f"Canvas shape wrong at frame {frame_index}: {canvas.shape}"
            total_det    += n_det
            total_tracks += n_trk
        except Exception as e:
            crash_frames += 1
            print(f"  [WARN] Frame {frame_index} error: {e}")

        elapsed    = time.perf_counter() - t0
        fps_inst   = 1.0 / elapsed if elapsed > 0 else 0.0
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps_inst

        if frame_index % 50 == 0:
            print(f"  {frame_index:>6}  {fps_smooth:>6.1f}  "
                  f"{n_det:>4}  {n_trk:>6}")

        frame_index += 1

    cap.release()
    total_time = time.perf_counter() - t_start
    avg_fps    = frame_index / total_time if total_time > 0 else 0

    print(f"\n  ── Summary ──────────────────────────────────")
    print(f"  Frames processed : {frame_index} / {total}")
    print(f"  Total time       : {total_time:.1f}s")
    print(f"  Average FPS      : {avg_fps:.1f}")
    print(f"  Crash frames     : {crash_frames}")
    print(f"  Total detections : {total_det}")
    print(f"  Total track ticks: {total_tracks}")

    assert crash_frames == 0, f"{crash_frames} frames crashed!"
    assert frame_index == total, \
        f"Only processed {frame_index}/{total} frames"
    print(f"\n  ✓ All {frame_index} frames processed without crash")


# ── Test 5: Output video save ─────────────────────────────────────────────────
def test_save_output():
    separator("TEST 5 — Save output video (20 frames)")

    video_path = "data/sample_frames/test_drone4.mp4.mp4"
    out_path   = "tests/output_test.mp4"
    if not os.path.exists(video_path):
        print(f"  [SKIP] Video not found at {video_path}")
        return

    import cv2
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    from integration.video_pipeline import build_pipeline, process_frame

    cap      = cv2.VideoCapture(video_path)
    src_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_wh = (src_w, src_h)

    detector, fe, tracker, kf_mgr, tp, fs, dash = build_pipeline(frame_wh)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, 20.0, (1280, 720))

    for frame_index in range(20):
        ret, frame = cap.read()
        if not ret:
            break
        canvas, *_ = process_frame(
            frame, frame_index,
            detector, fe, tracker, kf_mgr, tp, fs, dash, frame_wh,
        )
        writer.write(canvas)

    cap.release()
    writer.release()

    assert os.path.exists(out_path), "Output file not created"
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  ✓ Output saved: {out_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    separator("IAMARS — Full Integration Test Suite")
    try:
        classes  = test_imports()
        detector = test_model_loading(classes)
        test_single_frame_dummy(classes, detector)
        test_full_video_headless(classes)
        test_save_output()

        separator("ALL INTEGRATION TESTS PASSED ✓")
        print()
        print("  ══════════════════════════════════════════════════")
        print("  PIPELINE VALIDATED — READY FOR LIVE RUN")
        print()
        print("  Run command:")
        print("    python integration/video_pipeline.py")
        print()
        print("  With output save:")
        print("    python integration/video_pipeline.py --save output.mp4")
        print("  ══════════════════════════════════════════════════")

    except AssertionError as e:
        print(f"\n  [FAIL] {e}")
        sys.exit(1)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)