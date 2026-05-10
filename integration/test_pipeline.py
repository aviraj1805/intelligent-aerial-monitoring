"""
IAMARS — Intelligent Aerial Monitoring & Automated Response System
Pipeline Validation & Integration Test Script
Run: python tests/test_pipeline.py
"""

import sys
import os
import traceback
import time
import cv2
import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Helpers ───────────────────────────────────────────────────────────────────
PASS  = "\033[92m[PASS]\033[0m"
FAIL  = "\033[91m[FAIL]\033[0m"
INFO  = "\033[94m[INFO]\033[0m"
WARN  = "\033[93m[WARN]\033[0m"
SEP   = "-" * 60

results: list[tuple[str, bool, str]] = []   # (test_name, passed, detail)


def check(name: str, passed: bool, detail: str = ""):
    tag = PASS if passed else FAIL
    msg = f"  {tag}  {name}"
    if detail:
        msg += f"  —  {detail}"
    print(msg)
    results.append((name, passed, detail))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Module imports
# ─────────────────────────────────────────────────────────────────────────────
def test_imports():
    print(f"\n{SEP}")
    print("  SECTION 1 — Module Imports")
    print(SEP)

    modules = {
        "MultiModelDetector" : ("detection.multi_model_detector", "MultiModelDetector"),
        "FusionEngine"        : ("detection.fusion_engine",        "FusionEngine"),
        "ByteTracker"         : ("tracking.bytetracker",           "ByteTracker"),
        "KalmanFilter"        : ("estimation.kalman_filter",       "KalmanFilterManager"),
        "TrajectoryPredictor" : ("prediction.trajectory_predictor","TrajectoryPredictor"),
        "FireSolution"        : ("intercept.fire_solution",        "FireSolution"),
        "TacticalDashboard"   : ("visualization.tactical_dashboard","TacticalDashboard"),
        "IAMARSPipeline"      : ("integration.video_pipeline",     "IAMARSPipeline"),
    }

    imported = {}
    for label, (mod_path, cls_name) in modules.items():
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            imported[label] = cls
            check(f"import  {label}", True)
        except Exception as exc:
            check(f"import  {label}", False, str(exc))

    return imported


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Model files on disk
# ─────────────────────────────────────────────────────────────────────────────
def test_model_files():
    print(f"\n{SEP}")
    print("  SECTION 2 — Model Files")
    print(SEP)

    models = {
        "visiodect"     : os.path.join(ROOT, "models", "visiodect",         "best.pt"),
        "uav_IR"        : os.path.join(ROOT, "models", "uav_IR_detection",  "best.pt"),
        "uav_RGB"       : os.path.join(ROOT, "models", "uav_RGB_detection", "best.pt"),
        "audio"         : os.path.join(ROOT, "models", "audio_detection",   "best.pt"),
    }

    for name, path in models.items():
        exists = os.path.isfile(path)
        size   = f"{os.path.getsize(path) / 1e6:.1f} MB" if exists else "—"
        check(f"model   {name}", exists, path if not exists else size)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Video file
# ─────────────────────────────────────────────────────────────────────────────
def test_video_file():
    print(f"\n{SEP}")
    print("  SECTION 3 — Video File")
    print(SEP)

    video_path = os.path.join(ROOT, "data", "sample_frames", "test_drone4.mp4.mp4")
    exists = os.path.isfile(video_path)
    check("video file exists", exists, video_path)

    if not exists:
        return None

    cap = cv2.VideoCapture(video_path)
    opened = cap.isOpened()
    check("video opens with cv2", opened)

    if opened:
        fps    = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, _ = cap.read()
        check("first frame readable", ret, f"{w}x{h} @ {fps:.1f} fps  |  {frames} total frames")
        cap.release()
        return video_path if ret else None

    cap.release()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Subsystem instantiation
# ─────────────────────────────────────────────────────────────────────────────
def test_instantiation(imported: dict) -> dict:
    print(f"\n{SEP}")
    print("  SECTION 4 — Subsystem Instantiation")
    print(SEP)

    model_paths = {
        "visiodect" : os.path.join(ROOT, "models", "visiodect",         "best.pt"),
        "uav_ir"    : os.path.join(ROOT, "models", "uav_IR_detection",  "best.pt"),
        "uav_rgb"   : os.path.join(ROOT, "models", "uav_RGB_detection", "best.pt"),
        "audio"     : os.path.join(ROOT, "models", "audio_detection",   "best.pt"),
    }

    instances = {}

    constructors = [
        ("MultiModelDetector", lambda: imported["MultiModelDetector"]()),
        ("FusionEngine",        lambda: imported["FusionEngine"]()),
        ("ByteTracker",         lambda: imported["ByteTracker"]()),
        ("KalmanFilter",        lambda: imported["KalmanFilter"]()),
        ("TrajectoryPredictor", lambda: imported["TrajectoryPredictor"]()),
        ("FireSolution",        lambda: imported["FireSolution"]()),
        ("TacticalDashboard",   lambda: imported["TacticalDashboard"]()),
    ]

    for name, factory in constructors:
        if name not in imported:
            check(f"init    {name}", False, "class not imported")
            continue
        try:
            instances[name] = factory()
            check(f"init    {name}", True)
        except Exception as exc:
            check(f"init    {name}", False, str(exc))
            traceback.print_exc()

    return instances


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Short pipeline execution (5 frames)
# ─────────────────────────────────────────────────────────────────────────────
def test_pipeline_execution(instances: dict, video_path: str | None):
    print(f"\n{SEP}")
    print("  SECTION 5 — Pipeline Execution (5 frames)")
    print(SEP)

    if video_path is None:
        check("pipeline execution", False, "video file unavailable — skipping")
        return

    required = ["MultiModelDetector","FusionEngine","ByteTracker",
                "KalmanFilter","TrajectoryPredictor","FireSolution","TacticalDashboard"]
    missing  = [r for r in required if r not in instances]
    if missing:
        check("pipeline execution", False, f"missing subsystems: {missing}")
        return

    detector  = instances["MultiModelDetector"]
    fusion    = instances["FusionEngine"]
    tracker   = instances["ByteTracker"]
    kalman    = instances["KalmanFilter"]
    predictor = instances["TrajectoryPredictor"]
    fire_sol  = instances["FireSolution"]
    dashboard = instances["TacticalDashboard"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        check("pipeline execution", False, "could not reopen video")
        return

    TEST_FRAMES = 5
    processed   = 0
    errors      = []
    t0          = time.time()

    for i in range(TEST_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        try:
            frame_shape   = frame.shape
            model_results = detector.run_parallel(frame)
            fused         = fusion.fuse(model_results, frame_shape)
            tracked       = tracker.update(fused, frame_shape)
            smoothed      = kalman.update(tracked)
            predicted     = predictor.predict(smoothed)
            solutions     = fire_sol.compute(predicted, i + 1)
            out_frame     = dashboard.render(
                frame         = frame,
                model_results = model_results,
                fused         = fused,
                tracked       = tracked,
                smoothed      = smoothed,
                predicted     = predicted,
                solutions     = solutions,
                frame_index   = i + 1,
            )
            assert out_frame is not None, "dashboard.render returned None"
            assert isinstance(out_frame, np.ndarray), "dashboard output is not ndarray"
            processed += 1
        except Exception as exc:
            errors.append(f"frame {i+1}: {exc}")
            traceback.print_exc()

    cap.release()
    elapsed = time.time() - t0

    if errors:
        check("pipeline execution", False, f"{len(errors)} error(s): {errors[0]}")
    else:
        check("pipeline execution", True,
              f"{processed}/{TEST_FRAMES} frames OK  in {elapsed:.2f}s")

    # Per-stage timing info (informational only)
    print(f"\n  {INFO}  Avg time per frame: {elapsed/max(processed,1)*1000:.1f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — IAMARSPipeline integration smoke test
# ─────────────────────────────────────────────────────────────────────────────
def test_pipeline_class(imported: dict):
    print(f"\n{SEP}")
    print("  SECTION 6 — IAMARSPipeline Class Smoke Test")
    print(SEP)

    if "IAMARSPipeline" not in imported:
        check("IAMARSPipeline.initialize()", False, "class not imported")
        return

    try:
        pipeline = imported["IAMARSPipeline"]()
        ok = pipeline.initialize()
        check("IAMARSPipeline.initialize()", ok,
              "all subsystems loaded" if ok else "initialization returned False")

        if ok and pipeline.cap and pipeline.cap.isOpened():
            # Single frame through the unified pipeline
            ret, frame = pipeline.cap.read()
            if ret:
                try:
                    out = pipeline.process_frame(frame, 1)
                    check("IAMARSPipeline.process_frame()", out is not None)
                except Exception as exc:
                    check("IAMARSPipeline.process_frame()", False, str(exc))
            pipeline.cap.release()

    except Exception as exc:
        check("IAMARSPipeline smoke test", False, str(exc))
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
def print_summary():
    print(f"\n{'=' * 60}")
    print("  IAMARS VALIDATION SUMMARY")
    print('=' * 60)
    passed = sum(1 for _, p, _ in results if p)
    failed = sum(1 for _, p, _ in results if not p)
    total  = len(results)

    for name, p, detail in results:
        tag = PASS if p else FAIL
        print(f"  {tag}  {name}" + (f"  —  {detail}" if detail and not p else ""))

    print(f"\n  Total: {total}   {PASS} {passed}   {FAIL} {failed}")
    print('=' * 60)

    if failed == 0:
        print("\n  \033[92m✔  ALL TESTS PASSED — Pipeline is ready to run.\033[0m")
        print(f"  Run:  python integration/video_pipeline.py\n")
    else:
        print(f"\n  \033[91m✘  {failed} TEST(S) FAILED — Fix issues above before running pipeline.\033[0m\n")

    return failed == 0


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  IAMARS — Pipeline Validation Suite")
    print("=" * 60)

    imported  = test_imports()
    test_model_files()
    video_path = test_video_file()
    instances  = test_instantiation(imported)
    test_pipeline_execution(instances, video_path)
    test_pipeline_class(imported)

    all_passed = print_summary()
    sys.exit(0 if all_passed else 1)