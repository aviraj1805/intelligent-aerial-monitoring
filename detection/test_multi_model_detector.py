"""
Test: detection/multi_model_detector.py
Run from project root:  python tests/test_multi_model_detector.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detection.multi_model_detector import MultiModelDetector

REQUIRED_KEYS = {"name", "boxes", "scores", "labels", "weight"}


def separator(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def test_model_loading():
    separator("TEST 1 — Model Loading")
    detector = MultiModelDetector(device="cuda")

    loaded = len(detector.models)
    print(f"  Detection models loaded : {loaded} / 3")
    for m in detector.models:
        print(f"    ✓  {m['name']}  (weight={m['weight']}, conf={m['conf']})")

    audio_ok = detector._audio_model is not None
    print(f"  Audio model loaded      : {'✓ YES' if audio_ok else '✗ NO (check path)'}")

    assert loaded > 0, "No detection models loaded — check model paths!"
    return detector


def test_parallel_inference(detector: MultiModelDetector):
    separator("TEST 2 — Parallel Inference on Dummy Frame")

    # Simulate a 384x640 BGR frame (same as test video resolution)
    dummy_frame = np.random.randint(0, 255, (384, 640, 3), dtype=np.uint8)

    t0 = time.perf_counter()
    results = detector.run_parallel(dummy_frame)
    elapsed = time.perf_counter() - t0

    print(f"  run_parallel() returned : {len(results)} results in {elapsed:.3f}s")

    for r in results:
        assert r is not None, f"Result slot is None for a model!"
        assert REQUIRED_KEYS.issubset(r.keys()), f"Missing keys in result: {r.keys()}"

        boxes  = r["boxes"]
        scores = r["scores"]
        labels = r["labels"]

        assert isinstance(boxes,  np.ndarray), "boxes must be ndarray"
        assert isinstance(scores, np.ndarray), "scores must be ndarray"
        assert isinstance(labels, np.ndarray), "labels must be ndarray"

        assert boxes.ndim  == 2 and boxes.shape[1] == 4 or boxes.shape == (0, 4) or len(boxes) == 0, \
            f"boxes shape wrong: {boxes.shape}"

        print(f"    ✓  {r['name']:12s}  detections={len(boxes)}  "
              f"boxes={boxes.shape}  scores={scores.shape}  weight={r['weight']}")

    return results


def test_real_video_frame(detector: MultiModelDetector):
    separator("TEST 3 — Inference on Real Video Frame")

    video_path = "data/sample_frames/test_drone4.mp4.mp4"
    if not os.path.exists(video_path):
        print(f"  [SKIP] Video not found at {video_path}")
        return

    import cv2
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("  [SKIP] Could not read first frame from video")
        return

    print(f"  Frame shape : {frame.shape}")

    t0 = time.perf_counter()
    results = detector.run_parallel(frame)
    elapsed = time.perf_counter() - t0

    print(f"  Inference time : {elapsed:.3f}s")
    for r in results:
        print(f"    ✓  {r['name']:12s}  detections={len(r['boxes'])}")


def test_single_model_crash_isolation(detector: MultiModelDetector):
    separator("TEST 4 — Single Model Crash Isolation")

    # Corrupt one model temporarily to simulate a crash
    original_model = detector.models[0]["model"]
    detector.models[0]["model"] = None   # will cause AttributeError on predict()

    dummy_frame = np.random.randint(0, 255, (384, 640, 3), dtype=np.uint8)

    try:
        results = detector.run_parallel(dummy_frame)
        print(f"  Pipeline survived model crash ✓  ({len(results)} results returned)")
        for r in results:
            assert r is not None, "Crashed model slot returned None!"
            print(f"    {r['name']:12s}  boxes={r['boxes'].shape}")
    finally:
        detector.models[0]["model"] = original_model  # restore
        print("  Model restored ✓")


def test_audio_flag(detector: MultiModelDetector):
    separator("TEST 5 — Audio Classifier Flag")

    dummy_spec = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    conf = detector.run_audio(dummy_spec)

    assert isinstance(conf, float), "run_audio must return a float"
    assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"
    print(f"  run_audio() returned confidence = {conf:.4f}  ✓")


if __name__ == "__main__":
    separator("IAMARS — MultiModelDetector Test Suite")

    try:
        detector = test_model_loading()
        test_parallel_inference(detector)
        test_real_video_frame(detector)
        test_single_model_crash_isolation(detector)
        test_audio_flag(detector)

        separator("ALL TESTS PASSED ✓")
        print("  Ready to proceed to File 2 — fusion_engine.py\n")

    except AssertionError as e:
        print(f"\n  [FAIL] Assertion error: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n  [ERROR] Unexpected failure:")
        traceback.print_exc()
        sys.exit(1)