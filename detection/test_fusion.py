"""
Test: detection/fusion_engine.py
Run from project root:  python tests/test_fusion_engine.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detection.fusion_engine import FusionEngine
from detection.multi_model_detector import MultiModelDetector


def separator(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


# ── Test 1: Empty input (no detections from any model) ───────────────────────
def test_empty_input():
    separator("TEST 1 — Empty input (no detections)")
    fe = FusionEngine()
    model_results = [
        {"name": "visiodect", "boxes": np.empty((0,4),dtype=np.float32),
         "scores": np.empty((0,),dtype=np.float32),
         "labels": np.empty((0,),dtype=np.int32), "weight": 0.5},
        {"name": "uav_ir",    "boxes": np.empty((0,4),dtype=np.float32),
         "scores": np.empty((0,),dtype=np.float32),
         "labels": np.empty((0,),dtype=np.int32), "weight": 0.3},
        {"name": "uav_rgb",   "boxes": np.empty((0,4),dtype=np.float32),
         "scores": np.empty((0,),dtype=np.float32),
         "labels": np.empty((0,),dtype=np.int32), "weight": 0.2},
    ]
    result = fe.fuse(model_results, frame_shape=(384, 640, 3))
    assert result["boxes"].shape  == (0, 4), f"Expected (0,4), got {result['boxes'].shape}"
    assert result["scores"].shape == (0,),   f"Expected (0,), got {result['scores'].shape}"
    print("  ✓ No crash, empty arrays returned correctly")


# ── Test 2: All 3 models agree on same box ────────────────────────────────────
def test_all_models_agree():
    separator("TEST 2 — All 3 models detect the same box")
    fe = FusionEngine(iou_thr=0.45)

    # Same box (with tiny noise) from all 3 models
    box = np.array([[100, 80, 200, 160]], dtype=np.float32)
    model_results = [
        {"name": "visiodect", "boxes": box + np.array([[0,0,0,0]]),
         "scores": np.array([0.9]), "labels": np.array([0]), "weight": 0.5},
        {"name": "uav_ir",    "boxes": box + np.array([[2,1,-2,-1]]),
         "scores": np.array([0.8]), "labels": np.array([0]), "weight": 0.3},
        {"name": "uav_rgb",   "boxes": box + np.array([[-1,2,1,-2]]),
         "scores": np.array([0.75]),"labels": np.array([0]), "weight": 0.2},
    ]
    result = fe.fuse(model_results, frame_shape=(384, 640, 3))
    assert len(result["boxes"]) == 1, f"Expected 1 fused box, got {len(result['boxes'])}"
    fb = result["boxes"][0]
    print(f"  ✓ Fused box : {fb.astype(int)}  score={result['scores'][0]:.4f}")
    # Fused box should be close to original
    assert abs(fb[0] - 100) < 10, "x1 too far from expected"
    assert abs(fb[2] - 200) < 10, "x2 too far from expected"
    print("  ✓ Box coordinates within expected range")


# ── Test 3: Two distinct boxes (should stay separate) ────────────────────────
def test_two_separate_boxes():
    separator("TEST 3 — Two separate targets (should not merge)")
    fe = FusionEngine(iou_thr=0.45)

    box_a = np.array([[50,  50,  100, 100]], dtype=np.float32)
    box_b = np.array([[400, 300, 500, 400]], dtype=np.float32)

    model_results = [
        {"name": "visiodect", "boxes": np.vstack([box_a, box_b]),
         "scores": np.array([0.9, 0.85]), "labels": np.array([0, 0]), "weight": 0.5},
        {"name": "uav_ir",    "boxes": np.empty((0,4), dtype=np.float32),
         "scores": np.empty((0,)), "labels": np.empty((0,), dtype=np.int32), "weight": 0.3},
        {"name": "uav_rgb",   "boxes": np.empty((0,4), dtype=np.float32),
         "scores": np.empty((0,)), "labels": np.empty((0,), dtype=np.int32), "weight": 0.2},
    ]
    result = fe.fuse(model_results, frame_shape=(384, 640, 3))
    print(f"  Fused boxes count : {len(result['boxes'])}")
    for i, (b, s) in enumerate(zip(result["boxes"], result["scores"])):
        print(f"    Box {i}: {b.astype(int)}  score={s:.4f}")
    assert len(result["boxes"]) == 2, f"Expected 2 boxes, got {len(result['boxes'])}"
    print("  ✓ Two boxes kept separate")


# ── Test 4: Only one model fires ──────────────────────────────────────────────
def test_single_model_fires():
    separator("TEST 4 — Only one model detects (others empty)")
    fe = FusionEngine()
    model_results = [
        {"name": "visiodect", "boxes": np.array([[120,80,240,160]], dtype=np.float32),
         "scores": np.array([0.88]), "labels": np.array([0]), "weight": 0.5},
        {"name": "uav_ir",    "boxes": np.empty((0,4), dtype=np.float32),
         "scores": np.empty((0,)), "labels": np.empty((0,), dtype=np.int32), "weight": 0.3},
        {"name": "uav_rgb",   "boxes": np.empty((0,4), dtype=np.float32),
         "scores": np.empty((0,)), "labels": np.empty((0,), dtype=np.int32), "weight": 0.2},
    ]
    result = fe.fuse(model_results, frame_shape=(384, 640, 3))
    assert len(result["boxes"]) == 1
    print(f"  ✓ Single detection passed through: {result['boxes'][0].astype(int)}  score={result['scores'][0]:.4f}")


# ── Test 5: Full pipeline — real video frame ──────────────────────────────────
def test_real_video_frame():
    separator("TEST 5 — Full pipeline on real video frame")

    video_path = "data/sample_frames/test_drone4.mp4.mp4"
    if not os.path.exists(video_path):
        print(f"  [SKIP] Video not found at {video_path}")
        return

    import cv2
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("  [SKIP] Could not read frame")
        return

    detector = MultiModelDetector(device="cuda")
    fe       = FusionEngine(iou_thr=0.45, skip_box_thr=0.05)

    model_results = detector.run_parallel(frame)
    fused         = fe.fuse(model_results, frame_shape=frame.shape)

    print(f"  Frame shape   : {frame.shape}")
    print(f"  Raw detections per model:")
    for r in model_results:
        print(f"    {r['name']:12s}  boxes={len(r['boxes'])}")
    print(f"  Fused boxes   : {len(fused['boxes'])}")
    for i, (b, s) in enumerate(zip(fused["boxes"], fused["scores"])):
        print(f"    Fused box {i}: {b.astype(int)}  score={s:.4f}  label={fused['labels'][i]}")

    assert isinstance(fused["boxes"],  np.ndarray)
    assert isinstance(fused["scores"], np.ndarray)
    assert isinstance(fused["labels"], np.ndarray)
    print("  ✓ Fusion output shapes and types correct")


if __name__ == "__main__":
    separator("IAMARS — FusionEngine Test Suite")
    try:
        test_empty_input()
        test_all_models_agree()
        test_two_separate_boxes()
        test_single_model_fires()
        test_real_video_frame()

        separator("ALL TESTS PASSED ✓")
        print("  Ready to proceed to File 3 — bytetracker.py\n")

    except AssertionError as e:
        print(f"\n  [FAIL] {e}")
        sys.exit(1)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)