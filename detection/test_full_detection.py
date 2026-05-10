import numpy as np
import sys
sys.path.append(".")
from detection.multi_model_detector import MultiModelDetector
from detection.fusion_engine import FusionEngine
from tracking.bytetracker import DroneTracker

# Initialize all three components
detector = MultiModelDetector(config_path="D:\\intelligent-aerial-monitoring\\models\\model_configs.yaml")
fusion = FusionEngine(config_path="D:\\intelligent-aerial-monitoring\\models\\model_configs.yaml")
tracker = DroneTracker()

frame_shape = (640, 640, 3)

# Simulate 5 frames
fake_frames = [np.zeros((640, 640, 3), dtype=np.uint8)] * 5

print("Running Detection → Fusion → Tracking pipeline...\n")

for i, frame in enumerate(fake_frames):
    # Step 1: Parallel inference
    model_results = detector.run_parallel(frame)

    # Step 2: Fuse results
    boxes, scores = fusion.fuse(model_results, frame_shape)

    # Step 3: Track
    tracks = tracker.update(boxes, scores, frame_shape)

    print(f"Frame {i+1} → Detections: {len(boxes)} | Tracks: {len(tracks)}")

print("\n[PASS] Detection → Fusion → Tracking pipeline connected successfully")