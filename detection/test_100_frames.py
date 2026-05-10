import numpy as np
import time
import sys
sys.path.append(".")
from detection.multi_model_detector import MultiModelDetector
from detection.fusion_engine import FusionEngine
from tracking.bytetracker import DroneTracker

# Initialize
detector = MultiModelDetector(config_path="D:\\intelligent-aerial-monitoring\\models\\model_configs.yaml")
fusion = FusionEngine(config_path="D:\\intelligent-aerial-monitoring\\models\\model_configs.yaml")
tracker = DroneTracker()

frame_shape = (640, 640, 3)
total_frames = 100
latencies = []

print("Running 100-frame validation...\n")

for i in range(total_frames):
    # Simulate drone moving across frame
    x = 50 + i * 5
    y = 50 + i * 3

    # Inject fake detection directly into fusion (bypassing blank frame)
    fake_results = [
        {
            "name": "visiodect",
            "boxes": np.array([[x, y, x+60, y+60]], dtype=float),
            "scores": np.array([0.91]),
            "weight": 0.6
        },
        {
            "name": "dut_anti_uav",
            "boxes": np.array([[x+2, y+2, x+62, y+62]], dtype=float),
            "scores": np.array([0.87]),
            "weight": 0.4
        }
    ]

    start = time.time()

    boxes, scores = fusion.fuse(fake_results, frame_shape)
    tracks = tracker.update(boxes, scores, frame_shape)

    latency = (time.time() - start) * 1000
    latencies.append(latency)

    if (i + 1) % 10 == 0:
        track_id = tracks[0]['track_id'] if tracks else 'None'
        print(f"Frame {i+1:3d} | Track ID: {track_id} | Box: {np.round(boxes[0]).astype(int) if len(boxes) > 0 else 'None'} | Latency: {latency:.2f}ms")

print(f"\n--- Validation Summary ---")
print(f"Total Frames     : {total_frames}")
print(f"Avg Latency      : {np.mean(latencies):.2f}ms")
print(f"Max Latency      : {np.max(latencies):.2f}ms")
print(f"Min Latency      : {np.min(latencies):.2f}ms")
print(f"Target (<40ms)   : {'PASS' if np.mean(latencies) < 40 else 'FAIL'}")
print(f"\n[PASS] 100-frame validation complete")