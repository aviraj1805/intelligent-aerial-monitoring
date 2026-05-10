import cv2
import numpy as np
import sys
sys.path.append("detection")
from multi_model_detector import MultiModelDetector


# Create a blank test frame (black image)
frame = np.zeros((640, 640, 3), dtype=np.uint8)

# Load detector
detector = MultiModelDetector(config_path="D:\\intelligent-aerial-monitoring\\models\\model_configs.yaml")

# Run parallel inference
results = detector.run_parallel(frame)

# Print results
for r in results:
    print(f"\nModel : {r['name']}")
    print(f"Detections : {len(r['boxes'])}")
    print(f"Weight : {r['weight']}")

print("\n[PASS] MultiModelDetector working correctly")