import numpy as np
from fusion_engine import FusionEngine

# Simulate model results (fake detections)
model_results = [
    {
        "name": "visiodect",
        "boxes": np.array([[100, 150, 200, 250]], dtype=float),
        "scores": np.array([0.92]),
        "weight": 0.6
    },
    {
        "name": "dut_anti_uav",
        "boxes": np.array([[105, 155, 205, 255]], dtype=float),
        "scores": np.array([0.87]),
        "weight": 0.4
    }
]

frame_shape = (640, 640, 3)

fusion = FusionEngine(config_path="D:\\intelligent-aerial-monitoring\\models\\model_configs.yaml")
boxes, scores = fusion.fuse(model_results, frame_shape)

print(f"Fused Boxes  : {boxes}")
print(f"Fused Scores : {scores}")
print(f"Total Detections after Fusion : {len(boxes)}")
print("\n[PASS] FusionEngine working correctly")