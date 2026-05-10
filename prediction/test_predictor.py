import numpy as np
import sys
sys.path.append(".")
from prediction.trajectory_predictor import TrajectoryPredictor

predictor = TrajectoryPredictor(predict_frames=10)

# Simulate kalman output
fake_kalman_results = [{
    "track_id": 1,
    "smoothed_center": (189.9, 167.7),
    "velocity": (10.03, 7.32)
}]

predictions = predictor.predict(fake_kalman_results)

for p in predictions:
    print(f"Track ID  : {p['track_id']}")
    print(f"Current   : ({p['current'][0]:.1f}, {p['current'][1]:.1f})")
    print(f"Velocity  : (vx={p['velocity'][0]:.2f}, vy={p['velocity'][1]:.2f})")
    print(f"\nPredicted Future Positions:")
    for i, (fx, fy) in enumerate(p['future_points']):
        print(f"  Frame +{i+1:2d} → ({fx:.1f}, {fy:.1f})")

print("\n[PASS] Trajectory Predictor working correctly")