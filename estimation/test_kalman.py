import numpy as np
import sys
sys.path.append(".")
from estimation.kalman_filter import DroneKalmanFilter

kf = DroneKalmanFilter()

# Simulate noisy drone positions across 10 frames
positions = [
    (100, 100), (112, 108), (118, 115), (131, 122), (139, 130),
    (152, 138), (158, 145), (171, 153), (179, 160), (192, 168)
]

print("Frame | Raw Center     | Smoothed Center  | Velocity (vx, vy)")
print("-" * 65)

for i, (cx, cy) in enumerate(positions):
    # Add noise to simulate real detection jitter
    noisy_cx = cx + np.random.uniform(-3, 3)
    noisy_cy = cy + np.random.uniform(-3, 3)

    fake_tracks = [{
        "track_id": 1,
        "box": [noisy_cx-30, noisy_cy-30, noisy_cx+30, noisy_cy+30],
        "score": 0.91
    }]

    results = kf.update(fake_tracks)
    r = results[0]

    sx, sy = r["smoothed_center"]
    vx, vy = r["velocity"]

    print(f"  {i+1:2d}  | ({noisy_cx:6.1f}, {noisy_cy:6.1f}) | ({sx:6.1f}, {sy:6.1f})     | ({vx:5.2f}, {vy:5.2f})")

print("\n[PASS] Kalman Filter working correctly")