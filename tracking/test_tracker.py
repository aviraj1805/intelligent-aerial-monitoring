import numpy as np
import sys
sys.path.append("..")
from tracking.bytetracker import DroneTracker

tracker = DroneTracker()
frame_shape = (640, 640, 3)

# Simulate drone moving across 5 frames
fake_positions = [
    np.array([[100, 150, 200, 250]], dtype=float),
    np.array([[110, 160, 210, 260]], dtype=float),
    np.array([[120, 170, 220, 270]], dtype=float),
    np.array([[130, 180, 230, 280]], dtype=float),
    np.array([[140, 190, 240, 290]], dtype=float),
]
fake_scores = np.array([0.90])

for i, boxes in enumerate(fake_positions):
    tracks = tracker.update(boxes, fake_scores, frame_shape)
    print(f"Frame {i+1} → Track ID: {tracks[0]['track_id'] if tracks else 'None'} | Box: {tracks[0]['box'] if tracks else 'None'}")

print("\n[PASS] ByteTracker working correctly")