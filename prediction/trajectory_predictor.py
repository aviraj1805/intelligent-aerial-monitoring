import numpy as np


class TrajectoryPredictor:
    def __init__(self, predict_frames=10, dt=1/30):
        self.predict_frames = predict_frames
        self.dt = dt

    def predict(self, kalman_results):
        predictions = []

        for r in kalman_results:
            cx, cy = r["smoothed_center"]
            vx, vy = r["velocity"]

            future_points = []
            for t in range(1, self.predict_frames + 1):
                fx = cx + vx * t * self.dt * 30
                fy = cy + vy * t * self.dt * 30
                future_points.append((fx, fy))

            predictions.append({
                "track_id": r["track_id"],
                "current": (cx, cy),
                "velocity": (vx, vy),
                "future_points": future_points
            })

        return predictions