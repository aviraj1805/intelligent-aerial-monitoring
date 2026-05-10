"""
IAMARS — TrajectoryPredictor
Predicts the next N frame positions for each tracked target
using the current Kalman-smoothed position and velocity.

Prediction model: constant-velocity linear extrapolation
    future_cx[t] = cx + vx * t
    future_cy[t] = cy + vy * t
    (t = 1 .. horizon)
"""

import numpy as np


class TrajectoryPredictor:
    """
    Parameters
    ----------
    horizon : int — number of future frames to predict (default 10)
    """

    def __init__(self, horizon: int = 10):
        self.horizon = horizon

    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, smoothed: dict) -> dict:
        """
        Parameters
        ----------
        smoothed : dict from KalmanFilterManager.update()
            {boxes (K,4) xyxy, scores, labels, track_ids, velocities (K,2)}

        Returns
        -------
        Same dict + 'trajectories': list of K arrays, each (horizon, 2)
            trajectories[i][t] = (cx, cy) of track i at t frames ahead
        """
        boxes      = smoothed["boxes"]       # (K, 4)
        velocities = smoothed["velocities"]  # (K, 2)

        if len(boxes) == 0:
            return {**smoothed, "trajectories": []}

        trajectories = []

        for box, vel in zip(boxes, velocities):
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            vx, vy = float(vel[0]), float(vel[1])

            # t = 1 .. horizon  (t=0 is the current frame)
            t       = np.arange(1, self.horizon + 1, dtype=np.float32)
            fut_cx  = cx + vx * t
            fut_cy  = cy + vy * t

            traj = np.stack([fut_cx, fut_cy], axis=1)  # (horizon, 2)
            trajectories.append(traj)

        return {**smoothed, "trajectories": trajectories}

    # ─────────────────────────────────────────────────────────────────────────
    def predict_single(
        self,
        cx: float,
        cy: float,
        vx: float,
        vy: float,
    ) -> np.ndarray:
        """
        Predict trajectory for one target.

        Returns
        -------
        np.ndarray (horizon, 2) — future (cx, cy) positions
        """
        t      = np.arange(1, self.horizon + 1, dtype=np.float32)
        fut_cx = cx + vx * t
        fut_cy = cy + vy * t
        return np.stack([fut_cx, fut_cy], axis=1)