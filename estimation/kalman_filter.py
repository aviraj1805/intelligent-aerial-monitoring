import numpy as np


class DroneKalmanFilter:
    def __init__(self):
        # State: [x, y, vx, vy]
        self.kalman_filters = {}

    def _init_filter(self, x, y):
        kf = {
            "x": np.array([x, y, 0, 0], dtype=float),  # state
            "P": np.eye(4) * 100,                        # uncertainty
            "F": np.array([                              # transition matrix
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=float),
            "H": np.array([                              # measurement matrix
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ], dtype=float),
            "R": np.eye(2) * 5,                          # measurement noise
            "Q": np.eye(4) * 0.1                         # process noise
        }
        return kf

    def _predict(self, kf):
        kf["x"] = kf["F"] @ kf["x"]
        kf["P"] = kf["F"] @ kf["P"] @ kf["F"].T + kf["Q"]
        return kf

    def _update(self, kf, measurement):
        z = np.array(measurement, dtype=float)
        y = z - kf["H"] @ kf["x"]
        S = kf["H"] @ kf["P"] @ kf["H"].T + kf["R"]
        K = kf["P"] @ kf["H"].T @ np.linalg.inv(S)
        kf["x"] = kf["x"] + K @ y
        kf["P"] = (np.eye(4) - K @ kf["H"]) @ kf["P"]
        return kf

    def update(self, tracks):
        results = []

        for track in tracks:
            tid = track["track_id"]
            box = track["box"]

            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2

            if tid not in self.kalman_filters:
                self.kalman_filters[tid] = self._init_filter(cx, cy)

            kf = self.kalman_filters[tid]
            kf = self._predict(kf)
            kf = self._update(kf, [cx, cy])
            self.kalman_filters[tid] = kf

            results.append({
                "track_id": tid,
                "box": box,
                "smoothed_center": (kf["x"][0], kf["x"][1]),
                "velocity": (kf["x"][2], kf["x"][3])
            })

        return results