"""Stage 3: state estimation with a linear Kalman filter (NumPy).

A Kalman filter keeps a *belief* about a hidden state (here: position and
velocity) as a mean ``x`` and a covariance ``P`` (how unsure we are).
Every frame it does two steps:

predict:  x = F x           (move the state forward with a motion model)
          P = F P F^T + Q   (uncertainty grows; Q = how much the motion
                             model can be wrong, e.g. the drone turns)
update:   y = z - H x               (innovation: measurement minus prediction)
          S = H P H^T + R           (expected spread of y; R = sensor noise)
          K = P H^T S^-1            (Kalman gain: how much to trust z vs. x)
          x = x + K y
          P = (I - K H) P

If R is large (noisy detector) K is small and the filter trusts its
prediction. If Q is large (agile drone) K is large and it follows the
measurements. The same class is used for 2D image boxes in the video
pipeline and for 3D positions in the intercept simulation.
"""

from __future__ import annotations

import numpy as np


class LinearKalmanFilter:
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = np.asarray(F, float)
        self.H = np.asarray(H, float)
        self.Q = np.asarray(Q, float)
        self.R = np.asarray(R, float)
        self.x = np.asarray(x0, float).copy()
        self.P = np.asarray(P0, float).copy()

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z) -> np.ndarray:
        z = np.asarray(z, float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        return self.x


def constant_velocity_filter(
    z0, dim: int, dt: float, accel_std: float, meas_std: float, extra_dims: int = 0
) -> LinearKalmanFilter:
    """Constant-velocity filter in ``dim`` spatial dimensions.

    State = [p_1..p_dim, e_1..e_extra, v_1..v_dim] where ``e`` are extra
    measured quantities with no velocity (box width and height in 2D).
    Q comes from the standard "white-noise acceleration" model: an unknown
    acceleration with standard deviation ``accel_std`` acts during each step.
    """
    n_meas = dim + extra_dims
    n = n_meas + dim
    F = np.eye(n)
    for i in range(dim):
        F[i, n_meas + i] = dt
    H = np.zeros((n_meas, n))
    H[:, :n_meas] = np.eye(n_meas)

    q = accel_std**2
    Q = np.zeros((n, n))
    for i in range(dim):
        p, v = i, n_meas + i
        Q[p, p] = q * dt**4 / 4
        Q[p, v] = Q[v, p] = q * dt**3 / 2
        Q[v, v] = q * dt**2
    for j in range(dim, n_meas):  # box size drifts slowly
        Q[j, j] = (0.05 * meas_std) ** 2
    R = np.eye(n_meas) * meas_std**2

    x0 = np.zeros(n)
    x0[:n_meas] = z0
    P0 = np.diag([meas_std**2] * n_meas + [(accel_std * 10) ** 2] * dim)
    return LinearKalmanFilter(F, H, Q, R, x0, P0)


class BoxKalmanManager:
    """One Kalman filter per track ID, in image pixels.

    State [cx, cy, w, h, vx, vy]; units are pixels and pixels per second.
    When a track is missing for a frame the filter only *predicts*
    ("coasts") instead of being deleted, so its velocity estimate survives
    short gaps. Filters are deleted after ``max_coast`` missed frames.
    """

    def __init__(
        self,
        fps: float = 30.0,
        accel_std: float = 400.0,
        meas_std: float = 4.0,
        max_coast: int = 30,
    ):
        self.dt = 1.0 / fps
        self.accel_std = accel_std
        self.meas_std = meas_std
        self.max_coast = max_coast
        self.filters: dict[int, LinearKalmanFilter] = {}
        self.missed: dict[int, int] = {}
        self.age: dict[int, int] = {}

    def update(self, ids, xyxy) -> dict[int, np.ndarray]:
        """Advance all filters one frame. Returns {track_id: state} for ids seen now."""
        seen = set(int(i) for i in ids)
        for tid in list(self.filters):
            self.filters[tid].predict()
            if tid not in seen:
                self.missed[tid] += 1
                if self.missed[tid] > self.max_coast:
                    del self.filters[tid], self.missed[tid], self.age[tid]

        states = {}
        for tid, box in zip(ids, xyxy):
            tid = int(tid)
            z = np.array(
                [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]]
            )
            if tid not in self.filters:
                self.filters[tid] = constant_velocity_filter(
                    z, dim=2, dt=self.dt, accel_std=self.accel_std,
                    meas_std=self.meas_std, extra_dims=2,
                )
                self.age[tid] = 0
            else:
                self.filters[tid].update(z)
            self.missed[tid] = 0
            self.age[tid] += 1
            states[tid] = self.filters[tid].x.copy()
        return states

    def reset(self) -> None:
        self.filters.clear()
        self.missed.clear()
        self.age.clear()
