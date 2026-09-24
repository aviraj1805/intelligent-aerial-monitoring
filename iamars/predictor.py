"""Stage 4: kinematic trajectory prediction.

Constant-velocity extrapolation from the Kalman state:

    p(t) = p0 + v * t

It is the simplest physically meaningful model. It is accurate for short
horizons and straight flight, and it degrades when the drone turns or
accelerates. Works in any number of dimensions (2D pixels or 3D metres).
"""

from __future__ import annotations

import numpy as np


def predict_constant_velocity(position, velocity, horizon_s: float, dt: float) -> np.ndarray:
    """Return an (N, D) array of future positions at dt, 2dt, ..., horizon_s."""
    position = np.asarray(position, float)
    velocity = np.asarray(velocity, float)
    steps = max(1, int(round(horizon_s / dt)))
    t = np.arange(1, steps + 1) * dt
    return position[None, :] + t[:, None] * velocity[None, :]
