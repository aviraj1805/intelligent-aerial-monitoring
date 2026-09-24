"""Stage 5: intercept-point prediction (evaluated in simulation only).

Question answered: an interceptor leaves the sensor position (the origin)
at constant speed ``s`` in a straight line. Where should it aim so that it
arrives at the same place and time as the target?

If the target is at ``p`` and moves with constant velocity ``v``, its
future position is ``p + v t``. The interceptor covers ``s t`` metres in
time t. They meet when the distances match:

    |p + v t| = s t
    (v.v - s^2) t^2 + 2 (p.v) t + p.p = 0

This is a quadratic in t. The smallest positive root is the earliest
intercept time, and the aim point is ``p + v t``. There is no solution
when the target is faster than the interceptor and moving away.

This replaces the earlier pixel-space version, which had no metric units
and multiplied the projectile speed by the absolute video frame number.
A single camera cannot measure range, so the 3D version is only run on
simulated data (see iamars/simulation.py).
"""

from __future__ import annotations

import numpy as np


def solve_intercept(
    p, v, interceptor_speed: float, max_time: float = np.inf
) -> tuple[float, np.ndarray] | None:
    """Return (time_to_intercept_s, aim_point) or None if unreachable."""
    p = np.asarray(p, float)
    v = np.asarray(v, float)
    s = float(interceptor_speed)
    a = v @ v - s * s
    b = 2.0 * (p @ v)
    c = p @ p

    if abs(a) < 1e-9:  # target speed equals interceptor speed: linear equation
        roots = [-c / b] if abs(b) > 1e-12 else []
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        roots = [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]

    positive = [t for t in roots if t > 1e-9]
    if not positive:
        return None
    t = min(positive)
    if t > max_time:
        return None
    return float(t), p + v * t
