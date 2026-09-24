"""Kalman filter, trajectory prediction and intercept solver."""

import numpy as np
import pytest

from iamars.estimator import BoxKalmanManager, constant_velocity_filter
from iamars.intercept import solve_intercept
from iamars.predictor import predict_constant_velocity


def test_kalman_recovers_velocity_from_noisy_positions():
    rng = np.random.default_rng(0)
    dt, v_true = 1 / 30, np.array([5.0, -2.0, 1.0])
    kf = constant_velocity_filter(np.zeros(3), dim=3, dt=dt, accel_std=1.0, meas_std=0.5)
    for k in range(1, 150):
        kf.predict()
        kf.update(v_true * k * dt + rng.normal(0, 0.5, 3))
    assert np.allclose(kf.x[3:], v_true, atol=0.5)


def test_kalman_smooths_noise():
    rng = np.random.default_rng(1)
    dt = 1 / 30
    kf = constant_velocity_filter(np.zeros(2), dim=2, dt=dt, accel_std=0.5, meas_std=2.0)
    raw_err, kf_err = [], []
    for k in range(1, 200):
        truth = np.array([3.0, 1.0]) * k * dt
        z = truth + rng.normal(0, 2.0, 2)
        kf.predict()
        kf.update(z)
        if k > 50:
            raw_err.append(np.linalg.norm(z - truth))
            kf_err.append(np.linalg.norm(kf.x[:2] - truth))
    assert np.mean(kf_err) < 0.5 * np.mean(raw_err)


def test_box_manager_coasts_through_missed_frames():
    m = BoxKalmanManager(fps=30, max_coast=5)
    for k in range(20):
        m.update([1], [[100 + 3 * k, 50, 120 + 3 * k, 70]])
    for _ in range(3):  # track missing for 3 frames: filter kept
        m.update([], np.empty((0, 4)))
    assert 1 in m.filters
    vx = m.filters[1].x[4]
    assert 60 < vx < 120  # 3 px/frame * 30 fps = 90 px/s
    for _ in range(3):  # 6 misses > max_coast: filter deleted
        m.update([], np.empty((0, 4)))
    assert 1 not in m.filters


def test_constant_velocity_prediction():
    traj = predict_constant_velocity([0, 0], [10, 5], horizon_s=1.0, dt=0.5)
    assert np.allclose(traj, [[5, 2.5], [10, 5]])


def test_intercept_stationary_target():
    t, aim = solve_intercept([100.0, 0, 0], [0, 0, 0], interceptor_speed=50)
    assert t == pytest.approx(2.0)
    assert np.allclose(aim, [100, 0, 0])


def test_intercept_point_is_consistent():
    p, v, s = np.array([300.0, 100, 50]), np.array([-10.0, 5, 0]), 80.0
    t, aim = solve_intercept(p, v, s)
    assert np.allclose(aim, p + v * t)
    assert np.linalg.norm(aim) == pytest.approx(s * t)  # interceptor travels s*t


def test_intercept_impossible_when_target_is_faster_and_fleeing():
    assert solve_intercept([100.0, 0, 0], [60.0, 0, 0], interceptor_speed=50) is None


def test_intercept_respects_max_time():
    assert solve_intercept([1000.0, 0, 0], [0, 0, 0], 10, max_time=5) is None
