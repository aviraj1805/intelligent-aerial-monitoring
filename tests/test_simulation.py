"""3D simulation used for the intercept evaluation."""

import numpy as np

from iamars.simulation import SimConfig, run_trial, simulate_target, summarise


def test_straight_trajectory_is_constant_velocity():
    cfg = SimConfig()
    pos, vel = simulate_target("straight", cfg, np.random.default_rng(0), 3.0)
    assert np.allclose(vel, vel[0])
    assert np.allclose(np.diff(pos, axis=0), vel[:-1] * cfg.dt, atol=1e-9)


def test_turning_keeps_speed():
    pos, vel = simulate_target("turning", SimConfig(), np.random.default_rng(0), 3.0)
    speed = np.linalg.norm(vel, axis=1)
    assert np.allclose(speed, speed[0], rtol=1e-6)


def test_oracle_hits_straight_targets():
    cfg = SimConfig()
    rng = np.random.default_rng(0)
    trials = [run_trial("straight", cfg, rng) for _ in range(30)]
    s = summarise(trials, cfg.hit_radius)
    assert s["oracle"]["hit_rate_within_radius"] == 1.0


def test_simulation_is_reproducible():
    a = run_trial("jinking", SimConfig(), np.random.default_rng(7))
    b = run_trial("jinking", SimConfig(), np.random.default_rng(7))
    assert a == b
