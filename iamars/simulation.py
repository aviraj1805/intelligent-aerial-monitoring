"""3D simulation used to evaluate state estimation, prediction and intercept.

SIMULATION ONLY. Nothing here comes from real sensor data.

Each trial:
1. Generate a true 3D drone trajectory (straight, turning or jinking).
2. Produce noisy position measurements at 30 Hz (Gaussian, ``meas_std`` m),
   standing in for a sensor that measures 3D position (e.g. radar or stereo).
3. Run the 3D constant-velocity Kalman filter for ``warmup_s`` seconds.
4. From the filtered position and velocity, solve for the intercept point.
5. Compare the predicted intercept point with where the drone *actually*
   is at the predicted intercept time. That distance is the miss distance.

The "oracle" variant solves the intercept from the *true* state. Its error
comes only from the drone manoeuvring after the decision, which separates
motion-model error from estimation (noise) error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from iamars.estimator import constant_velocity_filter
from iamars.intercept import solve_intercept

SCENARIOS = ("straight", "turning", "jinking")


@dataclass
class SimConfig:
    dt: float = 1 / 30               # sensor rate 30 Hz
    warmup_s: float = 2.0            # filter runs this long before deciding
    interceptor_speed: float = 80.0  # m/s
    meas_std: float = 1.0            # m, per-axis position noise
    filter_accel_std: float = 2.0    # m/s^2, Kalman process noise
    max_intercept_s: float = 15.0
    hit_radius: float = 1.0          # m, success threshold
    range_min: float = 150.0         # m, initial horizontal range
    range_max: float = 400.0
    alt_min: float = 30.0            # m
    alt_max: float = 120.0
    speed_min: float = 8.0           # m/s, drone speed
    speed_max: float = 20.0
    turn_rate_max: float = 0.3       # rad/s, "turning" scenario
    jink_accel_std: float = 3.0      # m/s^2, "jinking" scenario

    def to_dict(self) -> dict:
        return asdict(self)


def simulate_target(scenario: str, cfg: SimConfig, rng: np.random.Generator, duration_s: float):
    """Return true positions and velocities, each of shape (T, 3)."""
    n = int(round(duration_s / cfg.dt)) + 1
    rng_r = rng.uniform(cfg.range_min, cfg.range_max)
    bearing = rng.uniform(0, 2 * np.pi)
    alt = rng.uniform(cfg.alt_min, cfg.alt_max)
    p = np.array([rng_r * np.cos(bearing), rng_r * np.sin(bearing), alt])

    speed = rng.uniform(cfg.speed_min, cfg.speed_max)
    # Heading roughly towards the sensor (+/- 60 deg), a typical approach.
    heading = bearing + np.pi + rng.uniform(-np.pi / 3, np.pi / 3)
    v = np.array([speed * np.cos(heading), speed * np.sin(heading), rng.normal(0, 0.5)])

    turn = rng.uniform(-cfg.turn_rate_max, cfg.turn_rate_max) if scenario == "turning" else 0.0
    a = np.zeros(3)
    pos, vel = np.zeros((n, 3)), np.zeros((n, 3))
    for k in range(n):
        pos[k], vel[k] = p, v
        if scenario == "turning":
            c, s = np.cos(turn * cfg.dt), np.sin(turn * cfg.dt)
            v = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])
        elif scenario == "jinking":
            # Ornstein-Uhlenbeck acceleration: random but smooth manoeuvres.
            a = a * 0.97 + rng.normal(0, cfg.jink_accel_std * 0.25, 3) * np.array([1, 1, 0.3])
            v = v + a * cfg.dt
        elif scenario != "straight":
            raise ValueError(f"unknown scenario {scenario}")
        p = p + v * cfg.dt
        p[2] = max(p[2], 5.0)  # stay above ground
    return pos, vel


def run_trial(scenario: str, cfg: SimConfig, rng: np.random.Generator) -> dict:
    duration = cfg.warmup_s + cfg.max_intercept_s + 1.0
    pos, vel = simulate_target(scenario, cfg, rng, duration)
    k0 = int(round(cfg.warmup_s / cfg.dt))

    meas = pos[: k0 + 1] + rng.normal(0, cfg.meas_std, (k0 + 1, 3))
    kf = constant_velocity_filter(meas[0], dim=3, dt=cfg.dt,
                                  accel_std=cfg.filter_accel_std, meas_std=cfg.meas_std)
    for z in meas[1:]:
        kf.predict()
        kf.update(z)
    p_est, v_est = kf.x[:3], kf.x[3:]

    def true_pos_at(t_s: float) -> np.ndarray:
        idx = min(t_s / cfg.dt, len(pos) - 1.001)
        i = int(idx)
        f = idx - i
        return pos[i] * (1 - f) + pos[i + 1] * f

    out = {
        "scenario": scenario,
        "est_pos_err_m": float(np.linalg.norm(p_est - pos[k0])),
        "est_vel_err_mps": float(np.linalg.norm(v_est - vel[k0])),
    }
    for tag, (p, v) in {"filtered": (p_est, v_est), "oracle": (pos[k0], vel[k0])}.items():
        sol = solve_intercept(p, v, cfg.interceptor_speed, cfg.max_intercept_s)
        if sol is None:
            out[f"{tag}_solved"] = False
            out[f"{tag}_miss_m"] = float("nan")
            out[f"{tag}_t_s"] = float("nan")
            continue
        t_hit, aim = sol
        out[f"{tag}_solved"] = True
        out[f"{tag}_t_s"] = t_hit
        out[f"{tag}_miss_m"] = float(np.linalg.norm(aim - true_pos_at(cfg.warmup_s + t_hit)))
    return out


def summarise(trials: list[dict], hit_radius: float) -> dict:
    """Aggregate a list of trial dicts into the reported metrics."""
    res = {"n_trials": len(trials)}
    res["est_pos_err_m_median"] = float(np.median([t["est_pos_err_m"] for t in trials]))
    res["est_vel_err_mps_median"] = float(np.median([t["est_vel_err_mps"] for t in trials]))
    for tag in ("filtered", "oracle"):
        solved = [t for t in trials if t[f"{tag}_solved"]]
        miss = np.array([t[f"{tag}_miss_m"] for t in solved])
        res[tag] = {
            "solved_fraction": len(solved) / len(trials),
            # Unsolved trials count as failures in the hit rate.
            "hit_rate_within_radius": float(np.sum(miss <= hit_radius) / len(trials)),
            "miss_m_median": float(np.median(miss)) if len(miss) else float("nan"),
            "miss_m_p90": float(np.percentile(miss, 90)) if len(miss) else float("nan"),
            "time_to_intercept_s_median": float(
                np.median([t[f"{tag}_t_s"] for t in solved])) if solved else float("nan"),
        }
    return res
