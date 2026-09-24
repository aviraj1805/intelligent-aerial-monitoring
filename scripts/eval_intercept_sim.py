"""Evaluate intercept-point prediction in 3D SIMULATION (no real data).

    python scripts/eval_intercept_sim.py              # 1,000 trials per scenario
    python scripts/eval_intercept_sim.py --trials 200 # quicker

Metric (defined once, used everywhere):
    hit rate = fraction of trials where the predicted intercept point is
               within 1.0 m of the simulated drone's true position at the
               predicted intercept time. Trials with no solution count as misses.

Also reported: median / 90th-percentile miss distance, and the same numbers for
an "oracle" that knows the true state (isolates manoeuvre error from noise).
A sweep over sensor noise shows how sensitive the result is to assumptions.
Writes results/intercept_simulation.json.
"""

import argparse
import json
from dataclasses import replace

import _bootstrap  # noqa: F401
import numpy as np

from iamars import config
from iamars.simulation import SCENARIOS, SimConfig, run_trial, summarise


def evaluate(cfg: SimConfig, trials: int, seed: int) -> dict:
    out = {}
    for i, sc in enumerate(SCENARIOS):
        rng = np.random.default_rng(seed + i)
        out[sc] = summarise([run_trial(sc, cfg, rng) for _ in range(trials)], cfg.hit_radius)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = SimConfig()
    res = {
        "label": "SIMULATION ONLY - synthetic trajectories and synthetic sensor noise",
        "metric": f"hit rate = share of trials with miss distance <= {base.hit_radius} m",
        "config": base.to_dict(),
        "trials_per_scenario": args.trials,
        "seed": args.seed,
        "results": evaluate(base, args.trials, args.seed),
        "noise_sweep": {},
    }
    for noise in (0.25, 0.5, 1.0, 2.0):
        r = evaluate(replace(base, meas_std=noise), args.trials, args.seed)
        res["noise_sweep"][f"{noise} m"] = {sc: r[sc]["filtered"]["hit_rate_within_radius"] for sc in SCENARIOS}

    config.RESULTS_DIR.mkdir(exist_ok=True)
    out = config.RESULTS_DIR / "intercept_simulation.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(res["label"])
    print(f"{'scenario':10s} {'hit<=1m':>8s} {'median miss':>12s} {'p90 miss':>9s} {'oracle hit':>10s}")
    for sc in SCENARIOS:
        f, o = res["results"][sc]["filtered"], res["results"][sc]["oracle"]
        print(f"{sc:10s} {f['hit_rate_within_radius']:8.3f} {f['miss_m_median']:11.2f}m "
              f"{f['miss_m_p90']:8.2f}m {o['hit_rate_within_radius']:10.3f}")
    print("noise sweep (filtered hit rate):", json.dumps(res["noise_sweep"]))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
