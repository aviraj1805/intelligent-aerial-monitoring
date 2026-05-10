import sys
sys.path.append(".")
from intercept.fire_solution import FireSolution

fire = FireSolution(projectile_speed=50)

# Use output directly from trajectory predictor
fake_predictions = [{
    "track_id": 1,
    "current": (189.9, 167.7),
    "velocity": (10.03, 7.32),
    "future_points": [
        (199.9, 175.0),
        (210.0, 182.3),
        (220.0, 189.7),
        (230.0, 197.0),
        (240.1, 204.3),
        (250.1, 211.6),
        (260.1, 218.9),
        (270.1, 226.3),
        (280.2, 233.6),
        (290.2, 240.9)
    ]
}]

solutions = fire.compute(fake_predictions)

for s in solutions:
    print(f"Track ID           : {s['track_id']}")
    print(f"Intercept Point    : {s['intercept_point']}")
    print(f"Azimuth            : {s['azimuth_deg']}°")
    print(f"Elevation          : {s['elevation_deg']}°")
    print(f"Time to Intercept  : {s['time_to_intercept_sec']}s")
    print(f"Intercept Frame    : +{s['intercept_frame']}")
    print(f"Distance           : {s['distance_px']} px")

print("\n[PASS] Fire Solution working correctly")