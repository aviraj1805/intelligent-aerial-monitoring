import numpy as np


class FireSolution:
    def __init__(self, projectile_speed=300, dt=1/30):
        # projectile_speed in pixels/second (tunable)
        self.projectile_speed = projectile_speed
        self.dt = dt

    def compute(self, predictions):
        solutions = []

        for p in predictions:
            cx, cy = p["current"]
            future_points = p["future_points"]

            best = None
            for i, (fx, fy) in enumerate(future_points):
                time_to_reach = (i + 1) * self.dt

                dx = fx - cx
                dy = fy - cy
                distance = np.sqrt(dx**2 + dy**2)

                projectile_distance = self.projectile_speed * (i + 1)

                if projectile_distance >= distance:
                    azimuth = np.degrees(np.arctan2(dx, -dy)) % 360
                    elevation = np.degrees(np.arctan2(-dy, max(distance, 1e-6)))

                    best = {
                        "track_id": p["track_id"],
                        "intercept_point": (fx, fy),
                        "azimuth_deg": round(azimuth, 2),
                        "elevation_deg": round(elevation, 2),
                        "time_to_intercept_sec": round(time_to_reach, 4),
                        "intercept_frame": i + 1,
                        "distance_px": round(distance, 2)
                    }
                    break

            if best is None:
                best = {
                    "track_id": p["track_id"],
                    "intercept_point": None,
                    "azimuth_deg": None,
                    "elevation_deg": None,
                    "time_to_intercept_sec": None,
                    "intercept_frame": None,
                    "distance_px": None
                }

            solutions.append(best)

        return solutions