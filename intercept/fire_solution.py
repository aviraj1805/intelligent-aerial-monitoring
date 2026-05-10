"""
IAMARS — FireSolution
Computes azimuth, elevation, and time-to-intercept for each tracked target.

Coordinate convention
---------------------
  Image centre  = weapon mount position (origin)
  +X right      = East
  +Y down       = increases elevation angle downward (drone above → negative elevation)

Azimuth   : angle from image +X axis, clockwise, degrees [0, 360)
Elevation : vertical angle from horizontal, degrees (+up / -down)

Time-to-intercept
-----------------
  Per spec: projectile_distance = projectile_speed * (frame_index + 1)
  We find the first predicted frame t where projectile_distance >= drone_distance.

Threat level
------------
  Based on drone distance from image centre, normalised to [0.0, 1.0].
  Closer = higher threat.
"""

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_PROJECTILE_SPEED = 300.0   # pixels per frame (tunable)
DEFAULT_FOCAL_LENGTH     = 800.0   # pixels (approximate, tunable)
MAX_THREAT_DISTANCE      = 1000.0  # pixels — distance considered "no threat"


class FireSolution:
    """
    Parameters
    ----------
    projectile_speed : float — pixels/frame (proxy for m/s, tunable)
    focal_length     : float — camera focal length in pixels (for elevation)
    frame_wh         : tuple — (W, H) of the video frame
    """

    def __init__(
        self,
        projectile_speed: float = DEFAULT_PROJECTILE_SPEED,
        focal_length:     float = DEFAULT_FOCAL_LENGTH,
        frame_wh:         tuple = (1920, 1080),
    ):
        self.projectile_speed = projectile_speed
        self.focal_length     = focal_length
        self.frame_wh         = frame_wh   # (W, H)

    # ─────────────────────────────────────────────────────────────────────────
    def compute(self, predicted: dict, frame_index: int) -> list[dict]:
        """
        Parameters
        ----------
        predicted   : dict from TrajectoryPredictor.predict()
                      {boxes, scores, labels, track_ids, velocities, trajectories}
        frame_index : int — current frame number (0-based)

        Returns
        -------
        list of dicts, one per tracked target:
            track_id        int
            cx, cy          float  — current box centre (pixels)
            azimuth         float  — degrees [0, 360)
            elevation       float  — degrees (+ = above horizon)
            distance        float  — pixels from image centre
            t_intercept     float  — frames until intercept (or inf)
            intercept_pos   (x,y)  — predicted intercept position (pixels)
            threat_level    float  — [0.0, 1.0]  1.0 = maximum threat
            velocity        (vx,vy)
        """
        boxes        = predicted["boxes"]        # (K, 4)
        track_ids    = predicted["track_ids"]    # (K,)
        velocities   = predicted["velocities"]   # (K, 2)
        trajectories = predicted["trajectories"] # list of K arrays (horizon, 2)

        W, H   = self.frame_wh
        origin = np.array([W / 2.0, H / 2.0])   # image centre = weapon mount

        solutions = []

        for i in range(len(boxes)):
            box  = boxes[i]
            tid  = int(track_ids[i])
            vel  = velocities[i]
            traj = trajectories[i] if i < len(trajectories) else None

            cx = float((box[0] + box[2]) / 2.0)
            cy = float((box[1] + box[3]) / 2.0)

            # ── Vector from origin to target ─────────────────────────────────
            dx = cx - origin[0]
            dy = cy - origin[1]   # positive = downward in image

            distance = float(np.sqrt(dx**2 + dy**2))

            # ── Azimuth (clockwise from +X / East) ───────────────────────────
            azimuth = float(np.degrees(np.arctan2(dy, dx))) % 360.0

            # ── Elevation (above horizon = positive) ─────────────────────────
            # dy negative → target above image centre → positive elevation
            elevation = float(np.degrees(np.arctan2(-dy, self.focal_length)))

            # ── Time-to-intercept ─────────────────────────────────────────────
            # projectile_distance = projectile_speed * (frame_index + t)
            # intercept when projectile_distance >= drone_distance at future t
            t_intercept   = float("inf")
            intercept_pos = (cx, cy)

            if traj is not None and distance > 0:
                for t_ahead, (fut_cx, fut_cy) in enumerate(traj, start=1):
                    fut_dx   = float(fut_cx) - origin[0]
                    fut_dy   = float(fut_cy) - origin[1]
                    drone_dist = np.sqrt(fut_dx**2 + fut_dy**2)

                    # Per spec: projectile_speed * (frame_index + 1)
                    # Here frame_index advances by t_ahead steps
                    proj_dist = self.projectile_speed * (frame_index + t_ahead)

                    if proj_dist >= drone_dist:
                        t_intercept   = float(t_ahead)
                        intercept_pos = (float(fut_cx), float(fut_cy))
                        break

            # ── Threat level ──────────────────────────────────────────────────
            threat_level = float(
                np.clip(1.0 - distance / MAX_THREAT_DISTANCE, 0.0, 1.0)
            )

            solutions.append({
                "track_id":      tid,
                "cx":            cx,
                "cy":            cy,
                "azimuth":       round(azimuth,   2),
                "elevation":     round(elevation, 2),
                "distance":      round(distance,  2),
                "t_intercept":   t_intercept,
                "intercept_pos": intercept_pos,
                "threat_level":  round(threat_level, 3),
                "velocity":      (round(float(vel[0]), 2), round(float(vel[1]), 2)),
            })

        return solutions