"""
IAMARS — KalmanFilter
Per-track Kalman filter for smooth position and velocity estimation.

State vector (6,):  [cx, cy, w, h, vx, vy]
    cx, cy  — box centre
    w,  h   — box width / height
    vx, vy  — centre velocity (pixels / frame)

Observation vector (4,):  [cx, cy, w, h]

CHANGELOG v1.1
--------------
Added KalmanFilterManager.predict_next():
    Returns the predicted next (cx, cy, w, h) for every active track
    without mutating any live filter state. Used by the ghost injection
    logic in video_pipeline.py to keep ByteTrack IDs stable during
    brief detection gaps.
"""

import copy
import numpy as np


class _SingleTrackKF:
    """
    Constant-velocity Kalman filter for one track.
    State  : [cx, cy, w, h, vx, vy]
    Observe: [cx, cy, w, h]
    """

    def __init__(self, cx: float, cy: float, w: float, h: float):
        # ── State transition matrix F (6×6) ──────────────────────────────────
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 4] = 1.0   # cx += vx
        self.F[1, 5] = 1.0   # cy += vy

        # ── Observation matrix H (4×6) ────────────────────────────────────────
        self.H = np.zeros((4, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # ── Process noise Q ───────────────────────────────────────────────────
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 10.0, 10.0]).astype(np.float64)

        # ── Measurement noise R ───────────────────────────────────────────────
        self.R = np.diag([5.0, 5.0, 5.0, 5.0]).astype(np.float64)

        # ── Initial state and covariance ─────────────────────────────────────
        self.x = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0]).astype(np.float64)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray):
        """z : observation [cx, cy, w, h]"""
        y  = z - self.H @ self.x                      # innovation
        S  = self.H @ self.P @ self.H.T + self.R      # innovation covariance
        K  = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x.copy()

    def peek_predict(self) -> np.ndarray:
        """
        Return the predicted next state WITHOUT mutating self.x or self.P.
        Used for ghost injection — never call predict() on a live filter
        unless you are also going to call update() immediately after.
        """
        x_next = self.F @ self.x
        return x_next.copy()


# ─────────────────────────────────────────────────────────────────────────────

class KalmanFilterManager:
    """
    Manages one _SingleTrackKF per active track ID.

    Usage
    -----
    kf_mgr  = KalmanFilterManager()
    smoothed = kf_mgr.update(tracked_dict)

    Returns a dict identical to tracked_dict but with smoothed boxes
    and an added 'velocities' (K, 2) array of [vx, vy] per track.
    """

    def __init__(self):
        self._filters: dict[int, _SingleTrackKF] = {}

    # ─────────────────────────────────────────────────────────────────────────
    def update(self, tracked: dict) -> dict:
        """
        Parameters
        ----------
        tracked : dict from ByteTracker.update()
            {boxes (K,4) xyxy, scores (K,), labels (K,), track_ids (K,)}

        Returns
        -------
        Same dict + 'velocities' (K, 2)  [vx, vy] per track.
        Boxes are replaced with Kalman-smoothed xyxy coords.
        """
        boxes     = tracked["boxes"]      # (K, 4)
        track_ids = tracked["track_ids"]  # (K,)

        if len(boxes) == 0:
            return {**tracked, "velocities": np.empty((0, 2), dtype=np.float32)}

        smoothed_boxes = np.zeros_like(boxes)
        velocities     = np.zeros((len(boxes), 2), dtype=np.float32)

        active_ids = set(track_ids.tolist())
        # Remove stale filters
        stale = [tid for tid in self._filters if tid not in active_ids]
        for tid in stale:
            del self._filters[tid]

        for i, (box, tid) in enumerate(zip(boxes, track_ids)):
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            w  = box[2] - box[0]
            h  = box[3] - box[1]

            if tid not in self._filters:
                # First observation — initialise filter
                self._filters[tid] = _SingleTrackKF(cx, cy, w, h)
                state = self._filters[tid].x
            else:
                kf = self._filters[tid]
                kf.predict()
                state = kf.update(np.array([cx, cy, w, h], dtype=np.float64))

            # Convert state back to xyxy
            s_cx, s_cy, s_w, s_h = state[0], state[1], state[2], state[3]
            s_w = max(s_w, 1.0)
            s_h = max(s_h, 1.0)
            smoothed_boxes[i] = [
                s_cx - s_w / 2,
                s_cy - s_h / 2,
                s_cx + s_w / 2,
                s_cy + s_h / 2,
            ]
            velocities[i] = [state[4], state[5]]

        return {
            "boxes":      smoothed_boxes.astype(np.float32),
            "scores":     tracked["scores"],
            "labels":     tracked["labels"],
            "track_ids":  tracked["track_ids"],
            "velocities": velocities,
        }

    # ─────────────────────────────────────────────────────────────────────────
    def predict_next(self) -> dict[int, tuple[float, float, float, float]]:
        """
        Return predicted next-frame (cx, cy, w, h) for every active track
        WITHOUT mutating any filter state.

        Called by video_pipeline._inject_ghost_detections() to build ghost
        bounding boxes that keep ByteTrack IDs alive during missed frames.

        Returns
        -------
        dict mapping track_id → (cx, cy, w, h)
        Empty dict if no active filters.
        """
        result = {}
        for tid, kf in self._filters.items():
            x_next = kf.peek_predict()          # non-mutating predict
            cx, cy = float(x_next[0]), float(x_next[1])
            w,  h  = float(x_next[2]), float(x_next[3])
            w = max(w, 20.0)
            h = max(h, 20.0)
            result[tid] = (cx, cy, w, h)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        self._filters.clear()

    def get_state(self, track_id: int) -> np.ndarray | None:
        """Return raw state vector for a given track ID, or None."""
        kf = self._filters.get(track_id)
        return kf.x.copy() if kf else None