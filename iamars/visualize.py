"""Draw tracks, trails, predicted trajectories and a telemetry panel.

Output is always a fixed-size canvas (default 1280x720): the video on the
left, a telemetry panel on the right. The previous dashboard sized its
canvas from the Windows display-scaling setting, so saved videos came out
empty whenever the scaling was not 100 %.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from iamars.pipeline import FrameResult

BG = (18, 20, 24)
TEXT = (220, 225, 230)
DIM = (130, 135, 140)
ACCENT = (0, 220, 120)     # track boxes (BGR)
PRED = (0, 165, 255)       # predicted trajectory
RAW = (200, 200, 200)      # raw detections
FONT = cv2.FONT_HERSHEY_SIMPLEX

_PALETTE = [(0, 220, 120), (255, 170, 0), (255, 90, 200), (0, 200, 255), (180, 120, 255)]


def _color(tid: int):
    return _PALETTE[tid % len(_PALETTE)]


class Annotator:
    def __init__(self, out_wh=(1280, 720), panel_w=320, trail_len=40, show_raw=True):
        self.W, self.H = out_wh
        self.panel_w = panel_w
        self.trails: dict[int, deque] = {}
        self.trail_len = trail_len
        self.show_raw = show_raw

    @property
    def size(self):
        return self.W, self.H

    def draw(self, frame: np.ndarray, res: FrameResult, fps: float | None = None,
             label: str = "") -> np.ndarray:
        vw, vh = self.W - self.panel_w, self.H
        fh, fw = frame.shape[:2]
        s = min(vw / fw, vh / fh)
        nw, nh = int(fw * s), int(fh * s)
        ox, oy = (vw - nw) // 2, (vh - nh) // 2

        canvas = np.full((self.H, self.W, 3), BG, np.uint8)
        view = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

        def P(x, y):
            return int(x * s), int(y * s)

        if self.show_raw:
            for b in res.detections.xyxy:
                cv2.rectangle(view, P(b[0], b[1]), P(b[2], b[3]), RAW, 1)

        active = set()
        for tr in res.tracks:
            active.add(tr.track_id)
            c = _color(tr.track_id)
            x1, y1 = P(tr.box[0], tr.box[1])
            x2, y2 = P(tr.box[2], tr.box[3])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            trail = self.trails.setdefault(tr.track_id, deque(maxlen=self.trail_len))
            trail.append((cx, cy))
            if len(trail) > 1:
                cv2.polylines(view, [np.array(trail, np.int32)], False, c, 2, cv2.LINE_AA)
            pts = np.array([P(x, y) for x, y in tr.trajectory], np.int32)
            for p in pts[::2]:
                cv2.circle(view, tuple(int(v) for v in p), 2, PRED, -1, cv2.LINE_AA)
            if len(pts):
                cv2.arrowedLine(view, (cx, cy), tuple(int(v) for v in pts[-1]), PRED, 1,
                                cv2.LINE_AA, tipLength=0.2)
            pad = 4
            cv2.rectangle(view, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), c, 2)
            tag = f"ID {tr.track_id}  {tr.conf:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, FONT, 0.5, 1)
            ty = max(y1 - pad - 6, th + 4)
            cv2.rectangle(view, (x1 - pad, ty - th - 4), (x1 - pad + tw + 6, ty + 3), c, -1)
            cv2.putText(view, tag, (x1 - pad + 3, ty), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        for tid in list(self.trails):  # forget trails of tracks gone for good
            if tid not in active and len(self.trails[tid]) and res.frame_index % 60 == 0:
                del self.trails[tid]

        canvas[oy:oy + nh, ox:ox + nw] = view
        self._panel(canvas, res, fps, label)
        return canvas

    def _panel(self, canvas, res: FrameResult, fps, label):
        x0 = self.W - self.panel_w + 14
        y = 30

        def line(text, color=TEXT, scale=0.5, dy=22, thick=1):
            nonlocal y
            cv2.putText(canvas, text, (x0, y), FONT, scale, color, thick, cv2.LINE_AA)
            y += dy

        cv2.line(canvas, (self.W - self.panel_w, 0), (self.W - self.panel_w, self.H), DIM, 1)
        line("IAMARS", ACCENT, 0.8, 26, 2)
        line("drone detection & tracking", DIM, 0.45, 30)
        line(f"frame      {res.frame_index:6d}")
        if fps is not None:
            line(f"speed      {fps:6.1f} FPS")
        line(f"latency    {res.timings_ms.get('total', 0):6.1f} ms")
        line(f"detections {len(res.detections):6d}")
        line(f"tracks     {len(res.tracks):6d}", dy=30)
        line("TRACK  CONF  SPEED px/s  AGE", DIM, 0.42)
        for tr in sorted(res.tracks, key=lambda t: t.track_id)[:12]:
            spd = float(np.hypot(*tr.velocity))
            line(f"{tr.track_id:5d}  {tr.conf:4.2f}  {spd:8.0f}   {tr.age:4d}",
                 _color(tr.track_id), 0.45, 20)
        y = self.H - 58
        line("box = ByteTrack + Kalman", DIM, 0.4, 16)
        line("orange = predicted path", PRED, 0.4, 16)
        if label:
            line(label, DIM, 0.4, 16)
