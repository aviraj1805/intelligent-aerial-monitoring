"""
IAMARS — Tactical Dashboard  (visualization/tactical_dashboard.py)
Rebuilt to match the IAMARS mockup design:
  Header  | Left Panel | Centre Video | Right Panel | Footer
All rendering is pure OpenCV — no GUI framework required.
"""

import cv2
import numpy as np
import time
from datetime import datetime
from collections import deque

# ── Canvas dimensions ────────────────────────────────────────────────────────
W, H          = 1280, 720
HEADER_H      = 46
FOOTER_H      = 32
BODY_H        = H - HEADER_H - FOOTER_H   # 642
LEFT_W        = 230
RIGHT_W       = 230
VIDEO_X0      = LEFT_W
VIDEO_X1      = W - RIGHT_W
VIDEO_Y0      = HEADER_H
VIDEO_Y1      = HEADER_H + BODY_H

# ── Colour palette (BGR) ─────────────────────────────────────────────────────
BG          = (14,  12,  10)
PANEL_BG    = (10,  11,   8)
METRIC_BG   = (20,  20,  13)
GREEN       = (136, 255,   0)
GREEN_DIM   = ( 85, 170,   0)
AMBER       = (  0, 170, 255)
RED         = ( 68,  68, 255)
WHITE       = (220, 220, 220)
BORDER      = ( 34,  51,   0)

F  = cv2.FONT_HERSHEY_SIMPLEX
FM = cv2.FONT_HERSHEY_DUPLEX


def _txt(img, text, x, y, color=GREEN, scale=0.38, thick=1, font=F):
    cv2.putText(img, str(text), (x, y), font, scale, color, thick, cv2.LINE_AA)


def _txt_right(img, text, x_right, y, color=GREEN, scale=0.38, thick=1):
    (tw, _), _ = cv2.getTextSize(str(text), F, scale, thick)
    cv2.putText(img, str(text), (x_right - tw, y), F, scale, color, thick, cv2.LINE_AA)


def _rect(img, x1, y1, x2, y2, color, thick=1):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)


def _fill(img, x1, y1, x2, y2, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)


class TacticalDashboard:
    """
    Renders the full 1280x720 IAMARS tactical display.
    render(frame, model_results, fused, tracked, smoothed,
           predicted, solutions, frame_index, audio_conf=0.0)
    """

    def __init__(self):
        self._start_time  = time.time()
        self._scan_y      = VIDEO_Y0
        self._frame_count = 0
        self._blink_state = True
        self._event_log   = deque(maxlen=8)
        self._event_log.append(("IAMARS SYSTEM ONLINE",  "active"))
        self._event_log.append(("MODELS LOADED: 4/4",    "active"))

    # ─────────────────────────────────────────────────────────────────────────
    def render(
        self,
        frame:         np.ndarray,
        model_results: list,
        fused:         dict,
        tracked:       dict,
        smoothed:      dict,
        predicted:     dict,
        solutions:     list,
        frame_index:   int,
        audio_conf:    float = 0.0,
    ) -> np.ndarray:

        self._frame_count = frame_index
        self._blink_state = (frame_index % 20) < 10

        self._scan_y += 3
        if self._scan_y > VIDEO_Y1:
            self._scan_y = VIDEO_Y0

        n_tracks = len((tracked or {}).get("track_ids", []))
        if n_tracks > 0 and frame_index % 40 == 1:
            self._event_log.append(("DETECTION LOCKED",   "active"))
            self._event_log.append(("TRACK INITIATED",    "warn"))
        if solutions and frame_index % 40 == 1:
            self._event_log.append(("INTERCEPT COMPUTED", "alert"))

        canvas = np.full((H, W, 3), BG, dtype=np.uint8)
        self._draw_header(canvas, frame_index)
        self._draw_left_panel(canvas, model_results, fused, audio_conf)
        self._draw_centre(canvas, frame, predicted, solutions, frame_index)
        self._draw_right_panel(canvas, solutions, smoothed)
        self._draw_footer(canvas, frame_index)
        return canvas

    # ── Header ────────────────────────────────────────────────────────────────
    def _draw_header(self, c, frame_index):
        _fill(c, 0, 0, W, HEADER_H, (9, 8, 6))
        cv2.line(c, (0, HEADER_H), (W, HEADER_H), BORDER, 1)

        _txt(c, "I A M A R S", 14, 30, GREEN, scale=0.68, thick=2, font=FM)
        if self._blink_state:
            cv2.circle(c, (152, 23), 5, RED, -1)
        _txt(c, "TACTICAL MONITOR", 165, 30, GREEN_DIM, scale=0.38)

        elapsed = time.time() - self._start_time
        _txt(c, f"T+{elapsed:07.2f}s  FRAME:{frame_index:05d}",
             W//2 - 80, 30, GREEN_DIM, scale=0.36)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        items = [
            (ts,               (0, 80, 40),  None),
            ("THREAT DETECTED", RED,         RED),
            ("MODELS ACTIVE",   AMBER,       AMBER),
            ("SYSTEM ONLINE",   GREEN,       GREEN),
        ]
        x = W - 12
        for label, tcol, dcol in items:
            _txt_right(c, label, x, 30, tcol, scale=0.30)
            (tw, _), _ = cv2.getTextSize(label, F, 0.30, 1)
            x -= tw + 16
            if dcol:
                cv2.circle(c, (x - 2, 24), 3, dcol, -1)
                x -= 12

    # ── Left panel ────────────────────────────────────────────────────────────
    def _draw_left_panel(self, c, model_results, fused, audio_conf):
        x0, x1 = 0, LEFT_W
        y0, y1 = HEADER_H, HEADER_H + BODY_H
        _fill(c, x0, y0, x1, y1, PANEL_BG)
        cv2.line(c, (x1, y0), (x1, y1), BORDER, 1)

        y = y0 + 12

        def section(label):
            nonlocal y
            y += 5
            _txt(c, label, x0+8, y, (0, 120, 50), scale=0.27)
            cv2.line(c, (x0+8, y+3), (x1-8, y+3), (0, 40, 15), 1)
            y += 11

        def metric_box(label, value, vcol=GREEN, big=False):
            nonlocal y
            _fill(c, x0+7, y, x1-7, y+32, METRIC_BG)
            _rect(c, x0+7, y, x1-7, y+32, (0, 35, 12))
            _txt(c, label, x0+12, y+10, (0, 100, 40), scale=0.25)
            _txt(c, value, x0+12, y+26, vcol, scale=0.46 if big else 0.34, thick=1)
            y += 36

        def model_row(name, status, ok=True):
            nonlocal y
            _fill(c, x0+7, y, x1-7, y+17, METRIC_BG)
            _rect(c, x0+7, y, x1-7, y+17, (0, 28, 10))
            _txt(c, name, x0+12, y+12, GREEN_DIM, scale=0.28)
            col = GREEN if ok else AMBER
            _txt_right(c, status, x1-10, y+12, col, scale=0.26)
            y += 20

        # Models
        section("DETECTION MODELS")
        det = {}
        for m in (model_results or []):
            if m:
                det[m.get("name","").upper()] = len(m.get("boxes", []))

        for name, default_st, ok in [
            ("VISIODECT", "READY",   True),
            ("UAV-IR",    "READY",   True),
            ("UAV-RGB",   "READY",   True),
            ("AUDIO",     "STANDBY", False),
        ]:
            key = name.replace("-","_")
            n = det.get(name, det.get(key, None))
            st = f"DET:{n}" if (ok and n is not None) else default_st
            model_row(name, st, ok)

        # Fusion
        y += 3
        section("FUSION STATUS")
        n_fused = len((fused or {}).get("boxes", []))
        _fill(c, x0+7, y, x1-7, y+17, METRIC_BG)
        _rect(c, x0+7, y, x1-7, y+17, (0,28,10))
        _txt(c, "WBF ACTIVE", x0+12, y+12, GREEN, scale=0.30, thick=1)
        y += 20
        metric_box("FUSED BOXES", str(n_fused), GREEN)
        _fill(c, x0+7, y, x1-7, y+26, METRIC_BG)
        _rect(c, x0+7, y, x1-7, y+26, (0,28,10))
        _txt(c, "iou_thr   : 0.45",  x0+12, y+11, (0,90,35), scale=0.26)
        _txt(c, "skip_thr  : 0.05",  x0+12, y+21, (0,90,35), scale=0.26)
        y += 30

        # Weights
        y += 3
        section("MODEL WEIGHTS")
        bw_total = x1 - x0 - 20
        for name, w in [("VISIODECT", 0.50), ("UAV-IR", 0.30), ("UAV-RGB", 0.20)]:
            _fill(c, x0+7, y, x1-7, y+15, METRIC_BG)
            _rect(c, x0+7, y, x1-7, y+15, (0,28,10))
            bw = int(bw_total * w)
            _fill(c, x0+9, y+2, x0+9+bw, y+13, (0,55,18))
            _txt(c, f"{name}: {w:.2f}", x0+12, y+11, GREEN_DIM, scale=0.27)
            y += 18

        # System info
        y += 4
        section("SYSTEM")
        for row in ["DEVICE : CUDA", "GPU    : RTX 2050", "VRAM   : 4 GB", "PyTorch: 2.5.1+cu121"]:
            _txt(c, row, x0+12, y, (0,90,35), scale=0.26)
            y += 12

    # ── Centre video ──────────────────────────────────────────────────────────
    def _draw_centre(self, c, frame, predicted, solutions, frame_index):
        vw = VIDEO_X1 - VIDEO_X0
        vh = BODY_H

        if frame is not None:
            resized = cv2.resize(frame, (vw, vh))
            c[VIDEO_Y0:VIDEO_Y1, VIDEO_X0:VIDEO_X1] = resized
        else:
            _fill(c, VIDEO_X0, VIDEO_Y0, VIDEO_X1, VIDEO_Y1, (10, 18, 10))

        # Subtle grid overlay
        for x in range(VIDEO_X0, VIDEO_X1, 60):
            cv2.line(c, (x, VIDEO_Y0), (x, VIDEO_Y1), (0, 25, 8), 1)
        for y in range(VIDEO_Y0, VIDEO_Y1, 60):
            cv2.line(c, (VIDEO_X0, y), (VIDEO_X1, y), (0, 25, 8), 1)

        # Corner brackets
        for px, py, dx, dy in [
            (VIDEO_X0+8, VIDEO_Y0+8,   1,  1),
            (VIDEO_X1-8, VIDEO_Y0+8,  -1,  1),
            (VIDEO_X0+8, VIDEO_Y1-8,   1, -1),
            (VIDEO_X1-8, VIDEO_Y1-8,  -1, -1),
        ]:
            cv2.line(c, (px, py), (px + dx*22, py),       GREEN_DIM, 2)
            cv2.line(c, (px, py), (px, py + dy*22),       GREEN_DIM, 2)

        # Scan line
        cv2.line(c, (VIDEO_X0, self._scan_y), (VIDEO_X1, self._scan_y), (50, 160, 50), 1)

        # Trajectory trail dots
        if predicted:
            for tid, pts in (predicted.get("predictions") or {}).items():
                for i, pt in enumerate(pts or []):
                    try:
                        px_ = int(pt[0] * vw / 1920) + VIDEO_X0 if pt[0] > 2 else int(pt[0] * vw) + VIDEO_X0
                        py_ = int(pt[1] * vh / 1080) + VIDEO_Y0 if pt[1] > 2 else int(pt[1] * vh) + VIDEO_Y0
                        r   = max(1, 4 - i)
                        cv2.circle(c, (px_, py_), r, AMBER, -1)
                    except Exception:
                        pass

        # Intercept cross
        if solutions:
            for sol in solutions:
                ix = sol.get("intercept_x", sol.get("ix", None))
                iy = sol.get("intercept_y", sol.get("iy", None))
                if ix is not None and iy is not None:
                    try:
                        sx = int(ix * vw / 1920) + VIDEO_X0 if ix > 2 else int(ix * vw) + VIDEO_X0
                        sy = int(iy * vh / 1080) + VIDEO_Y0 if iy > 2 else int(iy * vh) + VIDEO_Y0
                        cv2.line(c, (sx-16, sy), (sx+16, sy), RED, 2)
                        cv2.line(c, (sx, sy-16), (sx, sy+16), RED, 2)
                        if self._blink_state:
                            _txt(c, "INTERCEPT", sx+10, sy-8, RED, scale=0.28)
                    except Exception:
                        pass

        # Bottom info
        h_orig = frame.shape[0] if frame is not None else 0
        w_orig = frame.shape[1] if frame is not None else 0
        _txt(c, f"FRAME: {frame_index:04d} / 1000  |  20 FPS  |  {w_orig}x{h_orig}",
             VIDEO_X0+10, VIDEO_Y1-10, (0,90,35), scale=0.28)

        # Top-right timestamp
        _txt_right(c, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   VIDEO_X1-8, VIDEO_Y0+18, (0,110,50), scale=0.32)

    # ── Right panel ───────────────────────────────────────────────────────────
    def _draw_right_panel(self, c, solutions, smoothed):
        x0, x1 = VIDEO_X1, W
        y0, y1 = HEADER_H, HEADER_H + BODY_H
        _fill(c, x0, y0, x1, y1, PANEL_BG)
        cv2.line(c, (x0, y0), (x0, y1), BORDER, 1)

        y = y0 + 12

        def section(label):
            nonlocal y
            y += 5
            _txt(c, f"[ {label} ]", x0+8, y, GREEN, scale=0.30, thick=1)
            y += 10

        def metric_box(label, value, vcol=GREEN, big=True):
            nonlocal y
            _fill(c, x0+7, y, x1-7, y+32, METRIC_BG)
            _rect(c, x0+7, y, x1-7, y+32, (0,35,12))
            _txt(c, label, x0+12, y+10, (0,100,40), scale=0.25)
            _txt(c, value, x0+12, y+26, vcol, scale=0.48 if big else 0.34, thick=1)
            y += 36

        # Fire solution
        section("FIRE SOLUTION")
        if solutions:
            sol = solutions[0]
            az  = sol.get("azimuth",     sol.get("az",  0.0))
            el  = sol.get("elevation",   sol.get("el",  0.0))
            ti  = sol.get("t_intercept", sol.get("ti",  0.0))
            metric_box("AZIMUTH",     f"{az:.2f} deg", RED)
            metric_box("ELEVATION",   f"{el:.2f} deg", RED)
            metric_box("T-INTERCEPT", f"{ti:.3f}s",    AMBER)
        else:
            _fill(c, x0+7, y, x1-7, y+22, METRIC_BG)
            _rect(c, x0+7, y, x1-7, y+22, (0,35,12))
            _txt(c, "NO TARGET", x0+12, y+15, GREEN_DIM, scale=0.34)
            y += 26

        # Tracking
        y += 4
        section("TRACKING")
        track_ids = (smoothed or {}).get("track_ids", [])
        n_tracks  = len(track_ids)
        metric_box("ACTIVE TRACKS", str(n_tracks),
                   GREEN if n_tracks > 0 else GREEN_DIM, big=False)

        states = (smoothed or {}).get("states", {})
        for tid in list(track_ids)[:2]:
            st = states.get(tid, {})
            vx = st.get("vx", 0.0)
            vy = st.get("vy", 0.0)
            _fill(c, x0+7, y, x1-7, y+28, METRIC_BG)
            _rect(c, x0+7, y, x1-7, y+28, (0,35,12))
            _txt(c, f"TGT-{tid:03d}", x0+12, y+11, GREEN, scale=0.32)
            _txt(c, f"V: ({vx:.1f}, {vy:.1f}) px/f", x0+12, y+23, GREEN_DIM, scale=0.27)
            y += 32

        # Threat level
        y += 4
        section("THREAT LEVEL")
        threat = min(5, max(0, n_tracks * 2 + (1 if solutions else 0)))
        bar_cols = [(0,190,80),(0,190,80),(0,150,200),(0,150,200),(68,68,255),(68,68,255)]
        for i in range(5):
            bx = x0 + 12 + i * 14
            filled = i < threat
            col = bar_cols[min(threat,5)-1] if (filled and threat > 0) else (18,28,18)
            _fill(c, bx, y+2, bx+10, y+16, col)
            _rect(c, bx, y+2, bx+10, y+16, (0,55,20))
        tlab  = {0:"NONE",1:"LOW",2:"LOW",3:"MEDIUM",4:"HIGH",5:"HIGH"}.get(threat,"NONE")
        tcol  = RED if threat >= 4 else (AMBER if threat >= 2 else GREEN_DIM)
        _txt(c, tlab, x0+88, y+14, tcol, scale=0.32, thick=1)
        y += 24

        # System log
        y += 4
        section("SYSTEM LOG")
        col_map = {"active": GREEN, "warn": AMBER, "alert": RED}
        for msg, kind in list(self._event_log)[-7:]:
            if y > y1 - 10:
                break
            col = col_map.get(kind, GREEN_DIM)
            elapsed = int(time.time() - self._start_time)
            _txt(c, f"[{elapsed:05d}] {msg}", x0+10, y, col, scale=0.26)
            y += 13

    # ── Footer ────────────────────────────────────────────────────────────────
    def _draw_footer(self, c, frame_index):
        fy = H - FOOTER_H
        _fill(c, 0, fy, W, H, (9, 8, 6))
        cv2.line(c, (0, fy), (W, fy), BORDER, 1)

        stages = ["DETECT","FUSE","TRACK","KALMAN","PREDICT","INTERCEPT"]
        x = 12
        for i, stage in enumerate(stages):
            (tw, _), _ = cv2.getTextSize(stage, F, 0.28, 1)
            pad = 7
            bx1 = x;            by1 = fy + 6
            bx2 = x + tw + pad*2; by2 = fy + FOOTER_H - 6
            _fill(c, bx1, by1, bx2, by2, (0,40,13))
            _rect(c, bx1, by1, bx2, by2, GREEN)
            _txt(c, stage, bx1 + pad, by2 - 5, GREEN, scale=0.28)
            x = bx2 + 3
            if i < len(stages) - 1:
                _txt(c, ">", x+1, fy + FOOTER_H - 9, GREEN_DIM, scale=0.28)
                x += 11

        _txt_right(c, "IAMARS v1.0  |  DRDO DEMO BUILD  |  CONFIDENTIAL",
                   W - 10, fy + FOOTER_H - 8, (0,80,35), scale=0.27)

    # ── Public helpers ────────────────────────────────────────────────────────
    def log_event(self, msg: str) -> None:
        self._event_log.append((msg, "active"))

    def reset_tracks(self) -> None:
        self._event_log.append(("TRACKS RESET", "warn"))