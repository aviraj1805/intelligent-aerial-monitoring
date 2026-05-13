"""
IAMARS — TacticalDashboard  (v2.0 — DPI-FIXED)

WHAT CHANGED FROM v1.x
───────────────────────
ROOT CAUSE
  Windows at 125 % display scaling passes every imshow() frame through a
  1.25× OS bitmap-scaler before putting it on screen.  The result is blurry
  text, soft borders, and an undersized window.  OpenCV was compiled without
  the Qt backend and without a DPI-aware manifest, so it had no way to
  opt out of OS scaling.

FIX — THREE LAYERS

  Layer 1 — Windows DPI-awareness manifest (dpi_fix.py / ctypes call)
    SetProcessDpiAwareness(2) tells Windows "I handle my own DPI — do NOT
    scale my window".  Must be called BEFORE any cv2.imshow() or
    cv2.namedWindow() call, so it is called in __init__ here.

  Layer 2 — Physical-pixel canvas
    The dashboard now renders at PHYSICAL pixels, not logical pixels.
    On a 125 % display:
        logical  1280 × 720  →  physical  1600 × 900
    All layout constants (panel widths, header height, font positions,
    bar widths) are multiplied by DPI_SCALE so they occupy exactly the
    same fraction of the screen as before, just at higher pixel density.

  Layer 3 — Window size enforcement
    After namedWindow(), cv2.resizeWindow() is called with the LOGICAL
    dimensions (1280 × 720) so the window occupies the same screen real
    estate.  The canvas is rendered at 1600 × 900 physical pixels and
    the Win32 driver maps those 1:1 onto screen pixels — no interpolation,
    no blur.

UNCHANGED
  All rendering logic, colour palette, section layout, EMA smoother,
  log system, and public API (render / log_event / reset_tracks) are
  identical to v1.x.  Drop-in replacement.

Layout (logical, same as v1.x)
──────────────────────────────
  LEFT   panel  [  0..249]  model status + fusion + system
  CENTRE panel  [250..979]  header + video + bottom bar
  RIGHT  panel  [980..1279] fire solution + tracking + log
"""

import ctypes
import sys
import cv2
import numpy as np
import time
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Windows DPI awareness
# Must run before any cv2 window is created.
# ─────────────────────────────────────────────────────────────────────────────
def _set_dpi_aware() -> float:
    """
    Tell Windows this process manages its own DPI scaling.
    Returns the effective DPI scale factor (e.g. 1.25 for 125 %).
    Safe to call on non-Windows platforms — returns 1.0.
    """
    if sys.platform != "win32":
        return 1.0
    try:
        # PROCESS_PER_MONITOR_DPI_AWARE = 2  (best mode, Windows 8.1+)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            # Fallback: PROCESS_SYSTEM_DPI_AWARE = 1  (Windows Vista+)
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            return 1.0

    # Read the actual scale factor from the primary monitor
    try:
        dc   = ctypes.windll.user32.GetDC(0)
        dpi  = ctypes.windll.gdi32.GetDeviceCaps(dc, 88)  # LOGPIXELSX
        ctypes.windll.user32.ReleaseDC(0, dc)
        return round(dpi / 96.0, 4)          # 96 dpi = 100 % scale
    except Exception:
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Physical-pixel layout
# All constants are defined in LOGICAL pixels first, then scaled.
# ─────────────────────────────────────────────────────────────────────────────

# ── Detect DPI scale before anything else ────────────────────────────────────
DPI_SCALE: float = _set_dpi_aware()

def _px(logical: int) -> int:
    """Convert a logical pixel value to physical pixels."""
    return max(1, round(logical * DPI_SCALE))


# ── Colour palette (BGR) ──────────────────────────────────────────────────────
BG          = ( 10,  12,  14)
PANEL_BG    = ( 18,  20,  24)
HEADER_BG   = ( 14,  16,  20)
DARK_GREY   = ( 28,  30,  34)
GRID_COL    = ( 22,  38,  22)
GREY        = ( 75,  78,  84)
WHITE       = (210, 215, 220)
ACCENT      = (  0, 255, 136)
ACCENT_DIM  = (  0, 120,  64)
ACCENT_DARK = (  0,  48,  26)
RED         = ( 30,  40, 220)
RED_DIM     = ( 20,  28, 100)
ORANGE      = (  0, 148, 255)
YELLOW      = (  0, 210, 230)
CYAN        = (200, 230,   0)

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO   = cv2.FONT_HERSHEY_PLAIN

# ── Logical layout constants (unchanged from v1.x) ───────────────────────────
_LW, _LH        = 1280, 720          # logical canvas size
_LEFT_W         = 250
_RIGHT_W        = 300
_HEADER_H       = 40
_BOTTOM_H       = 50
_DIVIDER        = 1
_PAD            = 10
_LINE_TITLE     = 22
_LINE_BODY      = 16
_LINE_SMALL     = 14

# ── Physical layout constants (all passed to cv2 drawing calls) ───────────────
W           = _px(_LW)
H           = _px(_LH)
LEFT_W      = _px(_LEFT_W)
RIGHT_W     = _px(_RIGHT_W)
HEADER_H    = _px(_HEADER_H)
BOTTOM_H    = _px(_BOTTOM_H)
DIVIDER     = max(1, _px(_DIVIDER))
PAD         = _px(_PAD)
LINE_TITLE  = _px(_LINE_TITLE)
LINE_BODY   = _px(_LINE_BODY)
LINE_SMALL  = _px(_LINE_SMALL)

CX0 = LEFT_W
CX1 = W - RIGHT_W
CW  = CX1 - CX0

VX0 = CX0 + _px(4)
VX1 = CX1 - _px(4)
VY0 = HEADER_H + _px(4)
VY1 = H - BOTTOM_H - _px(4)
VW  = VX1 - VX0
VH  = VY1 - VY0

# ── Typography — scale font sizes with DPI ───────────────────────────────────
# cv2.putText scale is in "font units" not pixels; multiply by DPI_SCALE
# to keep characters the same apparent size on screen.
_S_TITLE  = round(0.46 * DPI_SCALE, 3)
_S_BODY   = round(0.39 * DPI_SCALE, 3)
_S_SMALL  = round(0.34 * DPI_SCALE, 3)
_S_BRAND  = round(0.38 * DPI_SCALE, 3)

# Line thickness also needs scaling so strokes don't look hairline-thin
_TH1 = max(1, round(1 * DPI_SCALE))   # normal stroke
_TH2 = max(1, round(2 * DPI_SCALE))   # bold / emphasis


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
def _sep(canvas, x0, x1, y, color=ACCENT_DIM):
    cv2.line(canvas, (x0 + PAD, y), (x1 - PAD, y), color, _TH1)


def _txt(canvas, text, x, y, scale=_S_BODY, color=WHITE, thickness=_TH1):
    cv2.putText(canvas, text, (x, y), FONT, scale, color,
                thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# TacticalDashboard
# ─────────────────────────────────────────────────────────────────────────────
class TacticalDashboard:
    """
    Drop-in replacement for v1.x TacticalDashboard.
    Public API is identical: render() / log_event() / reset_tracks()

    Parameters
    ----------
    frame_wh      : (W, H) of the source video frames
    max_log_lines : lines kept in the rolling system log
    window_name   : cv2 window title
    """

    _EMA_ALPHA       = 0.08
    _DISPLAY_REFRESH = 6
    _HOLD_FRAMES     = 15

    def __init__(
        self,
        frame_wh:      tuple = (1920, 1080),
        max_log_lines: int   = 14,
        window_name:   str   = "IAMARS \u2014 Tactical Dashboard",
    ):
        self.src_w, self.src_h  = frame_wh
        self.log: deque          = deque(maxlen=max_log_lines)
        self._frame_count        = 0
        self._blink_state        = True
        self._scan_y             = VY0
        self._start_time         = time.time()
        self._track_history: dict[int, deque] = {}
        self._window_name        = window_name

        self._stages = ["DETECT", "FUSE", "TRACK", "KALMAN", "PREDICT", "FIRE SOL"]

        self._ema:          dict[int, dict] = {}
        self._display:      dict[int, dict] = {}
        self._hold_ttl:     dict[int, int]  = {}
        self._panel_order:  list[int]       = []
        self._panel_refresh: int            = 0

        # ── Layer 3: create window at LOGICAL size ────────────────────────────
        # The canvas is physical pixels; the window is logical pixels.
        # Windows maps physical→logical 1:1 because we are DPI-aware,
        # so the rendered image fills the window without any OS scaling.
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, _LW, _LH)

        self.log_event("IAMARS SYSTEM ONLINE")
        self.log_event("MODELS LOADED: 4/4")
        self.log_event(f"DPI SCALE: {DPI_SCALE:.2f}x  "
                       f"CANVAS: {W}x{H}px")

    # ─────────────────────────────────────────────────────────────────────────
    # EMA smoother (unchanged logic)
    # ─────────────────────────────────────────────────────────────────────────
    def _update_solution_ema(self, solutions: list) -> None:
        a        = self._EMA_ALPHA
        now_tids = set()

        for sol in solutions:
            tid = int(sol["track_id"])
            now_tids.add(tid)
            vx, vy = float(sol["velocity"][0]), float(sol["velocity"][1])
            raw = {
                "azimuth":     float(sol["azimuth"]),
                "elevation":   float(sol["elevation"]),
                "t_intercept": float(sol["t_intercept"])
                               if sol["t_intercept"] != float("inf") else -1.0,
                "distance":    float(sol["distance"]),
                "vx":          vx,
                "vy":          vy,
                "threat":      float(sol["threat_level"]),
            }
            if tid not in self._ema:
                self._ema[tid]     = dict(raw)
                self._display[tid] = dict(raw)
            else:
                for k, v in raw.items():
                    self._ema[tid][k] = self._ema[tid][k] * (1 - a) + v * a
            self._hold_ttl[tid] = self._HOLD_FRAMES

        for tid in list(self._hold_ttl):
            if tid not in now_tids:
                self._hold_ttl[tid] -= 1
                if self._hold_ttl[tid] <= 0:
                    self._ema.pop(tid, None)
                    self._display.pop(tid, None)
                    self._hold_ttl.pop(tid, None)
                    if tid in self._panel_order:
                        self._panel_order.remove(tid)

        self._panel_refresh += 1
        if self._panel_refresh >= self._DISPLAY_REFRESH:
            self._panel_refresh = 0
            for tid, vals in self._ema.items():
                self._display[tid] = dict(vals)

        for tid in now_tids:
            if tid not in self._panel_order:
                self._panel_order.append(tid)

    # ─────────────────────────────────────────────────────────────────────────
    # Public render entry point
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

        self._scan_y += _px(3)
        if self._scan_y > VY1:
            self._scan_y = VY0

        self._update_solution_ema(solutions)

        # Physical-pixel canvas
        canvas = np.full((H, W, 3), BG, dtype=np.uint8)
        canvas[:, :LEFT_W]    = PANEL_BG
        canvas[:, W-RIGHT_W:] = PANEL_BG

        self._draw_header(canvas, frame_index)
        self._draw_centre(canvas, frame, predicted, solutions)
        self._draw_bottom_bar(canvas, frame_index)
        self._draw_left_panel(canvas, model_results, fused, audio_conf)
        self._draw_right_panel(canvas, solutions, smoothed)

        cv2.line(canvas, (LEFT_W,    0), (LEFT_W,    H), ACCENT_DIM, DIVIDER)
        cv2.line(canvas, (W-RIGHT_W, 0), (W-RIGHT_W, H), ACCENT_DIM, DIVIDER)

        return canvas

    # ─────────────────────────────────────────────────────────────────────────
    # Header
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_header(self, canvas, frame_index):
        cv2.rectangle(canvas, (CX0, 0), (CX1, HEADER_H), HEADER_BG, -1)
        cv2.line(canvas, (CX0, HEADER_H), (CX1, HEADER_H), ACCENT_DIM, _TH1)

        dot_x  = CX0 + _px(18)
        dot_y  = HEADER_H // 2
        dot_r  = _px(6)
        dot_col = RED if self._blink_state else (45, 45, 45)
        cv2.circle(canvas, (dot_x, dot_y), dot_r, dot_col, -1)

        _txt(canvas, "I A M A R S",
             CX0 + _px(34), dot_y + _px(6),
             round(0.72 * DPI_SCALE, 3), ACCENT, _TH2)

        _txt(canvas, "TACTICAL SURVEILLANCE SYSTEM",
             CX0 + _px(34), dot_y + _px(18),
             round(0.30 * DPI_SCALE, 3), ACCENT_DIM, _TH1)

        elapsed = time.time() - self._start_time
        ts      = f"T+{elapsed:07.2f}s   FRAME {frame_index:05d}"
        tw      = cv2.getTextSize(ts, FONT, round(0.40 * DPI_SCALE, 3), _TH1)[0][0]
        _txt(canvas, ts,
             CX1 - tw - _px(10), dot_y + _px(6),
             round(0.40 * DPI_SCALE, 3), GREY)

    # ─────────────────────────────────────────────────────────────────────────
    # Centre video panel
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_centre(self, canvas, frame, predicted, solutions):
        resized = cv2.resize(frame, (VW, VH), interpolation=cv2.INTER_LINEAR)

        # Tactical grid
        overlay = resized.copy()
        step = _px(60)
        for gx in range(0, VW, step):
            cv2.line(overlay, (gx, 0), (gx, VH), GRID_COL, _TH1)
        for gy in range(0, VH, step):
            cv2.line(overlay, (0, gy), (VW, gy), GRID_COL, _TH1)
        resized = cv2.addWeighted(resized, 0.84, overlay, 0.16, 0)

        # Centre crosshair
        vcx, vcy = VW // 2, VH // 2
        arm_c    = _px(18)
        cv2.line(resized, (vcx - arm_c, vcy), (vcx + arm_c, vcy), ACCENT_DIM, _TH1)
        cv2.line(resized, (vcx, vcy - arm_c), (vcx, vcy + arm_c), ACCENT_DIM, _TH1)
        cv2.circle(resized, (vcx, vcy), _px(3), ACCENT_DIM, _TH1)

        sx = VW / self.src_w
        sy = VH / self.src_h

        boxes     = predicted.get("boxes",       np.empty((0, 4), dtype=np.float32))
        track_ids = predicted.get("track_ids",   np.empty((0,),   dtype=np.int32))
        trajs     = predicted.get("trajectories", [])

        for i in range(len(boxes)):
            box = boxes[i]
            tid = int(track_ids[i])

            x1 = int(box[0] * sx);  y1 = int(box[1] * sy)
            x2 = int(box[2] * sx);  y2 = int(box[3] * sy)
            cx = (x1 + x2) // 2;   cy = (y1 + y2) // 2

            if tid not in self._track_history:
                self._track_history[tid] = deque(maxlen=30)
            self._track_history[tid].append((cx, cy))

            # Detection rectangle
            cv2.rectangle(resized, (x1, y1), (x2, y2), ACCENT, _TH2)

            # Corner brackets
            bl = _px(12)
            for (bx, by), (ddx1, ddy1), (ddx2, ddy2) in [
                ((x1, y1), ( bl,  0), ( 0,  bl)),
                ((x2, y1), (-bl,  0), ( 0,  bl)),
                ((x1, y2), ( bl,  0), ( 0, -bl)),
                ((x2, y2), (-bl,  0), ( 0, -bl)),
            ]:
                cv2.line(resized, (bx, by), (bx+ddx1, by+ddy1), WHITE, _TH2)
                cv2.line(resized, (bx, by), (bx+ddx2, by+ddy2), WHITE, _TH2)

            # Label tag
            label  = f"TGT #{tid}"
            lscale = round(0.42 * DPI_SCALE, 3)
            lw, lh = cv2.getTextSize(label, FONT, lscale, _TH1)[0]
            tag_h  = lh + _px(8)
            tag_y0 = max(y1 - tag_h - _px(2), 0)
            tag_y1 = tag_y0 + tag_h
            tag_x0 = min(x1, VW - lw - _px(10))
            tag_x0 = max(tag_x0, 0)
            cv2.rectangle(resized,
                          (tag_x0, tag_y0),
                          (tag_x0 + lw + _px(8), tag_y1),
                          ACCENT, -1)
            _txt(resized, label,
                 tag_x0 + _px(4), tag_y1 - _px(4),
                 lscale, (0, 0, 0), _TH1)

            # Trail
            trail = list(self._track_history.get(tid, []))
            for pt in trail[:-1]:
                cv2.circle(resized, pt, _px(2), ACCENT_DIM, -1)

            # Predicted trajectory
            if i < len(trajs):
                for t_pt in trajs[i]:
                    px2 = int(t_pt[0] * sx)
                    py2 = int(t_pt[1] * sy)
                    if 0 <= px2 < VW and 0 <= py2 < VH:
                        cv2.circle(resized, (px2, py2), _px(2), ORANGE, -1)

        # Intercept crosshair
        drawn_labels: list[tuple[int, int]] = []
        for sol in solutions:
            if sol["t_intercept"] != float("inf"):
                arm  = _px(16)
                ix   = int(sol["intercept_pos"][0] * sx)
                iy   = int(sol["intercept_pos"][1] * sy)
                ix   = max(arm, min(VW - arm - 1, ix))
                iy   = max(arm, min(VH - arm - 1, iy))
                cv2.line(resized, (ix-arm, iy), (ix+arm, iy), RED, _TH2)
                cv2.line(resized, (ix, iy-arm), (ix, iy+arm), RED, _TH2)
                cv2.circle(resized, (ix, iy), arm, RED, _TH1)
                if self._blink_state:
                    lbl  = "INTERCEPT"
                    lsc2 = round(0.42 * DPI_SCALE, 3)
                    lw2  = cv2.getTextSize(lbl, FONT, lsc2, _TH1)[0][0]
                    lx   = min(ix + arm + _px(4), VW - lw2 - _px(4))
                    ly   = iy + _px(5)
                    too_close = any(
                        abs(ly - py_) < _px(20) and abs(lx - px_) < lw2 + _px(4)
                        for (px_, py_) in drawn_labels
                    )
                    if not too_close:
                        _txt(resized, lbl, lx, ly, lsc2, RED)
                        drawn_labels.append((lx, ly))

        # Scan line
        syl = self._scan_y - VY0
        if _px(2) <= syl < VH - _px(2):
            fade_h = _px(6)
            y_lo   = max(0, syl - fade_h)
            y_hi   = min(VH, syl + fade_h + 1)
            strip  = resized[y_lo:y_hi].astype(np.float32)
            tint   = np.zeros_like(strip)
            tint[:, :, 1] = 48
            rows  = strip.shape[0]
            alpha = np.abs(np.linspace(-1, 1, rows)).reshape(rows, 1, 1)
            alpha = (1.0 - alpha) * 0.35
            resized[y_lo:y_hi] = np.clip(strip + tint * alpha, 0, 255).astype(np.uint8)

        canvas[VY0:VY1, VX0:VX1] = resized

        # Corner bracket decoration on video border
        p0   = (VX0, VY0);  p1 = (VX1 - 1, VY1 - 1)
        arm2 = _px(20)
        for (bx, by), (ddx1, ddy1), (ddx2, ddy2) in [
            (p0,             ( arm2,  0), ( 0,  arm2)),
            ((p1[0], p0[1]), (-arm2,  0), ( 0,  arm2)),
            ((p0[0], p1[1]), ( arm2,  0), ( 0, -arm2)),
            (p1,             (-arm2,  0), ( 0, -arm2)),
        ]:
            cv2.line(canvas, (bx, by), (bx+ddx1, by+ddy1), ACCENT, _TH2)
            cv2.line(canvas, (bx, by), (bx+ddx2, by+ddy2), ACCENT, _TH2)

    # ─────────────────────────────────────────────────────────────────────────
    # Bottom bar
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_bottom_bar(self, canvas, frame_index):
        y0 = H - BOTTOM_H
        cv2.rectangle(canvas, (CX0, y0), (CX1, H), HEADER_BG, -1)
        cv2.line(canvas, (CX0, y0), (CX1, y0), ACCENT_DIM, _TH1)

        n       = len(self._stages)
        avail_w = CW - _px(20)
        box_w   = min(_px(90), avail_w // n - _px(6))
        gap     = (avail_w - n * box_w) // (n + 1)
        x       = CX0 + gap
        by0     = y0 + _px(5)
        by1     = y0 + _px(30)

        for i, stage in enumerate(self._stages):
            bx0 = x;  bx1 = x + box_w
            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), DARK_GREY, -1)
            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), ACCENT, _TH1)
            tw = cv2.getTextSize(stage, FONT, _S_SMALL, _TH1)[0][0]
            tx = bx0 + (box_w - tw) // 2
            _txt(canvas, stage, tx, by1 - _px(6), _S_SMALL, ACCENT)
            if i < n - 1:
                ax = bx1 + max(gap // 2, _px(3))
                ay = (by0 + by1) // 2
                cv2.arrowedLine(canvas, (ax - _px(3), ay), (ax + _px(3), ay),
                                ACCENT_DIM, _TH1, tipLength=0.6)
            x += box_w + gap

        brand = "IAMARS v1.0  |  DRDO DEMO BUILD  |  CONFIDENTIAL"
        bsc   = round(0.28 * DPI_SCALE, 3)
        bw    = cv2.getTextSize(brand, FONT, bsc, _TH1)[0][0]
        bx    = CX0 + (CW - bw) // 2
        _txt(canvas, brand, bx, y0 + _px(45), bsc, GREY)

    # ─────────────────────────────────────────────────────────────────────────
    # Left panel
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_left_panel(self, canvas, model_results, fused, audio_conf):
        X0, X1 = 0, LEFT_W
        y = _px(16)

        def row(text, yy, color=WHITE, scale=_S_BODY, bold=False):
            _txt(canvas, text, X0 + PAD, yy, scale, color, _TH2 if bold else _TH1)

        def section(title, yy):
            row(title, yy, ACCENT, _S_TITLE, True)
            yy += LINE_TITLE - _px(4)
            _sep(canvas, X0, X1, yy)
            return yy + _px(10)

        y = section("[ MODEL STATUS ]", y)

        model_names = {
            "visiodect": "VISIODECT",
            "uav_ir":    "UAV-IR   ",
            "uav_rgb":   "UAV-RGB  ",
        }
        model_sub = {
            "visiodect": "YOLOv8n",
            "uav_ir":    "YOLOv8m",
            "uav_rgb":   "YOLOv8m",
        }
        for r in model_results:
            name  = model_names.get(r["name"], r["name"].upper()[:9])
            sub   = model_sub.get(r["name"], "")
            n_det = len(r["boxes"])
            row(f"  {name}  {sub}", y, ACCENT, _S_SMALL);       y += LINE_SMALL
            row(f"    STATUS: OK   DET: {n_det}", y, WHITE, _S_SMALL); y += LINE_BODY

        a_col = ACCENT if audio_conf > 0.5 else GREY
        row("  AUDIO CLASSIFIER", y, a_col, _S_SMALL);           y += LINE_SMALL
        row(f"    CONF: {audio_conf:.2f}", y, WHITE, _S_SMALL);   y += LINE_BODY + _px(4)

        _sep(canvas, X0, X1, y);  y += _px(8)
        y = section("[ FUSION STATUS ]", y)
        n_fused = len(fused.get("boxes", []))
        row("  WBF ACTIVE", y, ACCENT, _S_BODY);                  y += LINE_BODY
        row(f"  FUSED BOXES : {n_fused}", y, WHITE, _S_BODY);     y += LINE_BODY
        row("  IoU thr     : 0.35", y, GREY, _S_SMALL);           y += LINE_SMALL
        row("  Skip thr    : 0.01", y, GREY, _S_SMALL);           y += LINE_BODY + _px(4)

        _sep(canvas, X0, X1, y);  y += _px(8)
        y = section("[ MODEL WEIGHTS ]", y)
        BAR_MAX = LEFT_W - PAD * 2 - _px(60)
        for name, w in [("VISIODECT", 0.50), ("UAV-IR", 0.30), ("UAV-RGB", 0.20)]:
            bar_w = int(w * BAR_MAX)
            bx0   = X0 + PAD
            cv2.rectangle(canvas, (bx0, y), (bx0 + bar_w, y + _px(6)),
                          ACCENT_DIM, -1)
            cv2.rectangle(canvas, (bx0, y), (bx0 + BAR_MAX, y + _px(6)),
                          GREY, _TH1)
            row(f"  {name:<10} {w:.2f}", y + _px(18), WHITE, _S_SMALL)
            y += _px(28)

        y += _px(2)
        _sep(canvas, X0, X1, y);  y += _px(8)
        y = section("[ SYSTEM ]", y)
        for label, value in [
            ("DEVICE",  "CUDA"),
            ("GPU",     "RTX 2050"),
            ("VRAM",    "4 GB"),
            ("PyTorch", "2.5.1+cu121"),
            ("DPI",     f"{DPI_SCALE:.2f}x  {W}x{H}"),
        ]:
            row(f"  {label:<8} : {value}", y, WHITE, _S_SMALL);  y += LINE_SMALL

    # ─────────────────────────────────────────────────────────────────────────
    # Right panel
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_right_panel(self, canvas, solutions, smoothed):
        X0, X1   = W - RIGHT_W, W
        y        = _px(16)
        INNER_W  = RIGHT_W - PAD * 2

        def row(text, yy, color=WHITE, scale=_S_BODY, bold=False):
            _txt(canvas, text, X0 + PAD, yy, scale, color, _TH2 if bold else _TH1)

        def section(title, yy):
            row(title, yy, ACCENT, _S_TITLE, True)
            yy += LINE_TITLE - _px(4)
            _sep(canvas, X0, X1, yy)
            return yy + _px(10)

        y = section("[ FIRE SOLUTION ]", y)

        display_tids = [tid for tid in self._panel_order if tid in self._display]

        if not display_tids:
            row("  NO TARGET DETECTED", y, GREY, _S_BODY);  y += LINE_BODY
        else:
            MAX_PANEL_Y = H - _px(200)
            for tid in display_tids[:3]:
                if y > MAX_PANEL_Y:
                    row("  + more targets...", y, GREY, _S_SMALL)
                    y += LINE_SMALL
                    break

                d      = self._display[tid]
                tl     = max(0.0, min(1.0, d["threat"]))
                tl_col = RED if tl > 0.7 else YELLOW if tl > 0.4 else ACCENT
                col    = tl_col if tl > 0.4 else WHITE

                is_live   = tid in {int(s["track_id"]) for s in solutions}
                txt_alpha = WHITE if is_live else GREY

                row(f"  TARGET #{tid}", y, col, _S_BODY, True);  y += LINE_BODY

                t_str = (f"{d['t_intercept']:.1f} fr"
                         if d["t_intercept"] >= 0 else "N/A")
                data_rows = [
                    (f"    AZIMUTH  : {d['azimuth']:>7.2f} deg",  txt_alpha),
                    (f"    ELEVAT.  : {d['elevation']:>7.2f} deg", txt_alpha),
                    (f"    T-INTCPT : {t_str:>8}",                 YELLOW),
                    (f"    DIST     : {d['distance']:>7.1f} px",   txt_alpha),
                    (f"    VEL      : ({d['vx']:+.1f},{d['vy']:+.1f})", txt_alpha),
                ]
                for text, color in data_rows:
                    if y > MAX_PANEL_Y:
                        break
                    row(text, y, color, _S_SMALL);  y += LINE_SMALL

                if y <= MAX_PANEL_Y:
                    y += _px(2)
                    bar_w = int(tl * (INNER_W - 2))
                    bx0   = X0 + PAD
                    cv2.rectangle(canvas, (bx0, y),
                                  (bx0 + INNER_W, y + _px(8)), DARK_GREY, -1)
                    if bar_w > 0:
                        cv2.rectangle(canvas, (bx0, y),
                                      (bx0 + bar_w, y + _px(8)), tl_col, -1)
                    row(f"    THREAT : {tl:.2f}", y + _px(14), tl_col, _S_SMALL)
                    y += _px(24)
                    _sep(canvas, X0, X1, y, DARK_GREY);  y += _px(8)

        _sep(canvas, X0, X1, y);  y += _px(8)
        y = section("[ TRACKING ]", y)

        track_ids = smoothed.get("track_ids", [])
        n_tracked = len(track_ids)
        row(f"  ACTIVE TRACKS : {n_tracked}", y, WHITE, _S_BODY);  y += LINE_BODY

        for tid in list(track_ids)[:6]:
            row(f"    ID {int(tid):03d}  LOCKED", y, ACCENT, _S_SMALL)
            y += LINE_SMALL
        y += _px(6)

        remaining_lines = max(0, (H - _px(60) - y)) // LINE_SMALL
        if remaining_lines >= 3:
            _sep(canvas, X0, X1, y);  y += _px(8)
            y = section("[ SYSTEM LOG ]", y)
            for line in list(self.log)[-(remaining_lines):]:
                if y > H - _px(18):
                    break
                row(f" {line}", y, GREY, _S_SMALL);  y += LINE_SMALL

    # ─────────────────────────────────────────────────────────────────────────
    # Public utilities
    # ─────────────────────────────────────────────────────────────────────────
    def log_event(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")

    def reset_tracks(self) -> None:
        self._track_history.clear()
        self._ema.clear()
        self._display.clear()
        self._hold_ttl.clear()
        self._panel_order.clear()
        self._panel_refresh = 0