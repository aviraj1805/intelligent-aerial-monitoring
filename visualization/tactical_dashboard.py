"""
IAMARS — TacticalDashboard  (redesigned)
Full 1280×720 military/tactical dark-theme display.

Layout (fixed, pixel-perfect)
──────────────────────────────
  LEFT   panel  [  0 ..249,   0..719]  model status + fusion + system
  CENTRE panel  [250 ..979,   0..719]  header (0..39) + video (40..669) + bottom bar (670..719)
  RIGHT  panel  [980..1279,   0..719]  fire solution + tracking + log
"""

import cv2
import numpy as np
import time
from collections import deque

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

# ── Layout constants ──────────────────────────────────────────────────────────
W, H        = 1280, 720
LEFT_W      = 250
RIGHT_W     = 300
HEADER_H    = 40
BOTTOM_H    = 50
DIVIDER     = 1            # border line thickness

CX0 = LEFT_W              # centre panel x-start
CX1 = W - RIGHT_W         # centre panel x-end
CW  = CX1 - CX0           # 730

VX0 = CX0 + 4             # video area x-start (inside centre)
VX1 = CX1 - 4             # video area x-end
VY0 = HEADER_H + 4        # video area y-start (below header)
VY1 = H - BOTTOM_H - 4    # video area y-end   (above bottom bar)
VW  = VX1 - VX0           # video width
VH  = VY1 - VY0           # video height

# ── Typography helpers ────────────────────────────────────────────────────────
# All sizes tuned so text never exceeds panel width at the given scale.
_S_TITLE  = 0.46          # section titles
_S_BODY   = 0.39          # body text
_S_SMALL  = 0.34          # small / secondary
_S_BRAND  = 0.38          # bottom-bar branding

_LINE_TITLE = 22           # px between title and first body line
_LINE_BODY  = 16           # px between body lines
_LINE_SMALL = 14           # px between small lines
_PAD        = 10           # horizontal padding inside panels

# ── Separator helper ──────────────────────────────────────────────────────────
def _sep(canvas, x0, x1, y, color=ACCENT_DIM):
    cv2.line(canvas, (x0 + _PAD, y), (x1 - _PAD, y), color, 1)


# ── Text helper ──────────────────────────────────────────────────────────────
def _txt(canvas, text, x, y, scale=_S_BODY, color=WHITE, thickness=1):
    cv2.putText(canvas, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


class TacticalDashboard:
    """
    Parameters
    ----------
    frame_wh      : (W, H) of the source video frames
    max_log_lines : lines kept in the rolling system log
    """

    # EMA alpha — lower = smoother/slower display update (~12-frame lag at 0.08)
    _EMA_ALPHA       = 0.08
    # Commit display values every N frames (suppresses per-digit flicker)
    _DISPLAY_REFRESH = 6
    # Frames to keep showing a track's last known values after it disappears
    _HOLD_FRAMES     = 15

    def __init__(self, frame_wh: tuple = (1920, 1080), max_log_lines: int = 14):
        self.src_w, self.src_h = frame_wh
        self.log: deque         = deque(maxlen=max_log_lines)
        self._frame_count       = 0
        self._blink_state       = True
        self._scan_y            = VY0
        self._start_time        = time.time()
        self._track_history: dict[int, deque] = {}

        self._stages = ["DETECT", "FUSE", "TRACK", "KALMAN", "PREDICT", "FIRE SOL"]

        # Per-track EMA accumulators and latched display snapshots
        self._ema:     dict[int, dict] = {}   # live EMA floats
        self._display: dict[int, dict] = {}   # committed display values
        self._hold_ttl: dict[int, int] = {}   # hold-frames remaining per tid

        # Stable panel ordering so targets don't jump rows between frames
        self._panel_order:   list[int] = []
        self._panel_refresh: int       = 0

        self.log_event("IAMARS SYSTEM ONLINE")
        self.log_event("MODELS LOADED: 4/4")

    # ─────────────────────────────────────────────────────────────────────────
    # Solution EMA smoother — call once per frame before drawing
    # ─────────────────────────────────────────────────────────────────────────
    def _update_solution_ema(self, solutions: list) -> None:
        """
        Update per-track EMA accumulators from the raw solutions list.
        Commits stable display snapshots every _DISPLAY_REFRESH frames.
        Holds the last snapshot for _HOLD_FRAMES after a track vanishes.
        """
        a   = self._EMA_ALPHA
        now_tids = set()

        for sol in solutions:
            tid = int(sol["track_id"])
            now_tids.add(tid)
            vx, vy = float(sol["velocity"][0]), float(sol["velocity"][1])
            raw = {
                "azimuth":    float(sol["azimuth"]),
                "elevation":  float(sol["elevation"]),
                "t_intercept": float(sol["t_intercept"])
                               if sol["t_intercept"] != float("inf") else -1.0,
                "distance":   float(sol["distance"]),
                "vx":         vx,
                "vy":         vy,
                "threat":     float(sol["threat_level"]),
            }
            if tid not in self._ema:
                # First sight: seed EMA with raw value (no jump-in)
                self._ema[tid]     = dict(raw)
                self._display[tid] = dict(raw)
            else:
                for k, v in raw.items():
                    self._ema[tid][k] = self._ema[tid][k] * (1 - a) + v * a

            # Reset hold counter for active tracks
            self._hold_ttl[tid] = self._HOLD_FRAMES

        # Count-down hold timers for tracks no longer in solutions
        for tid in list(self._hold_ttl):
            if tid not in now_tids:
                self._hold_ttl[tid] -= 1
                if self._hold_ttl[tid] <= 0:
                    self._ema.pop(tid, None)
                    self._display.pop(tid, None)
                    self._hold_ttl.pop(tid, None)
                    if tid in self._panel_order:
                        self._panel_order.remove(tid)

        # Commit EMA → display every _DISPLAY_REFRESH frames
        self._panel_refresh += 1
        if self._panel_refresh >= self._DISPLAY_REFRESH:
            self._panel_refresh = 0
            for tid, vals in self._ema.items():
                self._display[tid] = dict(vals)

        # Maintain stable panel order: keep existing order, append new tids at end
        for tid in now_tids:
            if tid not in self._panel_order:
                self._panel_order.append(tid)

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
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
        if self._scan_y > VY1:
            self._scan_y = VY0

        # Update right-panel smoothed values before any drawing
        self._update_solution_ema(solutions)

        # ── Base canvas ──────────────────────────────────────────────────────
        canvas = np.full((H, W, 3), BG, dtype=np.uint8)
        canvas[:, :LEFT_W]    = PANEL_BG
        canvas[:, W-RIGHT_W:] = PANEL_BG

        # ── Sections ─────────────────────────────────────────────────────────
        self._draw_header(canvas, frame_index)
        self._draw_centre(canvas, frame, predicted, solutions)
        self._draw_bottom_bar(canvas, frame_index)
        self._draw_left_panel(canvas, model_results, fused, audio_conf)
        self._draw_right_panel(canvas, solutions, smoothed)

        # ── Panel dividers ────────────────────────────────────────────────────
        cv2.line(canvas, (LEFT_W,   0), (LEFT_W,   H), ACCENT_DIM, DIVIDER)
        cv2.line(canvas, (W-RIGHT_W,0), (W-RIGHT_W,H), ACCENT_DIM, DIVIDER)

        return canvas

    # ─────────────────────────────────────────────────────────────────────────
    # Header bar
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_header(self, canvas, frame_index):
        cv2.rectangle(canvas, (CX0, 0), (CX1, HEADER_H), HEADER_BG, -1)
        cv2.line(canvas, (CX0, HEADER_H), (CX1, HEADER_H), ACCENT_DIM, 1)

        # Blinking record dot
        dot_x, dot_y = CX0 + 18, HEADER_H // 2
        dot_col = RED if self._blink_state else (45, 45, 45)
        cv2.circle(canvas, (dot_x, dot_y), 6, dot_col, -1)

        # Title
        _txt(canvas, "I A M A R S", CX0 + 34, dot_y + 6, 0.72, ACCENT, 2)

        # Sub-label
        _txt(canvas, "TACTICAL SURVEILLANCE SYSTEM", CX0 + 34, dot_y + 18,
             0.30, ACCENT_DIM, 1)

        # Timestamp (right-aligned)
        elapsed = time.time() - self._start_time
        ts = f"T+{elapsed:07.2f}s   FRAME {frame_index:05d}"
        tw = cv2.getTextSize(ts, FONT, 0.40, 1)[0][0]
        _txt(canvas, ts, CX1 - tw - 10, dot_y + 6, 0.40, GREY)

    # ─────────────────────────────────────────────────────────────────────────
    # Centre video panel
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_centre(self, canvas, frame, predicted, solutions):
        resized = cv2.resize(frame, (VW, VH))

        # Tactical grid (very subtle)
        overlay = resized.copy()
        step = 60
        for gx in range(0, VW, step):
            cv2.line(overlay, (gx, 0), (gx, VH), GRID_COL, 1)
        for gy in range(0, VH, step):
            cv2.line(overlay, (0, gy), (VW, gy), GRID_COL, 1)
        resized = cv2.addWeighted(resized, 0.84, overlay, 0.16, 0)

        # Centre crosshair
        vcx, vcy = VW // 2, VH // 2
        cv2.line(resized, (vcx - 18, vcy), (vcx + 18, vcy), ACCENT_DIM, 1)
        cv2.line(resized, (vcx, vcy - 18), (vcx, vcy + 18), ACCENT_DIM, 1)
        cv2.circle(resized, (vcx, vcy), 3, ACCENT_DIM, 1)

        # Scale factors
        sx = VW / self.src_w
        sy = VH / self.src_h

        boxes      = predicted.get("boxes",      np.empty((0, 4), dtype=np.float32))
        track_ids  = predicted.get("track_ids",  np.empty((0,),   dtype=np.int32))
        trajs      = predicted.get("trajectories", [])

        for i in range(len(boxes)):
            box = boxes[i]
            tid = int(track_ids[i])

            x1 = int(box[0] * sx);  y1 = int(box[1] * sy)
            x2 = int(box[2] * sx);  y2 = int(box[3] * sy)
            cx = (x1 + x2) // 2;   cy = (y1 + y2) // 2

            # Update trail
            if tid not in self._track_history:
                self._track_history[tid] = deque(maxlen=30)
            self._track_history[tid].append((cx, cy))

            # Detection rectangle
            cv2.rectangle(resized, (x1, y1), (x2, y2), ACCENT, 2)

            # Corner brackets
            bl = 12
            for (bx, by), (ddx1, ddy1), (ddx2, ddy2) in [
                ((x1, y1), ( bl,  0), ( 0,  bl)),
                ((x2, y1), (-bl,  0), ( 0,  bl)),
                ((x1, y2), ( bl,  0), ( 0, -bl)),
                ((x2, y2), (-bl,  0), ( 0, -bl)),
            ]:
                cv2.line(resized, (bx, by), (bx+ddx1, by+ddy1), WHITE, 2)
                cv2.line(resized, (bx, by), (bx+ddx2, by+ddy2), WHITE, 2)

            # Label tag — above box, never clips at top or right edge
            label = f"TGT #{tid}"
            lw, lh = cv2.getTextSize(label, FONT, 0.42, 1)[0]
            tag_h  = lh + 8
            # Place 2 px above the box top; push down if that clips frame top
            tag_y0 = max(y1 - tag_h - 2, 0)
            tag_y1 = tag_y0 + tag_h
            # Right-clamp so tag never exits the video area
            tag_x0 = min(x1, VW - lw - 10)
            tag_x0 = max(tag_x0, 0)
            cv2.rectangle(resized, (tag_x0, tag_y0), (tag_x0 + lw + 8, tag_y1), ACCENT, -1)
            _txt(resized, label, tag_x0 + 4, tag_y1 - 4, 0.42, (0, 0, 0), 1)

            # Trail dots
            trail = list(self._track_history.get(tid, []))
            for pt in trail[:-1]:
                cv2.circle(resized, pt, 2, ACCENT_DIM, -1)

            # Predicted trajectory
            if i < len(trajs):
                traj = trajs[i]
                for t_pt in traj:
                    px = int(t_pt[0] * sx)
                    py = int(t_pt[1] * sy)
                    if 0 <= px < VW and 0 <= py < VH:
                        cv2.circle(resized, (px, py), 2, ORANGE, -1)

        # Intercept crosshair — one label per distinct pixel position
        drawn_labels: list[tuple[int, int]] = []
        for sol in solutions:
            if sol["t_intercept"] != float("inf"):
                arm = 16
                ix = int(sol["intercept_pos"][0] * sx)
                iy = int(sol["intercept_pos"][1] * sy)
                # Clamp to video bounds
                ix = max(arm, min(VW - arm - 1, ix))
                iy = max(arm, min(VH - arm - 1, iy))
                cv2.line(resized, (ix-arm, iy), (ix+arm, iy), RED, 2)
                cv2.line(resized, (ix, iy-arm), (ix, iy+arm), RED, 2)
                cv2.circle(resized, (ix, iy), arm, RED, 1)
                if self._blink_state:
                    lbl = "INTERCEPT"
                    lw2 = cv2.getTextSize(lbl, FONT, 0.42, 1)[0][0]
                    lx  = min(ix + arm + 4, VW - lw2 - 4)
                    ly  = iy + 5
                    # Only draw label if no existing label is within 20 px vertically
                    too_close = any(abs(ly - py) < 20 and abs(lx - px) < lw2 + 4
                                    for (px, py) in drawn_labels)
                    if not too_close:
                        _txt(resized, lbl, lx, ly, 0.42, RED)
                        drawn_labels.append((lx, ly))

        # Scan line — soft fade strip instead of a hard line
        syl = self._scan_y - VY0
        if 2 <= syl < VH - 2:
            fade_h = 6                              # half-height of the glow strip
            y_lo   = max(0, syl - fade_h)
            y_hi   = min(VH, syl + fade_h + 1)
            strip  = resized[y_lo:y_hi].astype(np.float32)
            tint   = np.zeros_like(strip)
            tint[:, :, 1] = 48                      # green channel tint
            # Build a vertical alpha gradient: bright at centre, fade to edges
            rows  = strip.shape[0]
            alpha = np.abs(np.linspace(-1, 1, rows)).reshape(rows, 1, 1)
            alpha = (1.0 - alpha) * 0.35            # max 35 % blend at centre
            resized[y_lo:y_hi] = np.clip(strip + tint * alpha, 0, 255).astype(np.uint8)

        # Paste video into canvas
        canvas[VY0:VY1, VX0:VX1] = resized

        # Corner bracket decoration on the video border
        p0 = (VX0, VY0);  p1 = (VX1 - 1, VY1 - 1)
        arm2 = 20
        for (bx, by), (ddx1, ddy1), (ddx2, ddy2) in [
            (p0,          ( arm2,  0), ( 0,  arm2)),
            ((p1[0],p0[1]),(-arm2,  0), ( 0,  arm2)),
            ((p0[0],p1[1]),( arm2,  0), ( 0, -arm2)),
            (p1,          (-arm2,  0), ( 0, -arm2)),
        ]:
            cv2.line(canvas, (bx, by), (bx+ddx1, by+ddy1), ACCENT, 2)
            cv2.line(canvas, (bx, by), (bx+ddx2, by+ddy2), ACCENT, 2)

    # ─────────────────────────────────────────────────────────────────────────
    # Bottom bar
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_bottom_bar(self, canvas, frame_index):
        y0 = H - BOTTOM_H
        cv2.rectangle(canvas, (CX0, y0), (CX1, H), HEADER_BG, -1)
        cv2.line(canvas, (CX0, y0), (CX1, y0), ACCENT_DIM, 1)

        # Pipeline stage boxes — evenly spaced, occupying top 28 px of bar
        n        = len(self._stages)
        avail_w  = CW - 20
        box_w    = min(90, avail_w // n - 6)
        gap      = (avail_w - n * box_w) // (n + 1)
        x        = CX0 + gap
        by0      = y0 + 5
        by1      = y0 + 30

        for i, stage in enumerate(self._stages):
            bx0 = x
            bx1 = x + box_w
            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), DARK_GREY, -1)
            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), ACCENT, 1)
            tw = cv2.getTextSize(stage, FONT, _S_SMALL, 1)[0][0]
            tx = bx0 + (box_w - tw) // 2
            _txt(canvas, stage, tx, by1 - 6, _S_SMALL, ACCENT)
            if i < n - 1:
                ax = bx1 + max(gap // 2, 3)
                ay = (by0 + by1) // 2
                cv2.arrowedLine(canvas, (ax - 3, ay), (ax + 3, ay),
                                ACCENT_DIM, 1, tipLength=0.6)
            x += box_w + gap

        # Branding — always on bottom sub-line (y0+33 .. y0+50), centred
        brand = "IAMARS v1.0  |  DRDO DEMO BUILD  |  CONFIDENTIAL"
        bw    = cv2.getTextSize(brand, FONT, 0.28, 1)[0][0]
        bx    = CX0 + (CW - bw) // 2
        _txt(canvas, brand, bx, y0 + 45, 0.28, GREY)

    # ─────────────────────────────────────────────────────────────────────────
    # Left panel
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_left_panel(self, canvas, model_results, fused, audio_conf):
        X0, X1 = 0, LEFT_W
        y = 16

        def row(text, yy, color=WHITE, scale=_S_BODY, bold=False):
            _txt(canvas, text, X0 + _PAD, yy, scale, color, 2 if bold else 1)

        def section(title, yy):
            row(title, yy, ACCENT, _S_TITLE, True)
            yy += _LINE_TITLE - 4
            _sep(canvas, X0, X1, yy)
            return yy + 10

        # ── MODEL STATUS ──────────────────────────────────────────────────────
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
            row(f"  {name}  {sub}", y, ACCENT, _S_SMALL);  y += _LINE_SMALL
            row(f"    STATUS: OK   DET: {n_det}", y, WHITE, _S_SMALL);  y += _LINE_BODY

        # Audio
        a_col = ACCENT if audio_conf > 0.5 else GREY
        row("  AUDIO CLASSIFIER", y, a_col, _S_SMALL);  y += _LINE_SMALL
        row(f"    CONF: {audio_conf:.2f}", y, WHITE, _S_SMALL);  y += _LINE_BODY + 4

        # ── FUSION ───────────────────────────────────────────────────────────
        _sep(canvas, X0, X1, y);  y += 8
        y = section("[ FUSION STATUS ]", y)
        n_fused = len(fused.get("boxes", []))
        row("  WBF ACTIVE", y, ACCENT, _S_BODY);  y += _LINE_BODY
        row(f"  FUSED BOXES : {n_fused}", y, WHITE, _S_BODY);  y += _LINE_BODY
        row("  IoU thr     : 0.45", y, GREY, _S_SMALL);  y += _LINE_SMALL
        row("  Skip thr    : 0.05", y, GREY, _S_SMALL);  y += _LINE_BODY + 4

        # ── MODEL WEIGHTS ─────────────────────────────────────────────────────
        _sep(canvas, X0, X1, y);  y += 8
        y = section("[ MODEL WEIGHTS ]", y)
        BAR_MAX = LEFT_W - _PAD * 2 - 60
        for name, w in [("VISIODECT", 0.50), ("UAV-IR", 0.30), ("UAV-RGB", 0.20)]:
            bar_w = int(w * BAR_MAX)
            bx0 = X0 + _PAD
            cv2.rectangle(canvas, (bx0, y), (bx0 + bar_w, y + 6), ACCENT_DIM, -1)
            cv2.rectangle(canvas, (bx0, y), (bx0 + BAR_MAX, y + 6), GREY, 1)
            row(f"  {name:<10} {w:.2f}", y + 18, WHITE, _S_SMALL);  y += 28

        # ── SYSTEM INFO ───────────────────────────────────────────────────────
        y += 2
        _sep(canvas, X0, X1, y);  y += 8
        y = section("[ SYSTEM ]", y)
        for label, value in [
            ("DEVICE",   "CUDA"),
            ("GPU",      "RTX 2050"),
            ("VRAM",     "4 GB"),
            ("PyTorch",  "2.5.1+cu121"),
        ]:
            row(f"  {label:<8} : {value}", y, WHITE, _S_SMALL);  y += _LINE_SMALL

    # ─────────────────────────────────────────────────────────────────────────
    # Right panel
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_right_panel(self, canvas, solutions, smoothed):
        X0 = W - RIGHT_W
        X1 = W
        y  = 16
        INNER_W = RIGHT_W - _PAD * 2

        def row(text, yy, color=WHITE, scale=_S_BODY, bold=False):
            _txt(canvas, text, X0 + _PAD, yy, scale, color, 2 if bold else 1)

        def section(title, yy):
            row(title, yy, ACCENT, _S_TITLE, True)
            yy += _LINE_TITLE - 4
            _sep(canvas, X0, X1, yy)
            return yy + 10

        # ── FIRE SOLUTION — draw from latched display snapshots ───────────────
        y = section("[ FIRE SOLUTION ]", y)

        # Build stable display list from panel_order, filtered to known display entries
        display_tids = [tid for tid in self._panel_order if tid in self._display]

        if not display_tids:
            row("  NO TARGET DETECTED", y, GREY, _S_BODY);  y += _LINE_BODY
        else:
            MAX_PANEL_Y = H - 200
            for tid in display_tids[:3]:
                if y > MAX_PANEL_Y:
                    row("  + more targets...", y, GREY, _S_SMALL);  y += _LINE_SMALL
                    break

                d   = self._display[tid]
                tl  = max(0.0, min(1.0, d["threat"]))
                tl_col = RED if tl > 0.7 else YELLOW if tl > 0.4 else ACCENT
                col    = tl_col if tl > 0.4 else WHITE

                # Fading indicator: dim text slightly if track is in hold (not live)
                is_live   = tid in {int(s["track_id"]) for s in solutions}
                txt_alpha = WHITE if is_live else GREY

                row(f"  TARGET #{tid}", y, col, _S_BODY, True);  y += _LINE_BODY

                t_str = (f"{d['t_intercept']:.1f} fr"
                         if d["t_intercept"] >= 0 else "N/A")
                data_rows = [
                    (f"    AZIMUTH  : {d['azimuth']:>7.2f} deg",   txt_alpha),
                    (f"    ELEVAT.  : {d['elevation']:>7.2f} deg",  txt_alpha),
                    (f"    T-INTCPT : {t_str:>8}",                  YELLOW),
                    (f"    DIST     : {d['distance']:>7.1f} px",    txt_alpha),
                    (f"    VEL      : ({d['vx']:+.1f},{d['vy']:+.1f})", txt_alpha),
                ]
                for text, color in data_rows:
                    if y > MAX_PANEL_Y:
                        break
                    row(text, y, color, _S_SMALL);  y += _LINE_SMALL

                if y <= MAX_PANEL_Y:
                    y += 2
                    bar_w = int(tl * (INNER_W - 2))
                    bx0   = X0 + _PAD
                    cv2.rectangle(canvas, (bx0, y), (bx0 + INNER_W, y + 8), DARK_GREY, -1)
                    if bar_w > 0:
                        cv2.rectangle(canvas, (bx0, y), (bx0 + bar_w, y + 8), tl_col, -1)
                    row(f"    THREAT : {tl:.2f}", y + 14, tl_col, _S_SMALL)
                    y += 24
                    _sep(canvas, X0, X1, y, DARK_GREY);  y += 8

        # ── TRACKING ──────────────────────────────────────────────────────────
        _sep(canvas, X0, X1, y);  y += 8
        y = section("[ TRACKING ]", y)

        track_ids = smoothed.get("track_ids", [])
        n_tracked = len(track_ids)
        row(f"  ACTIVE TRACKS : {n_tracked}", y, WHITE, _S_BODY);  y += _LINE_BODY

        for tid in list(track_ids)[:6]:
            row(f"    ID {int(tid):03d}  LOCKED", y, ACCENT, _S_SMALL);  y += _LINE_SMALL
        y += 6

        # ── SYSTEM LOG ────────────────────────────────────────────────────────
        remaining_lines = max(0, (H - 60 - y)) // _LINE_SMALL
        if remaining_lines >= 3:
            _sep(canvas, X0, X1, y);  y += 8
            y = section("[ SYSTEM LOG ]", y)
            log_entries = list(self.log)[-(remaining_lines):]
            for line in log_entries:
                if y > H - 18:
                    break
                row(f" {line}", y, GREY, _S_SMALL);  y += _LINE_SMALL

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