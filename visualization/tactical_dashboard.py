"""
IAMARS — TacticalDashboard
Full 1280×720 military/tactical dark-theme display.

Layout
------
  Left  panel (0..249,   0..719) : model status + fusion info
  Centre panel (250..979, 0..669): live video with overlays
  Right  panel (980..1279,0..719): fire solution + tracking + log
  Bottom bar   (250..979,670..719): pipeline stages + branding
  Header       (250..979,  0.. 39): title + blink dot + timestamp
"""

import cv2
import numpy as np
import time
import math
from collections import deque

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
BG          = (14,  12,  10)      # #0a0c0e near-black
ACCENT      = (0,  255, 136)      # bright green
ACCENT_DIM  = (0,  140,  75)
RED         = (0,   40, 220)
ORANGE      = (0,  140, 255)
YELLOW      = (0,  220, 220)
WHITE       = (220, 220, 220)
GREY        = (80,  80,  80)
DARK_GREY   = (30,  30,  30)
PANEL_BG    = (20,  22,  24)
GRID_COL    = (30,  45,  30)

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO   = cv2.FONT_HERSHEY_PLAIN

# ── Layout constants ──────────────────────────────────────────────────────────
W, H            = 1280, 720
LEFT_W          = 250
RIGHT_W         = 300
BOTTOM_H        = 50
HEADER_H        = 40
CENTER_X0       = LEFT_W
CENTER_X1       = W - RIGHT_W
CENTER_Y0       = HEADER_H
CENTER_Y1       = H - BOTTOM_H
CENTER_W        = CENTER_X1 - CENTER_X0   # 730
CENTER_H        = CENTER_Y1 - CENTER_Y0   # 630

# ── Video display area inside centre panel ────────────────────────────────────
VIDEO_X0        = CENTER_X0 + 4
VIDEO_Y0        = CENTER_Y0 + 4
VIDEO_X1        = CENTER_X1 - 4
VIDEO_Y1        = CENTER_Y1 - 4
VIDEO_W         = VIDEO_X1 - VIDEO_X0     # 722
VIDEO_H         = VIDEO_Y1 - VIDEO_Y0     # 622


class TacticalDashboard:
    """
    Parameters
    ----------
    frame_wh     : (W, H) of the source video frames
    max_log_lines: number of lines to keep in the system log
    """

    def __init__(
        self,
        frame_wh: tuple    = (1920, 1080),
        max_log_lines: int = 12,
    ):
        self.src_w, self.src_h = frame_wh
        self.log: deque        = deque(maxlen=max_log_lines)
        self._frame_count      = 0
        self._blink_state      = True
        self._scan_y           = VIDEO_Y0
        self._start_time       = time.time()

        # Track history for trajectory drawing  {track_id: deque of (cx,cy)}
        self._track_history: dict[int, deque] = {}

        # Pipeline stage names
        self._stages = [
            "DETECT", "FUSE", "TRACK", "KALMAN", "PREDICT", "FIRE SOL"
        ]

        self.log_event("IAMARS SYSTEM ONLINE")
        self.log_event("MODELS LOADED: 4/4")

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────
    def render(
        self,
        frame:        np.ndarray,
        model_results: list[dict],
        fused:         dict,
        tracked:       dict,
        smoothed:      dict,
        predicted:     dict,
        solutions:     list[dict],
        frame_index:   int,
        audio_conf:    float = 0.0,
    ) -> np.ndarray:
        """
        Compose and return the full 1280×720 tactical display.
        """
        self._frame_count = frame_index
        self._blink_state = (frame_index % 20) < 10   # blink every 10 frames

        # Update scan line
        self._scan_y += 4
        if self._scan_y > VIDEO_Y1:
            self._scan_y = VIDEO_Y0

        # ── Base canvas ──────────────────────────────────────────────────────
        canvas = np.full((H, W, 3), BG, dtype=np.uint8)

        # ── Panel backgrounds ─────────────────────────────────────────────────
        canvas[:, :LEFT_W]    = PANEL_BG
        canvas[:, W-RIGHT_W:] = PANEL_BG

        # ── Draw sections ─────────────────────────────────────────────────────
        self._draw_centre(canvas, frame, predicted, solutions, frame_index)
        self._draw_header(canvas, frame_index)
        self._draw_left_panel(canvas, model_results, fused, audio_conf)
        self._draw_right_panel(canvas, solutions, smoothed)
        self._draw_bottom_bar(canvas, frame_index)

        # ── Panel dividers ────────────────────────────────────────────────────
        cv2.line(canvas, (LEFT_W, 0),   (LEFT_W, H),   ACCENT_DIM, 1)
        cv2.line(canvas, (W-RIGHT_W, 0),(W-RIGHT_W, H),ACCENT_DIM, 1)

        return canvas

    # ─────────────────────────────────────────────────────────────────────────
    # Centre panel
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_centre(self, canvas, frame, predicted, solutions, frame_index):
        # Resize source frame to video display area
        resized = cv2.resize(frame, (VIDEO_W, VIDEO_H))

        # ── Tactical grid overlay ─────────────────────────────────────────────
        grid_step = 60
        overlay   = resized.copy()
        for gx in range(0, VIDEO_W, grid_step):
            cv2.line(overlay, (gx, 0), (gx, VIDEO_H), GRID_COL, 1)
        for gy in range(0, VIDEO_H, grid_step):
            cv2.line(overlay, (0, gy), (VIDEO_W, gy), GRID_COL, 1)
        resized = cv2.addWeighted(resized, 0.85, overlay, 0.15, 0)

        # ── Crosshair at video centre ─────────────────────────────────────────
        vcx, vcy = VIDEO_W // 2, VIDEO_H // 2
        cv2.line(resized, (vcx-20, vcy), (vcx+20, vcy), ACCENT_DIM, 1)
        cv2.line(resized, (vcx, vcy-20), (vcx, vcy+20), ACCENT_DIM, 1)

        # ── Scale factors: source→display ────────────────────────────────────
        sx = VIDEO_W / self.src_w
        sy = VIDEO_H / self.src_h

        # ── Bounding boxes + trajectory dots ─────────────────────────────────
        boxes      = predicted.get("boxes", np.empty((0,4)))
        track_ids  = predicted.get("track_ids", np.empty((0,), dtype=np.int32))
        velocities = predicted.get("velocities", np.empty((0,2)))
        trajs      = predicted.get("trajectories", [])

        for i in range(len(boxes)):
            box = boxes[i]
            tid = int(track_ids[i])

            x1 = int(box[0] * sx);  y1 = int(box[1] * sy)
            x2 = int(box[2] * sx);  y2 = int(box[3] * sy)
            cx = (x1 + x2) // 2;   cy = (y1 + y2) // 2

            # Update trail history
            if tid not in self._track_history:
                self._track_history[tid] = deque(maxlen=30)
            self._track_history[tid].append((cx, cy))

            # Green detection box
            cv2.rectangle(resized, (x1, y1), (x2, y2), ACCENT, 2)

            # Corner bracket accents
            bl = 12
            for (bx, by), (dx1, dy1), (dx2, dy2) in [
                ((x1,y1),(bl,0),(0,bl)), ((x2,y1),(-bl,0),(0,bl)),
                ((x1,y2),(bl,0),(0,-bl)),((x2,y2),(-bl,0),(0,-bl)),
            ]:
                cv2.line(resized,(bx,by),(bx+dx1,by+dy1),WHITE,2)
                cv2.line(resized,(bx,by),(bx+dx2,by+dy2),WHITE,2)

            # Label
            label = f"TARGET #{tid}"
            lw, lh = cv2.getTextSize(label, FONT, 0.45, 1)[0]
            cv2.rectangle(resized,(x1, y1-lh-6),(x1+lw+4, y1), ACCENT, -1)
            cv2.putText(resized, label, (x1+2, y1-3),
                        FONT, 0.45, (0,0,0), 1, cv2.LINE_AA)

            # Trail dots
            trail = list(self._track_history.get(tid, []))
            for pt in trail[:-1]:
                cv2.circle(resized, pt, 2, ACCENT_DIM, -1)

            # Predicted trajectory dots (orange)
            if i < len(trajs):
                traj = trajs[i]
                for t_pt in traj:
                    px = int(t_pt[0] * sx);  py = int(t_pt[1] * sy)
                    if 0 <= px < VIDEO_W and 0 <= py < VIDEO_H:
                        cv2.circle(resized, (px, py), 2, ORANGE, -1)

        # ── Intercept crosshair (red, blinking) ───────────────────────────────
        for sol in solutions:
            if sol["t_intercept"] != float("inf"):
                ix = int(sol["intercept_pos"][0] * sx)
                iy = int(sol["intercept_pos"][1] * sy)
                arm = 16
                cv2.line(resized,(ix-arm,iy),(ix+arm,iy), RED, 2)
                cv2.line(resized,(ix,iy-arm),(ix,iy+arm), RED, 2)
                cv2.circle(resized,(ix,iy), arm, RED, 1)
                if self._blink_state:
                    cv2.putText(resized,"INTERCEPT",(ix+arm+4,iy+5),
                                FONT, 0.45, RED, 1, cv2.LINE_AA)

        # ── Scanning line ─────────────────────────────────────────────────────
        scan_y_local = self._scan_y - VIDEO_Y0
        if 0 <= scan_y_local < VIDEO_H:
            cv2.line(resized,(0,scan_y_local),(VIDEO_W,scan_y_local),
                     (0, 60, 0), 1)

        # ── Paste resized frame onto canvas ───────────────────────────────────
        canvas[VIDEO_Y0:VIDEO_Y1, VIDEO_X0:VIDEO_X1] = resized

        # ── Corner brackets on centre panel border ────────────────────────────
        p0 = (VIDEO_X0, VIDEO_Y0);  p1 = (VIDEO_X1-1, VIDEO_Y1-1)
        arm = 20
        for (bx,by),(dx1,dy1),(dx2,dy2) in [
            (p0,(arm,0),(0,arm)),(( p1[0],p0[1]),(-arm,0),(0,arm)),
            ((p0[0],p1[1]),(arm,0),(0,-arm)),(p1,(-arm,0),(0,-arm)),
        ]:
            cv2.line(canvas,(bx,by),(bx+dx1,by+dy1),ACCENT,2)
            cv2.line(canvas,(bx,by),(bx+dx2,by+dy2),ACCENT,2)

    # ─────────────────────────────────────────────────────────────────────────
    # Header
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_header(self, canvas, frame_index):
        y0, y1 = 0, HEADER_H
        cv2.rectangle(canvas,(CENTER_X0,y0),(CENTER_X1,y1),(18,20,22),-1)

        # Blinking red dot
        if self._blink_state:
            cv2.circle(canvas,(CENTER_X0+16,y0+20),6,RED,-1)
        else:
            cv2.circle(canvas,(CENTER_X0+16,y0+20),6,(40,40,40),-1)

        # Title
        cv2.putText(canvas,"I A M A R S",(CENTER_X0+32,y0+26),
                    FONT,0.75,ACCENT,2,cv2.LINE_AA)

        # Timestamp
        elapsed = time.time() - self._start_time
        ts = f"T+{elapsed:07.2f}s  FRAME:{frame_index:05d}"
        cv2.putText(canvas,ts,(CENTER_X1-240,y0+26),FONT,0.45,WHITE,1,cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────────────
    # Left panel
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_left_panel(self, canvas, model_results, fused, audio_conf):
        x0, x1 = 0, LEFT_W
        y = 20

        def txt(text, yy, color=WHITE, scale=0.42, thickness=1):
            cv2.putText(canvas, text, (x0+10, yy), FONT, scale,
                        color, thickness, cv2.LINE_AA)

        # ── Section: MODEL STATUS ─────────────────────────────────────────────
        txt("[ MODEL STATUS ]", y, ACCENT, 0.45, 1);  y += 22
        cv2.line(canvas,(x0+8,y),(x1-8,y),ACCENT_DIM,1);  y += 10

        model_names = {
            "visiodect": "VISIODECT  YOLOv8n",
            "uav_ir":    "UAV-IR     YOLOv8m",
            "uav_rgb":   "UAV-RGB    YOLOv8m",
        }
        for r in model_results:
            name    = model_names.get(r["name"], r["name"].upper())
            n_det   = len(r["boxes"])
            ok      = True
            col     = ACCENT if ok else RED
            status  = f"OK  DET:{n_det}"
            txt(f"  {name}", y, col, 0.38);  y += 14
            txt(f"    {status}", y, WHITE, 0.36);  y += 18

        # Audio
        a_col = ACCENT if audio_conf > 0.5 else GREY
        txt(f"  AUDIO CLASSIFIER", y, a_col, 0.38);  y += 14
        txt(f"    CONF: {audio_conf:.2f}", y, WHITE, 0.36);  y += 24

        # ── Section: FUSION STATUS ────────────────────────────────────────────
        cv2.line(canvas,(x0+8,y),(x1-8,y),ACCENT_DIM,1);  y += 8
        txt("[ FUSION STATUS ]", y, ACCENT, 0.45, 1);  y += 20
        n_fused = len(fused.get("boxes",[]))
        txt(f"  WBF ACTIVE", y, ACCENT, 0.38);  y += 16
        txt(f"  FUSED BOXES : {n_fused}", y, WHITE, 0.38);  y += 16
        txt(f"  iou_thr     : 0.45", y, GREY, 0.36);  y += 16
        txt(f"  skip_thr    : 0.05", y, GREY, 0.36);  y += 24

        # ── Section: WEIGHTS ──────────────────────────────────────────────────
        cv2.line(canvas,(x0+8,y),(x1-8,y),ACCENT_DIM,1);  y += 8
        txt("[ MODEL WEIGHTS ]", y, ACCENT, 0.45, 1);  y += 20
        for name, w in [("VISIODECT","0.50"),("UAV-IR","0.30"),("UAV-RGB","0.20")]:
            bar_w = int(float(w) * (LEFT_W - 80))
            cv2.rectangle(canvas,(x0+10,y),(x0+10+bar_w,y+8),ACCENT_DIM,-1)
            txt(f"  {name}: {w}", y+10, WHITE, 0.36);  y += 22

        # ── Section: INFERENCE DEVICE ─────────────────────────────────────────
        y += 6
        cv2.line(canvas,(x0+8,y),(x1-8,y),ACCENT_DIM,1);  y += 8
        txt("[ SYSTEM ]", y, ACCENT, 0.45, 1);  y += 20
        txt("  DEVICE  : CUDA", y, WHITE, 0.38);  y += 16
        txt("  GPU     : RTX 2050", y, WHITE, 0.38);  y += 16
        txt("  VRAM    : 4 GB", y, WHITE, 0.38);  y += 16
        txt("  PyTorch : 2.5.1+cu121", y, WHITE, 0.36);  y += 16

    # ─────────────────────────────────────────────────────────────────────────
    # Right panel
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_right_panel(self, canvas, solutions, smoothed):
        x0 = W - RIGHT_W
        x1 = W
        y  = 20

        def txt(text, yy, color=WHITE, scale=0.42, thickness=1):
            cv2.putText(canvas, text, (x0+10, yy), FONT, scale,
                        color, thickness, cv2.LINE_AA)

        # ── Fire solution per target ──────────────────────────────────────────
        txt("[ FIRE SOLUTION ]", y, ACCENT, 0.45, 1);  y += 22
        cv2.line(canvas,(x0+8,y),(x1-8,y),ACCENT_DIM,1);  y += 10

        if not solutions:
            txt("  NO TARGET", y, GREY, 0.4);  y += 20
        else:
            for sol in solutions[:3]:   # show max 3 targets
                tid  = sol["track_id"]
                col  = ACCENT if sol["threat_level"] > 0.4 else WHITE
                txt(f"  TARGET #{tid}", y, col, 0.42, 1);  y += 16
                txt(f"    AZIMUTH  : {sol['azimuth']:7.2f} deg", y, WHITE, 0.38);  y += 14
                txt(f"    ELEVAT.  : {sol['elevation']:7.2f} deg", y, WHITE, 0.38);  y += 14
                t_str = f"{sol['t_intercept']:.1f} fr" \
                        if sol["t_intercept"] != float("inf") else "  N/A"
                txt(f"    T-INTCPT : {t_str}", y, YELLOW, 0.38);  y += 14
                txt(f"    DIST     : {sol['distance']:7.1f} px", y, WHITE, 0.38);  y += 14
                vx, vy = sol["velocity"]
                txt(f"    VEL      : ({vx:.1f},{vy:.1f})", y, WHITE, 0.38);  y += 16

                # Threat level bar
                tl  = sol["threat_level"]
                bar = int(tl * (RIGHT_W - 24))
                tl_col = RED if tl > 0.7 else YELLOW if tl > 0.4 else ACCENT
                cv2.rectangle(canvas,(x0+10,y),(x0+10+bar,y+8),tl_col,-1)
                txt(f"    THREAT: {tl:.2f}", y+16, tl_col, 0.37);  y += 30
                cv2.line(canvas,(x0+8,y),(x1-8,y),DARK_GREY,1);  y += 8

        # ── Tracking info ─────────────────────────────────────────────────────
        cv2.line(canvas,(x0+8,y),(x1-8,y),ACCENT_DIM,1);  y += 8
        txt("[ TRACKING ]", y, ACCENT, 0.45, 1);  y += 20
        n_tracked = len(smoothed.get("track_ids", []))
        txt(f"  ACTIVE TRACKS : {n_tracked}", y, WHITE, 0.38);  y += 16
        for tid in smoothed.get("track_ids", [])[:6]:
            txt(f"    ID {tid:03d}  LOCKED", y, ACCENT, 0.37);  y += 14
        y += 6

        # ── System log ────────────────────────────────────────────────────────
        cv2.line(canvas,(x0+8,y),(x1-8,y),ACCENT_DIM,1);  y += 8
        txt("[ SYSTEM LOG ]", y, ACCENT, 0.45, 1);  y += 18
        for line in list(self.log)[-10:]:
            txt(f"  {line}", y, GREY, 0.33);  y += 13
            if y > H - 60:
                break

    # ─────────────────────────────────────────────────────────────────────────
    # Bottom bar
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_bottom_bar(self, canvas, frame_index):
        y0 = H - BOTTOM_H
        cv2.rectangle(canvas,(CENTER_X0,y0),(CENTER_X1,H),(18,20,22),-1)
        cv2.line(canvas,(CENTER_X0,y0),(CENTER_X1,y0),ACCENT_DIM,1)

        stage_w = 100
        x       = CENTER_X0 + 10
        for i, stage in enumerate(self._stages):
            col    = ACCENT if True else GREY   # all stages active
            box_x1 = x + stage_w - 6
            cv2.rectangle(canvas,(x,y0+8),(box_x1,y0+30),DARK_GREY,-1)
            cv2.rectangle(canvas,(x,y0+8),(box_x1,y0+30),col,1)
            tw = cv2.getTextSize(stage,FONT,0.38,1)[0][0]
            cv2.putText(canvas,stage,(x+(stage_w-6-tw)//2,y0+23),
                        FONT,0.38,col,1,cv2.LINE_AA)
            if i < len(self._stages)-1:
                cv2.arrowedLine(canvas,(box_x1+1,y0+19),(box_x1+7,y0+19),
                                ACCENT_DIM,1,tipLength=0.4)
            x += stage_w

        # Branding
        brand = "IAMARS v1.0  |  DRDO DEMO BUILD  |  CONFIDENTIAL"
        bw    = cv2.getTextSize(brand,FONT,0.38,1)[0][0]
        cv2.putText(canvas,brand,(CENTER_X1-bw-10,y0+23),
                    FONT,0.38,GREY,1,cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────────────
    # Public utilities
    # ─────────────────────────────────────────────────────────────────────────
    def log_event(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")

    def reset_tracks(self) -> None:
        self._track_history.clear()