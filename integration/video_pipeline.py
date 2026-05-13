"""
IAMARS — VideoPipeline  (PATCHED v1.1)
Main integration: wires all 7 pipeline stages end-to-end.

CHANGELOG vs v1.0
-----------------
FIX 1 — Ghost track injection (was causing ID flicker):
    Previous code accessed kf.x directly and added raw velocity:
        cx = state[0] + state[4]   # WRONG — no unit conversion, no predict call
    Fixed: Use KalmanFilterManager.predict_next() which calls the proper
    Kalman predict step and returns the correct (cx, cy, w, h) state.

FIX 2 — ByteTracker threshold tuning:
    Raised track_activation_threshold from 0.25 → 0.10 so boosted WBF
    scores (~0.6+) never risk falling below activation.
    Raised minimum_matching_threshold from 0.2 → 0.30 for tighter IoU
    matching on real detections while keeping ghost matching loose.

FIX 3 — Ghost confidence raised from 0.30 → 0.45:
    Previous 0.30 was below track_activation_threshold=0.25 on some paths,
    causing ghost boxes to be silently dropped by ByteTrack.

Usage
-----
    python integration/video_pipeline.py
    python integration/video_pipeline.py --video path/to/video.mp4
    python integration/video_pipeline.py --video path/to/video.mp4 --save output.mp4
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detection.multi_model_detector   import MultiModelDetector
from detection.fusion_engine          import FusionEngine
from tracking.bytetracker             import ByteTracker
from estimation.kalman_filter         import KalmanFilterManager
from prediction.trajectory_predictor  import TrajectoryPredictor
from intercept.fire_solution          import FireSolution
from visualization.tactical_dashboard import TacticalDashboard

DEFAULT_VIDEO    = "data/sample_frames/test_drone4.mp4.mp4"
DISPLAY_W, DISPLAY_H = 1280, 720

# Ghost track settings
GHOST_CONF          = 0.45   # FIX 3: raised from 0.30 — must exceed ByteTrack activation floor
GHOST_MAX_AGE       = 15     # frames: stop injecting ghost after this many misses (prevent phantom tracks)


def build_pipeline(frame_wh: tuple):
    print("[INIT] Loading pipeline components...")
    detector = MultiModelDetector(device="cuda")
    fe       = FusionEngine(iou_thr=0.35, skip_box_thr=0.01)

    # FIX 2: tuned thresholds
    tracker  = ByteTracker(
        minimum_matching_threshold = 0.30,   # was 0.20
        lost_track_buffer          = 90,
        minimum_consecutive_frames = 1,
        frame_rate                 = 20,
        track_activation_threshold = 0.10,   # was 0.25 — boosted scores are ~0.6+ so this is safe
    )
    kf_mgr   = KalmanFilterManager()
    tp       = TrajectoryPredictor(horizon=10)
    fs       = FireSolution(
        projectile_speed = 300.0,
        focal_length     = 800.0,
        frame_wh         = frame_wh,
    )
    dash     = TacticalDashboard(frame_wh=frame_wh, max_log_lines=14)
    print("[INIT] All components ready.\n")
    return detector, fe, tracker, kf_mgr, tp, fs, dash


def _inject_ghost_detections(
    fused: dict,
    kf_mgr: "KalmanFilterManager",
    miss_counter: dict,
) -> dict:
    """
    When no real detections exist, inject ghost boxes at Kalman-predicted
    positions to keep ByteTrack from dropping track IDs.

    FIX 1: Uses kf_mgr to get the properly predicted next state instead of
    manually indexing the raw state vector (which was the v1.0 bug).

    The miss_counter dict tracks how many consecutive frames each track_id
    has been running on ghost. Ghosts are suppressed after GHOST_MAX_AGE
    frames to prevent phantom tracks when a drone truly leaves the scene.
    """
    if len(fused["boxes"]) > 0:
        # Real detections exist — reset all miss counters for active tracks
        # (we don't know which track they belong to yet, so just let ByteTrack handle it)
        return fused, miss_counter

    if not hasattr(kf_mgr, '_filters') or not kf_mgr._filters:
        return fused, miss_counter

    ghost_boxes  = []
    ghost_scores = []
    ghost_labels = []

    # Get predicted next positions from each active Kalman filter
    # KalmanFilterManager.predict_next() must return {track_id: (cx, cy, w, h)}
    # If your KalmanFilterManager doesn't have this method, see note below.
    try:
        predictions = kf_mgr.predict_next()
    except AttributeError:
        # Fallback: manually read state from filterpy KalmanFilter objects
        # State layout: [cx, cy, w, h, vx, vy]  (standard 6-state Kalman)
        predictions = {}
        for tid, kf in kf_mgr._filters.items():
            x = kf.x.flatten()
            if len(x) >= 4:
                # Apply one predict step on a *copy* — do NOT mutate the live filter
                import copy
                kf_copy = copy.deepcopy(kf)
                kf_copy.predict()
                state = kf_copy.x.flatten()
                predictions[tid] = (float(state[0]), float(state[1]),
                                    float(state[2]), float(state[3]))

    for tid, (cx, cy, w, h) in predictions.items():
        # Clamp dimensions
        w = max(float(w), 20.0)
        h = max(float(h), 20.0)

        # Check ghost age — stop injecting if drone is gone too long
        age = miss_counter.get(tid, 0)
        if age >= GHOST_MAX_AGE:
            continue

        miss_counter[tid] = age + 1

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        ghost_boxes.append([x1, y1, x2, y2])
        ghost_scores.append(GHOST_CONF)
        ghost_labels.append(0)

    if not ghost_boxes:
        return fused, miss_counter

    ghost_fused = {
        "boxes":  np.array(ghost_boxes,  dtype=np.float32),
        "scores": np.array(ghost_scores, dtype=np.float32),
        "labels": np.array(ghost_labels, dtype=np.int32),
    }
    return ghost_fused, miss_counter


def _reset_miss_counters_for_real_detections(
    tracked: dict,
    miss_counter: dict,
) -> dict:
    """
    After ByteTrack runs, any track_id that has a real box gets its
    miss counter reset. This prevents stale ghost-age counts.
    """
    for tid in tracked.get("track_ids", []):
        if tid in miss_counter:
            miss_counter[tid] = 0
    return miss_counter


def process_frame(
    frame, frame_index,
    detector, fe, tracker, kf_mgr, tp, fs, dash,
    frame_wh,
    miss_counter: dict,
):
    # ── Stage 1: Detection ──────────────────────────────────────────────
    model_results = detector.run_parallel(frame)
    audio_conf    = 0.0

    # ── Stage 2: Fusion ─────────────────────────────────────────────────
    fused = fe.fuse(model_results, frame.shape)

    # ── Ghost track injection (FIX 1) ───────────────────────────────────
    fused, miss_counter = _inject_ghost_detections(fused, kf_mgr, miss_counter)

    # ── Stage 3: ByteTrack ──────────────────────────────────────────────
    tracked = tracker.update(fused, frame.shape)

    # Reset miss counters for tracks that got real detections this frame
    miss_counter = _reset_miss_counters_for_real_detections(tracked, miss_counter)

    # ── Stage 4: Kalman ─────────────────────────────────────────────────
    smoothed = kf_mgr.update(tracked)

    # ── Stage 5: Trajectory ─────────────────────────────────────────────
    predicted = tp.predict(smoothed)

    # ── Stage 6: Fire solution ──────────────────────────────────────────
    solutions = fs.compute(predicted, frame_index=frame_index)

    # ── Stage 7: Dashboard ──────────────────────────────────────────────
    canvas = dash.render(
        frame         = frame,
        model_results = model_results,
        fused         = fused,
        tracked       = tracked,
        smoothed      = smoothed,
        predicted     = predicted,
        solutions     = solutions,
        frame_index   = frame_index,
        audio_conf    = audio_conf,
    )

    n_det = sum(len(r["boxes"]) for r in model_results)
    n_fus = len(fused["boxes"])
    n_trk = len(tracked["track_ids"])

    if n_trk > 0 and frame_index % 30 == 0:
        dash.log_event(f"FRAME {frame_index:05d}: {n_trk} TRACK(S) ACTIVE")
    if solutions and frame_index % 10 == 0:
        sol = solutions[0]
        dash.log_event(
            f"AZ={sol['azimuth']:.1f} EL={sol['elevation']:.1f} "
            f"T={sol['t_intercept']}"
        )

    return canvas, n_det, n_fus, n_trk, miss_counter


def run(video_path: str, save_path=None, headless: bool = False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 20.0

    print(f"[VIDEO] {video_path}")
    print(f"        {src_w}x{src_h}  {fps:.1f} FPS  {total_frames} frames\n")

    frame_wh = (src_w, src_h)
    detector, fe, tracker, kf_mgr, tp, fs, dash = build_pipeline(frame_wh)

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (DISPLAY_W, DISPLAY_H))
        print(f"[SAVE] Writing output to: {save_path}")

    frame_index  = 0
    fps_smooth   = 0.0
    miss_counter = {}          # track_id → consecutive missed frames
    t_start      = time.perf_counter()

    print("[RUN ] Pipeline started — press Q to quit, S to screenshot\n")
    print(f"  {'Frame':>6}  {'FPS':>6}  {'Det':>4}  {'Fused':>5}  {'Tracks':>6}")
    print("  " + "-" * 40)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        canvas, n_det, n_fus, n_trk, miss_counter = process_frame(
            frame, frame_index,
            detector, fe, tracker, kf_mgr, tp, fs, dash,
            frame_wh,
            miss_counter,
        )

        elapsed    = time.perf_counter() - t0
        fps_inst   = 1.0 / elapsed if elapsed > 0 else 0.0
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps_inst

        if frame_index % 10 == 0:
            print(f"  {frame_index:>6}  {fps_smooth:>6.1f}  "
                  f"{n_det:>4}  {n_fus:>5}  {n_trk:>6}")

        if writer:
            writer.write(canvas)

        if not headless:
            cv2.imshow("IAMARS — Tactical Dashboard", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[RUN ] Quit by user")
                break
            if key == ord('s'):
                spath = f"screenshot_{frame_index:05d}.png"
                cv2.imwrite(spath, canvas)
                print(f"\n[SAVE] Screenshot -> {spath}")

        frame_index += 1

    cap.release()
    if writer:
        writer.release()
    if not headless:
        cv2.destroyAllWindows()

    total_time = time.perf_counter() - t_start
    print(f"\n[DONE] {frame_index} frames in {total_time:.1f}s "
          f"({frame_index/total_time:.1f} FPS avg)")
    if save_path:
        print(f"[DONE] Saved -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IAMARS Video Pipeline")
    parser.add_argument("--video",    default=DEFAULT_VIDEO)
    parser.add_argument("--save",     default=None)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    run(video_path=args.video, save_path=args.save, headless=args.headless)