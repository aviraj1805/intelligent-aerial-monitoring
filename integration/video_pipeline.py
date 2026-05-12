"""
IAMARS — VideoPipeline
Main integration: wires all 7 pipeline stages end-to-end.

GHOST TRACK FIX:
  When all models miss the drone for a few frames, we inject a
  "ghost detection" at the Kalman-predicted position so ByteTrack
  never loses the track ID. Ghost boxes use a low confidence (0.3)
  so they don't interfere with real detections.

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

# Ghost track: inject predicted position when no detection for N frames
GHOST_CONF       = 0.30   # confidence assigned to ghost boxes
GHOST_BOX_SIZE   = 60     # pixel half-size of ghost box


def build_pipeline(frame_wh: tuple):
    print("[INIT] Loading pipeline components...")
    detector = MultiModelDetector(device="cuda")
    fe       = FusionEngine(iou_thr=0.35, skip_box_thr=0.01)
    tracker  = ByteTracker(
        minimum_matching_threshold = 0.2,
        lost_track_buffer          = 90,
        minimum_consecutive_frames = 1,
        frame_rate                 = 20,
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


def _inject_ghost_detections(fused: dict, kf_mgr: KalmanFilterManager) -> dict:
    """
    If fused has no boxes but Kalman filters exist for active tracks,
    inject ghost boxes at the predicted positions so ByteTrack keeps IDs.
    """
    if len(fused["boxes"]) > 0:
        return fused   # real detections exist — no ghost needed

    if not kf_mgr._filters:
        return fused   # no active tracks to ghost

    ghost_boxes  = []
    ghost_scores = []
    ghost_labels = []

    for tid, kf in kf_mgr._filters.items():
        # Predict one step forward
        state = kf.x   # [cx, cy, w, h, vx, vy]
        cx = state[0] + state[4]   # cx + vx
        cy = state[1] + state[5]   # cy + vy
        w  = max(state[2], 20.0)
        h  = max(state[3], 20.0)

        x1 = cx - w / 2;  y1 = cy - h / 2
        x2 = cx + w / 2;  y2 = cy + h / 2

        ghost_boxes.append([x1, y1, x2, y2])
        ghost_scores.append(GHOST_CONF)
        ghost_labels.append(0)

    if not ghost_boxes:
        return fused

    return {
        "boxes":  np.array(ghost_boxes,  dtype=np.float32),
        "scores": np.array(ghost_scores, dtype=np.float32),
        "labels": np.array(ghost_labels, dtype=np.int32),
    }


def process_frame(
    frame, frame_index,
    detector, fe, tracker, kf_mgr, tp, fs, dash,
    frame_wh,
):
    # Stage 1: Detection
    model_results = detector.run_parallel(frame)
    audio_conf    = 0.0

    # Stage 2: Fusion
    fused = fe.fuse(model_results, frame.shape)

    # Ghost track injection — keeps ByteTrack alive during missed frames
    fused = _inject_ghost_detections(fused, kf_mgr)

    # Stage 3: ByteTrack
    tracked = tracker.update(fused, frame.shape)

    # Stage 4: Kalman
    smoothed = kf_mgr.update(tracked)

    # Stage 5: Trajectory
    predicted = tp.predict(smoothed)

    # Stage 6: Fire solution
    solutions = fs.compute(predicted, frame_index=frame_index)

    # Stage 7: Dashboard
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

    return canvas, n_det, n_fus, n_trk


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

    frame_index = 0
    fps_smooth  = 0.0
    t_start     = time.perf_counter()

    print("[RUN ] Pipeline started — press Q to quit, S to screenshot\n")
    print(f"  {'Frame':>6}  {'FPS':>6}  {'Det':>4}  {'Fused':>5}  {'Tracks':>6}")
    print("  " + "-" * 40)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        canvas, n_det, n_fus, n_trk = process_frame(
            frame, frame_index,
            detector, fe, tracker, kf_mgr, tp, fs, dash,
            frame_wh,
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