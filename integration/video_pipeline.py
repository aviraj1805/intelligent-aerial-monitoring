"""
IAMARS — Intelligent Aerial Monitoring & Automated Response System
Step 8: Final Integration Video Pipeline
"""

import sys
import os
import time
import traceback
import cv2
import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Module imports ───────────────────────────────────────────────────────────
from detection.multi_model_detector import MultiModelDetector
from detection.fusion_engine import FusionEngine
from tracking.bytetracker import ByteTracker
from estimation.kalman_filter import KalmanFilterManager as KalmanFilter
from prediction.trajectory_predictor import TrajectoryPredictor
from intercept.fire_solution import FireSolution
from visualization.tactical_dashboard import TacticalDashboard

# ── Config ───────────────────────────────────────────────────────────────────
VIDEO_PATH = os.path.join(ROOT, "data", "sample_frames", "test_drone4.mp4.mp4")

DISPLAY_WINDOW = "IAMARS Tactical Dashboard"
FRAME_SKIP     = 1       # process every N-th frame (1 = all frames)
MAX_FRAMES     = None    # None = run until end of video
SHOW_FPS       = True


# ─────────────────────────────────────────────────────────────────────────────
class IAMARSPipeline:
    """End-to-end video processing pipeline for the IAMARS system."""

    def __init__(self):
        self.detector  = None
        self.fusion    = None
        self.tracker   = None
        self.kalman    = None
        self.predictor = None
        self.fire_sol  = None
        self.dashboard = None
        self.cap       = None
        self._initialized = False

    # ── Initialisation ────────────────────────────────────────────────────────
    def initialize(self) -> bool:
        """Load all subsystems in dependency order. Returns True on success."""
        print("[IAMARS] Initializing pipeline...")

        try:
            print("  [1/7] MultiModelDetector...")
            self.detector = MultiModelDetector()

            print("  [2/7] FusionEngine...")
            self.fusion = FusionEngine()

            print("  [3/7] ByteTracker...")
            self.tracker = ByteTracker()

            print("  [4/7] KalmanFilterManager...")
            self.kalman = KalmanFilter()

            print("  [5/7] TrajectoryPredictor...")
            self.predictor = TrajectoryPredictor()

            print("  [6/7] FireSolution...")
            self.fire_sol = FireSolution()

            print("  [7/7] TacticalDashboard...")
            self.dashboard = TacticalDashboard()

        except Exception as exc:
            print(f"[ERROR] Subsystem initialization failed: {exc}")
            traceback.print_exc()
            return False

        # ── Video capture ──────────────────────────────────────────────────
        print(f"  [CAP] Opening video: {VIDEO_PATH}")
        if not os.path.isfile(VIDEO_PATH):
            print(f"[ERROR] Video file not found: {VIDEO_PATH}")
            return False

        self.cap = cv2.VideoCapture(VIDEO_PATH)
        if not self.cap.isOpened():
            print("[ERROR] cv2.VideoCapture failed to open the video file.")
            return False

        fps   = self.cap.get(cv2.CAP_PROP_FPS)
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w     = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  [CAP] {w}x{h} @ {fps:.1f} fps  |  {total} frames")

        self._initialized = True
        print("[IAMARS] Pipeline initialized successfully.\n")
        return True

    # ── Single-frame processing ───────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Run the full processing chain on one frame. Returns annotated frame."""

        frame_shape = frame.shape  # (H, W, C)

        # 1. Detection — run_parallel(frame) -> list[dict]
        model_results = self.detector.run_parallel(frame)

        # 2. Fusion — fuse(model_results, frame_shape) -> dict
        fused = self.fusion.fuse(model_results, frame_shape)

        # 3. Tracking — update(fused, frame_shape) -> dict
        tracked = self.tracker.update(fused, frame_shape)

        # 4. Kalman — update(tracked) -> smoothed dict
        smoothed = self.kalman.update(tracked)

        # 5. Trajectory prediction — predict(smoothed) -> predicted dict
        predicted = self.predictor.predict(smoothed)

        # 6. Fire solution — compute(predicted, frame_index) -> list[dict]
        solutions = self.fire_sol.compute(predicted, frame_idx)

        # 7. Dashboard — render(frame, model_results, fused, tracked,
        #                       smoothed, predicted, solutions, frame_index)
        output_frame = self.dashboard.render(
            frame         = frame,
            model_results = model_results,
            fused         = fused,
            tracked       = tracked,
            smoothed      = smoothed,
            predicted     = predicted,
            solutions     = solutions,
            frame_index   = frame_idx,
        )

        return output_frame

    # ── Main run loop ─────────────────────────────────────────────────────────
    def run(self):
        """Read frames and drive the processing loop until completion."""
        if not self._initialized:
            print("[ERROR] Pipeline not initialized. Call initialize() first.")
            return

        print("[IAMARS] Starting video processing...  (press 'q' to quit)\n")

        frame_idx  = 0
        proc_count = 0
        t_start    = time.time()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("\n[IAMARS] End of video stream.")
                    break

                frame_idx += 1

                if MAX_FRAMES and frame_idx > MAX_FRAMES:
                    print(f"\n[IAMARS] Reached MAX_FRAMES limit ({MAX_FRAMES}).")
                    break

                if frame_idx % FRAME_SKIP != 0:
                    continue

                try:
                    output_frame = self.process_frame(frame, frame_idx)
                    proc_count  += 1
                except Exception as exc:
                    print(f"[WARN] Frame {frame_idx} processing error: {exc}")
                    traceback.print_exc()
                    output_frame = frame  # fall back to raw frame

                # ── FPS overlay ────────────────────────────────────────────
                if SHOW_FPS and proc_count > 0:
                    elapsed  = time.time() - t_start
                    fps_live = proc_count / elapsed
                    cv2.putText(
                        output_frame,
                        f"FPS: {fps_live:.1f}  Frame: {frame_idx}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(DISPLAY_WINDOW, output_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n[IAMARS] User requested quit.")
                    break

        except KeyboardInterrupt:
            print("\n[IAMARS] KeyboardInterrupt — shutting down.")

        finally:
            self._cleanup(frame_idx, proc_count, time.time() - t_start)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def _cleanup(self, total_frames: int, proc_frames: int, elapsed: float):
        print("\n[IAMARS] Cleaning up resources...")
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        fps_avg = proc_frames / elapsed if elapsed > 0 else 0
        print(f"[IAMARS] Done. Processed {proc_frames}/{total_frames} frames "
              f"in {elapsed:.1f}s  (avg {fps_avg:.1f} fps)")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    pipeline = IAMARSPipeline()
    if not pipeline.initialize():
        sys.exit(1)
    pipeline.run()


if __name__ == "__main__":
    main()