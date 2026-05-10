import sys
import cv2
import numpy as np
sys.path.append(".")

from detection.multi_model_detector import MultiModelDetector
from detection.fusion_engine import FusionEngine
from tracking.bytetracker import DroneTracker
from estimation.kalman_filter import DroneKalmanFilter
from prediction.trajectory_predictor import TrajectoryPredictor
from intercept.fire_solution import FireSolution
from visualization.dashboard import Dashboard

VIDEO_PATH = "data//sample_frames//test_drone4.mp4.mp4"

def run_video():
    detector = MultiModelDetector(config_path="D:\\intelligent-aerial-monitoring\\models\\model_configs.yaml")
    fusion = FusionEngine(config_path="D:\\intelligent-aerial-monitoring\\models\\model_configs.yaml")
    tracker = DroneTracker()
    kalman = DroneKalmanFilter()
    predictor = TrajectoryPredictor(predict_frames=10)
    fire = FireSolution(projectile_speed=50)
    dashboard = Dashboard()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Cannot open video file")
        return

    print("Running on real video... Press Q to quit\n")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            break

        frame_resized = cv2.resize(frame, (640, 384))
        frame_shape = frame_resized.shape

        # Real detection from your trained model
        model_results = detector.run_parallel(frame_resized)

        # Fusion
        boxes, scores = fusion.fuse(model_results, frame_shape)

        # Tracking
        tracks = tracker.update(boxes, scores, frame_shape)

        # Kalman
        kalman_results = kalman.update(tracks)

        # Prediction
        predictions = predictor.predict(kalman_results)

        # Intercept
        solutions = fire.compute(predictions)

        # Display
        display = dashboard.draw(frame_resized, tracks, kalman_results, predictions, solutions)

        # Show frame count
        cv2.putText(display, f"Frame: {frame_id}", (640-120, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

        dashboard.show(display)
        frame_id += 1

        key = dashboard.wait(30)
        if key == ord('q'):
            break

    cap.release()
    dashboard.close()
    print(f"Processed {frame_id} frames.")

if __name__ == "__main__":
    run_video()