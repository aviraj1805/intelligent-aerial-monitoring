import sys
import numpy as np
sys.path.append(".")

from detection.multi_model_detector import MultiModelDetector
from detection.fusion_engine import FusionEngine
from tracking.bytetracker import DroneTracker
from estimation.kalman_filter import DroneKalmanFilter
from prediction.trajectory_predictor import TrajectoryPredictor
from intercept.fire_solution import FireSolution
from visualization.dashboard import Dashboard


def run_demo():
    # Initialize all modules
    detector = MultiModelDetector(config_path="D:\\intelligent-aerial-monitoring\\models\\model_configs.yaml")
    fusion = FusionEngine(config_path="D:\\intelligent-aerial-monitoring\\models\\model_configs.yaml")
    tracker = DroneTracker()
    kalman = DroneKalmanFilter()
    predictor = TrajectoryPredictor(predict_frames=10)
    fire = FireSolution(projectile_speed=50)
    dashboard = Dashboard()

    frame_shape = (640, 640, 3)
    total_frames = 300

    print("Starting IAMARS pipeline... Press Q to quit\n")

    for i in range(total_frames):
        # Simulate drone moving across frame
        x = 50 + i * 2
        y = 50 + i * 1.5
        x = min(x, 580)
        y = min(y, 580)

        # Create black frame
        frame = np.zeros((640, 640, 3), dtype=np.uint8)

        # Draw simulated drone (white rectangle)
        cv_x, cv_y = int(x), int(y)
        import cv2
        cv2.rectangle(frame, (cv_x, cv_y), (cv_x+40, cv_y+40), (255, 255, 255), -1)

        # Inject fake detections
        fake_results = [{
            "name": "visiodect",
            "boxes": np.array([[x, y, x+40, y+40]], dtype=float),
            "scores": np.array([0.92]),
            "weight": 0.6
        }]

        # Pipeline
        boxes, scores = fusion.fuse(fake_results, frame_shape)
        tracks = tracker.update(boxes, scores, frame_shape)
        kalman_results = kalman.update(tracks)
        predictions = predictor.predict(kalman_results)
        solutions = fire.compute(predictions)

        # Visualize
        display = dashboard.draw(frame, tracks, kalman_results, predictions, solutions)
        dashboard.show(display)

        key = dashboard.wait(30)
        if key == ord('q'):
            break

    dashboard.close()
    print("Demo complete.")


if __name__ == "__main__":
    run_demo()