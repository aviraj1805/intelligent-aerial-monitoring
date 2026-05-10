import cv2
import numpy as np


class Dashboard:
    def __init__(self, window_name="IAMARS - Live Detection"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 900, 600)

    def draw(self, frame, tracks, kalman_results, predictions, solutions):
        display = frame.copy()

        # Draw tracked bounding boxes
        for track in tracks:
            box = track["box"].astype(int)
            tid = track["track_id"]
            cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(display, f"Target #{tid}", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw smoothed center
        for r in kalman_results:
            cx, cy = int(r["smoothed_center"][0]), int(r["smoothed_center"][1])
            cv2.circle(display, (cx, cy), 5, (255, 255, 0), -1)

        # Draw predicted trajectory
        for p in predictions:
            points = p["future_points"]
            for i in range(len(points) - 1):
                pt1 = (int(points[i][0]), int(points[i][1]))
                pt2 = (int(points[i+1][0]), int(points[i+1][1]))
                cv2.line(display, pt1, pt2, (0, 165, 255), 1)
            # Draw future dots
            for pt in points:
                cv2.circle(display, (int(pt[0]), int(pt[1])), 3, (0, 165, 255), -1)

        # Draw intercept crosshair
        for s in solutions:
            if s["intercept_point"] is not None:
                ix, iy = int(s["intercept_point"][0]), int(s["intercept_point"][1])
                cv2.drawMarker(display, (ix, iy), (0, 0, 255),
                               cv2.MARKER_CROSS, 20, 2)
                cv2.putText(display, "INTERCEPT", (ix + 10, iy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw data panel
        self._draw_panel(display, solutions, kalman_results)

        return display

    def _draw_panel(self, frame, solutions, kalman_results):
        # Panel background
        cv2.rectangle(frame, (0, 0), (280, 130), (20, 20, 20), -1)

        lines = ["IAMARS | LIVE"]
        if solutions and solutions[0]["azimuth_deg"] is not None:
            s = solutions[0]
            lines += [
                f"Azimuth   : {s['azimuth_deg']}",
                f"Elevation : {s['elevation_deg']}",
                f"T-Intercept: {s['time_to_intercept_sec']}s",
            ]
        if kalman_results:
            vx, vy = kalman_results[0]["velocity"]
            lines.append(f"Velocity  : ({vx:.1f}, {vy:.1f}) px/f")

        for i, line in enumerate(lines):
            color = (0, 255, 0) if i == 0 else (200, 200, 200)
            cv2.putText(frame, line, (10, 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def show(self, frame):
        cv2.imshow(self.window_name, frame)

    def wait(self, ms=30):
        return cv2.waitKey(ms) & 0xFF

    def close(self):
        cv2.destroyAllWindows()