import numpy as np
import supervision as sv


class DroneTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack(
            minimum_matching_threshold=0.3,
            lost_track_buffer=60,
            minimum_consecutive_frames=1,
        )

    def update(self, boxes, scores, frame_shape):
        if len(boxes) == 0:
            return []

        detections = sv.Detections(
            xyxy=boxes.astype(float),
            confidence=scores.astype(float),
            class_id=np.zeros(len(boxes), dtype=int)
        )

        tracks = self.tracker.update_with_detections(detections)

        results = []
        for i in range(len(tracks)):
            results.append({
                "track_id": int(tracks.tracker_id[i]),
                "box": tracks.xyxy[i],
                "score": float(tracks.confidence[i])
            })

        return results