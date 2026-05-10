import threading
import yaml
import numpy as np
from ultralytics import YOLO


class MultiModelDetector:
    def __init__(self, config_path="model_configs.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.models = []
        for m in config["models"]:
            print(f"[INFO] Loading model: {m['name']} from {m['path']}")
            self.models.append({
                "name": m["name"],
                "model": YOLO(m["path"]),
                "weight": m["weight"],
                "conf": m["conf_threshold"],
            })
            print(f"[INFO] {m['name']} loaded successfully")

        self.device = config["inference"]["device"]

    def _infer_single(self, model_cfg, frame, results_store, idx):
        result = model_cfg["model"].predict(
            frame,
            conf=model_cfg["conf"],
            device=self.device,
            verbose=False
        )[0]

        boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) > 0 else np.array([])
        scores = result.boxes.conf.cpu().numpy() if len(result.boxes) > 0 else np.array([])

        results_store[idx] = {
            "name": model_cfg["name"],
            "boxes": boxes,
            "scores": scores,
            "weight": model_cfg["weight"]
        }

    def run_parallel(self, frame):
        results = [None] * len(self.models)
        threads = []

        for i, m in enumerate(self.models):
            t = threading.Thread(
                target=self._infer_single,
                args=(m, frame, results, i)
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return results