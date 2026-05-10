"""
IAMARS — MultiModelDetector
Runs 3 detection models in parallel threads for WBF fusion.
Audio model is handled separately as a confidence flag only (no boxes).
"""

import threading
import numpy as np
from ultralytics import YOLO


# ── Model registry (only detection models go into WBF) ───────────────────────
DETECTION_MODELS = [
    {
        "name":   "visiodect",
        "path":   "models/visiodect/best.pt",
        "weight": 0.5,
        "conf":   0.10,
    },
    {
        "name":   "uav_ir",
        "path":   "models/uav_IR_detection/best.pt",
        "weight": 0.3,
        "conf":   0.10,
    },
    {
        "name":   "uav_rgb",
        "path":   "models/uav_RGB_detection/best.pt",
        "weight": 0.2,
        "conf":   0.10,
    },
]

AUDIO_MODEL_PATH = "models/audio_detection/best.pt"


class MultiModelDetector:
    """
    Loads 3 YOLO detection models and 1 YOLO audio classifier.
    call run_parallel(frame)  → list of per-model dicts (boxes, scores, labels, weight)
    call run_audio(spectrogram) → float confidence in [0, 1]
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.models: list[dict] = []
        self._audio_model = None

        # ── Load detection models ─────────────────────────────────────────────
        for cfg in DETECTION_MODELS:
            print(f"[INFO] Loading detection model : {cfg['name']}  ← {cfg['path']}")
            try:
                yolo = YOLO(cfg["path"])
                self.models.append(
                    {
                        "name":   cfg["name"],
                        "model":  yolo,
                        "weight": cfg["weight"],
                        "conf":   cfg["conf"],
                    }
                )
                print(f"[INFO] {cfg['name']} loaded OK")
            except Exception as exc:
                print(f"[WARN] Failed to load {cfg['name']}: {exc}")

        # ── Load audio classifier (no boxes, flag only) ───────────────────────
        print(f"[INFO] Loading audio classifier  ← {AUDIO_MODEL_PATH}")
        try:
            self._audio_model = YOLO(AUDIO_MODEL_PATH)
            print("[INFO] audio_detection loaded OK")
        except Exception as exc:
            print(f"[WARN] Failed to load audio model: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: single-model inference (called from thread)
    # ─────────────────────────────────────────────────────────────────────────
    def _infer_single(
        self,
        model_cfg: dict,
        frame: np.ndarray,
        results_store: list,
        idx: int,
    ) -> None:
        """Run one YOLO model on `frame`, store result at results_store[idx]."""
        boxes  = np.empty((0, 4), dtype=np.float32)
        scores = np.empty((0,),   dtype=np.float32)
        labels = np.empty((0,),   dtype=np.int32)

        try:
            result = model_cfg["model"].predict(
                frame,
                conf=model_cfg["conf"],
                device=self.device,
                verbose=False,
            )[0]

            # Boxes object/attributes may differ between ultralytics versions;
            # access defensively and fail each conversion separately so one
            # bad field doesn't crash the whole thread.
            if getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
                try:
                    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
                except Exception:
                    boxes = np.empty((0, 4), dtype=np.float32)

                try:
                    scores = result.boxes.conf.cpu().numpy().astype(np.float32)
                except Exception:
                    scores = np.empty((0,), dtype=np.float32)

                try:
                    labels = result.boxes.cls.cpu().numpy().astype(np.int32)
                except Exception:
                    labels = np.empty((0,), dtype=np.int32)

        except Exception as exc:
            # One model failing must NOT propagate; keep the slot as empty arrays
            print(f"[WARN] {model_cfg['name']} inference error: {exc}")

        results_store[idx] = {
            "name":   model_cfg["name"],
            "boxes":  boxes,   # shape (N, 4)  xyxy
            "scores": scores,  # shape (N,)
            "labels": labels,  # shape (N,)
            "weight": model_cfg["weight"],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Public: parallel detection across all 3 models
    # ─────────────────────────────────────────────────────────────────────────
    def run_parallel(self, frame: np.ndarray) -> list[dict]:
        """
        Runs all detection models in parallel threads.

        Returns
        -------
        list of dicts, one per model:
            {name, boxes (N,4), scores (N,), labels (N,), weight}
        """
        n = len(self.models)
        results: list = [None] * n
        threads: list[threading.Thread] = []

        for i, m in enumerate(self.models):
            t = threading.Thread(
                target=self._infer_single,
                args=(m, frame, results, i),
                daemon=True,
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Ensure no slot is None (safety net if thread died silently)
        for i, r in enumerate(results):
            if r is None:
                results[i] = {
                    "name":   self.models[i]["name"],
                    "boxes":  np.empty((0, 4), dtype=np.float32),
                    "scores": np.empty((0,),   dtype=np.float32),
                    "labels": np.empty((0,),   dtype=np.int32),
                    "weight": self.models[i]["weight"],
                }

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Public: audio classifier flag (returns scalar confidence, no boxes)
    # ─────────────────────────────────────────────────────────────────────────
    def run_audio(self, spectrogram: np.ndarray) -> float:
        """
        Classifies an audio spectrogram frame.

        Returns
        -------
        float — confidence that drone audio is present, or 0.0 on failure.
        """
        if self._audio_model is None:
            return 0.0
        try:
            result = self._audio_model.predict(
                spectrogram,
                device=self.device,
                verbose=False,
            )[0]

            probs = getattr(result, "probs", None)
            if probs is not None:
                # Prefer `top1conf` if provided, otherwise take max probability
                top1 = getattr(probs, "top1conf", None)
                if top1 is not None:
                    try:
                        return float(top1.cpu().numpy())
                    except Exception:
                        pass
                try:
                    arr = probs.cpu().numpy()
                    return float(arr.max())
                except Exception:
                    pass
        except Exception as exc:
            print(f"[WARN] Audio inference error: {exc}")

        return 0.0