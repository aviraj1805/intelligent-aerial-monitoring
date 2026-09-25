"""Central configuration: paths, model registry and default parameters.

Everything that used to be a hardcoded ``D:/...`` or ``F:/...`` path lives
here, relative to the repository root, so the code runs on any machine.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "weights"
DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
DATASETS_DIR = ROOT / "datasets"
RESULTS_DIR = ROOT / "results"
OUTPUTS_DIR = ROOT / "outputs"

# ── Model weights (hosted as GitHub Release assets, not in git) ────────────
GITHUB_REPO = "aviraj1805/intelligent-aerial-monitoring"
WEIGHTS_RELEASE_TAG = "weights-v1"

MODELS = {
    # Default detector (v2): YOLOv8n fine-tuned on VisioDECT with correct labels
    # and a leakage-aware block split (scripts/train_detector.py).
    "visiodect": {
        "file": "visiodect_yolov8n_v2.pt",
        "sha256": "c02c038f759afefa173f53ef131829e60d66a53e98c83d51f7ed559398cd3867",
        "arch": "YOLOv8n",
        "trained_on": "VisioDECT block split (RGB, 6 drone types, 3 weather conditions)",
    },
    # v1, kept only to reproduce the label-offset finding: its boxes are centred
    # on each drone's top-left corner (see docs/MODEL_CARD.md). Do not use.
    "visiodect_v1": {
        "file": "visiodect_yolov8n.pt",
        "sha256": "af46fe7f8e5b4d8e51a0103f78465f9dc020ed15e0c27c6e268b28a800847b65",
        "arch": "YOLOv8n",
        "trained_on": "VisioDECT random split, with shifted labels (bug)",
    },
    # Optional extra detectors contributed by teammates, used only with --fusion.
    "uav_rgb": {
        "file": "uav_rgb_yolov8m.pt",
        "sha256": "1b32ca95a98a857094f59981f7856f6e8e1896e96c71df9224a46f341fcf63f0",
        "arch": "YOLOv8m",
        "trained_on": "DUT-Anti-UAV (RGB), trained by a teammate",
    },
    "uav_ir": {
        "file": "uav_ir_yolov8m.pt",
        "sha256": "5f632057d5cc75d4be3dca764e3888bd3104554da8145840186a1ef95821cf44",
        "arch": "YOLOv8m",
        "trained_on": "Infrared drone dataset, trained by a teammate",
    },
}
DEFAULT_MODEL = "visiodect"

# Relative weight of each model's scores in weighted box fusion (--fusion only).
FUSION_WEIGHTS = {"visiodect": 2.0, "uav_rgb": 1.0, "uav_ir": 1.0}

# ── Sample video (downloaded, not committed) ───────────────────────────────
SAMPLE_VIDEO = SAMPLES_DIR / "magiclab_24_drones.mp4"
SAMPLE_SOURCE = {
    "title": "MagicLab - 24 Drone Flight.webm",
    "url": "https://upload.wikimedia.org/wikipedia/commons/0/0a/MagicLab_-_24_Drone_Flight.webm",
    "page": "https://commons.wikimedia.org/wiki/File:MagicLab_-_24_Drone_Flight.webm",
    "author": "Marco Tempest",
    "license": "CC BY 3.0",
    "license_url": "https://creativecommons.org/licenses/by/3.0/",
    "start_s": 190.5,  # one continuous shot: 15-25 drones fly in formation, then land
    "end_s": 226.0,
}

# ── Default pipeline parameters ─────────────────────────────────────────────
DETECT_CONF = 0.10  # low on purpose: ByteTrack uses 0.1-0.5 boxes in its 2nd pass
DETECT_IOU = 0.45   # NMS overlap threshold
IMG_SIZE = 640
TRACK_HIGH_THRESH = 0.5   # ByteTrack "high confidence" split (tuned on Anti-UAV val)
TRACK_BUFFER = 60         # frames a lost track is kept before deletion (tuned)
TRACK_MATCH_THRESH = 0.95 # supervision uses 1 - IoU: 0.95 means IoU >= 0.05 (tuned)
PREDICT_HORIZON = 15      # frames of trajectory prediction


def auto_device() -> str:
    """Return ``"0"`` (first CUDA GPU) if available, else ``"cpu"``."""
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"
