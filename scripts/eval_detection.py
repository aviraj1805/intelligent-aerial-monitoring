"""Measure detection accuracy (mAP) with Ultralytics validation on a held-out set.

    python scripts/prepare_visiodect.py --zip <VisioDECT.zip>
    python scripts/eval_detection.py --data datasets/visiodect.yaml

Writes results/detection_<name>.json with mAP@0.5, mAP@0.5:0.95, precision,
recall, image/instance counts and the hardware/software used.

--shift-half-box moves every predicted box right and down by half its width
and height before scoring. It exists only to document a bug in the first
model (v1), whose training labels put the box centre on the drone's top-left
corner. See docs/MODEL_CARD.md.
"""

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from iamars import config
from iamars.sysinfo import system_info
from iamars.weights import get_weights


def make_validator(shift_half_box: bool):
    from ultralytics.models.yolo.detect import DetectionValidator

    class Validator(DetectionValidator):
        def postprocess(self, preds):
            out = super().postprocess(preds)
            if shift_half_box:
                for p in out:
                    b = p["bboxes"]
                    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
                    b[:, [0, 2]] += (w / 2)[:, None]
                    b[:, [1, 3]] += (h / 2)[:, None]
            return out

    return Validator


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset yaml with a 'test' entry")
    ap.add_argument("--model", default=config.DEFAULT_MODEL, choices=list(config.MODELS))
    ap.add_argument("--weights", default=None, help="explicit .pt file (overrides --model)")
    ap.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--shift-half-box", action="store_true")
    ap.add_argument("--name", default=None, help="results file suffix")
    args = ap.parse_args()

    device = args.device or config.auto_device()
    weights = args.weights or str(get_weights(args.model))
    Validator = make_validator(args.shift_half_box)
    # Ultralytics defaults for mAP: conf=0.001, NMS IoU=0.7, all confidence levels swept.
    v = Validator(args=dict(
        model=weights, data=args.data, split="test", imgsz=args.imgsz, batch=args.batch,
        device=device, half=False, plots=False, verbose=False, workers=0,
        project=str(config.OUTPUTS_DIR / "val"), name="run", exist_ok=True,
    ))
    t0 = time.time()
    stats = v()
    elapsed = time.time() - t0

    name = args.name or Path(args.data).stem + ("_shifted" if args.shift_half_box else "")
    res = {
        "dataset": Path(args.data).stem,
        "split": "test",
        "weights": Path(weights).name,
        "shift_half_box": args.shift_half_box,
        "imgsz": args.imgsz,
        "images": int(v.seen),
        "instances": int(v.metrics.nt_per_class.sum()),
        "mAP50": round(stats["metrics/mAP50(B)"], 4),
        "mAP50_95": round(stats["metrics/mAP50-95(B)"], 4),
        "precision": round(stats["metrics/precision(B)"], 4),
        "recall": round(stats["metrics/recall(B)"], 4),
        "speed_ms_per_image": {k: round(val, 2) for k, val in v.speed.items()},
        "eval_seconds": round(elapsed, 1),
        "system": system_info(device),
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)
    out = config.RESULTS_DIR / f"detection_{name}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    keys = ("dataset", "weights", "shift_half_box", "images", "instances", "mAP50", "mAP50_95", "precision", "recall")
    print(json.dumps({k: res[k] for k in keys}, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
