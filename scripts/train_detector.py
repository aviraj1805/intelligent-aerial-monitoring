"""Fine-tune YOLOv8n (COCO-pretrained) on VisioDECT.

    python scripts/prepare_visiodect.py --zip <VisioDECT.zip>
    python scripts/train_detector.py                      # GPU if available
    python scripts/train_detector.py --epochs 1 --fraction 0.02   # quick smoke test
    python scripts/train_detector.py --resume                     # continue after interruption

Hyperparameters match the original v1 run (models card / git history):
SGD, lr0 0.01, momentum 0.937, weight decay 5e-4, batch 8, 640 px, mosaic.
The best checkpoint (by validation mAP@0.5:0.95) is copied to
weights/visiodect_yolov8n_v2.pt.

Transfer learning in one sentence: the network starts from weights learned on
the COCO dataset (80 everyday classes), so its early layers already detect
edges and shapes; fine-tuning only has to adapt it to one new class, "drone".
"""

import argparse
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401

from iamars import config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(config.DATASETS_DIR / "visiodect.yaml"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=1.0, help="use part of the train set")
    ap.add_argument("--name", default="visiodect_v2")
    ap.add_argument("--out", default=str(config.WEIGHTS_DIR / "visiodect_yolov8n_v2.pt"))
    ap.add_argument("--resume", action="store_true", help="continue from runs/<name>/weights/last.pt")
    args = ap.parse_args()

    from ultralytics import YOLO

    from iamars.keepawake import keep_awake

    with keep_awake():  # an idle-sleeping laptop killed the first run at epoch 19
        if args.resume:
            model = YOLO(str(config.ROOT / "runs" / args.name / "weights" / "last.pt"))
            model.train(resume=True)
        else:
            model = _train_new(YOLO, args)
    best = Path(model.trainer.best)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.out)
    print(f"best checkpoint -> {args.out}")


def _train_new(YOLO, args):
    model = YOLO("yolov8n.pt")  # COCO-pretrained, downloaded by Ultralytics
    model.train(
        data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        device=args.device or config.auto_device(), workers=args.workers, fraction=args.fraction,
        optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5,
        fliplr=0.5, mosaic=1.0, close_mosaic=10, seed=42, deterministic=True,
        project=str(config.ROOT / "runs"), name=args.name, exist_ok=True, plots=True,
    )
    return model


if __name__ == "__main__":
    main()
