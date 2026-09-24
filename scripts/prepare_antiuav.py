"""Build an out-of-distribution detection test set from Anti-UAV-RGBT (RGB videos).

The detector never saw Anti-UAV data, so this measures generalisation to a
different camera, background and drone set. Every --stride-th frame of each
test-split "visible.mp4" is saved with its box; frames where the drone is
absent become background images (empty label file).

    python scripts/prepare_antiuav.py --zip path/to/Anti-UAV-RGBT.zip
    python scripts/eval_detection.py --data datasets/antiuav_rgb_test.yaml
"""

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2

from iamars.config import DATASETS_DIR
from iamars.video import iter_frames

NL = chr(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--stride", type=int, default=25, help="keep every Nth frame")
    ap.add_argument("--max-videos", type=int, default=None)
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip)
    out = DATASETS_DIR / f"antiuav_rgb_{args.split}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    jsons = sorted(n for n in z.namelist() if n.startswith(f"{args.split}/") and n.endswith("visible.json"))
    jsons = jsons[: args.max_videos]
    n_img = n_box = n_bg = 0
    for j in jsons:
        ann = json.loads(z.read(j))
        vid = j.split("/")[1]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.mp4"
            path.write_bytes(z.read(j.replace("visible.json", "visible.mp4")))
            for f, frame in enumerate(iter_frames(path)):
                if f % args.stride or f >= len(ann["exist"]):
                    continue
                h, w = frame.shape[:2]
                name = f"{vid}_{f:05d}"
                cv2.imwrite(str(out / "images" / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                r = ann["gt_rect"][f]
                line = ""
                if ann["exist"][f] and len(r) == 4 and r[2] > 0 and r[3] > 0:
                    x, y, bw, bh = r
                    line = f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
                    n_box += 1
                else:
                    n_bg += 1
                (out / "labels" / f"{name}.txt").write_text(line, encoding="utf-8")
                n_img += 1
    yaml = DATASETS_DIR / f"antiuav_rgb_{args.split}.yaml"
    yaml.write_text(NL.join([f"path: {out.resolve().as_posix()}", "train: images", "val: images",
                             "test: images", "names:", "  0: drone", ""]), encoding="utf-8")
    print(f"videos: {len(jsons)}  images: {n_img}  boxes: {n_box}  background images: {n_bg}")
    print(f"dataset yaml: {yaml}")


if __name__ == "__main__":
    main()
