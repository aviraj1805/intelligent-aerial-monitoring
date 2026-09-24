"""Build the VisioDECT held-out test set in YOLO format from the public archive.

VisioDECT: 20,924 RGB images of 6 drone models in 3 weather conditions.
Source: IEEE DataPort, "VisioDECT Dataset: An Aerial Dataset for Scenario-Based
Multi-Drone Detection and Identification". Download the archive there, then:

    python scripts/prepare_visiodect.py --zip "path/to/VisioDECT.zip"            # train/val/test
    python scripts/prepare_visiodect.py --zip "path/to/VisioDECT.zip" --legacy   # old test split only

Default mode builds datasets/visiodect/{train,val,test} with a *block split*:
inside each of the 18 folders (6 drone models x 3 weather conditions) frames are
sorted by frame number and cut into blocks of 50 consecutive frames. Whole blocks
are assigned to train/val/test (70/20/10, seed 0). Consecutive video frames are
near-duplicates, so a random per-frame split would put near-copies of test
images in the training set and inflate the test score.

--legacy extracts only the 2,062 images in data/splits/visiodect_test.txt (the
random split used by the first model) to datasets/visiodect_test/. Labels are read from the VOC XML files, or from the
CSV file where a folder has no XML (Anafi-Extended/cloudy). All drone models are
mapped to the single class 0 = drone, as in training.
"""

import argparse
import csv
import io
import re
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from iamars.config import DATA_DIR, DATASETS_DIR

ROOT_IN_ZIP = "VisioDECT Dataset Upload/"


def voc_boxes(xml: str):
    out = []
    for obj in re.findall(r"<bndbox>(.*?)</bndbox>", xml, re.S):
        v = {k: float(re.search(rf"<{k}>([\d.]+)</{k}>", obj).group(1))
             for k in ("xmin", "ymin", "xmax", "ymax")}
        out.append((v["xmin"], v["ymin"], v["xmax"], v["ymax"]))
    return out


NL = chr(10)  # newline


def dataset_yaml(root: Path, train: str, val: str, test: str) -> str:
    lines = [f"path: {root.resolve().as_posix()}", f"train: {train}", f"val: {val}",
             f"test: {test}", "names:", "  0: drone", ""]
    return NL.join(lines)


FRAME_NO = re.compile(r"\((\d+)\)\.jpg$")


class LabelReader:
    """Reads VOC XML labels, falling back to the per-folder CSV file."""

    def __init__(self, z: zipfile.ZipFile):
        self.z = z
        self.names = set(z.namelist())
        self.csv_cache: dict[str, dict] = {}

    def _csv(self, folder: str) -> dict:
        # CSV rows: label,xmin,ymin,w,h,filename,img_w,img_h
        if folder not in self.csv_cache:
            rows = {}
            member = f"{ROOT_IN_ZIP}{folder}/csv.csv"
            if member in self.names:
                for r in csv.reader(io.StringIO(self.z.read(member).decode("utf-8", "replace"))):
                    if len(r) >= 6:
                        x, y, w, h = map(float, r[1:5])
                        rows.setdefault(r[5], []).append((x, y, x + w, y + h))
            self.csv_cache[folder] = rows
        return self.csv_cache[folder]

    def boxes(self, rel: str):
        """Corner boxes (x1, y1, x2, y2) in pixels and the label source."""
        model, _, weather, fname = rel.split("/")
        xml = f"{ROOT_IN_ZIP}{model}/labels/{weather.lower()}/voc/{fname[:-4]}.xml"
        if xml in self.names:
            return voc_boxes(self.z.read(xml).decode("utf-8", "replace")), "voc"
        return self._csv(f"{model}/labels/{weather.lower()}").get(fname, []), "csv"


def write_items(z, reader, items, out: Path) -> dict:
    """Extract images and write YOLO labels. Returns counts."""
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "boxes": 0, "voc": 0, "csv": 0}
    for rel in items:
        model, _, weather, fname = rel.split("/")
        data = z.read(ROOT_IN_ZIP + rel)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        boxes, src = reader.boxes(rel)
        stats[src] += 1
        name = f"{model}_{weather}_{fname[:-4]}"
        (out / "images" / f"{name}.jpg").write_bytes(data)
        lines = []
        for x1, y1, x2, y2 in boxes:
            x1, x2 = np.clip([x1, x2], 0, w)
            y1, y2 = np.clip([y1, y2], 0, h)
            # YOLO format: class x_centre y_centre width height, all divided by image size.
            lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        (out / "labels" / f"{name}.txt").write_text(NL.join(lines), encoding="utf-8")
        stats["images"] += 1
        stats["boxes"] += len(lines)
    return stats


def block_split(items, block=50, ratios=(0.7, 0.2, 0.1), seed=0):
    rng = np.random.default_rng(seed)
    by_folder: dict[str, list] = {}
    for rel in items:
        by_folder.setdefault(rel.rsplit("/", 1)[0], []).append(rel)
    splits = {"train": [], "val": [], "test": []}
    for folder in sorted(by_folder):
        frames = sorted(by_folder[folder], key=lambda r: int(FRAME_NO.search(r).group(1)))
        blocks = [frames[i:i + block] for i in range(0, len(frames), block)]
        order = rng.permutation(len(blocks))
        n_tr = int(round(ratios[0] * len(blocks)))
        n_va = int(round(ratios[1] * len(blocks)))
        for k, bi in enumerate(order):
            name = "train" if k < n_tr else "val" if k < n_tr + n_va else "test"
            splits[name].extend(blocks[bi])
    return splits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, help="VisioDECT archive (.zip)")
    ap.add_argument("--legacy", action="store_true", help="only the old random test split")
    ap.add_argument("--block", type=int, default=50, help="frames per block in the block split")
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip)
    reader = LabelReader(z)

    if args.legacy:
        split_file = DATA_DIR / "splits" / "visiodect_test.txt"
        items = [l.strip() for l in open(split_file, encoding="utf-8") if l.strip() and not l.startswith("#")]
        out = DATASETS_DIR / "visiodect_test"
        print(write_items(z, reader, items, out))
        yaml = DATASETS_DIR / "visiodect_test.yaml"
        yaml.write_text(dataset_yaml(out, "images", "images", "images"), encoding="utf-8")
        print(f"dataset yaml: {yaml}")
        return

    all_imgs = sorted(n[len(ROOT_IN_ZIP):] for n in reader.names
                      if n.startswith(ROOT_IN_ZIP) and n.endswith(".jpg") and "/images/" in n)
    labelled = [r for r in all_imgs if reader.boxes(r)[0]]
    print(f"images in archive: {len(all_imgs)}  with a label: {len(labelled)}  (unlabelled images are skipped)")
    splits = block_split(labelled, block=args.block)
    out = DATASETS_DIR / "visiodect"
    for name, items in splits.items():
        (DATA_DIR / "splits" / f"visiodect_block_{name}.txt").write_text(NL.join(items) + NL, encoding="utf-8")
        print(name, write_items(z, reader, items, out / name))
    yaml = DATASETS_DIR / "visiodect.yaml"
    yaml.write_text(dataset_yaml(out, "train/images", "val/images", "test/images"), encoding="utf-8")
    print(f"dataset yaml: {yaml}")


if __name__ == "__main__":
    main()
