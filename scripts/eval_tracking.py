"""Evaluate multi-object tracking (MOTA, IDF1, ID switches) against ground truth.

Datasets (download them yourself; they are research-use only and not redistributed):
  * Anti-UAV-RGBT (Anti-UAV300): RGB "visible.mp4" videos with one drone box per frame.
  * MultiUAV (Anti-UAV challenge, multi-UAV track): thermal videos, many drones, MOT format.

    # 1) cache detections (GPU recommended), 2) track + score from the cache
    python scripts/eval_tracking.py --dataset antiuav --zip Anti-UAV-RGBT.zip --split val --tune
    python scripts/eval_tracking.py --dataset antiuav --zip Anti-UAV-RGBT.zip --split test
    python scripts/eval_tracking.py --dataset multiuav --zip MultiUAV_Train.zip --max-videos 20

Metrics (py-motmetrics, a prediction matches ground truth when IoU >= 0.5):
  MOTA  = 1 - (misses + false positives + ID switches) / ground-truth boxes
  IDF1  = F1 score of boxes assigned to the *correct identity* over the whole video
  IDSW  = times a ground-truth drone's assigned track ID changes
--tune tries a small grid of ByteTrack settings and reports the best by IDF1;
use it on the validation split only, then evaluate the chosen setting on test.
"""

import argparse
import itertools
import json
import tempfile
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from iamars import config
from iamars.detector import Detections, YoloDetector
from iamars.estimator import BoxKalmanManager
from iamars.sysinfo import system_info
from iamars.tracker import ByteTracker
from iamars.video import iter_frames, video_info


# ── ground truth readers: return (video_member, {frame: [(id, x1, y1, x2, y2)]}) ──
def antiuav_videos(z: zipfile.ZipFile, split: str):
    for j in sorted(n for n in z.namelist() if n.startswith(f"{split}/") and n.endswith("visible.json")):
        d = json.loads(z.read(j))
        gt = {}
        for f, (e, r) in enumerate(zip(d["exist"], d["gt_rect"])):
            if e and len(r) == 4 and r[2] > 0 and r[3] > 0:
                gt[f] = [(1, r[0], r[1], r[0] + r[2], r[1] + r[3])]
        yield j.replace("visible.json", "visible.mp4"), gt


def multiuav_videos(z: zipfile.ZipFile, split: str):
    for t in sorted(n for n in z.namelist() if n.endswith(".txt") and "/TrainLabels/" in n):
        gt = {}
        for line in z.read(t).decode().splitlines():
            p = line.split(",")
            f, tid, x, y, w, h = int(p[0]) - 1, int(p[1]), *map(float, p[2:6])
            gt.setdefault(f, []).append((tid, x, y, x + w, y + h))
        yield t.replace("/TrainLabels/", "/TrainVideos/").replace(".txt", ".mp4"), gt


READERS = {"antiuav": antiuav_videos, "multiuav": multiuav_videos}


def video_info_from_zip(z, member) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "v.mp4"
        path.write_bytes(z.read(member))
        return video_info(path)["frames"]


def cache_detections(z, member, cache_file: Path, detector) -> dict:
    if cache_file.exists():
        d = np.load(cache_file, allow_pickle=True)
        return {"boxes": list(d["boxes"]), "confs": list(d["confs"]), "fps": float(d["fps"])}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "v.mp4"
        path.write_bytes(z.read(member))
        fps = video_info(path)["fps"]
        boxes, confs = [], []
        for frame in iter_frames(path):
            det = detector.detect(frame)
            boxes.append(det.xyxy)
            confs.append(det.conf)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, boxes=np.array(boxes, dtype=object),
                        confs=np.array(confs, dtype=object), fps=fps)
    return {"boxes": boxes, "confs": confs, "fps": fps}


def iou_distance(gt_xyxy: np.ndarray, hyp_xyxy: np.ndarray, min_iou: float = 0.5) -> np.ndarray:
    """1 - IoU for every (ground truth, hypothesis) pair; NaN = not allowed to match.

    Written here because motmetrics 1.4.0's helper uses a function removed in NumPy 2.
    """
    if len(gt_xyxy) == 0 or len(hyp_xyxy) == 0:
        return np.empty((len(gt_xyxy), len(hyp_xyxy)))
    a, b = gt_xyxy[:, None, :], hyp_xyxy[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    iou = inter / (area(a) + area(b) - inter + 1e-9)
    return np.where(iou >= min_iou, 1.0 - iou, np.nan)


def track_and_score(dets: dict, gt: dict, tracker_kwargs: dict, use_kalman_boxes: bool):
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    tracker = ByteTracker(frame_rate=dets["fps"], **tracker_kwargs)
    kf = BoxKalmanManager(fps=dets["fps"])
    for f, (b, c) in enumerate(zip(dets["boxes"], dets["confs"])):
        tr = tracker.update(Detections(np.asarray(b, np.float32).reshape(-1, 4), np.asarray(c, np.float32)))
        boxes = tr.xyxy
        if use_kalman_boxes and len(tr):
            st = kf.update(tr.ids, tr.xyxy)
            boxes = np.array([[s[0] - s[2] / 2, s[1] - s[3] / 2, s[0] + s[2] / 2, s[1] + s[3] / 2]
                              for s in (st[int(i)] for i in tr.ids)])
        g = gt.get(f, [])
        gt_ids = [x[0] for x in g]
        gt_xyxy = np.array([x[1:] for x in g], float).reshape(-1, 4)
        dist = iou_distance(gt_xyxy, np.asarray(boxes, float).reshape(-1, 4))
        acc.update(gt_ids, tr.ids.tolist(), dist)
    return acc


def summarise(accs, names):
    import motmetrics as mm

    mh = mm.metrics.create()
    metrics = ["num_frames", "mota", "idf1", "num_switches", "precision", "recall",
               "num_false_positives", "num_misses", "mostly_tracked", "num_unique_objects"]
    s = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    overall = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
               for k, v in s.loc["OVERALL"].to_dict().items()}
    return overall, s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(READERS), required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--weights", default=None, help="detector .pt (default: config default model)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tune", action="store_true", help="grid-search ByteTrack settings (val only)")
    ap.add_argument("--high-thresh", type=float, default=config.TRACK_HIGH_THRESH)
    ap.add_argument("--match-thresh", type=float, default=config.TRACK_MATCH_THRESH)
    ap.add_argument("--track-buffer", type=int, default=config.TRACK_BUFFER)
    ap.add_argument("--kalman-boxes", action="store_true", help="score Kalman-smoothed boxes")
    ap.add_argument("--oracle", action="store_true",
                    help="feed ground-truth boxes instead of detections: tracker-only upper bound")
    args = ap.parse_args()

    device = args.device or config.auto_device()
    detector = None if args.oracle else YoloDetector(weights_path=args.weights, device=device)
    weights_name = "ground_truth_boxes" if args.oracle else Path(detector.model.ckpt_path or "model").name
    cache_root = config.OUTPUTS_DIR / "det_cache" / f"{args.dataset}_{args.split}_{Path(weights_name).stem}"

    z = zipfile.ZipFile(args.zip)
    videos = list(READERS[args.dataset](z, args.split))[: args.max_videos]
    print(f"{len(videos)} videos")
    data = []
    for k, (member, gt) in enumerate(videos):
        if args.oracle:
            n = int(video_info_from_zip(z, member)) if not gt else max(gt) + 1
            dets = {"boxes": [np.array([x[1:] for x in gt.get(f, [])], np.float32).reshape(-1, 4) for f in range(n)],
                    "confs": [np.full(len(gt.get(f, [])), 0.9, np.float32) for f in range(n)], "fps": 30.0}
        else:
            dets = cache_detections(z, member, cache_root / (member.replace("/", "_") + ".npz"), detector)
        data.append((member, gt, dets))
        print(f"  [{k + 1}/{len(videos)}] {member}: {len(dets['boxes'])} frames")

    names = [m for m, _, _ in data]

    def run(kw):
        return summarise([track_and_score(d, g, kw, args.kalman_boxes) for _, g, d in data], names)

    base_kw = dict(high_thresh=args.high_thresh, match_thresh=args.match_thresh, track_buffer=args.track_buffer)
    result = {"dataset": args.dataset, "split": args.split, "videos": len(data),
              "detector_weights": weights_name, "iou_match_threshold": 0.5,
              "scored_boxes": "kalman" if args.kalman_boxes else "bytetrack"}
    if args.tune:
        grid = []
        for hi, mt, buf in itertools.product((0.25, 0.4, 0.5), (0.8, 0.9, 0.95), (30, 60)):
            kw = dict(high_thresh=hi, match_thresh=mt, track_buffer=buf)
            o, _ = run(kw)
            grid.append({**kw, "idf1": round(o["idf1"], 4), "mota": round(o["mota"], 4), "idsw": o["num_switches"]})
            print(f"  {kw} -> IDF1 {o['idf1']:.3f} MOTA {o['mota']:.3f} IDSW {o['num_switches']}")
        best = max(grid, key=lambda g: g["idf1"])
        result["grid"] = grid
        result["best_by_idf1"] = best
        base_kw = {k: best[k] for k in ("high_thresh", "match_thresh", "track_buffer")}

    overall, table = run(base_kw)
    result["tracker"] = base_kw
    result["overall"] = overall
    result["per_video"] = {n: {"idf1": round(float(table.loc[n, "idf1"]), 4),
                               "mota": round(float(table.loc[n, "mota"]), 4),
                               "idsw": int(table.loc[n, "num_switches"])} for n in names}
    result["system"] = system_info(device)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    suffix = ("_tune" if args.tune else "") + ("_oracle" if args.oracle else "")
    out = config.RESULTS_DIR / f"tracking_{args.dataset}_{args.split}{suffix}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    keys = ("num_frames", "mota", "idf1", "num_switches", "precision", "recall", "num_unique_objects")
    print("tracker:", base_kw)
    print(json.dumps({k: overall[k] for k in keys}, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
