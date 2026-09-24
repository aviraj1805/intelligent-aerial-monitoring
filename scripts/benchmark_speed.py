"""Benchmark pipeline speed (FPS and per-frame latency) on the sample video.

    python scripts/benchmark_speed.py --device 0      # GPU
    python scripts/benchmark_speed.py --device cpu

What is timed: detection + tracking + Kalman update + trajectory prediction for
one frame (the per-frame work of the pipeline). Video decoding, drawing and
encoding are excluded and reported separately as "with I/O" FPS.
The first --warmup frames are not counted (model loading, CUDA initialisation).
Frames are processed one at a time (batch size 1), as in live use.

Writes results/benchmark_<device>.json.
"""

import argparse
import json
import time

import _bootstrap  # noqa: F401
import numpy as np

from iamars import config
from iamars.detector import YoloDetector
from iamars.pipeline import Pipeline
from iamars.sysinfo import system_info
from iamars.video import iter_frames, video_info
from iamars.visualize import Annotator


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None, help="'cpu' or GPU index like '0'")
    ap.add_argument("--source", default=str(config.SAMPLE_VIDEO))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--frames", type=int, default=None, help="limit frames (default: whole video)")
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    device = args.device or config.auto_device()
    info = video_info(args.source)
    det = YoloDetector(weights_path=args.weights, device=device)
    pipe = Pipeline(det, fps=info["fps"])
    ann = Annotator()

    stages, loop_times = [], []
    frames = list(iter_frames(args.source, args.frames))  # decode up front
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        res = pipe.process(frame)
        ann.draw(frame, res)
        loop = time.perf_counter() - t0
        if i >= args.warmup:
            stages.append(res.timings_ms)
            loop_times.append(loop * 1e3)

    def stats(xs):
        xs = np.asarray(xs)
        return {"mean": round(float(xs.mean()), 2), "p50": round(float(np.median(xs)), 2),
                "p95": round(float(np.percentile(xs, 95)), 2)}

    per_stage = {k: stats([s[k] for s in stages]) for k in stages[0]}
    res = {
        "what": "per-frame pipeline latency: detect + track + estimate + predict (batch 1)",
        "source": f"{args.source} ({info['width']}x{info['height']}, {info['fps']:.2f} fps)",
        "model": det.name,
        "imgsz": det.imgsz,
        "frames_timed": len(stages),
        "warmup_frames": args.warmup,
        "latency_ms": per_stage,
        "pipeline_fps": round(1000.0 / per_stage["total"]["mean"], 1),
        "with_drawing_fps": round(1000.0 / float(np.mean(loop_times)), 1),
        "system": system_info(device),
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)
    tag = "cpu" if device == "cpu" else "gpu"
    out = config.RESULTS_DIR / f"benchmark_{tag}.json"
    if args.frames is None:
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("frames_timed", "latency_ms", "pipeline_fps", "with_drawing_fps")}, indent=2))
    print(f"device: {res['system'].get('device_used')}  cpu: {res['system']['cpu']}")
    print(f"saved -> {out}" if args.frames is None else "(partial run with --frames: not saved)")


if __name__ == "__main__":
    main()
