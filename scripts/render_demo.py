"""Render an annotated demo video (boxes, track IDs, trails, predicted paths).

    python scripts/render_demo.py                                  # sample video
    python scripts/render_demo.py --source my.mp4 --output out.mp4
    python scripts/render_demo.py --show                           # live window
    python scripts/render_demo.py --fusion                         # 3-model fusion
"""

import argparse
import json

import _bootstrap  # noqa: F401

from iamars import config
from iamars.detector import build_detector
from iamars.video import run_video


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(config.SAMPLE_VIDEO))
    ap.add_argument("--output", default=str(config.OUTPUTS_DIR / "demo_annotated.mp4"))
    ap.add_argument("--device", default=None, help="'cpu', '0' for GPU; default: auto")
    ap.add_argument("--fusion", action="store_true", help="use 3-model fusion")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--show", action="store_true", help="display while rendering")
    args = ap.parse_args()

    det = build_detector(fusion=args.fusion, device=args.device)
    label = f"model: {det.name}  device: {det.detectors[0].device if args.fusion else det.device}"

    def progress(i, n):
        if i % 50 == 0 or i == n:
            print(f"  frame {i}/{n}")

    summary = run_video(args.source, det, output=args.output, max_frames=args.max_frames,
                        show=args.show, label=label, progress=progress)
    print(json.dumps(summary, indent=2))
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
