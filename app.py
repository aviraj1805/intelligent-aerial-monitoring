"""IAMARS web demo (Gradio). Runs locally or on a free Hugging Face Space (CPU).

    python app.py        # then open http://127.0.0.1:7860
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import gradio as gr

from iamars import config
from iamars.detector import YoloDetector
from iamars.video import run_video, video_info

MAX_SECONDS = 20       # keep CPU processing time reasonable on free hosting
MAX_WIDTH = 960        # downscale larger videos before detection

_detector: YoloDetector | None = None


def get_detector() -> YoloDetector:
    global _detector
    if _detector is None:
        _detector = YoloDetector()
    return _detector


def ensure_sample() -> str | None:
    if not config.SAMPLE_VIDEO.exists():
        try:
            from iamars.assets import download_sample

            download_sample()
        except Exception as exc:  # the app still works with uploads
            print(f"[app] sample download failed: {exc}")
            return None
    return str(config.SAMPLE_VIDEO)


def process(video_path: str | None, seconds: float, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload a video or pick the example below.")
    info = video_info(video_path)
    max_frames = int(min(seconds, MAX_SECONDS) * info["fps"])
    det = get_detector()
    out = Path(tempfile.mkdtemp()) / "iamars_annotated.mp4"
    t0 = time.perf_counter()
    summary = run_video(
        video_path, det, output=out, max_frames=max_frames, max_width=MAX_WIDTH,
        label=f"YOLOv8n  device: {det.device}",
        progress=lambda i, n: progress(i / max(n, 1), desc=f"frame {i}/{n}"),
    )
    wall = time.perf_counter() - t0
    ms = summary.get("mean_ms", {})
    table = (
        "| | |\n|---|---|\n"
        f"| Frames processed | {summary['frames']} |\n"
        f"| Unique track IDs | {summary['unique_track_ids']} |\n"
        f"| Device | {det.device} |\n"
        f"| Mean latency per frame (detect + track + estimate + predict) | {ms.get('total', 0):.1f} ms |\n"
        f"| of which detection | {ms.get('detect', 0):.1f} ms |\n"
        f"| Wall time incl. video decode/encode | {wall:.1f} s |\n"
    )
    return str(out), table


DESCRIPTION = """
**IAMARS**: drone detection (YOLOv8n) → multi-object tracking (ByteTrack) →
Kalman-filter state estimation → constant-velocity trajectory prediction.

Boxes show tracked drones with their ID and confidence, coloured lines are the
recent path, orange dots are the predicted path for the next 0.5 s.
The detector was trained on small, distant drones (VisioDECT); it does poorly on
drones filmed close-up. On a CPU-only machine processing is slower than on a GPU
(measured on a laptop: 20.2 FPS on an i5-13420H CPU, 88.5 FPS on an RTX 2050; see the README).
"""


def build() -> gr.Blocks:
    sample = ensure_sample()
    with gr.Blocks(title="IAMARS drone tracking") as demo:
        gr.Markdown("# IAMARS: drone detection & tracking")
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                inp = gr.Video(label="Input video", sources=["upload"])
                secs = gr.Slider(2, MAX_SECONDS, value=10, step=1, label="Seconds to process")
                btn = gr.Button("Run pipeline", variant="primary")
            with gr.Column():
                out = gr.Video(label="Annotated output")
                stats = gr.Markdown()
        btn.click(process, [inp, secs], [out, stats])
        if sample:
            gr.Examples([[sample, 10]], [inp, secs], label="Example (MagicLab, CC BY 3.0, Marco Tempest)")
    return demo


if __name__ == "__main__":
    build().queue().launch()
