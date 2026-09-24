"""Video reading/writing and the ``run_video`` loop shared by scripts and app."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from iamars.pipeline import Pipeline
from iamars.visualize import Annotator


def video_info(path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def iter_frames(path, max_frames: int | None = None, start_s: float = 0.0):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    n = 0
    while max_frames is None or n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
        n += 1
    cap.release()


class VideoWriter:
    """H.264 MP4 writer (browser-playable) using the ffmpeg bundled with imageio-ffmpeg."""

    def __init__(self, path, fps: float, size_wh: tuple[int, int]):
        import imageio_ffmpeg

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._gen = imageio_ffmpeg.write_frames(
            str(path), size_wh, fps=fps, codec="libx264", pix_fmt_out="yuv420p",
            quality=7, macro_block_size=8, ffmpeg_log_level="error",
        )
        self._gen.send(None)
        self.size = size_wh

    def write(self, bgr: np.ndarray) -> None:
        assert (bgr.shape[1], bgr.shape[0]) == tuple(self.size), "frame size mismatch"
        self._gen.send(np.ascontiguousarray(bgr[:, :, ::-1]))

    def close(self) -> None:
        self._gen.close()


def run_video(
    source,
    detector,
    output=None,
    max_frames: int | None = None,
    show: bool = False,
    label: str = "",
    progress: Callable[[int, int], None] | None = None,
    max_width: int | None = None,
) -> dict:
    """Run the pipeline over a video. Optionally save/show the annotated result.

    ``max_width`` downscales wider frames first (faster on CPU).
    Returns a summary: frames processed, mean per-stage latency, track stats.
    """
    info = video_info(source)
    pipe = Pipeline(detector, fps=info["fps"])
    ann = Annotator()
    writer = VideoWriter(output, info["fps"], ann.size) if output else None

    timings, ids, fps_ema = [], set(), None
    total = min(info["frames"], max_frames) if max_frames else info["frames"]
    t_prev = t_start = time.perf_counter()
    for i, frame in enumerate(iter_frames(source, max_frames)):
        if max_width and frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(frame, (max_width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        res = pipe.process(frame)
        timings.append(res.timings_ms)
        ids.update(t.track_id for t in res.tracks)
        now = time.perf_counter()
        inst = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now
        fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst
        if writer or show:
            canvas = ann.draw(frame, res, fps=fps_ema, label=label)
            if writer:
                writer.write(canvas)
            if show:
                cv2.imshow("IAMARS", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        if progress:
            progress(i + 1, total)
    if writer:
        writer.close()
    if show:
        cv2.destroyAllWindows()

    summary = {"frames": len(timings), "unique_track_ids": len(ids), "video": info,
               "fps_overall": len(timings) / max(time.perf_counter() - t_start, 1e-9)}
    if timings:
        summary["mean_ms"] = {k: float(np.mean([t[k] for t in timings])) for k in timings[0]}
    return summary
