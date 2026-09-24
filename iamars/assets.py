"""Download the sample video (Wikimedia Commons) used by the demo and app."""

import tempfile
import urllib.request
from pathlib import Path

import cv2

from iamars import config
from iamars.video import VideoWriter

USER_AGENT = "IAMARS-portfolio/1.1 (https://github.com/aviraj1805/intelligent-aerial-monitoring)"


def download_sample(out: Path = config.SAMPLE_VIDEO, width: int = 1280) -> Path:
    if out.exists():
        print(f"[sample] already present: {out}")
        return out
    src = config.SAMPLE_SOURCE
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "source.webm"
        print(f"[sample] downloading {src['page']}")
        req = urllib.request.Request(src["url"], headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req) as r, open(raw, "wb") as f:
            f.write(r.read())

        cap = cv2.VideoCapture(str(raw))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (width, int(round(h * width / w / 8)) * 8)  # H.264 wants multiples of 8
        first, last = int(src["start_s"] * fps), int(src["end_s"] * fps)
        writer = VideoWriter(out, fps, size)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or i >= last:
                break
            if i >= first:
                writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
            i += 1
        writer.close()
        cap.release()

    (out.parent / "ATTRIBUTION.txt").write_text(
        f"{out.name}: segment {src['start_s']:.0f}-{src['end_s']:.0f} s of\n"
        f"'{src['title']}' by {src['author']}, {src['page']}\n"
        f"License: {src['license']} ({src['license_url']}). Trimmed and resized.\n",
        encoding="utf-8",
    )
    print(f"[sample] wrote {out}")
    return out
