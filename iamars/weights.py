"""Download model weights from the GitHub Release and verify their checksum."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from iamars.config import GITHUB_REPO, MODELS, WEIGHTS_DIR, WEIGHTS_RELEASE_TAG


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def release_url(filename: str) -> str:
    return (
        f"https://github.com/{GITHUB_REPO}/releases/download/"
        f"{WEIGHTS_RELEASE_TAG}/{filename}"
    )


def get_weights(name: str, weights_dir: Path = WEIGHTS_DIR) -> Path:
    """Return the local path of model ``name``, downloading it if needed."""
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Choose from {sorted(MODELS)}")
    spec = MODELS[name]
    path = Path(weights_dir) / spec["file"]
    if path.exists() and _sha256(path) == spec["sha256"]:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    url = release_url(spec["file"])
    print(f"[weights] downloading {spec['file']} from {url}")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    digest = _sha256(tmp)
    if digest != spec["sha256"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {spec['file']}: got {digest}")
    tmp.replace(path)
    return path
