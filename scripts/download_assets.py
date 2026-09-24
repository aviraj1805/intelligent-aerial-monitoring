"""Download model weights (GitHub Release) and the sample video (Wikimedia Commons).

    python scripts/download_assets.py            # default model + sample video
    python scripts/download_assets.py --all      # also the two fusion models (~100 MB)
"""

import argparse

import _bootstrap  # noqa: F401

from iamars import config
from iamars.assets import download_sample
from iamars.weights import get_weights


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="also download the fusion models")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    names = list(config.MODELS) if args.all else [config.DEFAULT_MODEL]
    for n in names:
        print(f"[weights] {n}: {get_weights(n)}")
    if not args.no_video:
        download_sample()


if __name__ == "__main__":
    main()
