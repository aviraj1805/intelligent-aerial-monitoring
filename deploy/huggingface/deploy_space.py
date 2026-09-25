"""Create or update the Hugging Face Space that hosts the IAMARS web demo.

    huggingface-cli login                                   # once, with a *write* token
    python scripts/render_demo.py                           # optional: pre-rendered output
    python deploy/huggingface/deploy_space.py --space <user>/iamars-drone-tracking

Uploads app.py, the iamars package, requirements, the Space config (README.md,
packages.txt), the v2 weights, the example clip and, if present, the pre-rendered
annotated demo video. Everything runs on the free CPU tier.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def stage_files(stage: Path) -> list[str]:
    files = {
        ROOT / "app.py": "app.py",
        ROOT / "requirements.txt": "requirements.txt",
        HERE / "README.md": "README.md",
        HERE / "packages.txt": "packages.txt",
        ROOT / "weights" / "visiodect_yolov8n_v2.pt": "weights/visiodect_yolov8n_v2.pt",
        ROOT / "data" / "samples" / "magiclab_24_drones.mp4": "data/samples/magiclab_24_drones.mp4",
        ROOT / "data" / "samples" / "ATTRIBUTION.txt": "data/samples/ATTRIBUTION.txt",
        ROOT / "outputs" / "demo_annotated.mp4": "data/samples/demo_annotated.mp4",
    }
    for py in sorted((ROOT / "iamars").glob("*.py")):
        files[py] = f"iamars/{py.name}"
    copied = []
    for src, dst in files.items():
        if not src.exists():
            print(f"[skip] {src.relative_to(ROOT)} not found")
            continue
        out = stage / dst
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        copied.append(dst)
    return copied


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, help="<hf-username>/<space-name>")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        copied = stage_files(stage)
        print(f"uploading {len(copied)} files")
        api = HfApi()
        api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(folder_path=str(stage), repo_id=args.space, repo_type="space",
                          commit_message="Deploy IAMARS demo from GitHub repo")
    owner, name = args.space.split("/")
    print(f"Space page: https://huggingface.co/spaces/{args.space}")
    print(f"Direct app: https://{owner.lower()}-{name.lower().replace('_', '-')}.hf.space")


if __name__ == "__main__":
    main()
