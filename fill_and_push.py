"""
IAMARS - Fill Empty Folders & Push to GitHub
Run this script from the ROOT of your project folder.
Usage: python fill_and_push.py
"""

import os
import subprocess

# ── Folder descriptions (IAMARS-specific) ──────────────────────────────────────
FOLDER_COMMENTS = {
    "data":          "# Stores raw and processed datasets used for drone detection model training and validation.",
    "detection":     "# YOLOv8-based drone detection module — runs inference on video frames and outputs bounding boxes.",
    "estimation":    "# Kalman Filter state estimation module — smooths noisy tracking data into position, velocity, and acceleration estimates.",
    "IAMARS_Dataset":"# VisDrone2019 dataset files and Roboflow-exported labels used to fine-tune the YOLOv8 detection model.",
    "iamars_env":    "# Python virtual environment folder — contains all installed project dependencies (do not edit manually).",
    "integration":   "# System integration layer — connects all pipeline modules (detection → tracking → estimation → prediction → intercept) into one real-time application.",
    "intercept":     "# Fire-control / intercept computation module — calculates azimuth, elevation, and firing time to intercept the drone.",
    "models":        "# Saved model weights and exported ONNX/TensorRT files for the trained YOLOv8 drone detection model.",
    "prediction":    "# Trajectory prediction module — estimates the drone's future position using kinematic equations and Kalman-derived state.",
    "results":       "# Output logs, evaluation metrics, mAP scores, and model performance reports generated during training and testing.",
    "simulator":     "# Mathematical drone flight simulator — generates synthetic 3D flight paths with Gaussian noise at 30 Hz for testing.",
    "tracking":      "# ByteTrack multi-object tracking module — assigns stable IDs to detected drones across video frames.",
    "visualization": "# Real-time 2D/3D visualisation dashboard — displays drone position, predicted path, intercept point, and live telemetry.",
}

PLACEHOLDER_FILENAME = ".gitkeep"


def is_folder_empty(folder_path):
    """Return True if a folder has no files (ignoring hidden system files)."""
    for root, dirs, files in os.walk(folder_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        if files:
            return False
    return True


def is_path_ignored(path):
    """Check if a path is ignored by git. Returns True if ignored."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", path],
        capture_output=True
    )
    return result.returncode == 0  # 0 = ignored, 1 = not ignored


def fill_empty_folders(project_root):
    """Find empty folders and add a .gitkeep with a descriptive comment."""
    filled = []

    for folder_name, comment in FOLDER_COMMENTS.items():
        folder_path = os.path.join(project_root, folder_name)

        if not os.path.isdir(folder_path):
            print(f"  [SKIP]   '{folder_name}' — folder not found.")
            continue

        if is_folder_empty(folder_path):
            placeholder = os.path.join(folder_path, PLACEHOLDER_FILENAME)
            with open(placeholder, "w", encoding="utf-8") as f:
                f.write(comment + "\n")
            print(f"  [FILLED] '{folder_name}' — added {PLACEHOLDER_FILENAME}")
            filled.append(folder_name)
        else:
            print(f"  [OK]     '{folder_name}' — already has content, skipping.")

    return filled


def git_push(project_root, filled_folders):
    """Stage (force if ignored), commit, and push the .gitkeep files."""
    os.chdir(project_root)

    print("\n── Git: staging .gitkeep files ──")

    for folder in filled_folders:
        rel_path = os.path.join(folder, PLACEHOLDER_FILENAME)

        if is_path_ignored(rel_path) or is_path_ignored(folder):
            # Force-add only the specific .gitkeep file inside the ignored folder
            cmd = ["git", "add", "-f", rel_path]
            print(f"  [FORCE]  '{rel_path}' — gitignored folder, force-adding .gitkeep only")
        else:
            cmd = ["git", "add", rel_path]
            print(f"  [ADD]    '{rel_path}'")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR staging '{rel_path}': {result.stderr.strip()}")

    print("\n── Git: committing ──")
    commit_msg = "chore: add .gitkeep placeholders to preserve empty project folders"
    result = subprocess.run(
        ["git", "commit", "-m", commit_msg],
        capture_output=True, text=True
    )

    if "nothing to commit" in result.stdout:
        print("  Nothing new to commit — all folders were already tracked.")
        return

    print(result.stdout.strip())

    print("\n── Git: pushing to origin main ──")
    push_result = subprocess.run(
        ["git", "push", "origin", "main"],
        capture_output=True, text=True
    )
    print(push_result.stdout.strip())

    if push_result.returncode != 0:
        print("  ERROR during push:")
        print(push_result.stderr.strip())
    else:
        print("  Push successful!")


def main():
    project_root = os.getcwd()
    print(f"\nIAMARS — Fill Empty Folders & Push")
    print(f"Project root: {project_root}\n")

    filled = fill_empty_folders(project_root)

    if not filled:
        print("\nNo empty folders found — nothing to push.")
        return

    print(f"\n{len(filled)} folder(s) filled: {filled}")

    answer = input("\nPush these changes to GitHub now? (y/n): ").strip().lower()
    if answer == "y":
        git_push(project_root, filled)
    else:
        print("Skipped. To push manually: git add -f <folder>/.gitkeep && git push origin main")


if __name__ == "__main__":
    main()