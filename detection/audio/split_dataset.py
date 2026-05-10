from pathlib import Path
import random
import shutil

random.seed(42)

SOURCE = {
    "drone": "data/spectrograms/drone",
    "non_drone": "data/spectrograms/non_drone"
}

for label, folder in SOURCE.items():

    files = list(Path(folder).glob("*.png"))

    random.shuffle(files)

    split_idx = int(len(files) * 0.8)

    train_files = files[:split_idx]
    val_files = files[split_idx:]

    train_dir = Path(f"dataset/train/{label}")
    val_dir = Path(f"dataset/val/{label}")

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for file in train_files:
        shutil.copy(file, train_dir / file.name)

    for file in val_files:
        shutil.copy(file, val_dir / file.name)

    print(f"\n{label}")
    print(f"Train: {len(train_files)}")
    print(f"Val: {len(val_files)}")

print("\nDONE SPLITTING DATASET")
