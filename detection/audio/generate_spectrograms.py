import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

INPUTS = {
    "drone": "data/balanced/drone",
    "non_drone": "data/balanced/non_drone"
}

OUTPUT_BASE = "data/spectrograms"

for label, folder in INPUTS.items():

    output_dir = Path(f"{OUTPUT_BASE}/{label}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(Path(folder).glob("*.wav"))

    for idx, file in enumerate(files):

        try:
            y, sr = librosa.load(file, sr=16000)

            mel = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=128
            )

            mel_db = librosa.power_to_db(mel, ref=np.max)

            plt.figure(figsize=(3,3))

            librosa.display.specshow(
                mel_db,
                sr=sr,
                x_axis=None,
                y_axis=None
            )

            plt.axis('off')

            output_file = output_dir / f"{label}_{idx:05d}.png"

            plt.savefig(
                output_file,
                bbox_inches='tight',
                pad_inches=0
            )

            plt.close()

            print(f"Saved: {output_file}")

        except Exception as e:
            print(f"Error processing {file}: {e}")

print("\nDONE GENERATING SPECTROGRAMS")
