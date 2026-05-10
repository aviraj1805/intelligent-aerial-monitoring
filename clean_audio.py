import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

# INPUT FOLDERS
DRONE_INPUTS = [
    "data/raw/archive/dataset/drone",
    "data/raw/archive1/Binary_Drone_Audio/yes_drone"
]

NON_DRONE_INPUTS = [
    "data/raw/archive/dataset/noise",
    "data/raw/archive1/Binary_Drone_Audio/unknown"
]

# OUTPUT FOLDERS
DRONE_OUTPUT = "data/clean/drone"
NON_DRONE_OUTPUT = "data/clean/non_drone"

Path(DRONE_OUTPUT).mkdir(parents=True, exist_ok=True)
Path(NON_DRONE_OUTPUT).mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000
TARGET_DURATION = 5  # seconds
TARGET_LENGTH = TARGET_SR * TARGET_DURATION

def process_audio(input_dirs, output_dir, prefix):

    count = 0

    for folder in input_dirs:

        files = list(Path(folder).glob("*.wav"))

        for file in files:

            try:
                # load audio
                y, sr = librosa.load(file, sr=TARGET_SR, mono=True)

                # skip silent files
                if np.max(np.abs(y)) < 0.01:
                    print(f"Skipping silent file: {file}")
                    continue

                # trim/pad audio
                if len(y) > TARGET_LENGTH:
                    y = y[:TARGET_LENGTH]
                else:
                    y = np.pad(y, (0, TARGET_LENGTH - len(y)))

                # normalize volume
                y = y / np.max(np.abs(y))

                output_path = f"{output_dir}/{prefix}_{count:05d}.wav"

                sf.write(output_path, y, TARGET_SR)

                print(f"Saved: {output_path}")

                count += 1

            except Exception as e:
                print(f"Error processing {file}: {e}")

print("\nProcessing DRONE audio...")
process_audio(DRONE_INPUTS, DRONE_OUTPUT, "drone")

print("\nProcessing NON-DRONE audio...")
process_audio(NON_DRONE_INPUTS, NON_DRONE_OUTPUT, "non_drone")

print("\nDONE CLEANING AUDIO")
