import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

DURATION = 5
SR = 16000

print("\nRecording started...")

audio = sd.rec(
    int(DURATION * SR),
    samplerate=SR,
    channels=1
)

sd.wait()

print("Recording finished.")

audio = np.squeeze(audio)

write("live.wav", SR, audio)

# Generate spectrogram
y, sr = librosa.load("live.wav", sr=16000)

mel = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_mels=128
)

mel_db = librosa.power_to_db(mel, ref=np.max)

plt.figure(figsize=(3,3))

librosa.display.specshow(
    mel_db,
    sr=sr
)

plt.axis('off')

plt.savefig(
    "live.png",
    bbox_inches='tight',
    pad_inches=0
)

plt.close()

print("\nGenerated spectrogram.")

# Run YOLO prediction
os.system(
    "yolo classify predict "
    "model=runs/classify/train/weights/best.pt "
    "source=live.png"
)
