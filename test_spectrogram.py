import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python test_spectrogram.py audio.wav")
    sys.exit()

audio_file = sys.argv[1]

# load audio
y, sr = librosa.load(audio_file, sr=16000)

# create mel spectrogram
mel = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_mels=128
)

mel_db = librosa.power_to_db(mel, ref=np.max)

# plot image
plt.figure(figsize=(3,3))

librosa.display.specshow(
    mel_db,
    sr=sr
)

plt.axis('off')

# save image
plt.savefig(
    "test.png",
    bbox_inches='tight',
    pad_inches=0
)

plt.close()

print("Saved test.png")
