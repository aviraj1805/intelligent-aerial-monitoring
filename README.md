## Audio Drone Detection

This module performs drone detection using audio classification with YOLOv8.

## Pipeline

Audio (.wav)
→ Cleaning
→ Mel Spectrogram Generation
→ YOLOv8 Classification
→ Drone / Non-Drone Prediction

## Features

- Audio preprocessing
- Spectrogram generation
- YOLOv8 classifier
- Real-time microphone detection
- External audio testing

## Validation Accuracy

~97.5%

## Main Scripts

- clean_audio.py
- generate_spectrograms.py
- split_dataset.py
- live_detection.py
- test_spectrogram.py

## Model Path

models/audio_detection/audio_yolov8_classifier.pt
