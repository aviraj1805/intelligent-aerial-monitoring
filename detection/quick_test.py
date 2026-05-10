from ultralytics import YOLO

model = YOLO("models/visiodect/best.pt")
results = model.predict(
    source="data/sample_frames/test_drone4.mp4.mp4",
    conf=0.15,
    save=True,
    show=False,
    max_det=10
)

detected = sum([len(r.boxes) for r in results])
print(f"Total detections across video: {detected}")