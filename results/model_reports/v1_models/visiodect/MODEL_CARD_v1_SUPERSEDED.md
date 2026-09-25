> **SUPERSEDED.** The metrics on this page were measured against labels whose box centre
> was the drone's top-left corner. Against correct labels this v1 model scores 0.000 mAP@0.5.
> See [docs/MODEL_CARD.md](../../../../docs/MODEL_CARD.md) for the current model and the bug analysis.

# MODEL CARD — IAMARS Drone Detector v1.0

**Project:** IAMARS — Intelligent Aerial Monitoring & Automated Response System  
**Step:** 2 — Drone Detection Model  
**Status:** ✅ Complete — Cleared for Step 3 Integration  
**Version:** v1.0  
**Date:** 2026-05-06

---

## Model Identity

| Field | Value |
|---|---|
| Model name | `drone_detector.pt` |
| Architecture | YOLOv8n (nano) |
| Base weights | COCO pretrained (`yolov8n.pt`) |
| Fine-tuned on | VisioDECT — 20,600 images |
| Framework | Ultralytics 8.3.0 / PyTorch 2.3.0 |
| Parameters | 2,684,563 |
| GFLOPs | 6.8 |

---

## Input Contract

| Field | Value |
|---|---|
| Input shape | `[1, 3, 640, 640]` — batch=1, RGB, H=W=640 |
| Colour order | RGB (not BGR) |
| Normalisation | Pixel values divided by 255.0 → range [0.0, 1.0] |
| Dtype | Float32 |
| Resize strategy | Letterbox (aspect-ratio preserved, padded to 640×640) |

> ⚠️ Ultralytics `model.predict()` handles all preprocessing automatically.
> Only applies if calling `drone_detector.onnx` directly.

---

## Output Contract

### PyTorch (`drone_detector.pt`) via Ultralytics

```python
results = model.predict(frame, conf=0.35, iou=0.45, device=0)
boxes   = results[0].boxes          # Ultralytics Boxes object

# Per detection:
boxes.xyxy        # Tensor [N, 4] — absolute pixel coords [x1, y1, x2, y2]
boxes.conf        # Tensor [N]    — confidence scores [0.0, 1.0]
boxes.cls         # Tensor [N]    — class index (always 0 = drone)
```

### ONNX (`drone_detector.onnx`) raw output

```
Output shape : [1, 5, 8400]
Transposed   : [8400, 5] → columns = [cx, cy, w, h, conf]
Coords       : normalised centre-format (0.0–1.0), relative to 640×640
Requires     : manual NMS before use
```

---

## Class Mapping

| Index | Class | Notes |
|---|---|---|
| `0` | `drone` | Only class — single-class model |

---

## Recommended Inference Settings

| Parameter | Recommended Value | Notes |
|---|---|---|
| `conf` threshold | **0.35** | Filters low-confidence detections pre-NMS |
| `iou` threshold | **0.45** | NMS overlap threshold — prevents duplicate boxes |
| `imgsz` | `640` | Must match training resolution |
| `device` | `0` (GPU) | Fall back to `"cpu"` if no GPU available |
| `half` | `False` | FP16 not stable on RTX 2050 |

> These values are tuned for the drone detection task. Do not lower `conf`
> below 0.25 — increases false positives significantly. Do not raise `iou`
> above 0.6 — merges overlapping true detections in dense scenes.

---

## Performance Metrics (Official — Held-Out Test Set)

| Metric | Value |
|---|---|
| **mAP@0.5** | **0.9841** |
| mAP@0.5:0.95 | 0.6259 |
| Precision | 0.9667 |
| Recall | 0.9628 |
| Test images | 2,063 |
| Test instances | 2,069 |

### Inference Speed (NVIDIA RTX 2050, 4GB VRAM)

| Stage | Time |
|---|---|
| Preprocess | 0.4 ms/image |
| Inference | 9.0 ms/image |
| Postprocess | 1.8 ms/image |
| **Total** | **~11.2 ms/image (~89 FPS)** |

---

## Detection Samples & Inference Results

### Multi-Platform Detection Performance

**Inference on diverse drone models and environmental conditions:**

The model demonstrates robust detection across multiple drone platforms (MAVIC, DJI Phantom, Anafi, EFT E410S) and lighting scenarios (sunny, cloudy, evening).

> 📸 **Detection visualization images** — Coming from inference pipeline output
> - MAVIC Air Evening, DJI Phantom Cloudy, and mixed platform results
> - Anafi Extended, EFT E410S, DJI Phantom Sunny results
> - High-density multi-drone detection on EFT E410S frames
> - Field deployment inference examples

---

## Model Evaluation Results

### Evaluation Metrics Visualizations

![Precision-Recall Curve](https://raw.githubusercontent.com/aviraj1805/intelligent-aerial-monitoring/main/models/iamars_drone_v1/eval/PR_curve.png)
*Figure 1: Precision-Recall curve showing model trade-offs at different confidence thresholds.*

![F1 Curve](https://raw.githubusercontent.com/aviraj1805/intelligent-aerial-monitoring/main/models/iamars_drone_v1/eval/F1_curve.png)
*Figure 2: F1 score vs confidence threshold — optimal operating point around 0.35 confidence.*

![Confusion Matrix](https://raw.githubusercontent.com/aviraj1805/intelligent-aerial-monitoring/main/models/iamars_drone_v1/eval/confusion_matrix.png)
*Figure 3: Confusion matrix (TP/FP/FN) on held-out test set showing high classification purity.*

---

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 50 |
| Batch size | 8 |
| Image size | 640 |
| Optimizer | SGD |
| lr0 / lrf | 0.01 / 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warmup epochs | 3 |
| Mosaic aug | 1.0 |
| Best epoch | 49/50 |
| Hardware | NVIDIA RTX 2050 4GB, CUDA 12.1 |

---

## Training Curves & Convergence Analysis

### Loss Curves (Train / Validation)

Stable convergence across all three YOLOv8 loss components:

- **Box Loss** (Localization): Drops from ~2.2 → 1.35, then plateaus — indicates tight bounding box predictions by epoch 49
- **Classification Loss**: Converges smoothly from ~2.1 → 0.6 — high-confidence single-class predictions
- **DFL Loss** (Distribution Focal): Descends from ~1.2 → 0.95 — stable anchor-free regression

### Performance Metrics (Validation)

Progressive improvement in detection quality across 50 epochs:

**Training curves show:**
- **Box Loss** (Localization): Drops from ~2.2 → 1.35 — tight bounding box predictions
- **Classification Loss**: Converges from ~2.1 → 0.6 — high-confidence predictions
- **DFL Loss** (Distribution Focal): Descends from ~1.2 → 0.95 — stable regression

**Metric Summary:**
- **Precision & Recall:** Both stabilize near **0.97–1.0** by epoch 30
- **mAP50:** Reaches **0.98+** — near-perfect detection at standard IoU
- **mAP50-95:** Settles at **0.62–0.65** — consistent performance at strict thresholds

> 📈 **Training metrics visualization** — Complete loss curves and performance graphs available in training logs

---

## File Manifest

| File | Purpose |
|---|---|
| `drone_detector.pt` | Primary model — use this for MVP pipeline |
| `drone_detector.onnx` | Portable export — use for edge/non-Python runtimes |
| `data.yaml` | Class map and dataset path config |
| `train_config.yaml` | Full training hyperparameter record |
| `eval/confusion_matrix.png` | TP/FP/FN visualisation |
| `eval/PR_curve.png` | Precision-Recall curve |
| `eval/F1_curve.png` | F1 vs confidence threshold |

---

## Known Limitations

- Model is single-class — outputs `drone` only, no background classification
- Optimised for aerial footage consistent with VisioDECT distribution
- Performance on thermal / night-vision / occluded frames not benchmarked
- Batch inference not tested — pipeline assumes `batch=1` real-time stream

---

*Generated by IAMARS Step 2 pipeline — Lead Model Training Engineer*
