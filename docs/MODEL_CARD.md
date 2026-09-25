# Model card — IAMARS drone detector v2 (`visiodect_yolov8n_v2.pt`)

Every number below is read from a JSON/CSV file in [`results/`](../results) that a
script in [`scripts/`](../scripts) produced. Re-run the script to reproduce it.

## Summary

| | |
|---|---|
| Task | Single-class object detection: `drone` |
| Architecture | YOLOv8n (Ultralytics 8.4.48), 3.0 M parameters, input 640 × 640 |
| Initialisation | COCO-pretrained `yolov8n.pt` (transfer learning) |
| Training data | VisioDECT, block split (see below) |
| Weights | GitHub Release `weights-v1`, file `visiodect_yolov8n_v2.pt` (SHA-256 `c02c038f…`) |
| Intended use | Research, education and portfolio demonstration. Not for operational use. |

## Data

**VisioDECT** ([IEEE DataPort](https://ieee-dataport.org/documents/visiodect-dataset-aerial-dataset-scenario-based-multi-drone-detection-and-identification)):
20,924 RGB images of 6 drone models (Anafi Extended, DJI FPV, DJI Phantom, EFT E410S,
Mavic Air 2, Mavic 2 Enterprise) in 3 conditions (cloudy, sunny, evening).
20,617 images have a box label and are used; all drone models are mapped to one class.

**Block split** ([`scripts/prepare_visiodect.py`](../scripts/prepare_visiodect.py),
lists in [`data/splits/`](../data/splits)). The images are consecutive video frames, so
neighbouring frames are almost identical. A random split would put near-copies of
test images into training and inflate the test score. Instead, inside each of the
18 model × weather folders, frames are cut into blocks of 50 consecutive frames and
whole blocks go to one split (seed 0):

| Split | Images |
|---|---|
| train | 14,515 |
| val | 4,302 |
| test | 1,800 |

## Training

[`scripts/train_detector.py`](../scripts/train_detector.py): 50 epochs, SGD
(lr0 0.01, momentum 0.937, weight decay 5e-4), batch 8, 640 px, mosaic augmentation
(off for the last 10 epochs), seed 42, on an NVIDIA RTX 2050 (4 GB).
The run was interrupted once by the laptop's idle sleep and resumed from the epoch-18
checkpoint. Wall time from the `time` column of
[`results/training_v2/results.csv`](../results/training_v2/results.csv): 4,933 s for
epochs 1–18 plus 11,161 s for epochs 19–50 (part of the second run was on battery power).
The checkpoint with the best validation mAP@0.5:0.95 was kept (epoch 50).
Loss and metric curves: [`results/training_v2/results.png`](../results/training_v2/results.png).

## Evaluation

[`scripts/eval_detection.py`](../scripts/eval_detection.py) runs Ultralytics validation
(confidence 0.001, NMS IoU 0.7, all thresholds swept). A prediction counts as correct
when its box overlaps a true box by IoU ≥ 0.5 (mAP@0.5), or averaged over IoU 0.5–0.95.

| Test set | Images | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Result file |
|---|---|---|---|---|---|---|
| VisioDECT block test (held out) | 1,800 | 0.965 | 0.596 | 0.960 | 0.957 | [json](../results/detection_v2_visiodect_block_test.json) |
| Anti-UAV-RGBT RGB test frames (never seen) | 3,424 | 0.291 | 0.143 | 0.571 | 0.274 | [json](../results/detection_v2_antiuav_rgb_test.json) |

For comparison, the teammate-trained YOLOv8m (`uav_rgb_yolov8m.pt`, trained on
DUT-Anti-UAV) scores mAP@0.5 0.793 and mAP@0.5:0.95 0.394 on the same Anti-UAV frames
([json](../results/detection_teammate_uav_rgb_antiuav_rgb_test.json)). It was not checked
whether DUT-Anti-UAV and Anti-UAV-RGBT share footage.

Validation at the kept epoch (from `results.csv`): mAP@0.5 0.969, mAP@0.5:0.95 0.614.

## Limitations

- **Domain gap.** Accuracy drops from 0.965 to 0.291 mAP@0.5 on footage from different
  cameras. On Anti-UAV frames the model often misses the drone and fires on the camera's
  on-screen crosshair overlay.
- **Distant drones only.** VisioDECT drones are small and far away; drones filmed close
  up (large in the frame) are mostly missed.
- **RGB only.** Not trained or evaluated on thermal video.
- **One dataset, six drone models.** Birds, planes and other flying objects were not
  part of training or testing, so the false-alarm rate on them is unknown.

## History: the v1 label bug

The first model (`visiodect_yolov8n.pt`, v1, trained on a random split) reported
mAP@0.5 0.98 and mAP@0.5:0.95 0.62 during training. Re-evaluating it against the
dataset's own VOC boxes on its original test split gives **0.000** mAP@0.5
([json](../results/detection_v1_legacy_split.json)). Its boxes are centred on each drone's
top-left corner: shifting every predicted box by half its width and height restores
mAP@0.5 0.986 / mAP@0.5:0.95 0.6475
([json](../results/detection_v1_legacy_split_shifted.json),
`prepare_visiodect.py --legacy`, then `eval_detection.py --data datasets/visiodect_test.yaml --model visiodect_v1 --shift-half-box`). So v1's training *and*
validation labels were converted with the box corner used as the box centre, and the
bug was invisible in the training metrics. v2 was retrained from scratch on labels
converted from the VOC corners, and the fix was confirmed by drawing predicted and
true boxes on test images. The superseded v1 card is kept in
[`results/model_reports/v1_models/`](../results/model_reports/v1_models).
