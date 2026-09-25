# Historical records of the v1 models

These folders are kept for provenance only.

- `iamars_baseline/` and `eval/`: Ultralytics training and test outputs of the v1 VisioDECT model.
  All v1 metrics here were measured against labels shifted by half a box (see docs/MODEL_CARD.md).
  `eval/iamars_test_eval9/predictions.json` is the source of the recovered v1 test split
  (`data/splits/visiodect_test.txt`).
- `v1_models/`: v1 model card (superseded), dataset and training configs of the v1 model and of
  the two teammate models (`uav_rgb`, `uav_ir`).

Current results are the JSON files one level up, in `results/`.
