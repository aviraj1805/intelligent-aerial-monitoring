# train_baseline.py
# IAMARS Step 2.3 — Baseline Training Run (GPU Forced Resume)

# Must be before ANY torch or ultralytics import
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import yaml
from ultralytics import YOLO

# ── Pre-flight GPU assertion ──────────────────────────────────────
assert torch.cuda.is_available(), (
    "❌ CUDA not available — check PyTorch build with: python -c 'import torch; print(torch.__version__)'"
)
print(f"✅ GPU confirmed : {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM          : {round(torch.cuda.get_device_properties(0).total_memory/1e9, 2)} GB")

# ── Patch args.yaml before resume reads it ───────────────────────
args_path = "runs/detect/iamars_baseline/args.yaml"
if os.path.exists(args_path):
    with open(args_path, "r") as f:
        args = yaml.safe_load(f)
    args["device"] = 0
    with open(args_path, "w") as f:
        yaml.dump(args, f)
    print(f"✅ args.yaml patched — device = {args['device']}")
else:
    print(f"⚠️  args.yaml not found at {args_path}")

# ── Load from checkpoint, NOT base weights ───────────────────────
WEIGHTS = "runs/detect/iamars_baseline/weights/last.pt"
assert os.path.exists(WEIGHTS), f"❌ Checkpoint not found: {WEIGHTS}"
model = YOLO(WEIGHTS)
print(f"✅ Checkpoint loaded : {WEIGHTS}")

# ── Resume training on GPU ────────────────────────────────────────
results = model.train(
    # --- Data ---
    data     = "D:/intelligent-aerial-monitoring/IAMARS_Dataset/data.yaml",

    # --- Core training ---
    epochs   = 50,
    batch    = 8,
    imgsz    = 640,

    # --- Optimizer ---
    optimizer    = "SGD",
    lr0          = 0.01,
    lrf          = 0.01,
    momentum     = 0.937,
    weight_decay = 0.0005,
    warmup_epochs   = 3,
    warmup_momentum = 0.8,

    # --- Augmentation ---
    hsv_h     = 0.015,
    hsv_s     = 0.7,
    hsv_v     = 0.4,
    degrees   = 0.0,
    translate = 0.1,
    scale     = 0.5,
    flipud    = 0.0,
    fliplr    = 0.5,
    mosaic    = 1.0,
    mixup     = 0.0,

    # --- Hardware ---
    device  = 0,
    workers = 4,

    # --- Output ---
    project    = "runs/detect",
    name       = "iamars_baseline",
    exist_ok   = True,
    pretrained = True,
    verbose    = True,
    seed       = 42,

    # --- Saving ---
    save        = True,
    save_period = 10,

    # --- Resume ---
    resume = True,
)
