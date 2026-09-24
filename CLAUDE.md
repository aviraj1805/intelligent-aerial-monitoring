# CLAUDE.md — working notes for AI assistants on this repo

IAMARS is a portfolio project: drone detection (YOLOv8n) → ByteTrack → Kalman
filter → constant-velocity trajectory prediction on video, plus a 3D
*simulation-only* intercept-point evaluation. The owner uses it for AI/ML
internship interviews, so **every number must be reproducible by a script**.

## Hard rules
- Never put a metric in README/docs unless a script in `scripts/` produced it and
  the JSON is in `results/`. No rounding up, no estimates, no "~30 FPS" guesses.
- Anything intercept-related is labelled "simulation". Use neutral wording
  ("intercept-point prediction"), never weapon/fire-control/threat language.
- Do not commit weights (`*.pt`), datasets, sample videos or `outputs/`.
  `INTERVIEW_NOTES.md` is personal and gitignored — never commit it.
- Work in small sub-goals: build → test → commit → next. Ask before deleting
  the owner's files, rewriting history or large rewrites.
- Commit on the `portfolio-cleanup` branch style: small commits, clear messages.

## Layout
```
iamars/            library code (import iamars.*)
  config.py        paths, model registry (+ SHA-256), default thresholds
  weights.py       download weights from GitHub Release "weights-v1"
  assets.py        download + trim the sample video (Wikimedia, CC BY 3.0)
  detector.py      YoloDetector (default) and FusionDetector (--fusion, optional)
  fusion.py        weighted box fusion via ensemble-boxes
  tracker.py       ByteTrack wrapper (supervision); match_thresh is 1 - IoU
  estimator.py     generic NumPy Kalman filter + per-track box manager
  predictor.py     constant-velocity extrapolation
  intercept.py     closed-form 3D intercept solver (simulation only)
  simulation.py    3D trajectories + trials for the intercept evaluation
  pipeline.py      Pipeline.process(frame) -> FrameResult with stage timings
  visualize.py     Annotator: fixed 1280x720 canvas (video + telemetry panel)
  video.py         run_video loop, H.264 writer (imageio-ffmpeg)
  sysinfo.py       hardware/library info stored with every result
scripts/           CLIs: download_assets, render_demo, prepare_visiodect,
                   prepare_antiuav, train_detector, eval_detection,
                   benchmark_speed, eval_tracking, eval_intercept_sim
tests/             pytest, no weights/GPU needed (FakeDetector)
app.py             Gradio demo (Hugging Face Spaces, CPU); deploy/huggingface/
launcher.py        optional PyQt6 desktop launcher
results/           committed JSON results (+ old training logs in model_reports/)
data/splits/       committed split lists (VisioDECT)
```

## Commands
```bash
pip install -r requirements.txt            # CPU; requirements-gpu.txt for CUDA 12.1
pip install -r requirements-dev.txt        # eval + tests
python scripts/download_assets.py          # weights + sample video
python scripts/render_demo.py              # outputs/demo_annotated.mp4
python -m pytest -q                        # ~2 s
python app.py                              # http://127.0.0.1:7860
```

## Gotchas learned the hard way
- **v1 detector label bug**: `visiodect_yolov8n.pt` (v1) was trained on labels
  whose box *centre* was the drone's top-left corner. It scores 0 mAP on correct
  labels, 0.986 mAP@0.5 if boxes are shifted by half their size
  (`eval_detection.py --shift-half-box`). v2 is retrained on correct labels.
- VisioDECT frames are consecutive video frames: use the **block split**
  (`prepare_visiodect.py`), not a random split, or test scores leak.
- supervision `minimum_matching_threshold` is a distance (1 - IoU): larger = looser.
  Pass real confidences to ByteTrack; detector conf must be ~0.1 so the
  low-confidence second association pass has boxes to use.
- Anti-UAV videos come from a pan-tilt camera: the target can jump >1 box width
  between frames, which breaks IoU association. Use `--oracle` to separate tracker
  errors from detector errors.
- Windows: Ultralytics DataLoader workers hang when code is piped via stdin;
  run real script files with `if __name__ == "__main__"`. Validation uses workers=0.
- On Windows, `platform.release()` says "10" on Windows 11; sysinfo handles it.
- motmetrics 1.4.0 `iou_matrix` breaks on NumPy 2; eval_tracking has its own.
- Ultralytics sets CUDA_VISIBLE_DEVICES when device="cpu"; don't query the GPU after.
- Wikimedia downloads need a User-Agent header (else HTTP 403 page saved as file).
