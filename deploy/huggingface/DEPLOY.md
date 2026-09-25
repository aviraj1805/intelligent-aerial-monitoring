# Hosting the IAMARS demo

Hugging Face requires a paid PRO plan to host Gradio or Docker apps (even on the
free CPU hardware). Static Spaces, which serve plain HTML, are free. So there are
three options:

## 1. Free showcase page (used for the live link)

A static Space that plays the annotated output video rendered by the real pipeline and
lists the measured results. The numbers are read from `results/*.json` when the page is
built, so the page cannot drift from the results.

```bash
huggingface-cli login                                  # once, with a write token
# main video: the four-drone sky clip (kept locally at data/samples/sky_drones.mp4)
python scripts/render_demo.py --source data/samples/sky_drones.mp4 --no-timing
# second example: the 24-drone clip
python scripts/render_demo.py --output outputs/magiclab_annotated.mp4 --no-timing
python deploy/huggingface/deploy_static_space.py --space <user>/iamars-drone-tracking
```

The sky clip is not in the repository; the page builder skips any video that is missing.

Live page: https://huggingface.co/spaces/AvirajV/iamars-drone-tracking

`--no-timing` hides the FPS/latency readout in the video panel, because speed at render
time depends on the machine's state (for example battery power). Measured speed is in
`results/benchmark_*.json`.

## 2. Interactive Gradio Space (needs Hugging Face PRO)

```bash
python deploy/huggingface/deploy_space.py --space <user>/<space-name>
```

Uploads `app.py`, the `iamars` package, requirements, the Space config (`README.md`,
`packages.txt`), the v2 weights, the example clip and the pre-rendered video.

## 3. Run it on your own computer

```bash
python app.py            # http://127.0.0.1:7860
python app.py --share    # also a temporary public *.gradio.live link while it runs
```

The `--share` tunnel can be blocked by some networks or antivirus software; on the
development laptop it failed with "Could not create share link".
