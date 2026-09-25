"""Build and publish a free *static* Hugging Face Space that shows the pipeline output.

Hugging Face requires a paid PRO plan to host Gradio/Docker apps, but static
Spaces (plain HTML) are free. This page plays the annotated demo video rendered by
the real pipeline and lists the measured results. Every number on the page is read
from results/*.json at build time, so the page cannot drift from the results.

    # main video: your clip, annotated (outputs/demo_annotated.mp4)
    python scripts/render_demo.py --source data/samples/sky_drones.mp4 --no-timing
    # second example: the 24-drone clip
    python scripts/render_demo.py --output outputs/magiclab_annotated.mp4 --no-timing
    python deploy/huggingface/deploy_static_space.py --space <user>/iamars-drone-tracking
    python deploy/huggingface/deploy_static_space.py --space <user>/<name> --build-only  # local preview

Videos that are missing locally are skipped.
"""

import argparse
import html
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = "https://github.com/aviraj1805/intelligent-aerial-monitoring"
BLOB = f"{REPO}/blob/main"
OUT = ROOT / "outputs"

_CC_BY = '<a href="https://creativecommons.org/licenses/by/3.0/" target="_blank" rel="noopener">CC BY 3.0</a>'
_MAGICLAB = ('<a href="https://commons.wikimedia.org/wiki/File:MagicLab_-_24_Drone_Flight.webm" '
             'target="_blank" rel="noopener">Wikimedia Commons</a>')

# (local file, published name, local poster, published poster, heading, caption HTML)
VIDEOS = [
    (OUT / "demo_annotated.mp4", "demo_annotated.mp4", OUT / "poster.jpg", "poster.jpg",
     "Four drones against open sky",
     "Pipeline output: each drone keeps its ID (1 to 4) for the whole clip. Boxes show track ID "
     "and confidence, lines show the recent path, orange marks the predicted path. Input clip "
     "supplied by the project author."),
    (OUT / "magiclab_annotated.mp4", "demo_magiclab.mp4", OUT / "poster_magiclab.jpg", "poster_magiclab.jpg",
     "Second example: 24 drones in formation",
     f'Pipeline output on a swarm. Source clip: "MagicLab – 24 Drone Flight" by Marco Tempest, '
     f"{_CC_BY}, via {_MAGICLAB}, trimmed and annotated."),
]


def available_videos() -> list[tuple]:
    return [v for v in VIDEOS if v[0].exists()]


def videos_html(start: int = 0, end: int | None = None) -> str:
    parts = []
    vids = available_videos()
    for i in range(start, min(end if end is not None else len(vids), len(vids))):
        _, name, poster_src, poster, heading, caption = vids[i]
        poster_attr = f' poster="{poster}"' if poster_src.exists() else ""
        head = f"<h2>{html.escape(heading)}</h2>" if i else f'<h2 class="first">{html.escape(heading)}</h2>'
        parts.append(f"""{head}
  <video src="{name}"{poster_attr} controls muted playsinline preload="metadata"></video>
  <p class="caption">{caption}</p>""")
    return "\n  ".join(parts)


def r(name: str) -> dict:
    return json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))


def link(path: str, text: str) -> str:
    return f'<a href="{BLOB}/{path}" target="_blank" rel="noopener">{html.escape(text)}</a>'


def table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def build_page() -> str:
    det_t, det_o = r("detection_v2_visiodect_block_test"), r("detection_v2_antiuav_rgb_test")
    gpu, cpu = r("benchmark_gpu"), r("benchmark_cpu")
    trk = {k: r(k) for k in ("tracking_antiuav_test", "tracking_antiuav_test_oracle", "tracking_multiuav_train_oracle")}
    sim = r("intercept_simulation")
    src = lambda n: link(f"results/{n}.json", "json")

    detection = table(
        ["Test set", "Images", "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "Source"],
        [["VisioDECT held-out test (block split)", f"{det_t['images']:,}", f"{det_t['mAP50']:.3f}",
          f"{det_t['mAP50_95']:.3f}", f"{det_t['precision']:.3f}", f"{det_t['recall']:.3f}",
          src("detection_v2_visiodect_block_test")],
         ["Anti-UAV-RGBT frames, never seen in training", f"{det_o['images']:,}", f"{det_o['mAP50']:.3f}",
          f"{det_o['mAP50_95']:.3f}", f"{det_o['precision']:.3f}", f"{det_o['recall']:.3f}",
          src("detection_v2_antiuav_rgb_test")]])

    speed = table(
        ["Device", "Mean latency / frame", "95th percentile", "Pipeline FPS", "Source"],
        [[gpu["system"]["gpu"], f"{gpu['latency_ms']['total']['mean']} ms", f"{gpu['latency_ms']['total']['p95']} ms",
          f"{gpu['pipeline_fps']}", src("benchmark_gpu")],
         [f"{cpu['system']['cpu']} (CPU only)", f"{cpu['latency_ms']['total']['mean']} ms",
          f"{cpu['latency_ms']['total']['p95']} ms", f"{cpu['pipeline_fps']}", src("benchmark_cpu")]])

    labels = {"tracking_antiuav_test": "Anti-UAV-RGBT test, IAMARS detector + ByteTrack",
              "tracking_antiuav_test_oracle": "Same videos, ground-truth boxes (tracker upper bound)",
              "tracking_multiuav_train_oracle": "MultiUAV swarm videos, ground-truth boxes"}
    tracking = table(
        ["Evaluation", "Videos", "MOTA", "IDF1", "ID switches", "Source"],
        [[labels[k], str(v["videos"]), f"{v['overall']['mota']:.3f}", f"{v['overall']['idf1']:.3f}",
          f"{int(v['overall']['num_switches']):,}", src(k)] for k, v in trk.items()])

    names = {"straight": "Straight", "turning": "Turning", "jinking": "Jinking (random manoeuvres)"}
    simt = table(
        ["Flight pattern", "Hit rate (≤ 1 m)", "Median miss", "Hit rate with true state"],
        [[names[k], f"{100 * v['filtered']['hit_rate_within_radius']:.1f} %", f"{v['filtered']['miss_m_median']:.2f} m",
          f"{100 * v['oracle']['hit_rate_within_radius']:.1f} %"] for k, v in sim["results"].items()])
    cfg = sim["config"]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IAMARS: drone detection and tracking</title>
<style>
  :root {{ --bg:#fbfbf9; --fg:#1d2226; --muted:#5b646b; --line:#dcdfe2; --accent:#0f7b57; --card:#ffffff; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#15181b; --fg:#e7eaec; --muted:#a3acb3; --line:#2d3338; --accent:#3fcf98; --card:#1c2024; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
  main {{ max-width:960px; margin:0 auto; padding:32px 20px 64px; }}
  h1 {{ font-size:1.9rem; margin:0 0 6px; letter-spacing:-.01em; }}
  h2 {{ font-size:1.25rem; margin:40px 0 10px; }}
  h2.first {{ margin-top:28px; }}
  p {{ margin:8px 0; }}
  .lead {{ color:var(--muted); font-size:1.05rem; }}
  .links a {{ display:inline-block; margin:10px 14px 0 0; }}
  a {{ color:var(--accent); }}
  video {{ width:100%; max-width:100%; border-radius:10px; border:1px solid var(--line); background:#000; margin-top:8px; }}
  .caption {{ color:var(--muted); font-size:.85rem; }}
  .pipeline {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:12px 0; }}
  .step {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px 12px; font-size:.92rem; }}
  .arrow {{ color:var(--muted); }}
  .scroll {{ overflow-x:auto; }}
  table {{ border-collapse:collapse; width:100%; font-size:.93rem; background:var(--card); }}
  th, td {{ border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; white-space:nowrap; }}
  th {{ color:var(--muted); font-weight:600; }}
  td:first-child {{ white-space:normal; min-width:220px; }}
  .note {{ color:var(--muted); font-size:.9rem; }}
  pre {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; overflow-x:auto; font-size:.85rem; }}
  .sim {{ border-left:3px solid var(--accent); padding-left:12px; }}
</style>
</head>
<body>
<main>
  <h1>IAMARS: drone detection, tracking and trajectory prediction</h1>
  <p class="lead">YOLOv8n finds small drones in video, ByteTrack keeps a stable ID for each one,
  a Kalman filter estimates position and velocity, and a constant-velocity model predicts the
  next half second. Every number on this page was produced by a script in the repository.</p>
  <div class="links">
    <a href="{REPO}" target="_blank" rel="noopener">Code on GitHub</a>
    {link("docs/MODEL_CARD.md", "Model card")}
    <a href="{REPO}/releases/tag/weights-v1" target="_blank" rel="noopener">Model weights</a>
  </div>

  {videos_html(0, 1)}

  <h2>Pipeline</h2>
  <div class="pipeline">
    <span class="step">Video frame</span><span class="arrow">→</span>
    <span class="step">YOLOv8n detection</span><span class="arrow">→</span>
    <span class="step">ByteTrack IDs</span><span class="arrow">→</span>
    <span class="step">Kalman filter</span><span class="arrow">→</span>
    <span class="step">Trajectory prediction</span>
  </div>

  <h2>Detection</h2>
  {detection}
  <p class="note">The second row measures the domain gap: the model was trained only on VisioDECT.</p>

  <h2>Speed</h2>
  {speed}
  <p class="note">Detect + track + Kalman + predict per frame, batch 1, 1280 × 720 video, laptop on AC power.</p>

  <h2>Tracking</h2>
  {tracking}
  <p class="note">MOTA and IDF1 from py-motmetrics (match at IoU ≥ 0.5). With perfect boxes the tracker keeps
  identities well on static-camera swarm videos but not on Anti-UAV, whose pan-tilt camera makes boxes jump.</p>

  {videos_html(1, 2)}

  <h2>Intercept-point prediction (simulation)</h2>
  <div class="sim">
  <p><strong>Simulation only.</strong> {sim['trials_per_scenario']:,} trials per pattern; 3D position measured
  at 30 Hz with {cfg['meas_std']} m noise; after {cfg['warmup_s']} s a Kalman filter estimate is used to aim a
  {cfg['interceptor_speed']:.0f} m/s interceptor. Hit = within {cfg['hit_radius']} m of the true position.</p>
  {simt}
  <p class="note">Source: {link("results/intercept_simulation.json", "results/intercept_simulation.json")}</p>
  </div>

  <h2>Run it yourself</h2>
  <pre>git clone {REPO}.git
cd intelligent-aerial-monitoring
pip install -r requirements.txt
python scripts/download_assets.py
python app.py            # interactive web demo at http://127.0.0.1:7860</pre>
  <p class="note">This page is static because hosted Python apps on Hugging Face require a paid plan.
  The interactive Gradio app runs locally with the commands above.</p>
</main>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, help="<hf-username>/<space-name>")
    ap.add_argument("--build-only", action="store_true", help="write the page to outputs/static_space/")
    args = ap.parse_args()

    if not available_videos():
        raise SystemExit("No rendered videos found; run scripts/render_demo.py first.")

    readme = "\n".join([
        "---", "title: IAMARS Drone Tracking", "emoji: 🛩️", "colorFrom: green", "colorTo: gray",
        "sdk: static", "app_file: index.html", "pinned: false", "---", "",
        f"Showcase page for IAMARS. Code and results: {REPO}", ""])

    def write(stage: Path):
        (stage / "index.html").write_text(build_page(), encoding="utf-8")
        (stage / "README.md").write_text(readme, encoding="utf-8")
        for src, name, poster_src, poster, _, _ in available_videos():
            shutil.copy2(src, stage / name)
            if poster_src.exists():
                shutil.copy2(poster_src, stage / poster)

    if args.build_only:
        out = ROOT / "outputs" / "static_space"
        out.mkdir(parents=True, exist_ok=True)
        write(out)
        print(f"built -> {out / 'index.html'}")
        return

    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        write(stage)
        api = HfApi()
        api.create_repo(args.space, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(folder_path=str(stage), repo_id=args.space, repo_type="space",
                          commit_message="Publish IAMARS showcase page")
    owner, name = args.space.split("/")
    print(f"Space page: https://huggingface.co/spaces/{args.space}")
    print(f"Direct page: https://{owner.lower()}-{name.lower().replace('_', '-')}.static.hf.space")


if __name__ == "__main__":
    main()
