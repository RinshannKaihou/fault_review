#!/usr/bin/env python3
"""Render the README animation at 2x pixel density and encode it.

  python3 docs/media/render_daily_loop.py [--skill DIR] [--chrome PATH] [--work DIR] [--skip-render] [--avif-only]

Per theme: build_daily_loop.py -> config -> live-panel page rendered at deviceScaleFactor 2 (the skill's render.py
is fixed at 1; a 1280 px render squeezed to 900 px and saved as lossy WebP is what made the first version blurry)
-> 12 fps PNG frames -> animated AVIF at 1800 px (README shows it at width 900, sharp on 2x screens) and a 900 px
WebP fallback for browsers without animated AVIF. Needs Pillow with AVIF (>= 11.3), ffmpeg with libwebp, Chrome.
"""
import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
THEMES = {"dark": "terminal-dark", "light": "light-pastel"}
FPS, AVIF_W, WEBP_W = 12, 1800, 900


def render(skill, chrome, cfg, frames):
    sys.path.insert(0, str(Path(skill) / "scripts"))
    import livepanel as lp
    page = str(Path(frames).parent / (Path(frames).name + ".html"))
    w, h, dur, _ = lp.canvas(lp.build_page(cfg, page, None))
    os.makedirs(frames, exist_ok=True)
    with lp.Chrome(lp.find_chrome(chrome), w, h) as br:
        br.cmd("Emulation.setDeviceMetricsOverride", {"width": w, "height": h, "deviceScaleFactor": 2, "mobile": False})
        br.open("file://" + os.path.abspath(page) + "?manual")
        for i in range(int(round(FPS * dur))):
            br.seek(i / FPS)
            Path(frames, f"f_{i:04d}.png").write_bytes(br.shot())
        err = br.eval("window.__error||''")
        if err:
            sys.exit("page errors: " + err)


def to_avif(frames, out):
    from PIL import Image
    files = sorted(glob.glob(os.path.join(frames, "f_*.png")))

    def load(f):
        im = Image.open(f).convert("RGB")
        return im.resize((AVIF_W, im.height * AVIF_W // im.width), Image.LANCZOS)

    dur = [(83, 83, 84)[i % 3] for i in range(len(files))]   # 12 fps: exactly 250 ms per 3 frames, loop stays 27 s
    # 4:4:4 keeps coloured text crisp; q70 is visually indistinguishable from the PNG frames at 2x
    load(files[0]).save(out, save_all=True, append_images=(load(f) for f in files[1:]), duration=dur, loop=0,
                        quality=70, speed=7, subsampling="4:4:4")


def to_webp(frames, out):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", os.path.join(frames, "f_%04d.png"),
                    "-vf", f"scale={WEBP_W}:-1:flags=lanczos", "-c:v", "libwebp_anim", "-loop", "0", "-lossless", "0",
                    "-quality", "60", "-compression_level", "6", "-preset", "drawing", out], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skill", default=os.path.expanduser("~/.claude/skills/live-panel-skill"))
    ap.add_argument("--chrome", help="Chrome / chrome-headless-shell (default: search PATH)")
    ap.add_argument("--work", default="/tmp/daily-loop", help="configs and PNG frames go here")
    ap.add_argument("--skip-render", action="store_true", help="reuse frames already in --work")
    ap.add_argument("--avif-only", action="store_true", help="leave the WebP fallbacks as they are")
    a = ap.parse_args()
    for short, preset in THEMES.items():
        cfg, frames = os.path.join(a.work, f"{short}.json"), os.path.join(a.work, short)
        if not a.skip_render:
            os.makedirs(a.work, exist_ok=True)
            subprocess.run([sys.executable, str(HERE / "build_daily_loop.py"), preset, cfg], check=True, stdout=subprocess.DEVNULL)
            render(a.skill, a.chrome, cfg, frames)
        to_avif(frames, str(HERE / f"daily-loop-{short}.avif"))
        if not a.avif_only:
            to_webp(frames, str(HERE / f"daily-loop-{short}.webp"))
        print(f"{short}: " + ", ".join(f"{p.name} {p.stat().st_size / 1e6:.1f} MB"
                                        for p in sorted(HERE.glob(f"daily-loop-{short}.*"))))


if __name__ == "__main__":
    main()
