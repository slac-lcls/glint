#!/usr/bin/env python3
"""Frame-by-frame headless-Chrome capture of a replay page, and GIF / still assembly.

This is the recipe docs/streaming_replay.md describes for the 480-frame README animation, committed
so the media can be regenerated. It drives the built replay page's own switches: `?print=1` hides the
transport and freezes the state, `?warm=W` pre-advances the clock by W seconds of playback, `?theme=`
picks the palette. Frame i (1-based on the clock) is reached by the W that makes the page's advance
loop `for(acc=0; acc<W; acc+=0.033) advance(0.033)` show exactly i frames -- computed here by replaying
that loop in IEEE doubles (frames shown = sum of floor(PLAY_FPS*0.033 + carry)), so a captured frame IS
trace record i-1 and never an interpolation. Never capture past the trace length: the page wraps.

One Chrome launch per frame (Chrome never exits on its own because of the rAF loop: poll for a
size-stable PNG, then kill), a fresh --user-data-dir per launch, N in parallel. A blank-frame guard
(grey-level std over two canvas regions) recaptures a frame whose canvases had not drawn.

  # 1. frames (every 2nd trace record, both themes), 4 Chrome in parallel
  python3 capture_gif.py frames glint_streaming_multicell_real.html --theme dark  --last 900 --step 2 --out cap_dark
  python3 capture_gif.py frames glint_streaming_multicell_real.html --theme light --last 900 --step 2 --out cap_light
  # 2. GIF: crop the footer-free top of the page, 900 px wide, ONE global 256-colour palette, no dither
  python3 capture_gif.py gif cap_dark  ../media/streaming_multicell_dark.gif  --crop 6,0,1314,780 --width 900 --fps 12
  # 3. stills at device-scale-factor 2 for the filmstrip (named <prefix><theme>_f<frame>.png)
  python3 capture_gif.py stills glint_streaming_multicell_real.html --theme dark --frames 0,5,146,420,900 --out stills --prefix multicell_

Frame numbers are checked against the page's trace (the `const TRACE=[...]` array the builder embeds):
a frame past the last record is refused before Chrome starts, because the page would wrap and the file
would carry a frame number it does not show.

Requires Google Chrome (macOS path below, or CHROME=...), Pillow and NumPy; no ffmpeg.
"""
import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
PLAY_FPS, DT = 12, 0.033                      # the pages' playback rate and advance step
W, H = 1320, 896                              # window; the crop box below is in these pixels at DSF 1


def sim_iters(n):
    """frames shown and clock after n iterations of advance(0.033) -- an exact replay of the JS."""
    carry = 0.0; frames = 0; t = 0.0
    for _ in range(n):
        nf = PLAY_FPS * DT + carry; N = math.floor(nf); carry = nf - N
        frames += N; t += DT
    return frames, t


def js_iter_count(warm):
    acc = 0.0; n = 0
    while acc < warm:
        acc += DT; n += 1
        if n > 20000:
            break
    return n


def warm_for_frame(i):
    """(warm string, iterations, clock) that shows exactly i frames with the clock nearest i/PLAY_FPS;
    frame 0 = the page with no ?warm."""
    if i == 0:
        return None, 0, 0.0
    target = i / PLAY_FPS
    best = None
    for n in range(1, 4 * i + 8):
        f, t = sim_iters(n)
        if f > i:
            break
        if f == i and (best is None or abs(t - target) < abs(best[1] - target)):
            best = (n, t)
    if best is None:
        raise RuntimeError(f"no iteration count shows frame {i}")
    n, t = best
    acc = 0.0; accs = [0.0]
    for _ in range(n):
        acc += DT; accs.append(acc)
    lo, hi = accs[n - 1], accs[n]              # need lo < warm <= hi
    for dec in (2, 3, 4):
        k0 = math.floor(lo * 10 ** dec) + 1
        for k in range(k0, k0 + 40):
            s = f"{k / 10 ** dec:.{dec}f}"
            if lo < float(s) <= hi and js_iter_count(float(s)) == n:
                return s, n, t
    raise RuntimeError(f"no ?warm value isolates frame {i}")


def region_std(png, box):
    im = Image.open(png).convert("L").crop(box)
    return float(np.asarray(im, dtype=np.float32).std())


def capture_one(page_url, theme, i, out_png, dsf=1, profiles=None, tag=""):
    warm, n, t = warm_for_frame(i)
    url = f"{page_url}?theme={theme}&print=1" + (f"&warm={warm}" if warm else "")
    prof = tempfile.mkdtemp(prefix=f"p{i:04d}{tag}_", dir=profiles)
    tmp_png = out_png + ".part.png"
    if os.path.exists(tmp_png):
        os.remove(tmp_png)
    cmd = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
           f"--user-data-dir={prof}", f"--force-device-scale-factor={dsf}",
           f"--window-size={W},{H}", "--virtual-time-budget=1500", f"--screenshot={tmp_png}", url]
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok = False
    while time.time() - t0 < 30:
        if os.path.exists(tmp_png) and os.path.getsize(tmp_png) > 1000:
            s1 = os.path.getsize(tmp_png); time.sleep(0.25)
            if os.path.getsize(tmp_png) == s1:
                ok = True; break
        if p.poll() is not None and not os.path.exists(tmp_png):
            break
        time.sleep(0.1)
    p.kill()
    try:
        p.wait(5)
    except Exception:                                    # noqa: BLE001
        pass
    shutil.rmtree(prof, ignore_errors=True)
    if ok:
        os.replace(tmp_png, out_png)
    return dict(frame=i, warm=warm, n_iter=n, clock=round(t, 3), ok=ok, secs=round(time.time() - t0, 2))


def page_url(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise SystemExit(f"page not found: {path}")
    return "file://" + path


def trace_length(path):
    """Number of records in the page's embedded trace (`const TRACE=[...];`), i.e. the last frame the page
    shows before it wraps."""
    with open(path) as fh:
        html = fh.read()
    k = html.find("const TRACE=")
    if k < 0:
        raise SystemExit(f"{path}: no `const TRACE=` array -- not a built replay page")
    end = html.find("];", k)
    return len(json.loads(html[k + len("const TRACE="):end + 1]))


def check_frames(path, frames):
    nf = trace_length(path)
    bad = [i for i in frames if i < 0 or i > nf]
    if bad:
        raise SystemExit(f"frame(s) {bad} outside this page's trace (0..{nf}); the page wraps past {nf}")
    return nf


def cmd_frames(a):
    url = page_url(a.page)
    os.makedirs(a.out, exist_ok=True)
    profiles = os.path.join(a.out, "_profiles"); os.makedirs(profiles, exist_ok=True)
    frames = list(range(a.first, a.last + 1, a.step))
    if a.last not in frames:
        frames.append(a.last)
    check_frames(a.page, frames)
    pipe_box = tuple(int(x) for x in a.pipe_box.split(","))
    stream_box = tuple(int(x) for x in a.stream_box.split(","))
    T0 = time.time()

    def job(i):
        out = os.path.join(a.out, f"f_{i:04d}.png")
        r = capture_one(url, a.theme, i, out, profiles=profiles)
        r["blank"] = False; r["recaptured"] = False
        if r["ok"]:
            ps, ss = region_std(out, pipe_box), region_std(out, stream_box)
            r["pipe_std"], r["stream_std"] = round(ps, 2), round(ss, 2)
            blank = ps < 2 or (i >= 1 and ss < 2)
            if blank:
                r["recaptured"] = True
                r2 = capture_one(url, a.theme, i, out, profiles=profiles, tag="r")
                if r2["ok"]:
                    ps, ss = region_std(out, pipe_box), region_std(out, stream_box)
                    r["pipe_std2"], r["stream_std2"] = round(ps, 2), round(ss, 2)
                    blank = ps < 2 or (i >= 1 and ss < 2)
                else:
                    blank = True
            r["blank"] = blank
            if blank:
                os.replace(out, out + ".BLANK")
        return r

    with ThreadPoolExecutor(a.parallel) as ex:
        results = list(ex.map(job, frames))
    shutil.rmtree(profiles, ignore_errors=True)
    wall = time.time() - T0
    with open(os.path.join(a.out, "capture_log.json"), "w") as fh:
        json.dump(dict(page=os.path.abspath(a.page), theme=a.theme, wall_s=round(wall, 1), parallel=a.parallel,
                       window=[W, H], dsf=1, pipe_box=pipe_box, stream_box=stream_box, step=a.step,
                       frames=results), fh, indent=1)
    good = sum(r["ok"] and not r["blank"] for r in results)
    print(f"theme={a.theme} frames {a.first}..{a.last} step {a.step}: {good} good, "
          f"{sum(r['blank'] for r in results)} blank-dropped, {sum(not r['ok'] for r in results)} failures, wall {wall:.0f}s")
    return 0 if good == len(frames) else 1


def cmd_gif(a):
    files = sorted(glob.glob(os.path.join(a.src, "f_*.png")))
    if not files:
        raise SystemExit(f"no f_*.png under {a.src}")
    crop = tuple(int(x) for x in a.crop.split(",")) if a.crop else None

    def load(f):
        im = Image.open(f).convert("RGB")
        if crop:
            im = im.crop(crop)
        if a.width:
            im = im.resize((a.width, round(im.height * a.width / im.width)), Image.LANCZOS)
        return im

    t = time.time()
    ims = [load(f) for f in files]
    samp = ims[::max(1, len(ims) // 9)]
    mosaic = Image.new("RGB", (samp[0].width, samp[0].height * len(samp)))
    for k, s in enumerate(samp):
        mosaic.paste(s, (0, k * s.height))
    pal = mosaic.quantize(256, method=Image.Quantize.MEDIANCUT)
    fr = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
    fr[0].save(a.out, save_all=True, append_images=fr[1:], duration=round(1000 / a.fps), loop=0,
               optimize=True, disposal=1)
    print(f"{a.out}: {len(fr)} frames {fr[0].size} at {a.fps} fps ({round(1000 / a.fps)} ms requested; GIF stores centiseconds) "
          f"-> {os.path.getsize(a.out) / 1e6:.2f} MB, {time.time() - t:.0f}s")
    return 0


def cmd_stills(a):
    url = page_url(a.page)
    os.makedirs(a.out, exist_ok=True)
    profiles = os.path.join(a.out, "_profiles"); os.makedirs(profiles, exist_ok=True)
    crop = tuple(int(x) for x in a.crop.split(",")) if a.crop else None
    frames = [int(x) for x in a.frames.split(",")]
    check_frames(a.page, frames)
    failed = []
    for i in frames:
        out = os.path.join(a.out, f"{a.prefix}{a.theme}_f{i:04d}.png")
        r = capture_one(url, a.theme, i, out, dsf=2, profiles=profiles)
        if r["ok"] and crop:
            im = Image.open(out); im.crop(tuple(2 * c for c in crop)).save(out)
        if not r["ok"]:
            failed.append(i)
        print(f"frame {i}: warm={r['warm']} clock={r['clock']} ok={r['ok']} -> {out}")
    shutil.rmtree(profiles, ignore_errors=True)
    if failed:
        print(f"FAILED: {len(failed)} still(s) not captured: {failed}")
    return 1 if failed else 0


def cmd_warm(a):
    for i in (int(x) for x in a.frames.split(",")):
        w, n, t = warm_for_frame(i)
        print(f"frame {i}: ?warm={w} ({n} iterations, clock {t:.3f} s)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames", help="capture one PNG per animation frame")
    f.add_argument("page"); f.add_argument("--theme", default="dark", choices=("dark", "light"))
    f.add_argument("--first", type=int, default=0); f.add_argument("--last", type=int, required=True)
    f.add_argument("--step", type=int, default=1); f.add_argument("--out", required=True)
    f.add_argument("--parallel", type=int, default=4)
    f.add_argument("--pipe-box", default="40,120,700,700", help="blank guard: schematic region x0,y0,x1,y1")
    f.add_argument("--stream-box", default="760,560,1290,690", help="blank guard: composition chart interior")
    f.set_defaults(fn=cmd_frames)
    g = sub.add_parser("gif", help="assemble captured frames into one GIF (global palette, no dither)")
    g.add_argument("src"); g.add_argument("out")
    g.add_argument("--crop", default=None, help="x0,y0,x1,y1 in capture pixels (footer-free top of the page)")
    g.add_argument("--width", type=int, default=900); g.add_argument("--fps", type=float, default=12)
    g.set_defaults(fn=cmd_gif)
    s = sub.add_parser("stills", help="device-scale-factor-2 stills of chosen frames")
    s.add_argument("page"); s.add_argument("--theme", default="dark", choices=("dark", "light"))
    s.add_argument("--frames", required=True, help="comma-separated frame numbers")
    s.add_argument("--out", required=True); s.add_argument("--crop", default=None)
    s.add_argument("--prefix", default="", help="filename prefix: <prefix><theme>_f<frame>.png (the committed "
                   "multicell stills use --prefix multicell_)")
    s.set_defaults(fn=cmd_stills)
    w = sub.add_parser("warm", help="print the ?warm= value for given frames")
    w.add_argument("--frames", required=True); w.set_defaults(fn=cmd_warm)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
