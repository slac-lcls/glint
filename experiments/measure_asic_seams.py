"""Measure the Jungfrau ASIC-seam artifact on raw psana frames -- the provenance for the numbers
quoted in experiments/test_asic_seam_mask.py.

THIS FILE EXISTS BECAUSE THE PREVIOUS NUMBERS COULD NOT BE RE-DERIVED. That docstring carried an
"~12% expected by area" that reconstructs from no natural definition of a seam (the interior-seam
area fraction is 0.49% at +-0 px, 1.5% at +-1, 2.4% at +-2, and only reaches 10% at +-10 px, which is
not a seam), and there was no script left to check it against. A measured claim in a docstring needs
the thing that measured it committed next to it, or it decays into folklore.

Not run by CI: it needs psana and a real Jungfrau run, neither of which the CPU runner has. Run it by
hand on an LCLS analysis node when the claim needs re-checking.

The claim being re-sourced: with the facility mask alone (psana `_mask_from_status()` &
`_mask_edges(width=2)`), peakfinder8 returns a large fraction of its peaks sitting exactly on an
interior ASIC seam -- far above chance -- and masking the seams collapses the median peak count.
The artifact is a property of the Jungfrau's tiling, not of any sample, so it must reproduce on any
Jungfrau 16M run; this measures it on one that carries no embargo.

DEFINITIONS, stated because the numbers mean nothing without them:
  interior seam   ss = 256 (mod 512), and fs in {256, 512, 768}. Module perimeters are EXCLUDED --
                  those are _mask_edges' job, and counting them would flatter the result.
  "exactly on"    distance to the nearest interior seam line <= 0.5 px
  "within +-1"    <= 1.0 px
  chance          the fraction of UNMASKED pixels at the same distance, computed from the actual
                  mask rather than analytically, so the radial cut and the bad-pixel map are
                  accounted for. This is the honest denominator.

The env's own compiled peakfinder8.so shadows the vendored module, and `import glint.peakfinder8`
pulls in glint/__init__ -> .detector -> torch, which psana2 does not have. So both modules are loaded
straight from their files, with `radial` pre-registered under the name peakfinder8.py falls back to.
"""
import importlib.util, sys, time
import numpy as np

H = "/sdf/home/s/smarches/h2h"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


_load("radial", f"{H}/glint/radial.py")
PF8 = _load("gpf8", f"{H}/glint/peakfinder8.py")

from psana import DataSource

RUN = int(sys.argv[1]) if len(sys.argv) > 1 else 13
NFRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 200
OUT = sys.argv[3] if len(sys.argv) > 3 else f"{H}/work/seam_r{RUN:04d}.npz"

SS, FS = 32 * 512, 1024
row_seams = np.arange(256, SS, 512)                      # interior only: 256, 768, 1280, ...
col_seams = np.array([256, 512, 768])


def seam_dist(ss, fs):
    dr = np.min(np.abs(ss[:, None] - row_seams[None, :]), axis=1)
    dc = np.min(np.abs(fs[:, None] - col_seams[None, :]), axis=1)
    return np.minimum(dr, dc)


def seam_mask(width=2):
    m = np.ones((SS, FS), bool)
    for b in row_seams:
        m[max(0, b - width):b + width + 1, :] = False
    for b in col_seams:
        m[:, max(0, b - width):b + width + 1] = False
    return m


ds = DataSource(exp="mfx101555026", run=RUN, max_events=NFRAMES)
r = next(ds.runs())
jf = r.Detector("jungfrau")
mstat = jf.raw._mask_from_status().astype(bool)
medge = jf.raw._mask_edges(width=2).astype(bool)
facility = (mstat & medge).reshape(SS, FS)

# radius in pixels about the detector centre, from the panel geometry psana carries
x, y, z = jf.raw._pixel_coords() if hasattr(jf.raw, "_pixel_coords") else (None, None, None)
if x is None:
    yy, xx = np.mgrid[0:SS, 0:FS]                        # fallback: slab coords are enough for rings
    rpix = np.hypot(yy - SS / 2.0, xx - FS / 2.0)
else:
    rpix = (np.sqrt(np.asarray(x, float) ** 2 + np.asarray(y, float) ** 2) / 75.0).reshape(SS, FS)

radial_ok = (rpix >= 50.0) & (rpix <= 3000.0)
mask_fac = facility & radial_ok                          # the "facility mask alone" arm
mask_seam = mask_fac & seam_mask(2)                      # the same, plus the seam mask

gy, gx = np.nonzero(mask_fac)
d_all = seam_dist(gy.astype(float), gx.astype(float))
chance = {t: float((d_all <= t).mean()) for t in (0.5, 1.0)}
print("frames requested %d | facility mask keeps %.4f of pixels, +seam mask %.4f (cost %.2f%%)"
      % (NFRAMES, mask_fac.mean(), mask_seam.mean(),
         100 * (mask_fac.sum() - mask_seam.sum()) / mask_fac.sum()), flush=True)
print("chance by area among UNMASKED pixels: <=0.5 px %.3f%%   <=1.0 px %.3f%%"
      % (100 * chance[0.5], 100 * chance[1.0]), flush=True)

base = dict(nbin=1000, thr_snr=7.0, min_snr=7.0, min_pix=4, max_pix=200,
            r_min=50.0, n_iter=3, thr_adu=50.0)
pf_fac = PF8.PeakFinder8(rpix, mask=mask_fac, **base)
pf_seam = PF8.PeakFinder8(rpix, mask=mask_seam, **base)

n_fac, n_seam, dists = [], [], []
t0 = time.time(); nfr = 0
for evt in r.events():
    cal = jf.raw.calib(evt)
    if cal is None:
        continue
    img = np.asarray(cal, dtype=np.float32).reshape(SS, FS)
    a = pf_fac.find(img); b = pf_seam.find(img)
    n_fac.append(int(a["x"].size)); n_seam.append(int(b["x"].size))
    if a["x"].size:
        dists.append(seam_dist(np.asarray(a["y"], float), np.asarray(a["x"], float)))
    nfr += 1
    if nfr % 25 == 0:
        print("  %d frames, %.0fs" % (nfr, time.time() - t0), flush=True)

d = np.concatenate(dists) if dists else np.zeros(0)
n_fac = np.array(n_fac); n_seam = np.array(n_seam)
print("\n=== mfx101555026 r%04d, %d frames, %d peaks (facility mask alone) ===" % (RUN, nfr, d.size))
for t in (0.5, 1.0):
    print("  within %.1f px of an interior seam: %.1f%%   (chance %.1f%%)"
          % (t, 100 * (d <= t).mean(), 100 * chance[t]))
print("  median peak-to-nearest-seam distance: %.1f px" % np.median(d))
print("  median peaks/frame  facility mask %.0f  ->  + seam mask %.0f"
      % (np.median(n_fac), np.median(n_seam)))
np.savez_compressed(OUT, d=d, n_fac=n_fac, n_seam=n_seam, nfr=nfr,
                    chance05=chance[0.5], chance10=chance[1.0])
print("saved ->", OUT)
