"""A/B: does common-mode correction change Jungfrau16M calib images, and GLINT's V4 peaks?

mfx101555026 r0013. Arms (separate Detector objects: psana caches calib kwargs per detector
and IGNORES a second, different set with only a warning):
  A  det.raw.calib(evt)                       -- what xtc_qreader.py calls today
  C  det.raw.calib(evt, cmpars=CM)            -- common mode ON
V4: GLINT peakfinder_v4 (copied from glint main b91ce63), numpy path, per-panel, production
settings from xtc_core (min_pix=3 son_min=15 thr_high=10 thr_low=5), mask = pixel_status==0 as
xtc_qreader.good_mask. Same finder both arms, so any peak difference is attributable to CM.
"""
import sys, time, json, socket, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from peakfinder_v4 import PeakFinderV4
import psana
from psana import DataSource

NEV = int(sys.argv[1]) if len(sys.argv) > 1 else 10
CMS = {"doc_default_7_3_200_10": (7, 3, 200, 10)}

print("host", socket.gethostname(), "psana", psana.__file__, "numpy", np.__version__, flush=True)
ds = DataSource(exp="mfx101555026", run=13, max_events=NEV)
run = next(ds.runs())
detA = run.Detector("jungfrau")
cc = detA.calibconst
print("calibconst keys:", sorted(cc.keys()), flush=True)
if "common_mode" in cc:
    cmc = np.asarray(cc["common_mode"][0]).ravel()
    print("experiment common_mode constants:", cmc.tolist(), flush=True)
    CMS["experiment_calibconst"] = tuple(float(x) for x in cmc[:4])
# psana returns the SAME cached Detector for repeated run.Detector() calls, and calib_jungfrau
# caches its kwargs on det.raw._odc at first entry, silently IGNORING later different kwargs.
# So keep one DetCache per arm and swap it in before each call; assert each holds what we expect.
raw = detA.raw
caches = {}
def calib_arm(arm, evt):
    kw = {} if arm == "A" else {"cmpars": CMS[arm]}
    raw._odc = caches.get(arm)
    out = raw.calib(evt, **kw)
    if arm not in caches:
        caches[arm] = raw._odc
        print("cache", arm, "cmps =", caches[arm].cmps, "kwa =", caches[arm].kwa, flush=True)
        if arm == "A": assert caches[arm].cmps is None
        else: assert tuple(caches[arm].cmps) == tuple(CMS[arm])
    assert caches[arm].kwa == kw, (arm, caches[arm].kwa, kw)
    return out

status = np.asarray(cc["pixel_status"][0])
finders = None
def peaks(frame):
    out = []
    for p, f in enumerate(finders):
        pk = f.find(np.asarray(frame[p], np.float32))
        for y, x, sn in zip(np.asarray(pk["y"]), np.asarray(pk["x"]), np.asarray(pk["snr"])):
            out.append((p, float(y), float(x), float(sn)))
    return out

def match(a, b, tol=1.5):
    """greedy nearest match within tol px on the same panel -> (matched_a_idx, matched_b_idx)"""
    usedb = set(); ma = []
    by = {}
    for j, (p, y, x, _) in enumerate(b):
        by.setdefault(p, []).append((j, y, x))
    for i, (p, y, x, _) in enumerate(a):
        best = None
        for j, yy, xx in by.get(p, []):
            if j in usedb: continue
            d = np.hypot(y - yy, x - xx)
            if d <= tol and (best is None or d < best[0]): best = (d, j)
        if best: usedb.add(best[1]); ma.append(i)
    return ma, usedb

def edge_dist(y, x):
    """px to the nearest Jungfrau BANK edge: banks are 256 rows x 64 cols (common-mode groups)"""
    return min(x % 64, 63 - x % 64, y % 256, 255 - y % 256)

SNR = {"matched": [], "onlyA": [], "onlyC": []}
EDGE = {"matched": [], "onlyA": [], "onlyC": []}

rows = []
for i, evt in enumerate(run.events()):
    t0 = time.time()
    fa = calib_arm("A", evt)
    if fa is None:
        print(f"ev {i}: calib None", flush=True); continue
    if finders is None:
        nseg, H, W = fa.shape
        good = ~(status.reshape(-1, nseg, H, W) != 0).any(axis=0)
        finders = [PeakFinderV4(good[p], dtype=np.float32, min_pix=3, son_min=15.0,
                                thr_high=10.0, thr_low=5.0) for p in range(nseg)]
        print("shape", fa.shape, "good frac %.4f" % good.mean(), flush=True)
    ta = time.time() - t0
    pa = peaks(fa)
    rec = {"ev": i, "t_calibA": round(ta, 2), "nA": len(pa)}
    for k, cm in CMS.items():
        t1 = time.time()
        fc = calib_arm(k, evt)
        tc = time.time() - t1
        live = (fa != 0) | (fc != 0)
        d = np.abs(fc.astype(np.float64) - fa)
        ch = d[live] > 1e-6
        pc = peaks(fc)
        ma, mb = match(pa, pc); m = len(ma)
        for i2, q in enumerate(pa):
            key = "matched" if i2 in set(ma) else "onlyA"
            SNR[key].append(q[3]); EDGE[key].append(edge_dist(q[1], q[2]))
        for j2, q in enumerate(pc):
            if j2 not in mb:
                SNR["onlyC"].append(q[3]); EDGE["onlyC"].append(edge_dist(q[1], q[2]))
        rec[k] = {"t_calib": round(tc, 2), "frac_px_changed": float(ch.mean()) if live.any() else 0.0,
                  "absdiff_median_changed": float(np.median(d[live][ch])) if ch.any() else 0.0,
                  "absdiff_p99": float(np.percentile(d[live], 99)), "absdiff_max": float(d.max()),
                  "nC": len(pc), "matched": m, "onlyA": len(pa) - m, "onlyC": len(pc) - m,
                  "hitA": len(pa) >= 6, "hitC": len(pc) >= 6}
    rec["t_total"] = round(time.time() - t0, 1)
    rows.append(rec)
    print(json.dumps(rec), flush=True)

print("SUMMARY", flush=True)
for key in SNR:
    v = np.asarray(SNR[key]); e = np.asarray(EDGE[key])
    if v.size:
        print(f"  {key:8s} n={v.size:5d} snr median {np.median(v):7.1f} p10 {np.percentile(v,10):6.1f}"
              f" frac snr<30 {np.mean(v<30):.3f}  frac within 4px of bank edge {np.mean(e<=4):.3f}", flush=True)
for k in CMS:
    nA = sum(r["nA"] for r in rows); nC = sum(r[k]["nC"] for r in rows); m = sum(r[k]["matched"] for r in rows)
    jac = m / (nA + nC - m) if (nA + nC - m) else 1.0
    print(k, CMS[k], "events", len(rows),
          "px_changed_mean %.4f" % np.mean([r[k]["frac_px_changed"] for r in rows]),
          "peaks A %d C %d matched %d onlyA %d onlyC %d Jaccard %.4f" % (nA, nC, m, nA - m, nC - m, jac),
          "hit flips %d" % sum(r["hitA"] != r["hitC"] for r in [dict(hitA=r[k]["hitA"], hitC=r[k]["hitC"]) for r in rows]),
          flush=True)
