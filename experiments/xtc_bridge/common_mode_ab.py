#!/usr/bin/env python3
"""Reproduce the Jungfrau psana2 common-mode A/B measurement.

This must run in the psana2 environment.  The Jungfrau calibration object caches
its first keyword arguments in ``raw._odc``; keep one cache per arm and swap it
before every call, otherwise both arms silently use the first calibration.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xtc_core  # noqa: E402


def _cache_cmpars(cache):
    """Return cached cmpars when psana exposes the usual dict-shaped cache."""
    if isinstance(cache, dict):
        kwa = cache.get("kwa", cache)
        if isinstance(kwa, dict):
            return kwa.get("cmpars")
    kwa = getattr(cache, "kwa", None)
    return kwa.get("cmpars") if isinstance(kwa, dict) else None


def _prime(raw, evt, cmpars):
    raw._odc = None
    if cmpars is None:
        raw.calib(evt)
    else:
        raw.calib(evt, cmpars=cmpars)
    cache = copy.deepcopy(raw._odc)
    assert _cache_cmpars(cache) == cmpars, (cache, cmpars)
    return cache


def _peaks(frame, finders):
    out = []
    for panel, finder in enumerate(finders):
        pk = finder.find(np.asarray(frame[panel]))
        for x, y, snr in zip(pk["x"], pk["y"], pk["snr"]):
            out.append((panel, float(x), float(y), float(snr)))
    return out


def _match(a, b, radius=1.5):
    unused = set(range(len(b)))
    matched = []
    only_a = []
    for pa, xa, ya, a_snr in a:
        candidates = [
            (i, (xb - xa) ** 2 + (yb - ya) ** 2)
            for i, (pb, xb, yb, _) in enumerate(b)
            if i in unused and pb == pa and (xb - xa) ** 2 + (yb - ya) ** 2 <= radius**2
        ]
        if not candidates:
            only_a.append((pa, xa, ya))
            continue
        i, distance = min(candidates, key=lambda item: item[1])
        unused.remove(i)
        matched.append((float(a_snr), float(b[i][3]), distance**0.5))
    return matched, only_a, [b[i] for i in sorted(unused)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="mfx101555026")
    parser.add_argument("--run", type=int, default=13)
    parser.add_argument("--events", type=int, default=200)
    parser.add_argument("--det", default="jungfrau")
    parser.add_argument("--output", type=Path, default=Path("common_mode_ab_run.json"))
    args = parser.parse_args()

    from psana import DataSource

    ds = DataSource(exp=args.exp, run=args.run)
    prun = next(ds.runs())
    detector = prun.Detector(args.det)
    assert prun.Detector(args.det) is detector
    raw = detector.raw
    cmpars = (7, 3, 200, 10)
    finders = None
    records = []
    off_cache = on_cache = None

    for event_index, evt in enumerate(prun.events()):
        if event_index >= args.events:
            break
        if off_cache is None:
            off_cache = _prime(raw, evt, None)
            on_cache = _prime(raw, evt, cmpars)
        raw._odc = off_cache
        off = np.asarray(raw.calib(evt), dtype=np.float32)
        raw._odc = on_cache
        on = np.asarray(raw.calib(evt, cmpars=cmpars), dtype=np.float32)
        if finders is None:
            try:
                status = detector.calibconst["pixel_status"][0].reshape(off.shape)
                good = ~(status != 0)
            except (KeyError, ValueError):
                good = np.ones_like(off, dtype=bool)
            PeakFinderV4 = xtc_core.load_peakfinder_v4().PeakFinderV4
            finders = [
                PeakFinderV4(good[p], min_pix=3, son_min=15, thr_high=10, thr_low=5)
                for p in range(off.shape[0])
            ]
        poff, pon = _peaks(off, finders), _peaks(on, finders)
        matched, only_off, only_on = _match(poff, pon)
        changed = np.abs(off - on)[good]
        changed_nonzero = changed[changed != 0]
        records.append(
            {
                "event": event_index,
                "changed_fraction": float(np.count_nonzero(changed) / changed.size),
                "median_abs_delta_kev": (
                    float(np.median(changed_nonzero)) if changed_nonzero.size else 0.0
                ),
                "peaks_off": len(poff),
                "peaks_on": len(pon),
                "matched_peaks": len(matched),
                "only_off": len(only_off),
                "only_on": len(only_on),
                "hit_off": len(poff) >= 6,
                "hit_on": len(pon) >= 6,
            }
        )

    if not records:
        raise RuntimeError("no events were calibrated")
    result = {
        "experiment": args.exp,
        "run": args.run,
        "detector": args.det,
        "events": len(records),
        "cmpars": list(cmpars),
        "settings": {"min_pix": 3, "son_min": 15, "thr_high": 10, "thr_low": 5},
        "matching_radius_px": 1.5,
        "records": records,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "events": len(records)}))


if __name__ == "__main__":
    main()
