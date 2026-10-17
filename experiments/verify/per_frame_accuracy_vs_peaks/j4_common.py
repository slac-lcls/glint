"""Shared helpers: pinned-tree path, commit check, default-settings guard, the strict bar."""
import glob
import os
import re
import sys

import numpy as np

GL = os.environ["VERIFY_GLINT"]          # the pinned GLINT tree (its COMMIT file must read 95d51ce); no default
COMMIT = "95d51ce"
# j4_bin.py's OUTDIR when none is given: $VERIFY_OUT/per_frame_accuracy_vs_peaks (none without VERIFY_OUT)
SCRATCH = (os.path.join(os.environ["VERIFY_OUT"], "per_frame_accuracy_vs_peaks")
           if "VERIFY_OUT" in os.environ else None)
# knobs that change speed or logging only, never which cell is returned
SPEED_ONLY = {"OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES"}


def pin():
    """Put the pinned tree first on sys.path and check its COMMIT file."""
    for p in (GL + "/experiments/xtc_bridge", GL + "/experiments", GL):
        if p not in sys.path:
            sys.path.insert(0, p)
    c = open(os.path.join(GL, "COMMIT")).read().strip()
    if c != COMMIT:
        raise SystemExit(f"pinned tree reports commit {c}, expected {COMMIT}")
    return c


def default_settings_guard():
    """Refuse to run if any environment variable the shipped modules read is set.

    Every tuning knob of glint_fast / glint_index / multishot / replica_gpu_batch is an
    os.environ lookup at import time, so 'default settings' means: none of them is set.
    Returns the sorted list of names checked, for the provenance record."""
    pat = re.compile(r'os\.environ(?:\.get|\.setdefault)?\(\s*["\']([A-Z0-9_]+)["\']'
                     r'|os\.environ\[\s*["\']([A-Z0-9_]+)["\']\s*\]')
    names = set()
    # the reader step imports experiments/xtc_bridge (GLINT_GEOM_MANIFEST and others), so it is scanned too
    for f in (glob.glob(os.path.join(GL, "glint", "*.py")) + [os.path.join(GL, "experiments", "scramble_qlist.py")]
              + glob.glob(os.path.join(GL, "experiments", "xtc_bridge", "*.py"))):
        for m in pat.finditer(open(f).read()):
            names.add(m.group(1) or m.group(2))
    bad = sorted(n for n in names if n in os.environ and n not in SPEED_ONLY)
    if bad:
        raise SystemExit("not default settings, these GLINT knobs are set: "
                         + ", ".join(f"{n}={os.environ[n]}" for n in bad))
    return sorted(names)


def cell_params(M):
    """a, b, c (A), alpha, beta, gamma (deg), volume (A^3) of a basis whose COLUMNS are the
    real-space axes (GLINT searches q . a = h)."""
    M = np.asarray(M, float)
    a, b, c = (M[:, i] for i in range(3))
    la, lb, lc = (np.linalg.norm(v) for v in (a, b, c))
    ang = lambda u, v: float(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))
    return la, lb, lc, ang(b, c), ang(a, c), ang(a, b), abs(float(np.linalg.det(M)))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))
