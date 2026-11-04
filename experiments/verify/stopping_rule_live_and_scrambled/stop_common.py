"""Shared code: the shipped stopping rule, taken from the pinned GLINT tree (95d51ce), never re-typed.

Nothing here changes GLINT. The rule's parameters are read from StreamDriver.__init__'s signature, the
live arms run StreamDriver itself (blind mode, push_q), and the replay arms rebuild the RunningConsensus
the driver builds (glint/stream_driver.py:1482-1484). Replay numbers are used only where the live driver
and the replay agree on the arrival order (parity check in stop_prep.py / stop_replay.py).
"""
import hashlib
import inspect
import json
import os
import platform
import socket
import sys
import time

import numpy as np

GLINT = os.environ.get("GLINT_ROOT") or os.environ["VERIFY_GLINT"]   # stop_env.sh exports GLINT_ROOT; no default
if GLINT not in sys.path:
    sys.path.insert(0, GLINT)

from glint.running_consensus import RunningConsensus                       # noqa: E402
from glint.multishot import (same_lattice, consensus_cell,                 # noqa: E402
                             CONSENSUS_MIN_FRAC, CONSENSUS_MIN_LEAD)
from glint.lattice import cell_to_Ar                                        # noqa: E402
from glint.geom import q_rows_ok                                            # noqa: E402
import glint.stream_driver as sd                                            # noqa: E402  (cupy/torch guarded)

OUT = os.environ.get("STOP_OUT") or os.path.join(os.environ["VERIFY_OUT"], "stopping_rule_live_and_scrambled")
GT = os.path.join(os.environ["VERIFY_INPUTS"], "glint_gt_mfxx49820")   # q6000_geom.npz, r0016_q6000.npz, calib/ (README)
BTX = "/sdf/data/lcls/ds/mfx/mfxx49820/results/btx"
SEED = 20261015                       # master seed; every rng below is default_rng([SEED, ...])

RULE_KEYS = ("lock_support", "lock_gap", "adaptive_gap", "warmup_nbest", "lock_frac", "lock_lead",
             "lock_pool_switch", "alias_gate", "min_peaks", "adaptive_relock")

# Real datasets in the order every table lists them. "kind" says where the q-frames come from.
DATASETS = ("r0278", "r0058", "r0016ref", "r0016ref_aug", "r0016unref", "cxidb480")


def commit():
    try:
        return open(os.path.join(GLINT, "COMMIT")).read().strip()
    except OSError:
        return "UNKNOWN"


def md5(path, n=None):
    h = hashlib.md5(open(path, "rb").read()).hexdigest()
    return h if n is None else h[:n]


def provenance(argv=None):
    """What every output file carries: commit, argv, host, env, Slurm ids, rule defaults."""
    import torch
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return dict(commit=commit(), glint_root=GLINT, argv=list(sys.argv if argv is None else argv),
                host=socket.gethostname(), python=platform.python_version(), torch=torch.__version__,
                device=dev, conda_env=os.environ.get("CONDA_DEFAULT_ENV"),
                slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
                time=time.strftime("%Y-%m-%d %H:%M:%S %Z"), rule=shipped_defaults(),
                crystfel="not used in this check", seed=SEED,
                env_overrides={k: os.environ[k] for k in os.environ
                               if k.startswith(("GLINT_", "STEPS", "M3_", "NTOP", "KEEP", "REFINER",
                                                "SCORER", "ANNEAL", "QPOW", "TOL", "QHI", "QLO"))})


def shipped_defaults():
    """StreamDriver's own defaults at the pinned commit, read from its signature."""
    p = inspect.signature(sd.StreamDriver.__init__).parameters
    return {k: p[k].default for k in RULE_KEYS}


def make_rule(variant, d=None):
    """RunningConsensus exactly as StreamDriver builds it.

    shipped : the rule as it ships -- support >= 3, lead in members >= g (g = 2, 3, 4 by the adaptive
              schedule), and once 72 cells are pooled also >= 2% of the pool and >= 1.5x the runner-up.
    gaponly : support and lead only (StreamDriver with lock_frac=0, lock_lead=1, lock_pool_switch=0,
              the bare rule its docstring names); the rule the seqstop_* keys were measured with.
    """
    d = d or shipped_defaults()
    if variant == "shipped":
        return RunningConsensus(min_support=d["lock_support"], gap=d["lock_gap"], adaptive=d["adaptive_gap"],
                                min_frac=d["lock_frac"], min_lead=d["lock_lead"],
                                pool_switch=d["lock_pool_switch"])
    if variant == "gaponly":
        return RunningConsensus(min_support=d["lock_support"], gap=d["lock_gap"], adaptive=d["adaptive_gap"])
    raise ValueError(variant)


def replay(per_frame, voted, order, variant, d=None, max_frames=None):
    """Feed frames in `order` the way StreamDriver._ingest_blind_q does: a frame with >= min_peaks
    usable rows votes (add_frame, even with no cells), the verdict is read after every pushed frame,
    and the first non-None verdict is the decision."""
    rc = make_rule(variant, d)
    k = 0
    for k, i in enumerate(order, 1):
        if voted[i]:
            rc.add_frame(per_frame[i])
        M, sup, run = rc.verdict()
        if M is not None:
            return dict(decided=True, pushed=k, voting=rc.nframes, npool=rc.npool, support=int(sup),
                        runner=int(run), M=np.asarray(M, float))
        if max_frames and k >= max_frames:
            break
    _, sup, run = rc.verdict()
    return dict(decided=False, pushed=k, voting=rc.nframes, npool=rc.npool, support=int(sup),
                runner=int(run), M=None)


# --- the live driver ---------------------------------------------------------------------------
# Peaks-only StreamDriver, as experiments/record_stream_replay.py builds it: a dummy one-panel
# detector, because push_q never uses the geometry before the decision.
N_PX = 400
PANELS = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=1.0 / (0.1 / 1000.0),
               cx=-(N_PX / 2.0 - 0.5), cy=-(N_PX / 2.0 - 0.5), coffset=0.0,
               min_fs=0, max_fs=N_PX - 1, min_ss=0, max_ss=N_PX - 1)]


class Recorder:
    """Wraps StreamDriver._blind_index (= glint_fast.index_blind_nbest) and keeps the cells it returned.
    The call and its arguments are passed through unchanged; this only lets a group be replayed under
    another rule without indexing it again."""

    def __init__(self, fn):
        self.fn, self.calls, self.secs = fn, [], 0.0

    def __call__(self, q, n):
        t = time.time()
        out = self.fn(q, n)
        self.secs += time.time() - t
        self.calls.append([np.asarray(c, float) for c, _ in out if c is not None])
        return out


def live_run(frames, order, stop_at_decision=True):
    """Push frames through a default StreamDriver(Mc=None) with push_q, in `order`.
    Returns the decision (or none) and, per pushed frame, the cells it voted with (None = did not vote)."""
    drv = sd.StreamDriver(None, PANELS, 0.1, 1.0, (N_PX, N_PX), dtype=np.uint16, use_gpu=False)
    rec = Recorder(drv._blind_index)
    drv._blind_index = rec
    cells = []
    k = 0
    for k, i in enumerate(order, 1):
        n0 = len(rec.calls)
        drv.push_q(frames[i])
        cells.append(rec.calls[-1] if len(rec.calls) > n0 else None)
        if drv.locked and stop_at_decision:
            break
    rc = drv._rc
    lead = rc.leaders()
    res = dict(decided=bool(drv.locked), pushed=int(drv.locked_after) if drv.locked else k,
               voting=int(rc.nframes), npool=int(rc.npool),
               support=int(drv.consensus_support) if drv.locked else int(lead[1]),
               members=int(drv.consensus_members) if drv.locked else None, runner=int(lead[2]),
               M=np.asarray(drv.Mc, float) if drv.locked else None, index_s=rec.secs)
    return res, cells


# --- reference cells and the correctness bar -----------------------------------------------------
def read_cell_file(path):
    """CrystFEL .cell -> (a, b, c, al, be, ga)."""
    v = {}
    for line in open(path):
        if "=" in line:
            k, x = [s.strip() for s in line.split("=", 1)]
            v[k] = x.split()[0]
    return tuple(float(v[k]) for k in ("a", "b", "c", "al", "be", "ga"))


def verdict_vs(M, ref):
    """The bar: glint.multishot.same_lattice(M, ref) at its defaults (rtol 0.05, ctol 0.06, vtol 0.10;
    Buerger-reduced lengths, angles and volume), the test the seqstop_* and poolgate_* keys used.
    Also the volume ratio, so a doubled cell reads as ~2."""
    if M is None or ref is None:
        return dict(same=None, vol_ratio=None)
    M = np.asarray(M, float); ref = np.asarray(ref, float)
    return dict(same=bool(same_lattice(M, ref)),
                vol_ratio=float(abs(np.linalg.det(M)) / abs(np.linalg.det(ref))))


def edges(M):
    return None if M is None else [round(float(x), 2) for x in np.sort(np.linalg.norm(np.asarray(M, float), axis=0))]


def clean(frames, min_peaks):
    """The rows push_q keeps, and whether the frame votes (>= min_peaks usable rows)."""
    out, voted = [], []
    for q in frames:
        q = np.asarray(q, float).reshape(-1, 3)
        q = q[q_rows_ok(q)]
        out.append(q)
        voted.append(len(q) >= min_peaks)
    return out, np.asarray(voted, bool)


def pack_cells(per_frame):
    """List of per-frame cell lists -> (flat (K,3,3) float64, nper (n,) int16)."""
    nper = np.array([0 if c is None else len(c) for c in per_frame], np.int16)
    flat = [np.asarray(m, float) for c in per_frame if c for m in c]
    return (np.asarray(flat, float).reshape(-1, 3, 3), nper)


def unpack_cells(flat, nper):
    off = np.concatenate([[0], np.cumsum(nper.astype(np.int64))])
    return [[flat[j] for j in range(off[i], off[i + 1])] for i in range(len(nper))]


def dump_json(path, obj):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=conv)
    os.replace(tmp, path)
