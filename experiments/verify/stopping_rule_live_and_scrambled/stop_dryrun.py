#!/usr/bin/env python3
"""Dry run (CPU, ~1-2 min on sdfiana027): the whole chain on stand-in inputs, into a throwaway STOP_OUT.

Not a measurement. It only proves that stop_prep.live / stop_scramble / stop_replay run end to end and that
the live StreamDriver and the replay agree. Stand-ins:
  r0278, r0058, r0016ref, r0016unref : the first N frames of the refined-r0016 cache (q6000_geom.npz),
                                       indexed here on CPU at warmup_nbest = 3
  cxidb480                           : experiments/nbest_120.npz (cxidb-120 cells, no indexing)

    STOP_OUT=<scratch>/dryrun STOP_DEV_N=2 python stop_dryrun.py [N]
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
assert os.environ.get("STOP_OUT", "").rstrip("/").endswith("dryrun"), "STOP_OUT must be a dryrun directory"
import stop_common as C                                                       # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
P = os.path.join(C.OUT, "prep")
os.makedirs(P, exist_ok=True)
os.makedirs(os.path.join(C.OUT, "replay"), exist_ok=True)

from glint.glint_fast import index_blind_nbest, LYSO                         # noqa: E402

z = np.load(os.path.join(C.GT, "q6000_geom.npz"))
fr = [np.asarray(z[f"q{i}"], float) for i in range(N)]
fr, voted = C.clean(fr, 6)
cells = [[np.asarray(c, float) for c, _ in index_blind_nbest(q, 3) if c is not None] if v else []
         for q, v in zip(fr, voted)]
flat, nper = C.pack_cells(cells)
Mb, sb = C.consensus_cell(list(flat), min_frac=C.CONSENSUS_MIN_FRAC, min_lead=C.CONSENSUS_MIN_LEAD)
common = dict(flat=flat, nper=nper, voted=voted, npk=np.array([len(q) for q in fr]), ms=np.zeros(N),
              events=np.arange(N), Mbatch=(np.zeros((3, 3)) if Mb is None else Mb), sup_batch=int(sb),
              batch_found=Mb is not None, lyso=np.asarray(LYSO, float), source="dryrun", source_md5="-",
              commit=C.commit())
for name in ("r0278", "r0058", "r0016ref", "r0016unref"):
    np.savez(os.path.join(P, f"votes_{name}.npz"), **common)
# q-frames the scramble arm reads, and r0016unref's so that stop_replay's device-parity check
# (stop_prep.load_frames) re-indexes the same stand-ins these votes came from, not the real unrefined cache
for name in ("r0278", "r0058", "r0016ref", "r0016unref"):
    np.savez(os.path.join(P, f"q_{name}.npz"), n=N, events=np.arange(N), lams=np.zeros(N),
             **{f"q{i}": q for i, q in enumerate(fr)})
b = np.load(os.path.join(C.GLINT, "experiments", "nbest_120.npz"))
pf = [list(b["cells"][b["fid"] == i]) for i in range(int(b["nframes"]))]
f2, n2 = C.pack_cells(pf)
np.savez(os.path.join(P, "votes_cxidb480.npz"), **dict(common, flat=f2, nper=n2, voted=np.ones(len(pf), bool),
                                                         npk=np.zeros(len(pf)), ms=np.zeros(len(pf)),
                                                         events=np.arange(len(pf)), lyso=b["lyso"]))
print(f"[dryrun] stand-ins written: {N} frames, {len(flat)} cells, batch {C.edges(Mb)} support {sb}", flush=True)

# live StreamDriver vs replay on one stand-in (the parity stop_prep.live checks), without the 480 file
res, live_cells = C.live_run(fr, np.arange(N))
rep = C.replay(cells, voted, np.arange(N), "shipped")
same = sum(1 for k, c in enumerate(live_cells) if c is not None and len(c) == len(cells[k])
           and all(C.same_lattice(x, y) for x, y in zip(c, cells[k])))
print(f"[dryrun] live decided={res['decided']}@{res['pushed']} support {res['support']} | replay "
      f"{rep['decided']}@{rep['pushed']} support {rep['support']} | cells identical on {same}/"
      f"{sum(c is not None for c in live_cells)} pushed frames", flush=True)
# the dry run exists to prove this agreement, so a mismatch fails it before the downstream steps run
diff = [k for k in ("decided", "pushed", "support") if res[k] != rep[k]]
if not ((res["M"] is None and rep["M"] is None) or
        (res["M"] is not None and rep["M"] is not None and C.same_lattice(res["M"], rep["M"]))):
    diff.append("decided cell")
if same != sum(c is not None for c in live_cells):
    diff.append("per-frame cells")
if diff:
    sys.exit(f"[dryrun] live and replay differ on {diff}")

py = sys.executable
subprocess.check_call([py, "-u", os.path.join(HERE, "stop_scramble.py"), "0", "1", "--G", "5", "--groups", "2,2,2",
                       "--long", "0", "--hours", "0.2", "--procs", "1"])
subprocess.check_call([py, "-u", os.path.join(HERE, "stop_replay.py"), "--procs", "4", "--orders", "8"])
print(open(os.path.join(C.OUT, "replay", "stop_tables.md")).read()[:3000])
