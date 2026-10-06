"""StreamDriver.flush() is exception-safe: one failed batch neither locks the ring nor merges a frame twice.

WHY THIS EXISTS (review r2, finding s2-03). flush() reset the ring (`_n`, the slot lists) only at its very end. An
exception part-way through a batch -- an on_event hook, StreamWriter.write on a full disk, an integrator error --
left `_n == B`, so every later push() raised IndexError (and was not counted), and a retried flush(), close() or
`with drv:` exit re-ran the whole batch, merging the frames already folded into `acc` a second time. Now the ring
is emptied in a finally, the failure is counted in stats()["flush_errors"] / ["flush_error_frames"], and the
exception still propagates.

Same CPU seam as test_stream_hit_ring.py (the repo's fit oracle as the known-cell indexer; shipped PeakFinderV4,
integrate_spots, MergeAccumulator).

  PYTHONPATH=. python experiments/test_flush_exception_safe.py      # exit 0 = all pass  (numpy only)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import test_stream_hit_ring as th                                   # noqa: E402
from test_cell_registry import SEED                                 # noqa: E402

B = 4
FAILS = []
FR = [th._frame("A", rng) for rng in [np.random.default_rng(SEED)] for _ in range(12)]


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'  ' + str(detail) if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def control(n):
    drv, _ = th._drv(B=B)
    for f in FR[:n]:
        drv.push(f)
    drv.close()
    return drv.acc.stats()["frames"]


def hook_once(at=2):
    st = {"n": 0}

    def hook(rec):
        if rec.get("outcome") == "integrated":
            st["n"] += 1
            if st["n"] == at:
                raise ConnectionError("monitor socket dropped")
    return hook


print("1. on_event raises once inside the first flush; the caller logs and keeps pushing")
ref = control(12)
drv, _ = th._drv(B=B, on_event=hook_once())
errs = []
for k, f in enumerate(FR):
    try:
        drv.push(f)
    except Exception as e:                                          # noqa: BLE001  (a log-and-continue DAQ loop)
        errs.append(type(e).__name__)
drv.close()
st = drv.stats()
check("only the one injected exception reaches the caller", errs == ["ConnectionError"], errs)
check("every frame is counted as pushed", drv.n_pushed == 12, drv.n_pushed)
expected = control(2) + ref - control(B)
check("merged prefix retained once and all later frames merged", st["frames"] == expected,
      f"{st['frames']} vs {expected}")
check("the failure is reported", st.get("flush_errors") == 1 and st.get("flush_error_frames") == B,
      (st.get("flush_errors"), st.get("flush_error_frames")))

print("\n2. the same fault with the driver as a context manager: __exit__ must not re-run the batch")
ref4 = control(4)
drv, _ = th._drv(B=B, on_event=hook_once())
try:
    with drv:
        for f in FR[:4]:
            drv.push(f)
except ConnectionError:
    pass
n_after = drv.acc.stats()["frames"]
check("frames merged <= physical frames", n_after <= 4 and n_after <= ref4, f"{n_after} (physical 4)")
check("ring is empty", drv._n == 0, drv._n)

print(f"\n{'FAILURES: ' + ', '.join(FAILS) if FAILS else 'ALL PASS'}")
sys.exit(1 if FAILS else 0)
