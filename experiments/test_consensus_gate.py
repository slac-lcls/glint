"""Does the consensus acceptance gate refuse a chance lock WITHOUT costing the benchmark?

The bug it exists for: `consensus_cell`'s only test was an ABSOLUTE `min_support=3`, but the pool it
scores is `n_frames * nbest` -- 360 hypotheses at the 120-frame benchmark, ~6300 on a multi-thousand
frame run. On mfxx49820 r0016 a 23-hypothesis cluster (0.4% of 6300) cleared 3, `hybrid_index` then
gated the whole rescue cascade on `Mc is not None` alone, and 2098 frames were rescued onto a
doubled-c lattice and reported as "95% indexed".

PART 1 (no GPU): _accept's defaults must be bit-identical to the old rule, the observed 23/6300 must
now be refused, healthy benchmark-scale clusters must survive, and a near-tie must be refused.
PART 2 (GPU): run the real 120-frame cxidb set through hybrid_index and assert the paper's blind
number does not move -- the whole point is that this gate is a no-op in the regime that is published.

  python test_consensus_gate.py [frames.txt]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from glint.multishot import _accept

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")


print("PART 1 -- gate logic (no GPU)")

# Defaults MUST reproduce `cnt >= min_support` exactly, or every existing caller silently changes.
mismatches = [(w, r, n) for w in range(0, 60) for r in (0, 1, 5, w) for n in (10, 360, 6300)
              if _accept(w, r, n, 3, 0.0, 1.0) != (w >= 3)]
check("defaults are an exact no-op vs the old rule", not mismatches, f"{mismatches[:3]}")

# The real failure, with the values hybrid_index now uses.
FRAC, LEAD = 0.02, 1.5
check("mfxx49820 r0016: 23 of ~6300 is REFUSED", not _accept(23, 10, 6300, 3, FRAC, LEAD))
check("120-frame benchmark: 90 of 360 survives", _accept(90, 10, 360, 3, FRAC, LEAD))
check("120-frame benchmark: a lean 40 of 360 survives", _accept(40, 5, 360, 3, FRAC, LEAD))
check("near-tie 500 vs 450 is REFUSED regardless of size", not _accept(500, 450, 6300, 3, FRAC, LEAD))
check("absolute floor still applies", not _accept(2, 0, 10, 3, FRAC, LEAD))
# A single cluster with no runner-up must not be refused BY the lead test alone.
check("no runner-up -> lead test does not fire", _accept(50, 0, 100, 3, 0.02, 1.5))

frames_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "frames_cxidb_clean.txt")
if not os.path.exists(frames_path):
    print(f"\nPART 2 SKIPPED -- {frames_path} not found (needs the real cxidb set + a GPU)")
    raise SystemExit(0 if ok else 1)

print(f"\nPART 2 -- real 120-frame benchmark ({os.path.basename(frames_path)})")
import glint.glint_fast as gf
from glint.hybrid_stream import hybrid_index, CONSENSUS_MIN_FRAC, CONSENSUS_MIN_LEAD

frames = list(gf.load(frames_path))
print(f"  loaded {len(frames)} frames; gate = min_frac {CONSENSUS_MIN_FRAC} / min_lead {CONSENSUS_MIN_LEAD}")
_, stats = hybrid_index(frames, nbest=3)
n_idx, sup, pool = stats["n_idx"], stats["support"], stats["n_pool"]
print(f"  indexed {n_idx}/{len(frames)}   support {sup}/{pool} "
      f"({100.0*sup/max(pool,1):.1f}%)   refused={stats['consensus_refused']}")
check("consensus was NOT refused on the benchmark", not stats["consensus_refused"])
check("blind indexed still 91/120 (the published number)", n_idx == 91, f"got {n_idx}")

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
raise SystemExit(0 if ok else 1)
