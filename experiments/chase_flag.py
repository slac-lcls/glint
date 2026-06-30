"""Chase the wrong_cell flag. The failure map: under a 2nd lattice, gen_miss~0 but single-pass
blind locks onto a confident SPURIOUS cell ~36% of frames (selection/pollution from the two
lattices' overlapping spots), not the true LYSO cell. Test whether deflate-and-reindex (index ->
strip that cell's inliers -> reindex residual) recovers a clean LYSO cell on those frames.

  python chase_flag.py [frames.txt] [N]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
os.environ.setdefault("QDIST", "1"); os.environ.setdefault("QDTOL", "0.004")
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)
import numpy as np
from glint_fast import index_blind_fast, load, matched, LYSO
from fftindex.multishot import same_lattice
from fat_ewald import sim_fat
from d34_deflate import deflate

GATE, MININL = 0.25, 10


def s_twin(q, frac, rng):                                        # add a 2nd LYSO lattice (new orientation)
    g2, _ = sim_fat(0.002, n_cap=max(4, int(frac * len(q))), spur=0.0, jitter=0.0, rng=rng)
    return np.vstack([q, g2]) if len(g2) else q


def single(q):
    """single-pass outcome: True=solved LYSO, False=confident wrong_cell, None=not confident."""
    M = index_blind_fast(q)
    if M is None:
        return None
    m = matched(M, q)
    if m / len(q) >= GATE and m >= MININL:
        return bool(same_lattice(M, LYSO))
    return None


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6][:N]
    n = len(frames)
    index_blind_fast(frames[0])
    print(f"N={n}  2nd-lattice (same LYSO cell) -- single-pass vs deflate")
    print(f"{'frac2':>6}{'1pass_solved':>14}{'1pass_wrong':>13}{'deflate>=1 LYSO':>17}{'deflate BOTH':>14}")
    for frac in (0.5, 1.0):
        rng = np.random.default_rng(0)
        sp_ok = sp_wrong = d_one = d_both = 0
        for q in frames:
            qq = s_twin(q, frac, rng)
            r = single(qq)
            sp_ok += r is True; sp_wrong += r is False
            cells = deflate(qq)
            nlyso = sum(bool(same_lattice(c, LYSO)) for c in cells)
            d_one += nlyso >= 1; d_both += nlyso >= 2
        print(f"{frac:>6.1f}{100*sp_ok//n:>13d}%{100*sp_wrong//n:>12d}%{100*d_one//n:>16d}%{100*d_both//n:>13d}%",
              flush=True)
