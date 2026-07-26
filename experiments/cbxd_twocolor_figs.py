"""Figures for the two-color convergent-beam CBXD forward-sim testbed (cbxd_twocolor.py).
Regenerates the committed PNGs next to this file.

  python cbxd_twocolor_figs.py ridges     # reflections-as-ridges shot (single vs two-colour)
  python cbxd_twocolor_figs.py solved      # blind-recovered orientation, predicted ridges over data
  python cbxd_twocolor_figs.py all
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cbxd_twocolor as tc

COL = ["#1b6ca8", "#d1495b"]                                   # 17.5 keV, 15.0 keV


def _ridges(R, Ks, na, nchi=3000):
    """in-cone Kossel arcs for orientation R as projected detector polylines (mrad) + colour idx
    + tangent-at-midpoint (unit, detector frame)."""
    chi = np.linspace(0, 2 * np.pi, nchi, endpoint=False)
    cc, ss = np.cos(chi), np.sin(chi)
    cosa, sina = np.cos(na), np.sin(na)
    out = []
    for h in tc.HS:
        G = R @ (tc.B @ h)
        for ci, K in enumerate(Ks):
            a, b = tc.kossel_basis(G, K)
            if a is None:
                continue
            kout = G / 2 + cc[:, None] * a + ss[:, None] * b
            kin = kout - G
            m = (kin[:, 2] > K * cosa) & (np.hypot(kin[:, 0], kin[:, 1]) < K * sina) & (kout[:, 2] > 0)
            idx = np.where(m)[0]
            if len(idx) < 4:
                continue
            P = 1e3 * kout[idx, :2] / kout[idx, 2:3]
            for seg in np.split(np.arange(len(idx)), np.where(np.diff(idx) > 3)[0] + 1):
                if len(seg) < 4:
                    continue
                poly = P[seg]
                t = poly[-1] - poly[0]; t = t / (np.linalg.norm(t) + 1e-9)
                out.append((poly, ci, poly[len(poly) // 2], t))
    return out


def _spurious(seed, kmax, n=60):
    rng = np.random.default_rng(seed)
    sp = rng.normal(size=(n, 3)); sp /= np.linalg.norm(sp, axis=1, keepdims=True); sp *= kmax
    sp = sp[sp[:, 2] > 0]
    return 1e3 * sp[:, :2] / sp[:, 2:3]


def fig_ridges(na=0.050, seed=7):
    """Simulated shot, reflections as ridges + tangent bars; single-colour vs two-colour."""
    R = tc.rand_rot(np.random.default_rng(seed))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), facecolor="white")
    TAN = 6.0; counts = []
    for ax, (title, Ks) in zip(axes, [("single-colour  (17.5 keV)", tc.KS_1),
                                      ("two-colour  (17.5 + 15.0 keV)", tc.KS_2)]):
        rs = _ridges(R, Ks, na); counts.append(len(rs))
        sp = _spurious(3, tc.KMAX)
        ax.scatter(sp[:, 0], sp[:, 1], s=6, c="#cccccc", alpha=0.6, lw=0, zorder=1)
        ax.add_collection(LineCollection([r[0] for r in rs], colors=[COL[r[1]] for r in rs],
                                         linewidths=2.0, zorder=2))
        ax.add_collection(LineCollection([[m - TAN * t, m + TAN * t] for _, _, m, t in rs],
                                         colors="k", linewidths=0.9, alpha=0.55, zorder=3))
        ax.plot(0, 0, "+", ms=13, mew=2, c="k", zorder=4)
        for ci, lab in [(0, "17.5 keV ridge"), (1, "15.0 keV ridge")][:len(Ks)]:
            ax.plot([], [], "-", c=COL[ci], lw=2.2, label=lab)
        ax.plot([], [], "-", c="k", lw=0.9, alpha=0.6, label="ridge tangent")
        ax.plot([], [], "o", c="#cccccc", ms=4, label="spurious")
        ax.set_title(title, fontsize=13); ax.set_aspect("equal")
        ax.set_xlim(-150, 150); ax.set_ylim(-150, 150)
        ax.set_xlabel("detector x  (mrad)"); ax.set_ylabel("detector y  (mrad)")
        ax.grid(alpha=0.15); ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle(f"Simulated convergent-beam CBXD shot -- reflections as ridges  "
                 f"(NA={na*1e3:.0f} mrad, cell 16/21/25 A)   ridges: {counts[0]} -> {counts[1]}",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(HERE, "cbxd_ridges.png"); fig.savefig(out, dpi=130)
    print("saved", out, "| ridges", counts)


def fig_solved(na=0.025, seed=7):
    """Blind-recover the orientation from a two-colour shot, overlay predicted ridges on the data."""
    R_true = tc.rand_rot(np.random.default_rng(seed))
    kobs, lab, col, cents, _ = tc.simulate(R_true, np.random.default_rng(3), 1.5e-4, na, tc.KS_2,
                                           nchi=1400, thin=2)
    R_hat, mask, frac, sym = None, None, -1.0, 180.0
    for att in range(4):                                        # keep-best over seeder restarts
        Rh = tc.seed_index(kobs, cents, na, tc.KS_2, np.random.default_rng(20 + att),
                           n_coarse=150000, keep=50)
        m = tc.score(Rh, kobs, 0.0025, na, tc.KS_2, ret_mask=True); f = m[lab].mean()
        print(f"  restart {att}: indexed real={100*f:.0f}%  symang={tc.ang_sym(R_true, Rh):.1f} deg")
        if f > frac:
            R_hat, mask, frac, sym = Rh, m, f, tc.ang_sym(R_true, Rh)
        if f > 0.85:
            break
    print(f"recovered (best): symang={sym:.2f} deg (mod 222)  indexed real={100*frac:.0f}%")
    P = 1e3 * kobs[:, :2] / kobs[:, 2:3]
    pred = _ridges(R_hat, tc.KS_2, na)
    fig, ax = plt.subplots(figsize=(8.4, 8.0), facecolor="white")
    ax.scatter(P[~lab, 0], P[~lab, 1], s=16, c="#d9d9d9", lw=0, zorder=1, label="observed spurious")
    oi, om = lab & mask, lab & ~mask
    ax.scatter(P[oi, 0], P[oi, 1], s=20, c="#333333", lw=0, zorder=3, label="observed Bragg (indexed)")
    ax.scatter(P[om, 0], P[om, 1], s=26, facecolors="none", edgecolors="#333333", lw=1.0, zorder=3,
               label="observed Bragg (missed)")
    ax.add_collection(LineCollection([p for p, _, _, _ in pred], colors=[COL[c] for _, c, _, _ in pred],
                                     linewidths=1.8, alpha=0.9, zorder=2))
    for ci, l in [(0, "predicted ridge 17.5 keV"), (1, "predicted ridge 15.0 keV")]:
        ax.plot([], [], "-", c=COL[ci], lw=2.0, label=l)
    ax.plot(0, 0, "+", ms=13, mew=2, c="k", zorder=4)
    ax.set_aspect("equal"); ax.set_xlim(-160, 160); ax.set_ylim(-160, 160)
    ax.set_xlabel("detector x  (mrad)"); ax.set_ylabel("detector y  (mrad)")
    ax.grid(alpha=0.15); ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(f"Solved two-colour CBXD shot: predicted ridges (recovered R) over observed data\n"
                 f"NA={na*1e3:.0f} mrad  ·  indexed {100*frac:.0f}% of Bragg peaks  ·  "
                 f"orientation error {sym:.1f}° (mod 222)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "cbxd_solved.png"); fig.savefig(out, dpi=135)
    print("saved", out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("ridges", "all"):
        fig_ridges()
    if which in ("solved", "all"):
        fig_solved()
