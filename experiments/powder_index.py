"""GLINT extended to POWDER auto-indexing. A powder pattern is the spherical average: orientation is gone,
only the ring radii q=|G_hkl| survive as a 1-D list. Indexing = recover the cell from {q_n}.

CUBIC is exactly GLINT's direct-sum objective in 1-D: (q_n a/2pi)^2 = h^2+k^2+l^2 must be INTEGER, so
score(a)=sum_n cos^2(pi (q_n a/2pi)^2) peaks at the true a. TET/HEX = a 2-parameter (a,c) metric grid,
each (a,c) scored by the fraction of observed rings matched to a predicted one (the metric-tensor fit).
Demonstrated blind on simple, low-overlap crystals. Front end = a radial average of the 2-D image
(pyFAI, the drp-benchmarks/radial_integration code); back end = Rietveld, as SFX feeds partialator."""
import numpy as np
TP = 2 * np.pi
rng = np.random.default_rng(0)


def _pairs(system, hm=8):
    """unique reduced invariants: tet (m=h^2+k^2, n=l^2), hex (m=h^2+hk+k^2, n=l^2)."""
    S = set()
    for h in range(-hm, hm + 1):
        for k in range(-hm, hm + 1):
            for l in range(0, hm + 1):
                if h == k == l == 0:
                    continue
                m = (h * h + k * k) if system == "tetragonal" else (h * h + h * k + k * k)
                S.add((m, l * l))
    P = np.array(sorted(S), float)
    return P[:, 0], P[:, 1]


def rings(system, p, dmin=0.9, merge=0.02, hm=8):
    a = p[0]; c = p[-1]
    if system == "cubic":
        m, nn = _pairs("tetragonal", hm)                       # h^2+k^2 and l^2 -> full N via m+nn
        d2 = (m + nn) / a**2                                   # (uses N=h^2+k^2+l^2)
    elif system in ("tetragonal", "hexagonal"):
        m, nn = _pairs(system, hm)
        coef = (4 / 3) if system == "hexagonal" else 1.0
        d2 = coef * m / a**2 + nn / c**2
    q = TP * np.sqrt(d2); q = q[TP / np.maximum(q, 1e-9) >= dmin]
    qs = np.sort(np.unique(np.round(q, 6)))
    out = [qs[0]]
    for x in qs[1:]:
        if x - out[-1] > merge:
            out.append(x)
    return np.array(out)


def observe(q, drop=0.1, spur=2, noise=0.004):
    keep = q[rng.random(len(q)) > drop] + 0.0
    keep = keep + rng.normal(0, noise, len(keep))
    sp = rng.uniform(q.min(), q.max(), spur)
    return np.sort(np.concatenate([keep, sp]))


def index_cubic(qo, amin=3.0, amax=9.0, n=6000, tol=0.02):
    Nmax = int((qo.max() * amax / TP) ** 2) + 2
    hm = int(np.sqrt(Nmax)) + 1
    Nset = np.array(sorted({h*h + k*k + l*l for h in range(hm+1) for k in range(hm+1) for l in range(hm+1)} - {0}), float)
    Nset = Nset[Nset <= Nmax]
    a = np.linspace(amin, amax, n); lo, hi = qo.min() - tol, qo.max() + tol
    best = (-1.0, 0.0)
    for ai in a:                                               # de Wolff FOM: predict all cubic rings, precision*recall
        qp = TP * np.sqrt(Nset) / ai; qp = qp[(qp >= lo) & (qp <= hi)]
        if len(qp) < len(qo):
            continue
        D = np.abs(qo[:, None] - qp[None, :])
        fom = (D.min(1) < tol).mean() * (D.min(0) < tol).mean()
        if fom > best[0]:
            best = (fom, ai)
    frac = (np.abs((qo * best[1] / TP) ** 2 - np.rint((qo * best[1] / TP) ** 2)) < 0.15).mean()
    return best[1], frac


def index_2p(system, qo, arange=(2.5, 7.5), crange=(2.5, 16.0), n=180, tol=0.02):
    m, nn = _pairs(system, 8); coef = (4 / 3) if system == "hexagonal" else 1.0
    aa = np.linspace(*arange, n); cc = np.linspace(*crange, n)
    lo, hi = qo.min() - tol, qo.max() + tol
    best = (-1.0, 0.0, 0.0)
    for a in aa:
        t1 = coef * m / a**2
        for c in cc:
            qp = TP * np.sqrt(t1 + nn / c**2)
            qp = np.unique(np.round(qp[(qp >= lo) & (qp <= hi)], 3))            # predicted rings in window
            if len(qp) < len(qo):
                continue
            D = np.abs(qo[:, None] - qp[None, :])
            recall = (D.min(1) < tol).mean()                                   # obs matched  (completeness)
            precision = (D.min(0) < tol).mean()                                # pred matched (anti-supercell)
            fom = recall * precision                                           # de Wolff-style: penalize extra rings
            if fom > best[0]:
                best = (fom, a, c)
    return best[1], best[2], best[0]


def report(name, system, ptrue):
    q = rings(system, ptrue); qo = observe(q)
    if system == "cubic":
        a, frac = index_cubic(qo)
        e = 100 * abs(a - ptrue[0]) / ptrue[0]
        print(f"{name:20s} true a={ptrue[0]:.3f}      -> a={a:.3f}       err {e:4.2f}%   idx {100*frac:3.0f}%  [{len(qo)} lines]")
    else:
        a, c, frac = index_2p(system, qo)
        te = 100 * abs(a - ptrue[0]) / ptrue[0]; ce = 100 * abs(c - ptrue[-1]) / ptrue[-1]
        print(f"{name:20s} true a={ptrue[0]:.3f} c={ptrue[-1]:.3f} -> a={a:.3f} c={c:.3f}  err {te:4.1f}/{ce:4.1f}%  match {100*frac:3.0f}%  [{len(qo)} lines]")


print("POWDER auto-indexing, BLIND from ring positions only (~10%% lines dropped, 2 spurious, noise 0.004 1/A):\n")
report("Si (cubic)",     "cubic",      (5.431,))
report("NaCl (cubic)",   "cubic",      (5.640,))
report("Po (cubic)",     "cubic",      (3.350,))
report("rutile (tet)",   "tetragonal", (4.593, 2.959))
report("anatase (tet)",  "tetragonal", (3.785, 9.514))
report("quartz (hex)",   "hexagonal",  (4.913, 5.405))
report("Mg (hex)",       "hexagonal",  (3.209, 5.211))


# ---- ORTHORHOMBIC a!=b!=c: distinct lines (no symmetry coincidence) -> the general, more informative case.
# q^2 = h^2 A + k^2 B + l^2 C, LINEAR in the metric (A,B,C)=(2pi/a)^2.. -> GLINT-style discrete multi-start
# over which low observed lines are the principal (100)/(010)/(001), solve, score with the M20-ish FOM.
from itertools import combinations, product
_HKL = np.array([[h, k, l] for h, k, l in product(range(6), repeat=3) if (h, k, l) != (0, 0, 0)], float)
_H2 = _HKL ** 2


def index_ortho(qo, tol=0.02, nlow=9):
    q2 = np.sort(qo ** 2); lows = q2[:nlow]
    lo, hi = qo.min() - tol, qo.max() + tol
    best = (-1.0, None)
    for A, B, C in combinations(lows, 3):                       # try low lines as the 3 principal reflections
        for perm in ((A, B, C),):                              # A,B,C symmetric under axis relabel -> one order
            pred2 = _H2 @ np.array(perm)                       # predicted q^2 for every hkl
            qp = np.sqrt(pred2[pred2 > 0]); qp = np.unique(np.round(qp[(qp >= lo) & (qp <= hi)], 3))
            if len(qp) < len(qo):
                continue
            D = np.abs(qo[:, None] - qp[None, :])
            fom = (D.min(1) < tol).mean() * (D.min(0) < tol).mean()
            if fom > best[0]:
                best = (fom, perm)
    A, B, C = sorted(best[1])                                  # smallest metric comp = longest axis
    return np.sort(TP / np.sqrt([A, B, C])), best[0]


def report_o(name, ptrue):
    q = rings("ortho_gen", None) if False else None
    # inline generic-ortho rings
    a, b, c = ptrue
    d2 = _H2 @ np.array([1/a**2, 1/b**2, 1/c**2]); qq = TP * np.sqrt(d2[d2 > 0])
    qq = qq[TP / qq >= 0.9]; qs = np.sort(np.unique(np.round(qq, 6)))
    out = [qs[0]]
    for x in qs[1:]:
        if x - out[-1] > 0.02:
            out.append(x)
    qo = observe(np.array(out))
    rec, fom = index_ortho(qo); tp = np.sort(ptrue)
    err = 100 * np.abs(rec - tp) / tp
    print(f"{name:26s} true {tp[0]:.2f}/{tp[1]:.2f}/{tp[2]:.2f} -> {rec[0]:.2f}/{rec[1]:.2f}/{rec[2]:.2f}  "
          f"err {err[0]:.1f}/{err[1]:.1f}/{err[2]:.1f}%  fom {fom:.2f}  [{len(qo)} lines]")


print("\nORTHORHOMBIC (a!=b!=c, distinct lines -- the general, more interesting case):")
report_o("generic 5.1/7.3/9.8", (5.1, 7.3, 9.8))
report_o("forsterite 4.75/10.2/5.98", (4.75, 10.20, 5.98))
report_o("aragonite 4.96/7.97/5.74", (4.96, 7.97, 5.74))
report_o("topaz 4.65/8.80/8.40", (4.65, 8.80, 8.40))
