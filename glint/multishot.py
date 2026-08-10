"""Multi-shot consensus indexing: pool many sparse shots of a shared cell.

A single very sparse shot (n < ~25) often can't be indexed unknown-cell. But many
shots share ONE cell whose metric tensor is rotation-invariant, so a consensus cell
emerges from the minority that do index; then KNOWN-cell indexing (constrained to
that cell) rescues shots that failed unknown-cell -- pushing below the single-shot
sparsity floor.

Lattices are compared by their length spectrum (the k shortest distinct lattice-
vector lengths) -- a rotation- and basis-choice-invariant signature, so no Niggli
reduction is needed for the prototype.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.spatial import cKDTree

from .index import IndexResult, _triplets, detect_batch, refine, search_basis
from .lattice import reduced_params
from .running_consensus import consensus_accept
from .peakfind import find_peaks_classical
from .transform import fft_volume


def reference_lattice(M_ref, qmax):
    """All reciprocal-lattice vectors of cell M_ref within |g| <= qmax (one orientation)."""
    B = np.linalg.inv(np.asarray(M_ref, float)).T
    hmax = int(np.ceil(qmax / np.min(np.linalg.norm(B, axis=0)))) + 1
    r = np.arange(-hmax, hmax + 1)
    H = np.array(np.meshgrid(r, r, r, indexing="ij")).reshape(3, -1).T
    H = H[np.any(H != 0, axis=1)]
    V = (B @ H.T).T
    return V[np.linalg.norm(V, axis=1) <= qmax]


def _frame(u, v):
    """Right-handed orthonormal frame from two non-parallel vectors (u along e1)."""
    nu = np.linalg.norm(u)
    if nu < 1e-9:
        return None
    e1 = u / nu
    w = v - (v @ e1) * e1
    nw = np.linalg.norm(w)
    if nw < 1e-9:
        return None
    e2 = w / nw
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def index_known_pairangle(g, qmax, M_ref, Vref=None, tree=None, tol_frac=0.02,
                          len_tol=0.04, ang_tol=3.0, n_seed=7):
    """Taketwo-style known-cell indexer: match observed spot PAIRS to the known
    reciprocal lattice by length+angle, solve the orientation, and vote.

    Peak-independent (no FFT volume), so it indexes far sparser shots than the
    FFT-peak path, and returns M_ref's lattice by construction (no lattice-equality
    test needed). Seeds from the shortest observed spots (reliable low orders). Pass
    a cached (Vref, tree) when indexing many shots against one consensus cell.
    """
    g = np.asarray(g, float)
    n = len(g)
    if n < 3:
        return IndexResult(None, None, None, None, 0)
    tol = tol_frac * qmax
    if Vref is None:
        Vref = reference_lattice(M_ref, qmax)
    if tree is None:
        tree = cKDTree(Vref)
    Lref = np.linalg.norm(Vref, axis=1)
    Lobs = np.linalg.norm(g, axis=1)
    seeds = np.argsort(Lobs)[:n_seed]
    # restrict seed matching to low-order ref vectors (seeds are the shortest spots);
    # shrinks the candidate shells without dropping the correct match -- the speed knob.
    short = Lref < 1.2 * Lobs[seeds[-1]]
    win = 0.8 * n

    best = (0, None)
    for ii in range(len(seeds)):
        for jj in range(ii + 1, len(seeds)):
            i, j = seeds[ii], seeds[jj]
            qi, qj = g[i], g[j]
            ao = np.degrees(np.arccos(np.clip(qi @ qj / (Lobs[i] * Lobs[j]), -1, 1)))
            ca = np.where(short & (np.abs(Lref - Lobs[i]) < len_tol * Lobs[i]))[0]
            cb = np.where(short & (np.abs(Lref - Lobs[j]) < len_tol * Lobs[j]))[0]
            if len(ca) == 0 or len(cb) == 0:
                continue
            cosm = (Vref[ca] @ Vref[cb].T) / (Lref[ca][:, None] * Lref[cb][None, :])
            angm = np.degrees(np.arccos(np.clip(cosm, -1, 1)))
            ai, bi = np.where(np.abs(angm - ao) < ang_tol)
            for a, b in zip(ca[ai], cb[bi]):
                Fr, Fo = _frame(Vref[a], Vref[b]), _frame(qi, qj)
                if Fr is None or Fo is None:
                    continue
                R = Fo @ Fr.T
                nm = int((tree.query((R.T @ g.T).T)[0] < tol).sum())
                if nm > best[0]:
                    best = (nm, R)
            if best[0] >= win:                              # early-out on a strong fit
                break
        if best[0] >= win:
            break

    nm, R = best
    if R is None or nm < max(4, 0.4 * n):
        return IndexResult(None, None, None, None, 0)
    M, C, hkl, inl = refine(g, R @ np.asarray(M_ref, float), tol)
    return IndexResult(M, C, hkl, inl, int(inl.sum()))


def _kabsch(P, Q):
    """Rotation R minimizing |Q - R P| (both point sets through the origin -> no
    translation): R = U diag(1,1,det) V^T from SVD of the cross-covariance Q P^T."""
    U, _, Vt = np.linalg.svd(Q @ P.T)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def _refine_orientation(g, C_ref, R0, tol, n_iter=12):
    """Iterated Kabsch (the constrained gradient descent): assign each spot an hkl
    on the KNOWN reciprocal cell R C_ref, solve the optimal rotation from the inliers,
    repeat. Cell is pinned (3 DOF, orientation only) so it can't drift off M_ref.
    Returns (R, n_inliers)."""
    R = R0
    nm = 0
    for _ in range(n_iter):
        RC = R @ C_ref
        h = np.rint(np.linalg.solve(RC, g.T).T)            # hkl = (R C_ref)^{-1} g
        resid = np.linalg.norm(g - (RC @ h.T).T, axis=1)
        inl = resid < tol
        if inl.sum() < 3:
            break
        nm = int(inl.sum())
        R = _kabsch(C_ref @ h[inl].T, g[inl].T)
    return R, nm


def index_known_gd(g, qmax, M_ref, Vref=None, tol_frac=0.025, len_tol=0.06,
                   ang_tol=5.0, n_seed=12, min_frac=0.4):
    """Known-cell indexer: pair-angle SEED -> iterated-Kabsch REFINE (cell fixed).

    Improves on index_known_pairangle on sparse frames: every pair-angle candidate
    orientation is polished by _refine_orientation (which grows inliers from an
    approximately-right seed), instead of being scored raw and accepted/rejected on
    one free-basis refine. Returns M_ref's lattice rotated into the frame."""
    g = np.asarray(g, float)
    n = len(g)
    if n < 3:
        return IndexResult(None, None, None, None, 0)
    tol = tol_frac * qmax
    M_ref = np.asarray(M_ref, float)
    C_ref = np.linalg.inv(M_ref).T
    if Vref is None:
        Vref = reference_lattice(M_ref, qmax)
    Lref = np.linalg.norm(Vref, axis=1)
    Lobs = np.linalg.norm(g, axis=1)
    seeds = np.argsort(Lobs)[:n_seed]
    short = Lref < 1.3 * Lobs[seeds[-1]]

    best = (0, None)
    for ii in range(len(seeds)):
        for jj in range(ii + 1, len(seeds)):
            i, j = seeds[ii], seeds[jj]
            qi, qj = g[i], g[j]
            ao = np.degrees(np.arccos(np.clip(qi @ qj / (Lobs[i] * Lobs[j]), -1, 1)))
            ca = np.where(short & (np.abs(Lref - Lobs[i]) < len_tol * Lobs[i]))[0]
            cb = np.where(short & (np.abs(Lref - Lobs[j]) < len_tol * Lobs[j]))[0]
            if len(ca) == 0 or len(cb) == 0:
                continue
            cosm = (Vref[ca] @ Vref[cb].T) / (Lref[ca][:, None] * Lref[cb][None, :])
            angm = np.degrees(np.arccos(np.clip(cosm, -1, 1)))
            ai, bi = np.where(np.abs(angm - ao) < ang_tol)
            if len(ai) > 6:                                # refine only the closest-angle
                keep = np.argsort(np.abs(angm[ai, bi] - ao))[:6]
                ai, bi = ai[keep], bi[keep]
            for a, b in zip(ca[ai], cb[bi]):
                Fr, Fo = _frame(Vref[a], Vref[b]), _frame(qi, qj)
                if Fr is None or Fo is None:
                    continue
                R, nm = _refine_orientation(g, C_ref, Fo @ Fr.T, tol)
                if nm > best[0]:
                    best = (nm, R)
                if best[0] >= 0.85 * n:
                    break
            if best[0] >= 0.85 * n:
                break
        if best[0] >= 0.85 * n:                            # early-out on a strong fit
            break

    nm, R = best
    if R is None or nm < max(4, min_frac * n):
        return IndexResult(None, None, None, None, 0)
    M = R @ M_ref
    h = np.rint(np.linalg.solve(R @ C_ref, g.T).T)
    inl = np.linalg.norm(g - (R @ C_ref @ h.T).T, axis=1) < tol
    return IndexResult(M, np.linalg.inv(M).T, h, inl, int(inl.sum()))


def cell_signature(M, k=6, dedup=0.5):
    """k shortest distinct lattice-vector lengths of basis M (rotation/basis invariant)."""
    r = np.arange(-3, 4)
    N = np.array(np.meshgrid(r, r, r, indexing="ij")).reshape(3, -1).T
    N = N[np.any(N != 0, axis=1)]
    L = np.sort(np.linalg.norm((np.asarray(M, float) @ N.T).T, axis=1))
    keep = [L[0]]
    for v in L[1:]:
        if v - keep[-1] > dedup:
            keep.append(v)
        if len(keep) >= k:
            break
    return np.array(keep)


def same_lattice(M1, M2, rtol=0.05, ctol=0.06, vtol=0.10):
    """True iff M1, M2 describe the SAME lattice (rotation/basis invariant).

    Compares Buerger-reduced cell lengths + angles AND volume. The earlier
    length-spectrum signature dedup'd near-equal lengths within 0.5A, so for a
    near-degenerate cell (e.g. tetragonal lyso a=b) any refine drift that split a,b
    inserted an extra signature entry and shifted the positional comparison ->
    spurious reject (injected true axes scored only 23/49 vs the correct 49/49).
    Reduced-cell + volume is degeneracy-robust and also rejects super/sub-cells that
    share short lattice vectors (e.g. a basis built from lyso face diagonals)."""
    if M1 is None or M2 is None:
        return False
    if abs(abs(np.linalg.det(M1)) - abs(np.linalg.det(M2))) > vtol * abs(np.linalg.det(M2)):
        return False
    (l1, c1), (l2, c2) = reduced_params(M1), reduced_params(M2)
    return bool(np.all(np.abs(l1 - l2) <= rtol * l2) and np.all(np.abs(c1 - c2) <= ctol))


def _select_known(vecs, g, tol_abs, Lref, sig_ref, topk=12, len_tol=0.08,
                  min_inlier_frac=0.4):
    """Cell-constrained re-selection of an ALREADY-DETECTED peak list: among triplets
    of the top peaks, take the one that reproduces the known lattice and indexes the
    most spots. The blind miss is a SELECTION error (the true axes are in the peak
    list, the parsimony search just assembled a sub-lattice), so knowing the cell
    fixes it with no new FFT.

    Cheap by construction: keep only peaks whose length matches a known axis length
    (Lref) -- that cuts the top-12 to a handful, so only a few triplets are refined --
    and compare each refined cell to the PRE-COMPUTED reference signature sig_ref
    (no per-triplet recompute of M_ref's spectrum, which was the hot spot)."""
    cand = np.asarray(vecs[:topk], float)
    if len(cand) >= 3:
        lens = np.linalg.norm(cand, axis=1)
        keep = np.any(np.abs(lens[:, None] - Lref[None, :]) <= len_tol * Lref[None, :], axis=1)
        if keep.sum() >= 3:
            cand = cand[keep]                            # length-gate -> few triplets
    best = None
    n_obs = len(g)
    for M0 in _triplets(cand, 0.1):
        M, C, hkl, inl = refine(g, M0, tol_abs)
        if C is None or inl.sum() < min_inlier_frac * n_obs:
            continue
        s = cell_signature(M)
        k = min(len(s), len(sig_ref))
        if k < 3 or not np.all(np.abs(s[:k] - sig_ref[:k]) <= 0.05 * sig_ref[:k]):
            continue
        ni = int(inl.sum())
        if best is None or ni > best[0]:
            best = (ni, M, C, hkl, inl)
    if best is None:
        return None
    _, M, C, hkl, inl = best
    return M, C, hkl, inl


def index_shots_batch_known(gs, qmaxs, M_ref, n="auto", min_len=3.0, topk=12,
                            tol_frac=0.02, min_inlier_frac=0.4, gpu=True,
                            dtype="float32", n_max=256, coarse=True, gd_fallback=True):
    """Phase-2 of the two-phase deployment: index every frame against the KNOWN cell
    (from consensus) at ~blind speed, via a 3-tier cascade that pays for accuracy
    only where it's needed:

      1. fast numba blind search on the batched-detect peaks -- if it already lands
         on the known lattice, done (the ~70-95% blind gets, at 0.5 ms);
      2. else a cheap cell-constrained RE-SELECTION of the same peaks -- rescues
         SELECTION errors (true axes detected but assembled into a sub-lattice),
         no second FFT;
      3. else, only for the rare frames both miss (a DETECTION error -- a true axis
         not in the peak list at all), the spot-space iterated-Kabsch matcher
         index_known_gd, which doesn't depend on the FFT peaks. It's ~50 ms but pays
         on only a few % of frames, so the per-frame average stays near blind speed.
    """
    peaks, n = detect_batch(gs, qmaxs, n=n, min_len=min_len, gpu=gpu, dtype=dtype,
                            n_max=n_max, coarse=coarse)
    M_ref = np.asarray(M_ref, float)
    sig_ref = cell_signature(M_ref)                       # known-lattice spectrum, once
    Lref = np.linalg.norm(M_ref, axis=0)                  # known axis lengths, once
    out = []
    for g, q, (vecs, _amps) in zip(gs, qmaxs, peaks):
        g = np.asarray(g, float)
        tol_abs = tol_frac * q
        cands = vecs[:topk]
        res = search_basis(cands, np.zeros((0, 3)), g, tol_abs, n_real=topk,
                           min_inlier_frac=min_inlier_frac) if len(cands) >= 3 else None
        if res is not None and same_lattice(res[0], M_ref):
            M, C, hkl, inl = res                          # tier 1: blind nailed it
        else:
            sel = _select_known(vecs, g, tol_abs, Lref, sig_ref, topk=topk,
                                min_inlier_frac=min_inlier_frac)
            if sel is not None:
                M, C, hkl, inl = sel                      # tier 2: re-select the peaks
            elif gd_fallback:
                r = index_known_gd(g, q, M_ref)           # tier 3: spot-space rescue
                if r.M is None:
                    out.append(IndexResult(None, None, None, None, 0, vecs))
                    continue
                M, C, hkl, inl = r.M, r.C, r.hkl, r.inliers
            else:
                out.append(IndexResult(None, None, None, None, 0, vecs))
                continue
        out.append(IndexResult(M, C, hkl, inl, int(inl.sum()), vecs))
    return out


def _grp_reduced(reps, RP, rtol, ctol, vtol):
    """Greedy same_lattice grouping over (rep_index, weight) pairs using cached RP[idx]=((l,c),det).

    Returns [rep_index, total_weight, member_indices] for the winner, plus the runner-up weight.

    The runner-up is what tells a decisive lock from a coin-flip between two near-equal clusters --
    the winner's own count cannot, and `max()` alone threw that information away. MEMBERSHIP is kept
    for the same reason: `rep_index` is whichever member happened to CREATE the group, an arbitrary
    choice that is fine for deciding which lattice won and poor for deciding what it is.

    THAT ARBITRARY CHOICE IS ALSO AN ORDER DEPENDENCE, because the group test compares against the
    SEED (`RP[grp[0]]`) and tolerance matching is not transitive: whichever hypothesis arrives first
    defines the group, so the same SET of hypotheses partitions differently under a different input
    order. Measured on mfxl1038923 by permuting the pool order alone (nothing else changed): support
    624-979 on r0278 (1.57x) and 608-1268 on r0058 (2.09x), and on r0058 4 of 20 orderings REFUSED
    outright, because a fragmented cluster falls under min_frac/min_lead. The winning LATTICE is
    robust to this (0 of 36 answering permutations disagreed, axes within 1.2%) -- it is the SUPPORT
    that moves, which is the number the acceptance gates read and the paper quotes.

    SO THE GROUPS ARE SEEDED DENSEST-NEIGHBOURHOOD-FIRST, ties broken on the reduced-cell key, which
    makes the partition a function of the SET alone. GLINT_CONSENSUS_STABLE=0 restores the old
    order-dependent behaviour for reproducing pre-2026-08-08 results.

    Seeding centrally is the point, not merely seeding deterministically. Sorting by the cell key is
    also deterministic and is strictly WORSE than the arbitrary order it replaces (measured on the
    test pool: support 56, against 81-133 for random orders), because a lexicographic sort starts at
    the EDGE of the distribution and a boundary seed reaches only one side of its own cluster,
    slicing it into tolerance-wide slabs. Seeding from the most populated neighbourhood instead gives
    129 on that pool -- the top of the random range -- and drops the runner-up cluster from 38 to 8,
    i.e. it stops fragmenting the true cluster, which improves the min_lead margin as well.

    Density is counted on a coarse grid at the grouping tolerances (geometric in the lengths, linear
    in the angle cosines), which is O(n) rather than the O(n^2) of true neighbour counting. The grid
    is only a seeding heuristic; group membership is still decided by the same tolerance test, so a
    hypothesis landing on the wrong side of a grid boundary costs nothing but seed priority.

    COST OF THE SWITCH, measured: support moves 891 -> 846 on r0278 and 1117 -> 1060 on r0058 (~5%),
    same lattice, cells within 0.7%. A support is now a property of the data instead of the order it
    was read in, which is what makes it quotable at all."""
    if os.environ.get("GLINT_CONSENSUS_STABLE", "1") == "1":
        def _key(i):
            (l, c), d = RP[i]
            return (tuple(l), tuple(c), d)

        def _cellof(i):
            (l, c), _ = RP[i]
            return (tuple(int(round(float(np.log(x)) / np.log1p(rtol))) for x in l)
                    + tuple(int(round(float(y) / ctol)) for y in c))

        dens = {}
        for idx, w in reps:
            dens[_cellof(idx)] = dens.get(_cellof(idx), 0) + w
        reps = sorted(reps, key=lambda t: (-dens[_cellof(t[0])], _key(t[0])))
    groups = []                                               # [rep_index, total_weight, members]
    for idx, w in reps:
        (li, ci), di = RP[idx]
        for grp in groups:
            (lj, cj), dj = RP[grp[0]]
            if (abs(di - dj) <= vtol * dj and bool(np.all(np.abs(li - lj) <= rtol * lj))
                    and bool(np.all(np.abs(ci - cj) <= ctol))):
                grp[1] += w; grp[2].append(idx)
                break
        else:
            groups.append([idx, w, [idx]])
    ranked = sorted(groups, key=lambda g: g[1], reverse=True)
    return ranked[0], (ranked[1][1] if len(ranked) > 1 else 0)


def _accept(win, runner, n_pool, min_support, min_frac, min_lead, min_gap=0):
    """Thin alias for running_consensus.consensus_accept -- see there for what each test means.

    The rule deliberately lives in running_consensus (the lighter module, on the streaming hot path)
    and is imported here rather than duplicated: the two consensus paths previously enforced
    different SUBSETS of it and disagreed on the same data, which is the bug this indirection exists
    to prevent recurring. Keep this a pass-through; do not add a test here without adding it there.
    """
    return consensus_accept(win, runner, n_pool, min_support, min_frac, min_lead, min_gap)


def _rep_of(valid, ridx, members, medoid):
    """Which member of the winning group to report as THE cell.

    `ridx` is whichever member created the group, i.e. an artifact of iteration order. The medoid is
    the member closest to the group's median fingerprint, so it is what the group agrees on rather
    than what it happened to start from. Measured on mfxx49820 r0016 (833 agreeing hypotheses): the
    arbitrary rep gave 38.3/79.1/80.3 against a truth of 38.4/79.3/79.5 (max dev 1.01%), the medoid
    38.3/79.3/79.7 (0.26%). Same lattice either way -- this cannot change WHICH lattice won, only
    which member speaks for it."""
    if not medoid or len(members) < 2:
        return valid[ridx]
    # explicit None test, NOT `x or fallback`: these are numpy arrays, and `or` on one raises
    # "truth value of an array is ambiguous" rather than falling through
    med = group_medoid([valid[i] for i in members])
    return valid[ridx] if med is None else med


def _consensus_exact(valid, min_support, rtol, ctol, vtol, min_frac=0.0, min_lead=1.0,
                     min_gap=0, medoid=True):
    """One Buerger reduction per hypothesis, then greedy same_lattice grouping (reference / CONSENSUS_EXACT=1)."""
    RP = [(reduced_params(M), abs(np.linalg.det(M))) for M in valid]
    (ridx, cnt, mem), runner = _grp_reduced(list(enumerate([1] * len(valid))), RP, rtol, ctol, vtol)
    ok = _accept(cnt, runner, len(valid), min_support, min_frac, min_lead, min_gap)
    rep = _rep_of(valid, ridx, mem, medoid)
    return (rep if ok else None), cnt


def _consensus_fast(valid, min_support, rtol, ctol, vtol, min_frac=0.0, min_lead=1.0,
                    min_gap=0, medoid=True, sig_tol=0.5):
    """Bucket hypotheses by a cheap rotation/perm-invariant signature (sqrt-eig of the Gram G=M^T M, no
    Buerger), run ONE reduced_params per bucket, then group the bucket reps weighted by bucket count.
    ~10x fewer Buerger reductions; sig_tol keeps genuinely-distinct lattices in separate buckets."""
    buckets = {}                                              # sigkey -> [rep_index, count]
    order = []
    for i, M in enumerate(valid):
        ev = np.sqrt(np.clip(np.linalg.eigvalsh(M.T @ M), 0.0, None))   # sorted principal-axis lengths
        k = tuple(int(round(float(x) / sig_tol)) for x in ev)
        b = buckets.get(k)
        if b is None:
            buckets[k] = [i, 1]; order.append(k)
        else:
            b[1] += 1
    reps = [tuple(buckets[k]) for k in order]                 # [(rep_index, count), ...]
    RP = {idx: (reduced_params(valid[idx]), abs(np.linalg.det(valid[idx]))) for idx, _ in reps}
    (ridx, cnt, mem), runner = _grp_reduced(reps, RP, rtol, ctol, vtol)
    ok = _accept(cnt, runner, len(valid), min_support, min_frac, min_lead, min_gap)
    rep = _rep_of(valid, ridx, mem, medoid)
    return (rep if ok else None), cnt


def group_medoid(cells):
    """The most TYPICAL cell of an already-same-lattice group, by reduced-param distance.

    `_grp_reduced` returns whichever member happened to create the group -- an arbitrary choice that
    is fine for deciding WHETHER a lattice won, and poor for deciding WHAT it is. This picks the
    member closest to the group's median fingerprint instead.

    It returns a REAL member rather than a synthesized average on purpose: `reduced_params` sorts
    lengths and cosines INDEPENDENTLY, so the axis-to-angle pairing is destroyed and a cell cannot be
    rebuilt from an averaged fingerprint without inventing a pairing. A medoid needs no such
    reconstruction and carries a valid orientation.

    Distance mixes a RELATIVE length term with an ABSOLUTE cosine term, mirroring how same_lattice
    already compares the two (rtol on lengths, ctol on cosines), so the units are commensurate.
    """
    cells = [c for c in cells if c is not None]
    if not cells:
        return None
    if len(cells) == 1:
        return cells[0]
    RP = [reduced_params(np.asarray(c, float)) for c in cells]
    L = np.array([r[0] for r in RP]); C = np.array([r[1] for r in RP])
    Lm, Cm = np.median(L, axis=0), np.median(C, axis=0)
    d = (np.abs(L - Lm) / np.maximum(Lm, 1e-9)).max(1) + np.abs(C - Cm).max(1)
    return cells[int(np.argmin(d))]


def consensus_cell(Ms, min_support=3, rtol=0.05, ctol=0.06, vtol=0.10, min_frac=0.0,
                   min_lead=1.0, min_gap=0, medoid=True):
    """Group recovered bases by shared lattice; return (representative M, support).

    min_frac / min_lead default to OFF (0.0 / 1.0), so every existing caller keeps its exact
    behaviour; see _accept for what they test and why an absolute min_support alone is not enough
    once the pooled-hypothesis count changes by an order of magnitude.

    Default = per-hypothesis path (`_consensus_exact`): reduced_params cached once per hypothesis (was
    ~0.9s of repeated Buerger reductions in same_lattice; now ~5x less), PROVABLY bit-exact. CONSENSUS_FAST=1
    opts into the signature-bucketed path (`_consensus_fast`): buckets by the cheap Gram-eigenvalue signature
    and runs one reduction per distinct lattice (~2x more on real data, most hypotheses being distinct
    spurious cells) -- helps the one-time latency, negligible on bulk throughput; approximate-by-design."""
    valid = [M for M in Ms if M is not None]
    if not valid:
        return None, 0
    if os.environ.get("CONSENSUS_FAST", "0") == "1" and len(valid) > 8:
        return _consensus_fast(valid, min_support, rtol, ctol, vtol, min_frac, min_lead, min_gap,
                               medoid)
    return _consensus_exact(valid, min_support, rtol, ctol, vtol, min_frac, min_lead, min_gap,
                            medoid)


def index_known(g, qmax, M_ref, tol_frac=0.02, topk=15, min_inlier_frac=0.5):
    """Index a shot constrained to the known lattice M_ref.

    Same FFT + triplet-refine machinery as index_shot, but instead of parsimony it
    accepts the triplet that reproduces M_ref's lattice (length-spectrum match) and
    indexes the most spots -- so a sparse shot whose true axes are present but not the
    parsimonious choice still gets indexed.
    """
    g = np.asarray(g, float)
    tol_abs = tol_frac * qmax
    # known cell -> size the grid to fit its longest axis exactly (no escalation needed;
    # estimate_grid_n undersizes on sparse shots, which is why an unconstrained index
    # would miss the long axis here).
    Lmax = np.max(np.linalg.norm(np.asarray(M_ref, float), axis=0))
    n = int(np.clip(round(4 * qmax * 1.7 * Lmax), 96, 320))
    vol, x = fft_volume(g, qmax, n=n + n % 2)
    vecs, amps = find_peaks_classical(vol, x, min_len=3.0)
    best = None  # (n_inliers, M, C, hkl, inl)
    for M0 in _triplets(vecs[:topk], 0.1):
        M, C, hkl, inl = refine(g, M0, tol_abs)
        if C is None or inl.sum() < min_inlier_frac * len(g):
            continue
        if not same_lattice(M, M_ref):                 # constrain to the known cell
            continue
        if best is None or int(inl.sum()) > best[0]:
            best = (int(inl.sum()), M, C, hkl, inl)
    if best is None:
        return IndexResult(None, None, None, None, 0)
    # NOTE v0 limitation: same_lattice is a length-spectrum signature, not a true
    # lattice-equality test, so it occasionally accepts a signature-collision cell at
    # higher spot counts (-> regression vs single-shot at n>=28). A Niggli-reduced
    # metric-tensor equality (and ideally a peak-independent pair-angle known-cell
    # matcher) is the proper fix.
    _, M, C, hkl, inl = best
    return IndexResult(M, C, hkl, inl, int(inl.sum()))
