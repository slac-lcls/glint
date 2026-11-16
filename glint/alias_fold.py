"""Fold a consensus refusal that is only an alias tie: leader vs its own index-2 relative, same frames.

The pooled vote (glint.multishot.consensus_cell) groups hypotheses by reduced cell AND volume, so a
lattice and its index-n super- or sub-cell are always separate groups. When the N-best search returns
both on the same frames -- it does: a doubled axis indexes exactly the same peaks, they just sit on every
other node -- the two groups split one population's votes, and the runner-up ratio test (min_lead) can
refuse a run whose answer is not in doubt. This has been seen on a real run: the runner-up was an
index-2 supercell of the leader, proposed on nearly all of the leader's own frames (none after shuffling
the frame labels), with a lead below the 1.5 the rule asks for.

The vote cannot settle which of the two is the lattice -- both index the same peaks -- but the alias
gate can: AliasGate.confirm_frames scores coverage x occupancy per frame over the derivative family,
and a super-cell loses on occupancy while a sub-cell loses on coverage. So, ON A REFUSAL ONLY:

  1. regroup the pool exactly as consensus_cell's exact path does, keeping every group and the frame
     each hypothesis came from;
  2. for each group whose representative is an index <= max_index relative of the leader's AND whose
     frames are mostly the leader's frames (`min_shared`), ask the gate about both cells on the shared
     frames, each frame in its own orientation;
  3. if exactly one of the pair is confirmed and the other refused, drop the loser from contention.
     Anything else (both confirmed, both refused, an abstention) leaves the pair alone;
  4. re-run the SAME acceptance rule (running_consensus.consensus_accept) on what is left, with the
     pool size unchanged, so a dropped group's hypotheses still count against min_frac.

What this does NOT do: it never adds a loser's votes to the winner (that would assume the answer), it
never runs on an accepted vote (the post-accept gate in hybrid_index covers that), and it never runs
when any test other than min_lead failed. Off unless hybrid_index(alias_fold=True), which also needs an
alias_gate.

Measured: nothing on real data yet. The unit tests in experiments/test_alias_fold.py are synthetic.
"""
import numpy as np

from .alias_gate import AliasGate, derivative_lattices
from .multishot import _grp_reduced, _rep_of, same_lattice
from .lattice import reduced_params
from .running_consensus import consensus_accept, consensus_failed_tests


def _strict_gate(gate):
    """The caller's gate with adopt forced off: here only the verdict is read, never a swapped cell."""
    return AliasGate(max_index=gate.max_index, margin=gate.margin, hkl_tol=gate.hkl_tol, adopt=False,
                     min_coverage=gate.min_coverage, frac_beat=gate.frac_beat,
                     min_frames=gate.min_frames)


def _is_relative(rep_lead, det_lead, rep_g, det_g, max_index, vtol):
    """True if rep_g is (within same_lattice tolerance) an index 2..max_index sub- or super-cell of
    rep_lead. Volume ratio first: it is free and rejects nearly every unrelated group."""
    r = det_g / det_lead
    if not any(abs(r - k) <= vtol * k or abs(r - 1.0 / k) <= vtol / k for k in range(2, max_index + 1)):
        return False
    return any(same_lattice(D, rep_g) for D in derivative_lattices(rep_lead, max_index)[1:])


def fold_alias_relatives(pool, pool_fid, frames, gate, min_support=3, min_frac=0.0, min_lead=1.0,
                         min_gap=0, rtol=0.05, ctol=0.06, vtol=0.10, medoid=True, max_groups=6,
                         min_shared=0.5, max_gate_frames=200):
    """Re-decide a refused vote after removing alias relatives the gate can tell apart.

    pool       hypotheses as handed to consensus_cell (None entries allowed, dropped as there).
    pool_fid   the frame index each hypothesis came from, parallel to `pool`.
    frames     the frames' reciprocal peaks, indexed by pool_fid.
    gate       an AliasGate; its thresholds are used, its adopt flag is not.
    max_gate_frames  the gate scores at most this many shared frames, evenly spaced over them in frame
               order (deterministic). tightness enumerates the node grid per call, ~0.2 s per frame for
               the pair on a 56/70/102 A cell at |q| <= 0.5, so an uncapped 1000-frame pair costs minutes.
               None = all.

    Returns (Mc or None, support, info). `info["pairs"]` records every relative examined and why it was
    or was not dropped; `info["failed"]` is the acceptance rule's verdict after the fold."""
    valid, fid = [], []
    for c, f in zip(pool, pool_fid):
        if c is not None:
            valid.append(np.asarray(c, float)); fid.append(int(f))
    info = {"pairs": [], "dropped": [], "n_valid": len(valid)}
    if not valid:
        info["failed"] = ["min_support"]
        return None, 0, info
    RP = [(reduced_params(M), abs(np.linalg.det(M))) for M in valid]
    groups = []
    _grp_reduced(list(enumerate([1] * len(valid))), RP, rtol, ctol, vtol, groups_out=groups)
    g_strict = _strict_gate(gate)

    def rep(g):
        return _rep_of(valid, g[0], g[2], medoid)

    def frames_of(g):
        out = {}
        for i in g[2]:
            out.setdefault(fid[i], i)                      # first hypothesis of this group on that frame
        return out

    alive = list(range(len(groups)))
    changed = True
    while changed:                                         # a dropped leader promotes its relative; re-check
        changed = False
        lead = alive[0]
        R0 = rep(groups[lead]); d0 = abs(np.linalg.det(R0)); F0 = frames_of(groups[lead])
        for j in [k for k in alive[1:] if k < max_groups]:   # only the top groups are examined;
                                                                # every group still counts as a runner-up
            if any(p["groups"] == (lead, j) for p in info["pairs"]):
                continue
            Rj = rep(groups[j]); dj = abs(np.linalg.det(Rj))
            if not _is_relative(R0, d0, Rj, dj, gate.max_index, vtol):
                continue
            Fj = frames_of(groups[j])
            shared = sorted(set(F0) & set(Fj))
            rec = {"groups": (lead, j), "support": (groups[lead][1], groups[j][1]),
                   "frames": (len(F0), len(Fj)), "shared": len(shared),
                   "volume_ratio": float(dj / d0)}
            if len(shared) < max(min_shared * len(Fj), g_strict.min_frames):
                rec["outcome"] = "not on the same frames"
                info["pairs"].append(rec)
                continue
            if max_gate_frames and len(shared) > max_gate_frames:
                pick = np.linspace(0, len(shared) - 1, int(max_gate_frames)).round().astype(int)
                shared = [shared[k] for k in np.unique(pick)]
            rec["gated_frames"] = len(shared)
            g_strict.confirm_frames(R0, [(frames[f], valid[F0[f]]) for f in shared])
            v0 = g_strict.info["verdict"]; rec["leader_gate"] = dict(g_strict.info)
            g_strict.confirm_frames(Rj, [(frames[f], valid[Fj[f]]) for f in shared])
            vj = g_strict.info["verdict"]; rec["relative_gate"] = dict(g_strict.info)
            if v0 == "confirm" and vj == "refuse":
                rec["outcome"] = "relative dropped"
                alive.remove(j); info["dropped"].append(j)
            elif vj == "confirm" and v0 == "refuse":
                rec["outcome"] = "leader dropped"
                alive.remove(lead); info["dropped"].append(lead)
                alive.sort(key=lambda k: -groups[k][1])
                changed = True
            else:
                rec["outcome"] = f"undecided (leader {v0}, relative {vj})"
            info["pairs"].append(rec)
            if changed:
                break

    win = groups[alive[0]][1]
    runner = groups[alive[1]][1] if len(alive) > 1 else 0
    n = len(valid)
    info["failed"] = consensus_failed_tests(win, runner, n, min_support, min_frac, min_lead, min_gap)
    info.update(support=int(win), runner=int(runner), leader_group=alive[0])
    ok = consensus_accept(win, runner, n, min_support, min_frac, min_lead, min_gap)
    return (rep(groups[alive[0]]) if ok else None), int(win), info
