"""Productized GLINT hybrid -> CrystFEL .stream (the LUTE/LCLS deliverable, fully blind).

Pipeline (no cell prior):
  1. N-BEST blind-index every frame (top-3 distinct cells)     -> per-frame hypotheses
  2. consensus over the POOLED N-best hypotheses               -> the run cell Mc (sturdier)
  3. per frame pick the consensus-consistent N-best cell       -> recovers ambiguity-demoted truth
  4. cell-GENERAL known-cell GPU rescue for the rest           -> recovered orientations
  5. write a .stream where every indexed crystal shares Mc's metric (consensus-consistent)

The rescue is cell-general (runs on ANY protein); N-best keeps the reachable-but-not-top-1
hypotheses that single-shot orientation ambiguity demotes, arbitrated by cross-frame consensus.

  python hybrid_stream.py [frames.txt] [N] [out.stream]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np
from functools import partial
from glint.glint_fast import GATE_FRAC, GATE_MIN, index_blind_fast, index_blind_nbest, load, matched_strict
from glint.replica_gpu import index_known_gpu_cell
from glint.multishot import consensus_cell, group_medoid, same_lattice
# The pooled-vote acceptance gate (min_frac / min_lead, env-overridable) LIVES in glint.multishot
# next to consensus_cell, so torch-free code can read it; re-exported here because callers and
# experiments/test_consensus_gate.py import it from this module (and rebind it here for A/Bs --
# hybrid_index reads THIS module's globals at call time, so that still works).
from glint.multishot import CONSENSUS_MIN_FRAC, CONSENSUS_MIN_LEAD   # noqa: F401  (re-export)
from glint.stream import write_stream


def _hkl(q, M):
    H = q @ M; r = np.rint(H); inl = np.abs(H - r).max(1) < 0.15
    return r[inl].astype(int), q[inl], int(inl.sum())


GATES = ("none", "strict", "floor")

# Chance floors for gate "floor", (a, b, c): a registration is kept only if matched_strict >= a*n + b + c*sqrt(n)
# for a frame of n peaks, on top of the strict bar. Each one is the 99th percentile of what glint_cli's --cell
# route (hybrid_index: blind N-best, then the per-frame index_known_gpu_cell rescue) matches on azimuth-scrambled
# copies of ONE dataset's frames. DATASET-SPECIFIC: the null moves with the cell, the peak finder, the detector
# and the search depth, so on other data re-fit with experiments/cli_gate_null.py and pass the coefficients.
#   cxidb17: cxidb-17 lysozyme, q480_fix.txt (480 frames, pf8 peaks), --cell 79.02 79.02 37.98 90 90 90,
#            32 copies per frame, CPU torch: 0.26 % of the copies pass (1.0 % below 60 peaks), 344 of the 354 strict
#            frames are kept; experiments/RESULTS_cli_gate_null.md. Not StreamDriver's NULL_FLOOR_CXIDB17, which
#            was fitted for the driver's batched engine and count; the two come out close but were fitted apart.
CLI_NULL_FLOORS = {"cxidb17": (0.0246, 5.39, 1.188)}


def resolve_floor(floor):
    """A floor for gate "floor" -> (a, b) or (a, b, c) as floats. Accepts a name in CLI_NULL_FLOORS or an
    ordered sequence of 2 or 3 finite numbers (a list, tuple or array; a set has no order and is refused, as are
    NaN / inf, strings that are not a known name, and any other length)."""
    if isinstance(floor, str):
        if floor not in CLI_NULL_FLOORS:
            raise ValueError(f"unknown floor name {floor!r}; known: {sorted(CLI_NULL_FLOORS)}, or give a, b[, c]")
        return tuple(float(v) for v in CLI_NULL_FLOORS[floor])
    if not isinstance(floor, (list, tuple, np.ndarray)) or np.ndim(floor) != 1:
        raise ValueError(f"floor must be a name or an ordered (a, b[, c]), got {floor!r}")
    if any(isinstance(v, (bool, np.bool_)) for v in floor):
        raise ValueError(f"floor coefficients must be numbers, not booleans: {floor!r}")
    try:
        fl = tuple(float(v) for v in floor)
    except (TypeError, ValueError):
        raise ValueError(f"floor coefficients must be numbers, got {floor!r}") from None
    if len(fl) not in (2, 3) or not all(np.isfinite(fl)):
        raise ValueError(f"floor must be 2 or 3 finite numbers (a, b[, c]), got {floor!r}")
    return fl


def floor_value(n, floor):
    """The chance floor at n peaks: a*n + b (+ c*sqrt(n)) for floor = (a, b[, c]) (see resolve_floor)."""
    fl = resolve_floor(floor)
    n = np.asarray(n, float)
    return fl[0] * n + fl[1] + (fl[2] * np.sqrt(n) if len(fl) > 2 else 0.0)


def gate_results(results, frames, gate="none", floor=None):
    """Withdraw, in place, the registration of every frame that fails `gate`; return how many were withdrawn.

    Without a gate (``"none"``, the default and the historical output) every registration the indexer returns
    is written as a crystal. With a known cell that is nearly every frame: the rescue's only test is
    `same_lattice(M, Mc)`, which a known-cell search passes by construction, since it returns the cell it was
    asked for. On LUTE's SFX test runs this wrote 28 % of mfx100848724 r51 as crystals where CrystFEL and
    cctbx index about 1 %, and 98 % of mfxl1038923 r58.

    ``"strict"`` applies the paper's scoring bar to what gets WRITTEN: at least GATE_MIN peaks matched and at
    least GATE_FRAC of the frame's peaks (matched_strict, |q @ M - round(q @ M)| < GATE_TOL). A withdrawn frame
    is written exactly as an unindexed one (M, hkl None; q the frame's peaks), so --integrate and --tofile skip
    it too. The strict bar is not a null-calibrated one: on SPARSE lattice-free frames it passes often, because
    the search maximises the matched count over many orientations and 25 % of a few dozen peaks is within reach
    of chance. With --cell on cxidb-17 it writes 29 of 480 azimuth-scrambled frames (all at 38-60 peaks) and
    23 % of scrambled copies below 60 peaks, none above 90 (experiments/RESULTS_cli_gate_null.md).

    ``"floor"`` is strict plus a chance floor: matched_strict >= a*n + b (+ c*sqrt(n)) for a frame of n peaks,
    `floor` a name in CLI_NULL_FLOORS or the coefficients (resolve_floor). Fitted at the 99th percentile of the
    route's matched count on azimuth-scrambled frames, it holds chance accepts to at most about 1 % at every
    peak count for the dataset it was fitted on; the constants are dataset-specific (see CLI_NULL_FLOORS). `floor` must be
    given with gate "floor" and only with it, so a floor never goes silently unused."""
    if gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}, got {gate!r}")
    if len(results) != len(frames):
        raise ValueError(f"results ({len(results)}) and frames ({len(frames)}) must align one to one")
    if gate == "floor":
        if floor is None:
            raise ValueError('gate "floor" needs floor= (a name in CLI_NULL_FLOORS or a, b[, c])')
        fl = resolve_floor(floor)
    elif floor is not None:
        raise ValueError(f'floor= is used only by gate "floor", not {gate!r}')
    if gate == "none":
        return 0
    n = 0
    for r, q in zip(results, frames):
        M = r.get("M")
        if M is None:
            continue
        q = np.asarray(q, float)
        m = matched_strict(np.asarray(M, float), q)
        if m >= GATE_MIN and m >= GATE_FRAC * len(q) and (gate == "strict" or m >= floor_value(len(q), fl)):
            continue
        r.update(M=None, hkl=None, q=q)
        n += 1
    return n



# hybrid_index(select=...): how each frame's registration is chosen among its consensus-consistent candidates.
SELECTS = ("first", "matched")


def _escalation_config(escalate):
    """hybrid_index's `escalate` -> the deep arm's settings, or None when it is off.

    None or False: off. True: the measured defaults. A dict, even an empty one: the defaults with its keys
    overridden. Anything else raises, as do unknown keys and values the arm cannot run with (topa, nc,
    k_null and round_copies integers >= 1, seed an integer >= 0, batch a bool), before any frame is searched.
    batch (default True) runs the deep searches batched (glint.retry_cascade.escalate_batch over
    replica_gpu_batch.index_known_deep_batch), copies round_copies at a time; batch=False is the per-frame
    arm (arm_known_deep), same rule. A bad setting has to fail
    here: inside the arm an indexer exception reads as a frame the deep search missed, so a value like
    topa=1.5 would otherwise switch the escalation off without a word."""
    from glint.retry_cascade import DEEP_K_NULL, DEEP_NC, DEEP_ROUND_COPIES, DEEP_SEED, DEEP_TOPA
    if escalate is None or isinstance(escalate, (bool, np.bool_)):
        if not escalate:
            return None
        escalate = {}
    if not isinstance(escalate, dict):
        raise TypeError(f"hybrid_index(escalate=...): None, True/False or a dict, got {type(escalate).__name__}")
    cfg = dict(topa=DEEP_TOPA, nc=DEEP_NC, k_null=DEEP_K_NULL, seed=DEEP_SEED, batch=True,
               round_copies=DEEP_ROUND_COPIES)
    unknown = set(escalate) - set(cfg)
    if unknown:
        raise ValueError(f"hybrid_index(escalate=...): unknown keys {sorted(unknown)}")
    cfg.update(escalate)
    if not isinstance(cfg["batch"], (bool, np.bool_)):
        raise ValueError(f"hybrid_index(escalate=...): batch must be True or False, got {cfg['batch']!r}")
    cfg["batch"] = bool(cfg["batch"])
    for key, lo in (("topa", 1), ("nc", 1), ("k_null", 1), ("seed", 0), ("round_copies", 1)):
        v = cfg[key]
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < lo:
            raise ValueError(f"hybrid_index(escalate=...): {key} must be an integer >= {lo}, got {v!r}")
        cfg[key] = int(v)
    return cfg

def hybrid_index(frames, images=None, Mc_known=None, warmup=True, nbest=3, cascade=None,
                 alias_gate=None, triage_topk=None, escalate=None, select="first", laue=None):
    """Fully-blind hybrid. (1) N-BEST blind-index every frame (top-`nbest` distinct cells, not just
    argmax). (2) consensus over the POOLED N-best hypotheses (aliases scatter, truth clusters ->
    sturdier cell). (3) per frame pick the highest-scored N-best cell consistent with the consensus
    Mc -- this recovers reachable-but-not-top-1 cells that single-shot orientation ambiguity demotes
    (+5 gated, corroborated). (4) cell-general known-cell GPU rescue for the rest. (5) OPTIONAL
    external-indexer `cascade` (e.g. ffbidx) on whatever is still unindexed -- free insurance for the
    marginal regime, redundant when consensus is strong. Returns write_stream dicts + stats. nbest=1
    reduces to top-1 consensus. Pass Mc_known to skip consensus and rescue against a supplied cell.
    `cascade` = callable(frames_subset, Mc) -> {local_index: M}. frames: list of (N,3) q (1/A).

    (6) OPTIONAL escalation (`escalate`, off by default): every frame still failing the observable gate
    (>= GATE_MIN matched and >= GATE_FRAC of its peaks) gets a DEEP known-cell search, accepted only if the
    fit beats all of its own azimuth-scrambled copies (glint.retry_cascade.arm_known_deep's rule). True for
    the measured defaults, or a dict overriding topa / nc / k_null / seed / batch / round_copies (checked
    before any search, see _escalation_config). By default the searches are batched
    (glint.retry_cascade.escalate_batch over replica_gpu_batch.index_known_deep_batch: all misses in one
    pass, then the scrambled copies 8 at a time for the frames still undecided) -- on the 480 set the stage
    went from about 18 s per-frame to 0.93 s with the same accepts (exp/batched-escalation
    RESULTS_batched_deep.md); batch=False runs the per-frame arm. On the cxidb-17 480 set the escalation took
    the strict count from 366 to 387 (RESULTS_escalation_k32.md, job 39181473). An escalated frame is labelled `escalated` with its null
    record, and both stream writers carry that into the chunk as glint/escalated and the null's numbers
    (glint.stream.escalation_lines). It is A lattice of the frame, not necessarily the one the shipped
    search would have found: on a double hit it can be the other crystal.

    `select` chooses each frame's registration in steps (3)-(4). "first" (the default, the rule above): the
    first consensus-consistent N-best cell, the known-cell search only when there is none. "matched": the
    known-cell search runs on EVERY frame and the frame keeps whichever consensus-consistent candidate (its
    N-best cells, then the known-cell fit) matches the most peaks (matched_strict; ties keep that order). On
    the cxidb-17 480 set this took 366 to 370 (exp/joint-ceiling RESULTS.md, "ADMM-lite round 0"); see
    experiments/select_matched/ for the preregistered check. It costs one known-cell search per frame."""
    if select not in SELECTS:
        raise ValueError(f"hybrid_index(select=...): one of {SELECTS}, got {select!r}")
    esc_cfg = _escalation_config(escalate)
    kc = {} if laue is None else {"laue": laue}             # known-cell engines: see replica_gpu._both_hands
    n = len(frames)
    images = images or [{"image": "glint.cxi", "event": i} for i in range(n)]
    if warmup and n:
        index_blind_nbest(frames[0], nbest)

    NB = [index_blind_nbest(q, nbest) for q in frames]           # [(cell,score),...] per frame
    top1 = [nb[0][0] if nb else None for nb in NB]
    n_blind = sum(1 for nb in NB if nb)

    n_pool = n_refine = 0
    vote_idx = list(range(n))          # bound on BOTH branches; the known-cell path never votes
    if Mc_known is not None:
        Mc, support = np.asarray(Mc_known, float), -1
    else:                                                        # consensus over the POOLED hypotheses
        # TRIAGE (opt-in). The streaming path votes over the top-`warm_topk` frames by peak count
        # (warmup_batch); this path has always pooled EVERY frame, which is the worse position to be
        # in and not the safer one. The random-agreement floor grows with the number of hypotheses
        # drawn, so pooling more frames does not buy a sturdier cell past a point -- it buys a bigger
        # lottery. Measured on mfxx49820 r0016 (2228 frames, unrefined geometry): pooling all 6294
        # hypotheses locked a WRONG doubled-c cell on a 23-vote cluster, while the triaged vote
        # REFUSED at every k from 16 to 512, because 1-2 votes out of 48-1536 can never clear
        # min_support AND the runner-up margin. Triage also picks the STRONGEST diffraction, so the
        # frames it does keep are the ones most likely to index correctly.
        #
        # Default None = pool everything, i.e. bit-identical to before. Set it and the vote is
        # restricted; the per-frame N-best pick and the rescue below still see every frame, so this
        # changes only WHO VOTES, never who gets indexed.
        if triage_topk:
            from glint.warmup_batch import triage_order            # the SAME selector streaming uses
            vote_idx = triage_order([len(q) for q in frames], int(triage_topk), floor=1)
        pool = [c for i in vote_idx for c, _ in NB[i]]
        # A bare min_support=3 is an ABSOLUTE floor, and this pool is not a fixed size: it is
        # n_frames x nbest. At 120 frames that is 360 hypotheses; at 2200 it is ~6300, where a
        # 23-cluster (0.4% of the pool) clears 3 by chance and every downstream frame then gets
        # rescued onto that wrong cell -- observed on mfxx49820 r0016, which reported "95% indexed"
        # against a doubled-c lattice. Gate on the SHARE of the pool and on the margin over the
        # runner-up as well. Both are no-ops on the small-N regime the benchmarks use.
        Mc, support = consensus_cell(pool, min_frac=CONSENSUS_MIN_FRAC, min_lead=CONSENSUS_MIN_LEAD)
        n_pool = len(pool)
        # TRIAGE-THEN-REFINE. Deciding on 32 frames and REPORTING a cell derived from 32 frames are
        # separate choices, and only the first needs to be small. Measured on mfxx49820 r0016 with
        # the good geometry: the triaged vote's cell came out 37.9/79.6/81.0 against the full vote's
        # 38.3/79.1/80.3 (truth 38.4/79.3/79.5) -- both same_lattice, but the triaged one demonstrably
        # less accurate, because fewer members means less averaging.
        #
        # So: the DECISION stays triaged (that is the defence), then the cell is re-picked as the
        # medoid of every hypothesis on EVERY frame that agrees with it. That is strictly more data
        # for the same decision, and it cannot change which lattice was chosen -- `same_lattice` is
        # the filter, so the medoid is by construction the same lattice as Mc.
        if triage_topk and Mc is not None:
            agree = [c for nb in NB for c, _ in nb if same_lattice(c, Mc)]
            if len(agree) > support:                    # more agreeing cells than the vote itself saw
                Mr = group_medoid(agree)
                if Mr is not None and same_lattice(Mr, Mc):
                    Mc, n_refine = Mr, len(agree)
        # Deterministic complement to the statistical gate above: the vote share cannot tell a cell
        # from its own index<=N super-cell, because a doubled axis collects exactly the same peaks --
        # they just sit on every OTHER node. AliasGate scores coverage*occupancy over the derivative
        # lattices, so a super-cell is caught by its systematically absent nodes. Opt-in: default None
        # leaves this path bit-identical.
        #
        # NB an earlier version of this comment said the streaming driver "has had this since the
        # alias-gate work". That is WRONG and worth recording: stream_driver's `_alias_gate` is used
        # only inside `_watchdog` (stream_driver.py:1030,1046), and `_watchdog`'s single call site
        # (:1114) sits in the `elif slots:` arm of `if slots and not self.adaptive_relock`, with
        # `adaptive_relock=False` the shipped default (:479). So the gate gets nowhere near the
        # PRIMARY blind lock (`_push_blind` -> `_lock`) that sets self.Mc -- passing
        # `alias_gate=AliasGate()` to a default StreamDriver is a silent no-op. Neither path was
        # protected where it mattered; this call is the first place the gate guards a primary lock.
        if Mc is not None and alias_gate is not None:
            # drawn from the SAME frames that voted, so the alias check and the statistical check
            # are answering about one population rather than two.
            #
            # PER FRAME, in each frame's own orientation -- NOT `confirm(Mc, np.vstack(voters))`, which
            # is what this line used to do and which cannot work: coverage and occupancy are both
            # computed in the leader's frame, so a pooled cloud of many orientations puts every
            # candidate at chance coverage and leaves the score a 1/V preference for smaller cells.
            # Measured on the cxidb-62 lock, that refused the TRUE cell (every half-volume derivative
            # scored 1.5-1.8x the leader) while the same gate per frame confirmed it 30/40. The frame's
            # own agreeing N-best hypothesis IS the leader's lattice in that frame's orientation, so
            # the correct input is already in hand -- no re-indexing.
            voters = [(frames[i], next(c for c, _ in NB[i] if same_lattice(c, Mc)))
                      for i in vote_idx
                      if any(same_lattice(c, Mc) for c, _ in NB[i])]
            if voters:
                Mc = alias_gate.confirm_frames(Mc, voters)       # may return a tighter alias, or None

    results = []; n_idx = n_resc = n_nb = n_kc = n_swap = 0
    for q, nb, t1, meta in zip(frames, NB, top1, images):
        M = None
        if Mc is not None and select == "matched":              # best-matching consistent candidate, KC included
            cands = [c for c, _ in nb if same_lattice(c, Mc)]
            nblind = len(cands)
            Mr = index_known_gpu_cell(q, Mc, **kc); n_kc += 1
            if Mr is not None and same_lattice(Mr, Mc):
                cands.append(Mr)
            if cands:
                j = int(np.argmax([matched_strict(c, q) for c in cands]))   # first max: blind order, then KC
                M = cands[j]
                if j == nblind:
                    n_resc += 1                                  # the known-cell fit
                else:
                    n_nb += (M is not t1)
                n_swap += (nblind > 0 and j > 0)                 # the shipped rule would have kept cands[0]
        if Mc is not None and select == "first":                 # pick best consensus-consistent N-best hypothesis
            for c, _ in nb:
                if same_lattice(c, Mc):
                    M = c
                    n_nb += (c is not t1)                        # recovered via a non-top-1 hypothesis
                    break
        if M is None and Mc is not None and select == "first":   # cell-general GPU known-cell rescue (ffbidx-style)
            Mr = index_known_gpu_cell(q, Mc, **kc); n_kc += 1
            if Mr is not None and same_lattice(Mr, Mc):
                M = Mr; n_resc += 1
        if M is None and Mc is None:                             # no consensus formed -> top-1 fallback
            M = t1
        if M is not None:
            hkl, qin, _ = _hkl(q, M); n_idx += 1
        else:
            hkl, qin = None, q
        results.append({"image": meta["image"], "event": meta["event"], "M": M, "q": qin, "hkl": hkl})
    n_casc = 0                                                   # (5) optional external cascade on the still-unindexed
    if cascade is not None and Mc is not None:
        resid = [i for i, r in enumerate(results) if r["M"] is None]
        if resid:
            cm = cascade([results[i]["q"] for i in resid], Mc)  # {local_index: M}
            for k, i in enumerate(resid):
                Mx = cm.get(k)
                if Mx is not None and same_lattice(Mx, Mc):
                    hkl, qin, _ = _hkl(results[i]["q"], Mx)
                    results[i].update({"M": Mx, "q": qin, "hkl": hkl})
                    n_idx += 1; n_casc += 1
    esc_stats = None
    if esc_cfg is not None and Mc is not None:                   # (6) optional escalation on the misses
        from glint.retry_cascade import arm_known_deep

        def _obs(M, q):
            m = matched_strict(M, q)
            return m >= GATE_MIN and m >= GATE_FRAC * len(q)

        def _gate(M, q):
            return _obs(M, q) and same_lattice(M, Mc)

        misses = [i for i, q in enumerate(frames) if results[i]["M"] is None or not _obs(results[i]["M"], q)]
        n_esc = searches = 0
        qm = [np.asarray(frames[i], float) for i in misses]
        if esc_cfg["batch"]:
            from glint.replica_gpu_batch import index_known_deep_batch
            from glint.retry_cascade import escalate_batch

            def _search(qs, Mcell):
                return index_known_deep_batch(qs, Mcell, esc_cfg["topa"], esc_cfg["nc"], return_errors=True, **kc)
            Ms, recs = escalate_batch(qm, Mc, _search, matched_strict, _gate,
                                      seeds=[[esc_cfg["seed"], i] for i in misses], k_null=esc_cfg["k_null"],
                                      round_copies=esc_cfg["round_copies"])
        else:
            Ms, recs = [], []
            for i, q in zip(misses, qm):
                M, rec = arm_known_deep(q, Mc, partial(index_known_gpu_cell, **kc) if kc else index_known_gpu_cell,
                                        matched_strict, _gate,
                                        seed=[esc_cfg["seed"], i], topa=esc_cfg["topa"], nc=esc_cfg["nc"],
                                        k_null=esc_cfg["k_null"])
                Ms.append(M); recs.append(rec)
        for i, q, M, rec in zip(misses, qm, Ms, recs):
            searches += rec["searches"]
            if M is not None:
                hkl, qin, _ = _hkl(q, M)
                if results[i]["M"] is None:
                    n_idx += 1
                results[i].update({"M": M, "q": qin, "hkl": hkl, "escalated": True, "escalation": rec})
                n_esc += 1
        esc_stats = dict(n_escalation_candidates=len(misses), n_escalated=n_esc,
                         escalation_searches=searches, escalation=esc_cfg)
    edges = np.round(np.sort(np.linalg.norm(Mc, axis=0)), 1) if Mc is not None else None
    stats = {"n": n, "n_blind": n_blind, "support": support, "edges": edges, "n_nbest": n_nb,
             "n_resc": n_resc, "n_casc": n_casc, "n_idx": n_idx, "Mc": Mc,
             "select": select, "n_kc_searches": n_kc, "n_select_swaps": n_swap,
             "n_pool": n_pool, "support_frac": (support / n_pool) if n_pool else None,
             "n_voters": len(vote_idx) if Mc_known is None else 0, "triage_topk": triage_topk,
             "n_refine": n_refine,
             "consensus_refused": bool(Mc is None and n_pool)}
    if esc_stats is not None:
        stats.update(esc_stats)
    return results, stats


def dense_index(frames, images=None, warmup=True):
    """Dense/rotation path: each frame self-indexes via the local-cluster-FFT front end
    (`index_blind_cluster_seeded`) -- no cross-frame consensus needed because a rotation cloud is
    3D-complete. Below CLUSTER_MIN rlps the front end auto-falls-back to the Fibonacci grid, so this is
    safe on mixed data; the CLI picks this path only when the median rlp count is dense."""
    from glint.glint_fast import index_blind_cluster_seeded
    n = len(frames)
    images = images or [{"image": "glint.cxi", "event": i} for i in range(n)]
    if warmup and n:
        index_blind_cluster_seeded(frames[0])
    results = []; n_idx = 0
    for q, meta in zip(frames, images):
        M = index_blind_cluster_seeded(q)
        if M is not None:
            hkl, qin, _ = _hkl(q, M); n_idx += 1
        else:
            hkl, qin = None, q
        results.append({"image": meta["image"], "event": meta["event"], "M": M, "q": qin, "hkl": hkl})
    stats = {"n": n, "mode": "dense", "n_blind": n_idx, "support": -1, "edges": None,
             "n_nbest": 0, "n_resc": 0, "n_casc": 0, "n_idx": n_idx, "Mc": None}
    return results, stats


def _floor_note(stats):
    fl = stats.get("gate_floor")
    if fl is None:
        return ""
    return f"; floor {fl[0]:g} n + {fl[1]:g}" + (f" + {fl[2]:g} sqrt(n)" if len(fl) > 2 else "")


def _report(stats, out):
    n = max(stats["n"], 1)
    if stats.get("mode") == "dense":
        print(f"=== GLINT dense/rotation (local-cluster FFT), N={stats['n']} ===")
        if "n_gated" in stats:
            print(f"  gate ({stats['gate']})      : {stats['n_gated']} registrations withdrawn (written as unindexed)"
                  + _floor_note(stats))
        print(f"  indexed            : {stats['n_idx']}/{stats['n']} ({100*stats['n_idx']//n}%)  -> {out}")
        return
    print(f"=== GLINT hybrid (blind+consensus+general-rescue), N={stats['n']} ===")
    print(f"  blind indexed      : {stats['n_blind']}/{stats['n']} ({100*stats['n_blind']//n}%)")
    frac = stats.get("support_frac")
    share = f" = {100*frac:.1f}% of {stats['n_pool']} pooled" if frac is not None else ""
    if stats.get("consensus_refused"):
        # Loud on purpose: this is the run reporting that it does NOT have a cell, and every
        # subsequent frame fell back to its own top-1 rather than being rescued onto a shared one.
        print(f"  consensus REFUSED  : best cluster {stats['support']}{share} -- below the "
              f"min_frac/min_lead gate; per-frame top-1 fallback, NO rescue")
    else:
        print(f"  consensus cell     : {stats['edges']} A  support {stats['support']}{share}")
    if stats.get("n_nbest"):
        print(f"  N-best recovered   : {stats['n_nbest']} (consensus-consistent non-top-1 hypothesis)")
    if stats.get("select") == "matched":
        print(f"  known-cell picks   : {stats['n_resc']} (select=matched: the known-cell fit matched the most peaks; "
              f"{stats['n_select_swaps']} frames changed pick, {stats['n_kc_searches']} known-cell searches)")
    else:
        print(f"  rescued failures   : {stats['n_resc']}")
    if stats.get("n_casc"):
        print(f"  cascade recovered  : {stats['n_casc']} (external fallback on still-unindexed)")
    if "n_escalated" in stats:
        c = stats["n_escalation_candidates"]
        print(f"  escalated          : {stats['n_escalated']} of {c} gate-failing frames "
              f"(deep search + scrambled null, {stats['escalation_searches'] / max(c, 1):.1f} searches each)")
    if "n_gated" in stats:
        print(f"  gate ({stats['gate']})      : {stats['n_gated']} registrations withdrawn (written as unindexed)"
              + _floor_note(stats))
    print(f"  FINAL indexed      : {stats['n_idx']}/{stats['n']} ({100*stats['n_idx']//n}%)  -> {out}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = sys.argv[3] if len(sys.argv) > 3 else "glint_hybrid.stream"
    frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6]
    if N: frames = frames[:N]
    images = [{"image": os.path.basename(path), "event": i} for i in range(len(frames))]
    results, stats = hybrid_index(frames, images)
    write_stream(results, out)
    _report(stats, out)


def index_known_fast(frames, Mc, batch=32, images=None, laue=None):
    """Throughput-critical KNOWN-CELL steady state: batch the known-cell engine across frames
    (glint.replica_gpu_batch), skipping the blind N-best pass entirely -- use once consensus (or a
    supplied prior) fixes the cell. ~7x faster than per-frame index_known_gpu_cell rescue and
    self-contained (no ffbidx handoff). Returns (results, stats) in the same shape as hybrid_index;
    for max known-cell ACCURACY use hybrid_index(..., Mc_known=Mc) instead (runs the N-best pass)."""
    from glint.replica_gpu_batch import index_known_gpu_cell_batch
    n = len(frames)
    images = images or [{"image": "glint.cxi", "event": i} for i in range(n)]
    Ms = []
    for i in range(0, n, batch):
        Ms += index_known_gpu_cell_batch(frames[i:i + batch], Mc, **({} if laue is None else {"laue": laue}))
    Mcn = np.asarray(Mc, float)
    results = []; n_idx = 0
    for M, q, meta in zip(Ms, frames, images):
        if M is not None and same_lattice(M, Mcn):
            hkl, qin, _ = _hkl(q, M); n_idx += 1
        else:
            M, hkl, qin = None, None, q
        results.append({"image": meta["image"], "event": meta["event"], "M": M, "q": qin, "hkl": hkl})
    stats = {"n": n, "n_idx": n_idx, "mode": "known_fast", "Mc": Mcn,
             "edges": np.round(np.sort(np.linalg.norm(Mcn, axis=0)), 1)}
    return results, stats
