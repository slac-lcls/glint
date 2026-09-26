#!/usr/bin/env python3
"""Where a per-lattice replay's numbers come from: two recorder runs on the same input compared frame by
frame, and a chance test for every frame the per-lattice path rescued.

The two runs are experiments/record_stream_replay.py outputs of the same arm, one without and one with
--per-lattice. The report gives
  * the totals of both (strict whole-frame, the per-lattice column, accepts, misses, watchdog rescues,
    relocks);
  * every frame whose outcome, cell or whole-frame strict verdict differs between the two;
  * every rescued_per_lattice frame: both lattices' counts, which one was kept, and whether the kept
    lattice passes the strict gate on the whole frame anyway;
  * every frame only the per-lattice column credits;
  * CHANCE: for each rescued frame, the driver's own known-cell search (replica_gpu_batch.index_fused,
    the call StreamDriver._index_integrate makes) run on --null azimuth-scrambled copies of the frame's
    peaks (multilattice.scramble_azimuth: |q| and the Ewald geometry kept, lattice coherence destroyed).
    The count it reaches there is what a registration explains with no crystal behind it; the rescued
    frame's first registration (n1) is placed in that distribution. The search runs against the cell the
    replay locked (header primary_cell), with the live window stream_driver.HKL_TOL.

  PYTHONPATH=. python experiments/per_lattice_report.py --base default.json --pl per_lattice.json \\
      --input q480_fix.txt --null 32 --out report.json

CPU is fine (index_fused falls back to the graph path); the null is seeded per frame.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def _totals(d):
    t, c = d["header"]["totals"], d["header"]["counters"]
    return dict(strict=t["strict_ok"], per_lattice=t.get("strict_ok_per_lattice"), indexed=t["indexed"], miss=t["miss"],
                watchdog_rescues=c["n_watchdog_rescued"], relocks=c["n_relock"],
                rescued_per_lattice=t.get("rescued_per_lattice", 0))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="recorder JSON, per_lattice off")
    ap.add_argument("--pl", required=True, help="recorder JSON, per_lattice on (same input and arm otherwise)")
    ap.add_argument("--input", required=True, help="the q list both runs replayed (FRAME-block .txt)")
    ap.add_argument("--null", type=int, default=32, help="azimuth-scrambled copies per rescued frame (0: skip)")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    B, P = json.load(open(a.base)), json.load(open(a.pl))
    hb, hp = B["header"], P["header"]
    assert [i["digest"] for i in hb["inputs"]] == [i["digest"] for i in hp["inputs"]], "different inputs"
    kb = {k: v for k, v in hb["driver_kw"].items() if k not in ("per_lattice", "per_lattice_below")}
    kp = {k: v for k, v in hp["driver_kw"].items() if k not in ("per_lattice", "per_lattice_below")}
    assert kb == kp, f"different arms: {kb} vs {kp}"
    # driver_kw leaves out the roster, and the scoring references, the gate, the ingest mode and the geometry
    # live beside it; any of them differing would be misread below as an effect of per_lattice.
    for key in ("refs", "roster", "gate", "ingest", "geometry"):
        assert hb.get(key) == hp.get(key), f"different {key}: {hb.get(key)} vs {hp.get(key)}"
    assert not hb["driver_kw"].get("per_lattice") and hp["driver_kw"].get("per_lattice"), "base must be off, pl on"
    rb = {r["i"]: r for r in B["records"]}
    rp = {r["i"]: r for r in P["records"]}
    n = len(rb)
    # The digests say both runs read the same pools; they do not say both replayed the same frames in the same
    # order -- a --schedule (or its seed) interleaves the pools, and it lives outside driver_kw. Records are
    # aligned by i below, so the frame behind each i must match before any delta means anything.
    assert hb.get("schedule") == hp.get("schedule"), "different schedules: records would pair different frames"
    assert sorted(rb) == sorted(rp) == list(range(n)), "record indices differ or are not 0..n-1"
    ident = lambda r: (r.get("truth"), r.get("src_index"), r.get("src"), r.get("src_event"))
    bad = [i for i in range(n) if ident(rb[i]) != ident(rp[i])]
    assert not bad, f"{len(bad)} frames differ in (truth, src_index, src, src_event), first at i={bad[0]}"
    rep = dict(inputs=hp["inputs"], driver_kw=hp["driver_kw"], git=dict(base=hb["provenance"]["git"], pl=hp["provenance"]["git"],
                                                                base_dirty=hb["provenance"].get("git_dirty"),
                                                                pl_dirty=hp["provenance"].get("git_dirty")),
               totals=dict(base=_totals(B), pl=_totals(P)))

    changed = []
    for i in range(n):
        x, y = rb[i], rp[i]
        if x["o"] != y["o"] or x.get("cell_name") != y.get("cell_name") or x["ok"] != y["ok"]:
            changed.append(dict(i=i, base=[x["o"], x.get("cell_name"), x["ok"]], pl=[y["o"], y.get("cell_name"), y["ok"]]))
    rep["changed"] = changed

    rescued, only_pl = [], []
    for i in range(n):
        y = rp[i]
        sl = y.get("second_lattice")
        if y["o"] == "rescued_per_lattice":
            rescued.append(dict(i=i, npk=y["npk"], n1=sl["n1"], m2=sl["m2"], dominant=sl["dominant"], kept=sl["kept"],
                                misorientation=sl["misorientation"], frac_first=round(sl["n1"] / sl["n_peaks"], 4),
                                strict_whole_frame=y["ok"], frac_whole_frame=y["frac"], frac_per_lattice=y.get("frac_pl"),
                                base_outcome=rb[i]["o"], base_cell=rb[i].get("cell_name")))
        if y.get("ok_pl") and not y["ok"]:
            only_pl.append(dict(i=i, o=y["o"], npk=y["npk"], n1=sl["n1"], m2=sl["m2"], dominant=sl["dominant"],
                                kept=sl["kept"], misorientation=sl["misorientation"], frac_whole_frame=y["frac"],
                                frac_per_lattice=y.get("frac_pl")))
    rep["rescued_per_lattice"] = rescued
    rep["per_lattice_only"] = only_pl

    if a.null and rescued:
        import glint.replica_gpu_batch as rgb
        from glint.glint_fast import load
        from glint.lattice import cell_to_Ar
        from glint.multilattice import claimed_mask, scramble_azimuth
        from glint.stream_driver import HKL_TOL
        from record_stream_replay import digest
        # The null is attached to frame ids, so --input must BE the replayed stream, frame for frame: one q
        # input read in file order (no schedule), whose digest is the one the recorder wrote.
        if len(hp["inputs"]) != 1 or hp.get("schedule") is not None or hp["inputs"][0].get("kind") != "q":
            raise SystemExit("--null needs a single q-list input replayed in file order; this run used "
                             f"{len(hp['inputs'])} input(s), schedule={'yes' if hp.get('schedule') else 'no'}")
        frames = [np.asarray(q, float) for q in load(a.input)]
        assert len(frames) == n, (len(frames), n)
        assert digest(frames) == hp["inputs"][0]["digest"], \
            f"--input digest {digest(frames)} is not the replayed input's {hp['inputs'][0]['digest']}"
        assert all(rp[i].get("src_index") == i for i in range(n)), "replay order is not file order"
        Mc = np.asarray(cell_to_Ar(*hp["primary_cell"]), float)

        def count(q, M):
            return 0 if M is None else int(claimed_mask(q, M, HKL_TOL).sum())

        for r in rescued:
            q = frames[r["i"]]
            rng = np.random.default_rng([a.seed, r["i"]])
            qs = [scramble_azimuth(q, rng) for _ in range(a.null)]
            null = np.array([count(s, M) for s, M in zip(qs, rgb.index_fused(qs, Mc, B=a.null))])
            r["null"] = dict(k=a.null, median=float(np.median(null)), p95=float(np.percentile(null, 95)),
                             max=int(null.max()), frac_ge_n1=round(float((null >= r["n1"]).mean()), 3))
            r["n_known_cell_real"] = count(q, rgb.index_fused([q], Mc, B=1)[0])

    tb, tp = rep["totals"]["base"], rep["totals"]["pl"]
    print(f"base: strict {tb['strict']}/{n}  indexed {tb['indexed']}  miss {tb['miss']}  watchdog {tb['watchdog_rescues']}  relocks {tb['relocks']}")
    print(f"pl:   strict {tp['strict']}/{n}  per-lattice {tp['per_lattice']}/{n}  indexed {tp['indexed']}  miss {tp['miss']}  "
          f"watchdog {tp['watchdog_rescues']}  relocks {tp['relocks']}  rescued_per_lattice {tp['rescued_per_lattice']}")
    print(f"frames that differ ({len(changed)}):")
    for c in changed:
        print(f"  {c['i']:4d}  {c['base'][0]:>18s} {str(c['base'][1]):>5s} ok={c['base'][2]}  ->  "
              f"{c['pl'][0]:>20s} {str(c['pl'][1]):>5s} ok={c['pl'][2]}")
    print(f"rescued_per_lattice ({len(rescued)}): kept the residual's lattice in "
          f"{sum(r['kept'] == 'second' for r in rescued)}, strict on the whole frame in {sum(r['strict_whole_frame'] for r in rescued)}")
    for r in rescued:
        nl = r.get("null")
        ns = (f"  null(k={nl['k']}) median {nl['median']:.0f} p95 {nl['p95']:.0f} max {nl['max']}  P(null>=n1) {nl['frac_ge_n1']:.2f}"
              f"  known-cell real {r['n_known_cell_real']}") if nl else ""
        print(f"  {r['i']:4d}  npk {r['npk']:4d}  n1 {r['n1']:3d} ({r['frac_first']:.3f})  m2 {r['m2']:3d}  kept {r['kept']:6s} "
              f"{r['misorientation']:6.1f} deg  whole-frame {r['frac_whole_frame']:.3f} ok={r['strict_whole_frame']}  "
              f"was {r['base_outcome']}/{r['base_cell']}{ns}")
    print(f"per-lattice column only ({len(only_pl)}):")
    for r in only_pl:
        print(f"  {r['i']:4d}  {r['o']:20s} npk {r['npk']:4d}  n1 {r['n1']:3d}  m2 {r['m2']:3d}  kept {r['kept']:6s} "
              f"{r['misorientation']:6.1f} deg  whole-frame {r['frac_whole_frame']:.3f}  per-lattice {r['frac_per_lattice']}")
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
