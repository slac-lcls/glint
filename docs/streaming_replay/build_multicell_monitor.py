#!/usr/bin/env python3
"""Build `glint_streaming_multicell_real.html` -- the streaming monitor replaying a RECORDED run of the
SHIPPED `glint.stream_driver.StreamDriver` over a TWO-SPECIES stream: real lysozyme frames (cxidb-17)
and real Proteinase K frames (cxidb-45), interleaved by a seeded schedule, indexed cold with no cell
handed in, with the adaptive re-lock ON. The trace is what experiments/record_stream_replay.py wrote.

Third instance of the builder pattern (build_stream_monitor.py -> build_cxidb480_monitor.py -> this):
take the sim template, swap its simulation engine for a player over the recorded trace, STRIP every
panel the recorded run cannot honestly fill, and KEEP only what the driver's own record supports.

What this run adds over the 480 replay, and therefore what this build KEEPS that the 480 build removed:

  * Active cells -- the driver's cell REGISTRY (glint#199): one card per registered cell, named from
    the roster the recorder handed the driver (`roster={lyso, prok}` -- names only, never a cell to
    index against), with its lock frame, source (blind warm-up / watchdog re-lock), measured cell,
    the frames attributed to it and its share of the last 200 attributions. The cards appear when the
    driver registers the cell, not before: "Discovering..." until the lock, the second card at the
    re-lock. Cell values are the medians over the frames the driver attributed to that cell, printed
    as measured; nothing is jittered.
  * Planted schedule -- the interleaving is scripted, so it is shown AS the ground truth it is,
    labelled planted: the scene the playhead is in and its species weights. It is the only planted
    quantity on the page, and it says so.
  * Stream composition per CELL -- one stacked bar per 8 pushed frames: frames the driver attributed
    to each cell (solid = cleared the strict bar, faint = accepted below it), refused (grey); under it
    a thin ribbon of the PLANTED species per bin, so measured attribution and planted truth sit one
    above the other and the first-fit misassignments are visible as colour disagreement.

What still goes, for the same reasons as before: detector geometry (this is a q-level replay: no
pixels, nothing predicted, `geom_refine` off), operational QA / spurious meter (no injections, no
lock probe), the blind fleet, the second-lattice KPI, the veto lane, the Inject button.

Honesty, in one place: the PEAKS are real (cxidb-17: CrystFEL peakfinder8 lists at the published fixed
wavelength; cxidb-45: the deposit's top-170 reciprocal-lattice-point lists, the same lists DIALS was
given in the paper's head-to-head), the CELLS are real (recovered blind), the INTERLEAVING is planted,
and no frame is integrated (index-only: the driver was fed q). The strict count is scored against
each frame's OWN species' reference, so a ProK frame claimed by the lysozyme cell scores zero -- and
the page shows how many were (the first-fit cascade's cost, measured, not hidden).

  python3 build_multicell_monitor.py [trace.json] [--control lyso=gate480.json --control prok=prok907.json]
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
# The SIM template this builder retrofits is NOT in the repository (see build_cxidb480_monitor.py); set
# GLINT_REPLAY_TEMPLATE to rebuild. The shipped HTML next to this file is the finished output.
TEMPLATE = os.environ.get("GLINT_REPLAY_TEMPLATE", "glint_streaming_live_demo.html")
OUT = os.path.join(HERE, "glint_streaming_multicell_real.html")
DEFAULT_TRACE = os.path.join(HERE, "multicell_strace_a100.json")

# The build refuses anything that is not the recorded run this page's footer describes.
EXPECT = dict(n=900, strict_ok=582, n_relock=1, n_watchdog_rescued=15, n_rescued=10)

# roster name -> (display name, dataset line, template colour slot)
NAMES = {
    "lyso": ("Lysozyme", "cxidb-17 (CC0) · pf8 peak lists", "--s1"),
    "prok": ("Proteinase K", "cxidb-45 (CC0) · top-170 lists", "--s2"),
}
SLOT_COLOURS = ["--s1", "--s2", "--s3", "--s4"]


def must_replace(txt, a, b, what):
    """Fail-fast replacement -- a half-applied build must not be possible."""
    if a not in txt:
        raise SystemExit(f"anchor not found ({what}): {a[:70]!r}")
    return txt.replace(a, b, 1)


def replace_function(txt, name, new_src):
    """Swap a whole `function NAME(...){...}` by brace matching."""
    m = re.search(r"\n(\s*)function %s\s*\([^)]*\)\s*\{" % re.escape(name), txt)
    if not m:
        raise SystemExit(f"function {name} not found")
    i = txt.index("{", m.end() - 1)
    depth, j = 0, i
    while j < len(txt):
        if txt[j] == "{":
            depth += 1
        elif txt[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    return txt[:m.start()] + "\n" + new_src.rstrip() + txt[j + 1:]


# --------------------------------------------------------------------- trace -> player records --
def convert(tr, controls):
    """The recorder's (header, records) -> what the player needs: compact per-frame records, the cell
    table with measured 6-parameter cells, the schedule with frame ranges, the planted/attributed
    confusion, and the solo controls scored on the same frames (when given)."""
    from glint.lattice import cell_params
    h, recs = tr["header"], tr["records"]
    species = [m["name"] for m in h["inputs"]]
    sp_index = {s: i for i, s in enumerate(species)}
    cells = h["cells"]
    ncell = len(cells)
    # lock / relock frames and the retro rescues they carried
    lock_i = next(r["i"] for r in recs if r["o"] == "warmup_lock")
    relock_is = [r["i"] for r in recs if r.get("relock")]
    wu_resc = [r for r in recs if r.get("resc") == "rescued_warmup"]
    rl_resc = [r for r in recs if r.get("resc") == "rescued_relock"]

    def relock_for(rec_i):
        later = [k for k in relock_is if k >= rec_i]
        return later[0] if later else (relock_is[-1] if relock_is else None)

    rl_by_frame = {}
    for r in rl_resc:
        k = relock_for(r["i"])
        rl_by_frame.setdefault(k, []).append(r)
    cell_id = {c["name"]: c["id"] for c in cells}

    out = []
    for r in recs:
        o = r["o"]
        acc = o in ("indexed", "rescued_watchdog", "rescued_cascade")
        d = dict(wu=int(r["wu"]), o=int(acc), ok=int(r.get("ok", 0)) if acc else 0, bl=int(o == "blank"),
                 c=(r["cell"] if (acc and r.get("cell") is not None) else -1), t=sp_index[r["truth"]],
                 buf=int(r.get("buf") or 0), n=int(r.get("npk") or 0))
        if r["wu"]:
            d["sup"] = int(r.get("sup") or 0); d["lead"] = int(r.get("lead") or 0)
        if r.get("resc"):
            # a retroactive rescue (at the lock, or at a re-lock): the record keeps its terminal outcome
            # (a warm-up vote or a miss) and ALSO the cell that later claimed it and whether the frame
            # clears the strict bar under that cell -- the player re-colours the frame's bar when the
            # rescue fires (r.rl on the lock / re-lock record lists the rescued records)
            d["rsc"] = cell_id[r["resc_cell"]]; d["rok"] = int(r.get("ok", 0))
        if o == "warmup_lock":
            d["lock"] = 1; d["rs"] = len(wu_resc); d["rso"] = sum(1 for x in wu_resc if x.get("ok"))
            d["rl"] = [x["i"] for x in wu_resc]
        if r.get("wresc"):
            d["wresc"] = 1
        if r.get("relock"):
            d["relock"] = 1
            d["rc"] = next((c["id"] for c in cells if c["name"] == r.get("relock_cell")), ncell - 1)
            rr = rl_by_frame.get(r["i"], [])
            d["rlr"] = len(rr); d["rlo"] = sum(1 for x in rr if x.get("ok"))
            d["rl"] = [x["i"] for x in rr]
        if r.get("flush"):
            d["flush"] = 1
        out.append(d)
    # every rescued record is listed on exactly one lock / re-lock record, and the lists agree with the
    # driver's counters (the header's EXPECT check pins n_rescued; n_warmup_rescued is pinned here)
    listed = [j for d in out for j in d.get("rl", [])]
    assert sorted(listed) == sorted(i for i, d in enumerate(out) if "rsc" in d), "rescue lists != rescued records"
    for k, d in enumerate(out):                                  # the rescuing cell is the one that fired
        want = 0 if d.get("lock") else d.get("rc")
        assert all(out[j]["rsc"] == want and j <= k for j in d.get("rl", [])), f"record {k}: rescue list"
    assert len(wu_resc) == h["counters"]["n_warmup_rescued"], (len(wu_resc), h["counters"]["n_warmup_rescued"])
    assert len(rl_resc) == h["counters"]["n_rescued"], (len(rl_resc), h["counters"]["n_rescued"])

    # cells: measured parameters = median over EVERY frame the driver attributed to each cell -- the
    # frames it accepted live (their terminal M) and the frames a lock / re-lock rescued retroactively
    # (their M_retro); the count must equal the registry's n_frames
    cell_meta = []
    for c in cells:
        Ms = [np.asarray(r["M"], float) for r in recs if r.get("cell") == c["id"] and r.get("M") is not None
              and r["o"] in ("indexed", "rescued_watchdog", "rescued_cascade")]
        Ms += [np.asarray(r["M_retro"], float) for r in recs
               if r.get("resc_cell") == c["name"] and r.get("M_retro") is not None]
        assert len(Ms) == c["n_frames"], f"cell {c['name']}: {len(Ms)} matrices for n_frames {c['n_frames']}"
        params = np.median(np.array([cell_params(M) for M in Ms]), axis=0) if Ms else np.full(6, np.nan)
        # per-frame M carries the frame's own axis order; print in the registry's standardized order
        # (its `axes`), carrying each axis's angle with it
        if Ms and c.get("axes"):
            perm, taken = [], set()
            for ax in c["axes"]:
                j = int(np.argmin([abs(params[i] - ax) if i not in taken else np.inf for i in range(3)]))
                perm.append(j); taken.add(j)
            params = np.array([params[perm[0]], params[perm[1]], params[perm[2]],
                               params[3 + perm[0]], params[3 + perm[1]], params[3 + perm[2]]])
        by_truth = {}
        for r in recs:
            if (r.get("resc_cell") or r.get("cell_name")) == c["name"]:
                by_truth[r["truth"]] = by_truth.get(r["truth"], 0) + 1
        disp, ds, cc = NAMES.get(c["name"], (f"cell {c['id']}", "not in the roster · unnamed", None))
        cell_meta.append(dict(id=c["id"], name=c["name"], display=disp, dataset=ds,
                              cc=cc or SLOT_COLOURS[c["id"] % 4], source=c["source"],
                              locked_at=c["locked_at"], n_frames=c["n_frames"],
                              cell=[round(float(v), 2) for v in params], by_truth=by_truth,
                              in_roster=c["name"] in NAMES, strict=sum(1 for r in recs if r.get("ok") and
                                                                      (r.get("resc_cell") or r.get("cell_name")) == c["name"])))
    # schedule with frame ranges
    scenes = []
    pos = 0
    for sc in h["schedule"]["scenes"]:
        n = int(sc["length"])
        scenes.append(dict(name=sc["name"], start=pos, end=pos + n, weights=sc["weights"], hold=sc.get("hold", 1)))
        pos += n
    # solo controls on the same frames (src_index < n used)
    ctrl = {}
    used = {s: 1 + max(r["src_index"] for r in recs if r["truth"] == s) for s in species}
    for s, path in controls.items():
        c = json.load(open(path))
        rs = c["records"]
        ctrl[s] = dict(strict=sum(r["ok"] for r in rs if r["src_index"] < used[s]), n=used[s],
                       file=os.path.basename(path), digest=c["header"]["inputs"][0]["digest"])
    return dict(records=out, species=species, cells=cell_meta, scenes=scenes, used=used, controls=ctrl,
                lock_i=lock_i, relock_is=relock_is, n_wu_resc=len(wu_resc), n_rl_resc=len(rl_resc))


# --------------------------------------------------------------------- the trace player ---------
def engine(tr, cv):
    h = tr["header"]
    TRACE = json.dumps(cv["records"], separators=(",", ":"))
    CELLS = json.dumps(cv["cells"], separators=(",", ":"))
    SCENES = json.dumps(cv["scenes"], separators=(",", ":"))
    sp_meta = [dict(id=s, name=NAMES.get(s, (s, "", None))[0], cc=NAMES.get(s, (s, "", SLOT_COLOURS[i % 4]))[2] or SLOT_COLOURS[i % 4])
               for i, s in enumerate(cv["species"])]
    meta = dict(n=h["n"], batch_B=h["driver_kw"]["B"], cell_window=h["driver_kw"].get("cell_window", 200),
                totals=h["totals"], by_species=h["by_species"], confusion=h["confusion"], counters=h["counters"],
                gate=h["gate"], driver_kw=h["driver_kw"], roster=h["roster"], provenance=h["provenance"],
                consensus_support=3, used=cv["used"], controls=cv["controls"])
    META = json.dumps(meta, separators=(",", ":"))
    return f"""  // ---- RECORDED run of the SHIPPED glint.stream_driver.StreamDriver over TWO real datasets -------
  // Input: cxidb-17 lysozyme (CrystFEL peakfinder8 lists at the published fixed wavelength) and
  // cxidb-45 Proteinase K (the deposit's top-170 reciprocal-lattice-point lists), interleaved by a
  // seeded schedule (experiments/schedules/lyso_prok_switch.json). No pixels: the driver is fed q.
  // Config: the published 480 arm (warmup_rescue + adaptive_relock) + rescue_buffer=64 + a naming
  // roster {{lyso, prok}}. The roster names cells; it never hands the indexer a cell.
  // Every quantity below is read out of the recorded trace. The only planted quantity on the page is
  // the schedule, and it is labelled planted.
  const META={META};
  const SPECIES={json.dumps(sp_meta, separators=(",", ":"))};
  const NS=SPECIES.length;
  const CELLS={CELLS};             // the driver's registry at the end of the run (names, lock frames, measured cells)
  const NC=CELLS.length;
  const SCENES={SCENES};           // the planted schedule, with frame ranges
  const TRACE={TRACE};
  const NF_TRACE=TRACE.length;
  const B_RING=META.batch_B;
  const SUPPORT=META.consensus_support;
  const WIN=META.cell_window;      // the driver's recent-share window (cell_window)
  const BIN=8, PLAY_FPS=12;

  let S;
  function freshBin(){{ return {{total:0,gen:0,bl:0,idx:new Array(NC).fill(0),buf:new Array(NC).fill(0),truth:new Array(NS).fill(0)}}; }}
  function fresh(){{
    S={{ tf:0, t:0, frames:0, pushed:0, accepted:0, strict:0, refused:0, blank:0, rescued:0, warmup:0,
      ring:0, mbuf:0, wbuf:0, locked:false, support:0, lead:0, lockFrame:0,
      idxR:0, okR:0, phase:'warmup', phaseHold:0, rescueFlash:0, flushFlash:0, batchFlash:0,
      wdog:0, relocks:0, relockFrame:0, relockCell:-1, rlResc:0, wdogFlash:0, relockFlash:0,
      cn:new Array(NC).fill(0), cs:new Array(NC).fill(0), win:[],
      seen:new Array(NC).fill(false),
      bins:[], binsDropped:0, curBin:freshBin(), histIdx:[], histOk:[], histRing:[] }};
  }}
  fresh();
  function resetForLoop(){{ const t=S.t; fresh(); S.t=t; }}

  function binAdd(cat,c,t){{ const b=S.curBin;
    if(cat==='gen')b.gen++; else if(cat==='bl')b.bl++; else if(cat==='idx')b.idx[c]++; else b.buf[c]++;
    if(t>=0) b.truth[t]++;
    if(++b.total>=BIN){{ S.bins.push(b); if(S.bins.length>240){{S.bins.shift(); S.binsDropped++;}} S.curBin=freshBin(); }} }}
  // A retroactive rescue: trace record j (played earlier as a warm-up vote or a miss -> grey) was
  // re-indexed against cell c when the lock / re-lock fired. Move its unit from grey to the cell's
  // colour in the bin it fell in (record j is the (j % BIN)-th entry of bin floor(j/BIN): every
  // record adds exactly one unit to exactly one bin).
  function amend(j,c,ok){{ const bi=Math.floor(j/BIN)-S.binsDropped; if(bi<0) return;
    const b = bi===S.bins.length ? S.curBin : S.bins[bi]; if(!b||b.gen<=0) return;
    b.gen--; if(ok) b.idx[c]++; else b.buf[c]++; }}
  function ema(k,x){{ const a=0.06; S[k]+=(x-S[k])*a; }}
  function attribute(c,n){{ for(let i=0;i<n;i++){{ S.win.push(c); if(S.win.length>WIN)S.win.shift(); }} S.cn[c]+=n; S.seen[c]=true; }}
  function share(c){{ if(!S.win.length) return 0; let k=0; for(const x of S.win) if(x===c)k++; return k/S.win.length; }}
  function sceneAt(f){{ for(const s of SCENES) if(f>=s.start && f<s.end) return s; return SCENES[SCENES.length-1]; }}

  function playFrame(){{
    if(S.tf>=NF_TRACE){{ S.tf=0; resetForLoop(); }}
    const r=TRACE[S.tf++]; S.frames++; S.pushed++;
    if(r.wu){{
      S.warmup++; S.wbuf=r.buf; S.support=r.sup||0; S.lead=r.lead||0;
      S.phase='warmup'; binAdd('gen',0,r.t);   // grey until the lock decides which votes it rescues
      ema('idxR',0); ema('okR',0);
    }} else {{
      S.ring++; S.mbuf=r.buf;           // r.buf = misses held in the rescue buffer (the driver's own count)
      if(r.o){{ S.accepted++; S.strict+=r.ok; attribute(r.c,1); S.cs[r.c]+=r.ok; binAdd(r.ok?'idx':'buf',r.c,r.t); }}
      else if(r.bl){{ S.blank++; binAdd('bl',0,r.t); }}
      else {{ S.refused++; binAdd('gen',0,r.t); }}
      ema('idxR', r.o?1:0); ema('okR', r.ok?1:0);
    }}
    if(r.lock){{
      // the blind consensus fired: cell 0 is registered (named from the roster) and the frames spent on
      // discovery are re-indexed against it -- those rescues are attributed to cell 0 here, as the driver did
      S.locked=true; S.lockFrame=S.frames; S.seen[0]=true;
      S.rescued+=r.rs; S.strict+=r.rso; S.cs[0]+=r.rso; attribute(0,r.rs); S.wbuf=0;
      for(const j of (r.rl||[])) amend(j, TRACE[j].rsc, TRACE[j].rok);
      S.rescueFlash=1; S.phase='rescuing'; S.phaseHold=Math.round(PLAY_FPS*1.4);
    }}
    if(r.wresc){{ S.wdog++; S.wdogFlash=1; }}
    if(r.relock){{
      // adaptive_relock: the buffered misses voted a recurring cell the active set did not explain. It is
      // ADDED (the first cell stays), named from the roster, and the misses in the rescue buffer are
      // re-indexed against it -- the retroactive rescue the rescue_buffer exists for.
      S.relocks++; S.relockFrame=S.frames; S.relockCell=r.rc; S.seen[r.rc]=true;
      S.rlResc+=r.rlr; S.strict+=r.rlo; S.cs[r.rc]+=r.rlo; attribute(r.rc,r.rlr);
      for(const j of (r.rl||[])) amend(j, TRACE[j].rsc, TRACE[j].rok);
      S.relockFlash=1; S.phase='relock'; S.phaseHold=Math.round(PLAY_FPS*2.0); S.mbuf=0;
    }}
    if(r.flush){{ S.batchFlash=1; S.ring=0; }}
  }}

  let carry=0;
  function advance(dtSim){{
    let nf=PLAY_FPS*dtSim+carry; const N=Math.floor(nf); carry=nf-N;
    for(let i=0;i<N;i++) playFrame();
    S.t+=dtSim;
    S.rescueFlash*=Math.pow(0.5, dtSim/0.9);
    S.batchFlash *=Math.pow(0.5, dtSim/0.35);
    S.wdogFlash  *=Math.pow(0.5, dtSim/0.7);
    S.relockFlash*=Math.pow(0.5, dtSim/1.6);
    if(S.phaseHold>0){{ S.phaseHold-=N; if(S.phaseHold<0)S.phaseHold=0; }}
    else S.phase = S.locked ? 'steady' : 'warmup';
    S.histIdx.push(S.idxR); S.histOk.push(S.okR); S.histRing.push(S.ring/B_RING);
    const H=260; if(S.histIdx.length>H){{S.histIdx.shift();S.histOk.shift();S.histRing.shift();}}
  }}
"""


# ------------------------------------------------------------------- replacement render code ----
DRAW_PIPELINE = r"""  function drawPipeline(){
    const w=pipe.clientWidth, h=pipe.clientHeight; if(!w||!h||!S) return;
    if(pipe.width!==Math.round(w*DPR)||pipe.height!==Math.round(h*DPR)){ pipe.width=Math.round(w*DPR); pipe.height=Math.round(h*DPR); }
    pcx.setTransform(DPR,0,0,DPR,0,0); pcx.clearRect(0,0,w,h);
    const acc=col('--accent'), good=col('--good'), bcol=col('--buf'), gold=col('--s4'),
          coral=col('--serious'), ink=col('--ink'), mut=col('--muted'), hair=col('--hair'),
          surf=col('--surface'), FONT=cssv('--mono');
    const K=Math.min(1.3, Math.max(1, h/232));
    const R1=h*0.235, R2=h*0.715;
    // The SHIPPED driver's dataflow. Row 1: the locked steady state -- a batch of B frames is tried
    // against the ACTIVE SET of cells, first fit wins. Row 2: the blind warm-up that found the first
    // cell, and the watchdog that found the second.
    const SRC=[w*0.075,R1], PK=[w*0.235,R1], RING=[w*0.455,R1], GL=[w*0.665,R1], IDX=[w*0.885,R1],
          BLIND=[w*0.245,R2], CONS=[w*0.485,R2], DROP=[w*0.80,R2];
    const pipes={ i1:{a:SRC,b:PK,c:mut,wd:3}, feed:{a:PK,b:RING,c:acc,wd:4.5},
      batch:{a:RING,b:GL,c:acc,wd:5}, fast:{a:GL,b:IDX,c:good,wd:5},
      drop:{a:GL,b:DROP,c:mut,wd:2.6},
      warm:{a:PK,b:BLIND,c:gold,wd:3.2}, vote:{a:BLIND,b:CONS,c:gold,wd:3.2},
      lock:{a:CONS,b:GL,c:good,wd:3}, resc:{a:CONS,b:IDX,c:col('--rescue'),wd:2.8},
      wdog:{a:DROP,b:IDX,c:coral,wd:2.4},
      wvote:{a:DROP,b:CONS,c:coral,wd:2.0} };
    pcx.lineCap='round';
    for(const k in pipes){ const p=pipes[k]; pcx.strokeStyle=hair; pcx.globalAlpha=0.5; pcx.lineWidth=p.wd;
      pcx.setLineDash(k==='i1'?[4,3]:[]);
      pcx.beginPath(); pcx.moveTo(p.a[0],p.a[1]); pcx.lineTo(p.b[0],p.b[1]); pcx.stroke(); }
    pcx.setLineDash([]); pcx.globalAlpha=1;
    const idx=clamp(S.idxR,0,1), ok=clamp(S.okR,0,1), warm=S.locked?0:1,
          rfill=clamp(S.ring/B_RING,0,1), wfill=clamp(S.wbuf/Math.max(S.lockFrame||5,1),0,1);
    const spawn=(k,rate,sz,cl)=>{ if(rate<=0)return; pipeAcc[k]=(pipeAcc[k]||0)+rate;
      while(pipeAcc[k]>=1){ pipeAcc[k]-=1; const p=pipes[k];
        pipeParts.push({a:p.a,b:p.b,t:0,spd:0.02+rand()*0.012,c:cl||p.c,r:sz}); } };
    // the two datasets feed the same lane: colour the particles by the cell the playhead is attributing to
    const lastC = S.win.length ? S.win[S.win.length-1] : 0;
    spawn('i1',0.30,1.9); spawn('feed',0.55,2.3);
    spawn('batch',0.10+1.5*S.batchFlash,2.5); spawn('fast',0.06+0.85*idx,2.4, col(CELLS[lastC].cc));
    spawn('drop',0.02+0.9*(1-idx),1.6);
    spawn('warm',0.55*warm,2.1); spawn('vote',0.5*warm,2.1);
    spawn('wdog',1.6*S.wdogFlash,2.4); spawn('wvote',1.2*S.relockFlash,2.2);
    if(S.rescueFlash>pipePrevF+0.25){
      for(let i=0;i<9;i++) pipeParts.push({a:CONS,b:GL,t:i*0.05,spd:0.03,c:good,r:2.6});
      for(let i=0;i<7;i++) pipeParts.push({a:CONS,b:IDX,t:i*0.05,spd:0.028,c:col('--rescue'),r:2.7}); }
    pipePrevF=S.rescueFlash;
    if(S.relockFlash>pipePrevR+0.25){
      // the re-lock's retroactive rescue: buffered misses re-indexed against the NEW cell
      for(let i=0;i<Math.min(S.rlResc,12);i++) pipeParts.push({a:DROP,b:IDX,t:i*0.05,spd:0.028,c:col(CELLS[Math.max(S.relockCell,0)].cc),r:2.7}); }
    pipePrevR=S.relockFlash;
    for(let i=pipeParts.length-1;i>=0;i--){ const q=pipeParts[i]; q.t+=q.spd; if(q.t>=1){ pipeParts.splice(i,1); continue; }
      const x=q.a[0]+(q.b[0]-q.a[0])*q.t, y=q.a[1]+(q.b[1]-q.a[1])*q.t;
      pcx.globalAlpha=0.9; pcx.fillStyle=q.c; pcx.beginPath(); pcx.arc(x,y,q.r,0,7); pcx.fill(); }
    pcx.globalAlpha=1; if(pipeParts.length>500) pipeParts.splice(0,pipeParts.length-500);
    pcx.fillStyle=mut; pcx.font='600 '+(7*K).toFixed(1)+'px '+FONT; pcx.textAlign='center'; pcx.textBaseline='middle';
    const mid=(p,dx,dy,t)=>pcx.fillText(t,(p.a[0]+p.b[0])/2+dx,(p.a[1]+p.b[1])/2+dy);
    mid(pipes.feed,0,-25,'q-vectors'); mid(pipes.batch,0,-25,'batch of '+B_RING);
    mid(pipes.drop,-26,17,'missed'); mid(pipes.warm,-6,4,'blind'); mid(pipes.lock,-18,8,'lock');
    mid(pipes.resc,12,-9,'rescue');
    pcx.fillStyle=S.wdogFlash>0.05?coral:mut;  mid(pipes.wdog,24,40,'watchdog');
    pcx.fillStyle=S.relockFlash>0.05?coral:mut; mid(pipes.wvote,34,20,'re-lock vote');
    pcx.fillStyle=mut;
    const node=(P,rx,ry,lab,glow,gc,fs,dash)=>{ rx*=K; ry*=K; fs=(fs||9.5)*K;
      pcx.fillStyle=surf; pcx.strokeStyle=gc; pcx.lineWidth=1.5*K;
      if(glow>0.02){ pcx.shadowColor=gc; pcx.shadowBlur=3+11*glow; } rr(P[0]-rx,P[1]-ry,rx*2,ry*2,6); pcx.fill();
      pcx.shadowBlur=0; if(dash)pcx.setLineDash([3,2.5]); pcx.stroke(); pcx.setLineDash([]);
      pcx.fillStyle=dash?mut:ink; pcx.font='600 '+fs.toFixed(1)+'px '+FONT; pcx.fillText(lab,P[0],P[1]); };
    node(SRC,34,12,'cxidb-17 + 45',0.12,mut,7.5,true);
    node(PK,47,14,'recorded peaks',0.5,acc,8.5);
    node(GL,34,14,'GLINT · '+(S.locked?(S.seen.filter(Boolean).length+' cell'+(S.seen.filter(Boolean).length>1?'s':'')):'no cell'),idx,good,9);
    node(IDX,29,13,'indexed',0.3+0.5*ok,good,9);
    node(BLIND,40,13,'blind warm-up',0.15+0.8*warm,gold,8.5);
    node(CONS,42,13,S.locked?'consensus ✓':'consensus',
         S.locked?(0.85+0.6*S.relockFlash):0.2+0.6*warm,
         S.relockFlash>0.05?coral:(S.locked?good:gold),8.5);
    node(DROP,26,11,'missed',0.1+0.7*S.wdogFlash,S.wdogFlash>0.05?coral:mut,8,true);
    pcx.fillStyle=mut; pcx.font='600 '+(6.8*K).toFixed(1)+'px '+FONT; pcx.textAlign='center'; pcx.textBaseline='middle';
    pcx.fillText('real peaks from two deposits, replayed as q-vectors in a planted order — no pixels, no peak finder, nothing integrated',
                 w/2, h-7);
    const twd=60*K,tht=30*K,tx=RING[0]-twd/2,ty=RING[1]-tht/2;
    pcx.fillStyle=surf; pcx.strokeStyle=bcol; pcx.lineWidth=1.5; rr(tx,ty,twd,tht,4); pcx.fill();
    pcx.save(); rr(tx,ty,twd,tht,4); pcx.clip(); pcx.globalAlpha=0.5; pcx.fillStyle=bcol;
    pcx.fillRect(tx,ty+tht*(1-rfill),twd,tht*rfill); pcx.globalAlpha=1; pcx.restore();
    if(S.batchFlash>0.05){ pcx.save(); pcx.globalAlpha=0.55*S.batchFlash; pcx.strokeStyle=acc;
      pcx.lineWidth=2.5; rr(tx-2,ty-2,twd+4,tht+4,5); pcx.stroke(); pcx.restore(); }
    rr(tx,ty,twd,tht,4); pcx.stroke();
    pcx.textAlign='center'; pcx.textBaseline='middle';
    pcx.fillStyle=ink; pcx.font='700 '+(12*K).toFixed(1)+'px '+FONT; pcx.fillText(S.ring,RING[0],RING[1]-3*K);
    pcx.fillStyle=mut; pcx.font='600 '+(7*K).toFixed(1)+'px '+FONT; pcx.fillText('device ring',RING[0],RING[1]+9*K);
    const hwd=30*K,hht=22*K,hx=CONS[0]+58*K,hy=CONS[1]-hht/2;
    pcx.fillStyle=surf; pcx.strokeStyle=S.locked?col('--rescue'):gold; pcx.lineWidth=1.4;
    rr(hx,hy,hwd,hht,4); pcx.fill();
    pcx.save(); rr(hx,hy,hwd,hht,4); pcx.clip(); pcx.globalAlpha=0.45;
    pcx.fillStyle=gold; pcx.fillRect(hx,hy+hht*(1-wfill),hwd,hht*wfill); pcx.globalAlpha=1; pcx.restore();
    rr(hx,hy,hwd,hht,4); pcx.stroke();
    pcx.fillStyle=ink; pcx.font='700 '+(10*K).toFixed(1)+'px '+FONT; pcx.fillText(S.wbuf,hx+hwd/2,hy+hht/2-2*K);
    pcx.fillStyle=mut; pcx.font='600 '+(6.5*K).toFixed(1)+'px '+FONT; pcx.fillText('held',hx+hwd/2,hy+hht/2+8*K);
    // ---- live counters, every one of them a trace readout ----
    pcx.font='700 '+(8*K).toFixed(1)+'px '+FONT;
    pcx.fillStyle=acc;  pcx.fillText(S.pushed+' / '+NF_TRACE+' pushed', PK[0], R1+34*K);
    pcx.fillStyle=good; pcx.fillText(S.accepted+' accepted', GL[0], R1+22*K);
    pcx.fillStyle=mut;  pcx.font='700 '+(7*K).toFixed(1)+'px '+FONT;
    pcx.fillText('first fit · gate ≥'+Math.round(META.driver_kw.min_inlier_frac*100)+'%', GL[0], R1+32*K);
    pcx.fillStyle=good; pcx.font='700 '+(9.5*K).toFixed(1)+'px '+FONT;
    pcx.fillText(S.strict+' / '+NF_TRACE, IDX[0], R1+23*K);
    pcx.fillStyle=mut;  pcx.font='700 '+(6.6*K).toFixed(1)+'px '+FONT;
    pcx.fillText('strict · own species', IDX[0], R1+33*K);
    pcx.fillText('≥25% spots · ≥10 refl', IDX[0], R1+42*K);
    if(S.rescued>0){ pcx.fillStyle=col('--rescue'); pcx.font='700 '+(7.6*K).toFixed(1)+'px '+FONT;
      pcx.fillText('↑'+S.rescued+' rescued at lock', IDX[0], R1+53*K); }
    if(S.wdog>0){ pcx.fillStyle=coral; pcx.font='700 '+(7.6*K).toFixed(1)+'px '+FONT;
      pcx.fillText('↑'+S.wdog+' watchdog rescue'+(S.wdog>1?'s':''), IDX[0], R1+63*K); }
    if(S.rlResc>0){ pcx.fillStyle=col(CELLS[Math.max(S.relockCell,0)].cc); pcx.font='700 '+(7.6*K).toFixed(1)+'px '+FONT;
      pcx.fillText('↑'+S.rlResc+' rescued at re-lock', IDX[0], R1+73*K); }
    pcx.fillStyle=mut; pcx.font='700 '+(7*K).toFixed(1)+'px '+FONT;
    pcx.fillText(S.refused+' of '+NF_TRACE, DROP[0], DROP[1]+16*K);
    // the rescue buffer: misses held for the retroactive re-index a re-lock performs (rescue_buffer)
    const RB=META.driver_kw.rescue_buffer||0;
    if(RB>0){ const bwd=44*K,bht=12*K,bx=DROP[0]-bwd/2,by=DROP[1]+24*K, f=clamp(S.mbuf/RB,0,1);
      pcx.fillStyle=surf; pcx.strokeStyle=S.relockFlash>0.05?coral:mut; pcx.lineWidth=1.2; rr(bx,by,bwd,bht,3); pcx.fill();
      pcx.save(); rr(bx,by,bwd,bht,3); pcx.clip(); pcx.globalAlpha=0.5; pcx.fillStyle=coral; pcx.fillRect(bx,by,bwd*f,bht); pcx.globalAlpha=1; pcx.restore();
      rr(bx,by,bwd,bht,3); pcx.stroke();
      pcx.fillStyle=mut; pcx.font='600 '+(6.5*K).toFixed(1)+'px '+FONT; pcx.fillText('rescue buffer '+S.mbuf+'/'+RB, DROP[0], by+bht+7*K); }
    pcx.fillStyle=S.locked?good:gold; pcx.font='700 '+(7.8*K).toFixed(1)+'px '+FONT;
    pcx.fillText(S.locked?('locked at frame '+S.lockFrame+' · '+CELLS[0].display):('support '+S.support+'/'+SUPPORT),
                 CONS[0], CONS[1]+21*K);
    if(S.relocks>0){ const rc=CELLS[Math.max(S.relockCell,0)]; pcx.fillStyle=coral; pcx.font='700 '+(7.4*K).toFixed(1)+'px '+FONT;
      pcx.fillText('re-lock ×'+S.relocks+' · frame '+S.relockFrame+' · +'+rc.display,
                   CONS[0], CONS[1]+31*K); }
    pcx.textAlign='left'; pcx.textBaseline='alphabetic';
  }
"""

RENDER = r"""  function render(){
    $('frames').textContent=S.frames.toLocaleString();
    $('clk').textContent=S.t.toFixed(1);
    const ph=$('phase'),pt=$('phaseTxt');
    ph.className='phase '+(S.phase==='warmup'||S.phase==='relock'?'discovering':S.phase);
    const rc = S.relockCell>=0 ? CELLS[S.relockCell] : null;
    pt.textContent = S.phase==='warmup'?'Blind warm-up — no cell yet'
                   : S.phase==='rescuing'?'Locked + rescuing warm-up frames'
                   : S.phase==='relock'?('Watchdog re-lock — '+(rc?rc.display:'second cell')+' added')
                   : ('Steady — '+S.seen.filter(Boolean).length+' cell'+(S.seen.filter(Boolean).length>1?'s':'')+' in the active set');
    ph.title = S.phase==='warmup'
        ? 'Mc=None: the driver has no cell. It indexes the opening frames one at a time and pools their candidates in a running cross-frame consensus.'
      : S.phase==='rescuing'
        ? 'The consensus fired. The frames spent on discovery are re-indexed against the freshly locked cell (warmup_rescue).'
      : S.phase==='relock'
        ? 'adaptive_relock: the misses the batch pass could not explain voted a recurring cell of their own. It is ADDED to the active set, named from the roster, and the misses still in the rescue buffer are re-indexed against it.'
        : 'Every pushed frame is tried against the active cells, first fit wins; the registry counts what each cell claims.';
    // planted schedule
    const sc=sceneAt(S.frames);
    const mb=$('mixbar'); mb.innerHTML='';
    const ws=SPECIES.map((s,i)=>[sc.weights[s.id]||0,i]).filter(o=>o[0]>0).sort((a,b)=>b[0]-a[0]);
    const wsum=ws.reduce((a,o)=>a+o[0],0)||1;
    ws.forEach(([wgt,i])=>{ const sp=document.createElement('span'); sp.style.width=(100*wgt/wsum)+'%';
      sp.style.background=col(SPECIES[i].cc); sp.style.boxShadow='inset -1px 0 0 var(--surface)'; mb.appendChild(sp); });
    const si=SCENES.indexOf(sc);
    $('mixNote').textContent='scene '+(si+1)+'/'+SCENES.length+' · '+sc.name+' · frames '+sc.start+'–'+sc.end+(sc.hold>1?' · redrawn every '+sc.hold:'');
    const ml=$('mixlist'); ml.innerHTML='';
    SPECIES.forEach((s,i)=>{ const wgt=sc.weights[s.id]||0, on=wgt>0;
      const el=document.createElement('span'); el.className='mitem'+(on?'':' off');
      el.innerHTML=`<span class="sw" style="background:${col(s.cc)}"></span>${s.name} <b>${on?(100*wgt/wsum).toFixed(0)+'%':'—'}</b>`;
      ml.appendChild(el); });
    renderCells();
    drawSched();
    drawStream();
  }
  function drawSched(){
    // the whole planted schedule as a strip: one segment per scene, its height split by the species
    // weights, the playhead where the replay is. Planted, and labelled so.
    const [w,h]=sched._d||fit(sched); scx.clearRect(0,0,w,h);
    const total=SCENES[SCENES.length-1].end, H=h-16;
    for(const sc of SCENES){ const x0=w*sc.start/total, x1=w*sc.end/total;
      const ws=SPECIES.map((s,i)=>[sc.weights[s.id]||0,i]); const wsum=ws.reduce((a,o)=>a+o[0],0)||1; let y=0;
      for(const [wgt,i] of ws){ if(wgt<=0)continue; const hh=H*wgt/wsum; scx.globalAlpha=0.85; scx.fillStyle=col(SPECIES[i].cc);
        scx.fillRect(x0,y,Math.max(1,x1-x0-1),hh); y+=hh; }
      scx.globalAlpha=1; scx.fillStyle=col('--muted'); scx.font='600 7.5px '+cssv('--mono'); scx.textAlign='center'; scx.textBaseline='top';
      if(x1-x0>34) scx.fillText(sc.name.replace('_',' '), (x0+x1)/2, H+3);
    }
    const px=w*Math.min(S.frames,total)/total;
    scx.fillStyle=col('--ink'); scx.fillRect(px-1,0,2,H);
    scx.beginPath(); scx.moveTo(px-5,H+1); scx.lineTo(px+5,H+1); scx.lineTo(px,H-5); scx.closePath(); scx.fill();
  }
"""

RENDER_CELLS = r"""  function renderCells(){
    const host=$('cells'); if(!host) return;
    let html='';
    for(let k=0;k<NC;k++){
      const cd=CELLS[k];
      if(k>0 && !S.seen[k]){ html+=`<div class="cell empty">— a second cell would appear here at a re-lock —</div>`; continue; }
      const locked = S.seen[k] && S.locked;
      const c = locked ? col(cd.cc) : col('--muted');
      const cs = locked ? cd.cell.map((v,i)=> isFinite(v)? v.toFixed(2) : '—') : ['—','—','—','—','—','—'];
      const nm  = locked ? cd.display : 'Discovering…';
      const sys = locked ? (cd.in_roster ? 'roster name · '+cd.dataset : cd.dataset) : 'no cell yet · blind warm-up';
      const pill = !locked ? `<span class="pill disc">CONSENSUS ${Math.min(S.support,SUPPORT)}/${SUPPORT}</span>`
                 : (cd.source==='relock' ? `<span class="pill locked" style="color:var(--serious);border-color:color-mix(in srgb,var(--serious) 45%,var(--hair))">RE-LOCK ✓ f${cd.locked_at}</span>`
                                          : `<span class="pill locked">LOCKED ✓ f${cd.locked_at}</span>`);
      const sh = locked ? (100*share(k)).toFixed(0)+'%' : '—';
      // what this cell claimed, by PLANTED species -- the first-fit misassignments are visible here
      const claims = Object.entries(cd.by_truth).sort((a,b)=>b[1]-a[1]).map(([t,n])=>`${n} ${NAMESHORT[t]||t}`).join(' · ');
      html+=`<div class="cell${locked?'':' pend'}" style="--cc:${c}">
        <div class="rail"></div>
        <div class="chead"><span class="chip"></span>
          <div><div class="cname">${nm}</div><div class="csys">${sys}</div></div>${pill}</div>
        <div class="params">
          <div><span class="lbl">a b c&nbsp;</span>${cs[0]} · ${cs[1]} · ${cs[2]} <span class="lbl">Å</span></div>
          <div><span class="lbl">α β γ&nbsp;</span>${cs[3]} · ${cs[4]} · ${cs[5]} <span class="lbl">°</span></div>
        </div>
        <div class="foot"><span class="share" title="share of the last ${WIN} attributions">${sh}</span>
          <span class="fmeta">${S.cn[k].toLocaleString()} frames · ${S.cs[k]} strict${locked?'':' · holding'}<br>${locked?('claimed at end: '+claims):('pushed '+S.pushed.toLocaleString())}</span></div>
      </div>`;
    }
    host.innerHTML=html;
  }
  const NAMESHORT={lyso:'lyso',prok:'ProK'};
"""

DRAW_STREAM = r"""  function drawStream(){
    // One bar per BIN pushed frames. Bottom-up: frames attributed to each cell that cleared the strict
    // bar (solid, cell colour), accepted below it (faint), then grey: missed at the live gate, or a
    // warm-up vote. A retroactive rescue (lock, re-lock) re-colours its frame's unit when it fires, so
    // the bars show the driver's FINAL attribution, the one the registry counts. Under the bars, a thin
    // ribbon of the PLANTED species per bin -- the truth the attribution is judged against.
    const [w,h]=wf._d||fit(wf); wfx.clearRect(0,0,w,h);
    const arr = S.curBin.total>0 ? S.bins.concat([S.curBin]) : S.bins;
    const n=arr.length; if(!n) return;
    const VIS=Math.min(n,64), start=n-VIS, bw=w/VIS;
    const RIB=7, hb=h-RIB-3;
    const cg=col('--muted');
    const accOf=b=>{ if(b.total<=0)return 0; let s=0; for(let k=0;k<NC;k++)s+=b.idx[k]+b.buf[k]; return s/b.total; };
    const okOf =b=>{ if(b.total<=0)return 0; let s=0; for(let k=0;k<NC;k++)s+=b.idx[k]; return s/b.total; };
    const Y=f=>hb-2-(hb-4)*clamp(f,0,1);
    for(let k=start;k<n;k++){
      const b=arr[k], T=b.total; if(T<=0) continue;
      const x=(k-start)*bw, bwid=Math.max(1,bw-0.7); let a=0;
      const seg=(cn,color,alpha)=>{ if(cn<=0)return; const y0=Y(a), y1=Y(a+cn/T);
        wfx.fillStyle=color; wfx.globalAlpha=alpha; wfx.fillRect(x,y1,bwid,y0-y1+0.4); a+=cn/T; };
      for(let c=0;c<NC;c++) seg(b.idx[c], col(CELLS[c].cc), 1);
      for(let c=0;c<NC;c++) seg(b.buf[c], col(CELLS[c].cc), 0.32);
      seg(b.gen, cg, 0.5);
      // planted ribbon
      let ax=x; const tt=b.truth.reduce((s,v)=>s+v,0)||1;
      for(let s=0;s<NS;s++){ const wd=bwid*b.truth[s]/tt; if(wd<=0)continue;
        wfx.globalAlpha=0.9; wfx.fillStyle=col(SPECIES[s].cc); wfx.fillRect(ax,h-RIB,wd,RIB); ax+=wd; }
    }
    wfx.globalAlpha=1;
    const env=(frac,color,wd,dash)=>{ wfx.beginPath(); wfx.setLineDash(dash||[]);
      for(let k=start;k<n;k++){ const X=(k-start+0.5)*bw, Yy=Y(frac(arr[k])); (k===start)?wfx.moveTo(X,Yy):wfx.lineTo(X,Yy); }
      wfx.strokeStyle=color; wfx.lineWidth=wd; wfx.lineJoin='round'; wfx.stroke(); wfx.setLineDash([]); };
    env(accOf, col('--ink2'), 1.4, [3,2]);
    env(okOf,  col('--accent'), 2, []);
  }
"""

BUILD_LEGEND = r"""  function buildLegend(){
    const L=$('legend'); if(!L) return; L.innerHTML='';
    const add=html=>{const k=document.createElement('span');k.className='k';k.innerHTML=html;L.appendChild(k);};
    for(let k=0;k<NC;k++) add(`<span class="sw" style="background:${col(CELLS[k].cc)}"></span>attributed to ${CELLS[k].display}, strict`);
    add(`<span class="sw" style="background:${col(CELLS[0].cc)};opacity:.4"></span>accepted, below the bar`);
    add(`<span class="sw" style="background:${col('--muted')};opacity:.55"></span>missed (or a warm-up vote never rescued)`);
    add(`<span class="swl" style="background:${col('--accent')}"></span>strict rate`);
    add(`<span class="swl" style="background:${col('--ink2')};opacity:.8"></span>live accept rate`);
    const note=document.createElement('span'); note.className='k'; note.style.opacity='.7';
    note.textContent='one bar = 8 pushed frames · bottom ribbon = planted species'; L.appendChild(note);
  }
"""


def strip_panels(txt):
    """Remove the DOM of every panel the recorded run cannot fill, plus the controls that mean nothing
    on a replay; keep cells, the (planted) mixture, the schematic and the composition chart."""
    a = txt.index('      <div class="panel">\n        <div class="ph"><h2>Detector geometry')
    b = txt.index('      <div class="panel">\n        <div class="ph"><h2>Operational QA')
    txt = txt[:a] + txt[b:]
    a = txt.index('      <div class="panel">\n        <div class="ph"><h2>Operational QA')
    b = txt.index('      <div class="panel">\n        <div class="ph"><h2>Active cells')
    txt = txt[:a] + txt[b:]
    txt = must_replace(txt, "  const geobox=$('geobox'), gbx=geobox.getContext('2d');",
                       "  // geometry panel removed: a q-level replay refines no geometry\n"
                       "  const sched=$('sched'), scx=sched.getContext('2d');   // the planted schedule's timeline", "geobox handle")
    txt = must_replace(txt, "    wf._d=fit(wf); }", "    wf._d=fit(wf); sched._d=fit(sched); }", "fitAll sched")
    txt = replace_function(txt, "drawGeoBox", "  function drawGeoBox(){}   // geometry panel removed")
    txt = replace_function(txt, "renderQA", "  function renderQA(){}      // no injections on a benchmark replay")
    txt = replace_function(txt, "drawSpurious", "  function drawSpurious(){} // lock_probe=False -- no spurious meter is computed")
    txt = must_replace(txt, "  function buildGeo(){}", "  function buildGeo(){}  // no geometry box on this build", "buildGeo")
    a = txt.index("  const GEO_XC=TRUE_GEO.x")
    b = txt.index("  function hexa(", a)
    txt = txt[:a] + "  // geometry-box constants removed with the panel (they read the sim's TRUE_GEO)\n" + txt[b:]
    txt = must_replace(txt, '<h2>Active cells · at most 4 in play</h2><span class="note" id="cellNote">multi-cell active set</span>',
                       '<h2>Active cells · the driver\'s registry</h2><span class="note" id="cellNote">named from a roster · discovered blind</span>',
                       "cells header")
    txt = must_replace(txt, '<h2>Sample mixture</h2><span class="note" id="mixNote">— active</span>',
                       '<h2>Planted schedule · ground truth</h2><span class="note" id="mixNote">—</span>', "mixture header")
    txt = must_replace(txt, ".main{display:grid;grid-template-columns:1.32fr 1fr;",
                       ".main{display:grid;grid-template-columns:1.25fr 1fr;", "main grid")
    # layout: schematic wide on the left; cells over the planted schedule over the composition on the right
    m = re.search(r'  <div class="main">\n(.*?)\n  </div>\n\n  <p class="foot">', txt, re.S)
    if not m:
        raise SystemExit("main block not found -- cannot rebuild the layout")
    lines, panels, cur = m.group(1).split("\n"), [], None
    for ln in lines:
        if ln == '      <div class="panel">':
            cur = [ln]
        elif cur is not None:
            cur.append(ln)
            if ln == "      </div>":
                panels.append("\n".join(cur) + "\n"); cur = None
    by = {}
    for p in panels:
        key = ("cells" if 'id="cells"' in p else "mix" if 'id="mixbar"' in p else "pipe" if 'id="pipe"' in p else
               "wf" if 'id="wf"' in p else None)
        if key:
            by[key] = p
    if set(by) != {"cells", "mix", "pipe", "wf"}:
        raise SystemExit(f"expected exactly the cells/mix/pipe/wf panels, got {sorted(by)}")
    by["mix"] = must_replace(by["mix"], '        <div class="mixlist" id="mixlist"></div>\n',
                             '        <div class="mixlist" id="mixlist"></div>\n'
                             '        <div class="cwrap" style="margin-top:10px"><canvas id="sched" height="54" style="height:54px"></canvas></div>\n',
                             "schedule timeline canvas")
    main = ('  <div class="main">\n'
            '    <div class="col">\n' + by["pipe"] + by["wf"] + '    </div>\n\n'
            '    <div class="col">\n' + by["cells"] + by["mix"] + '    </div>\n'
            '  </div>\n')
    txt = txt[:m.start()] + main + '\n  <p class="foot">' + txt[m.end():]
    txt = must_replace(txt, '<canvas id="pipe" height="200" style="height:200px">',
                       '<canvas id="pipe" height="340" style="height:340px">', "pipe height")
    txt = must_replace(txt, '<canvas id="wf" height="110" style="height:110px">',
                       '<canvas id="wf" height="128" style="height:128px">', "wf height")
    txt = must_replace(txt,
                       '<span class="note"><span style="color:var(--ink2)">hit</span> &amp; '
                       '<span style="color:var(--accent)">indexed</span> share · newest →</span>',
                       '<span class="note">per cell · <span style="color:var(--accent)">strict</span> &amp; '
                       '<span style="color:var(--ink2)">accepted</span> share · newest →</span>',
                       "composition note")
    txt = must_replace(txt, '      <button id="inject" class="btn accent">＋ Inject sample</button>\n', "", "inject button")
    txt = must_replace(txt, "  $('inject').addEventListener('click',()=>{ injectSample();",
                       "  const _inj=$('inject');   // absent on replay builds\n"
                       "  if(_inj) _inj.addEventListener('click',()=>{ injectSample();", "inject handler")
    return txt


def footer(tr, cv):
    h = tr["header"]; p = h["provenance"]; kw = h["driver_kw"]; t = h["totals"]; c = h["counters"]
    bs = h["by_species"]; conf = h["confusion"]
    cells = cv["cells"]
    ctrl = cv["controls"]
    lyso, prok = bs.get("lyso", {}), bs.get("prok", {})
    conf_prok = conf.get("prok", {}); conf_lyso = conf.get("lyso", {})
    misassigned = conf_prok.get("lyso", 0)
    ctrl_line = ""
    if ctrl:
        parts = []
        for s, d in ctrl.items():
            nm = NAMES.get(s, (s,))[0]
            parts.append(f"{nm} <b>{d['strict']}/{d['n']}</b> alone (<code>{d['file']}</code>) vs "
                         f"<b>{bs[s]['ok']}/{bs[s]['n']}</b> in the mixture")
        ctrl_line = ("    <br><br>\n    <b>Solo controls on the same frames.</b> " + "; ".join(parts) +
                     ". Lysozyme pays a couple of frames for the mixture. Proteinase K pays about a hundred, and most of\n"
                     f"    that is one mechanism: <b>{misassigned} ProK frames were claimed first by the lysozyme cell</b> — the\n"
                     "    first-fit cascade tries cell 0 first and a dense 170-peak frame gives it enough chance near-integer hits to\n"
                     "    pass the live gate — and none of them pass the strict bar under that cell. That is the measured case for\n"
                     "    best-fit assignment; this run does not do it, and the page shows the cost rather than hiding it.\n")
    relock_cells = [x for x in cells if x["source"] == "relock"]
    rl = relock_cells[0] if relock_cells else None
    return (
        "  <p class=\"foot\">\n"
        "    <b>Recorded replay — not a live beamline, and not a simulation.</b> Two real datasets, one stream: the\n"
        f"    <b>cxidb-17 lysozyme</b> frames ({cv['used'].get('lyso','?')} of the 480-frame extension, CXIDB entry 17, CC0) and the\n"
        f"    <b>cxidb-45 Proteinase K</b> frames ({cv['used'].get('prok','?')} of 907, CXIDB entry 45, CC0), interleaved by a seeded schedule\n"
        f"    (<code>experiments/schedules/lyso_prok_switch.json</code>, seed {h['schedule'].get('seed')}: {len(cv['scenes'])} scenes, frames taken in file\n"
        "    order, none reused). <b>The interleaving is the one planted thing on this page</b>, and it is drawn as the\n"
        "    ground truth it is: the schedule panel and the ribbon under the composition chart. The peaks are real, the\n"
        "    cells are recovered blind, the order is scripted.\n"
        "    <br><br>\n"
        "    <b>Where the peaks come from.</b> Lysozyme: <b>CrystFEL 0.12.0 peakfinder8</b> peaks read out of the blind\n"
        "    <code>indexamajig</code> stream for the 480 images and converted to q through the package's own bridge at the\n"
        "    published fixed wavelength (1.322216 Å). Proteinase K: the deposit's reciprocal-lattice-point lists, <b>the\n"
        "    170 brightest peaks per frame</b> — the same lists DIALS was given in the paper's head-to-head, so the cap is\n"
        "    part of the record, not a choice made here. No pixels are replayed, <b>no peak finder runs</b>, and\n"
        "    <b>nothing is integrated</b>: the driver is fed q-vectors and every frame is index-only.\n"
        "    <br><br>\n"
        f"    The indexer is the <b>shipped</b> <code>glint.stream_driver.StreamDriver</code> at commit <code>{p.get('git','')[:12]}</code>,\n"
        f"    cold-started with no cell (<code>Mc=None</code>), at the published 480 arm — <code>B={kw['B']}</code>, <code>dmin={kw['dmin']}</code>,\n"
        f"    <code>tol={kw['tol']}</code>, <code>warmup_nbest={kw['warmup_nbest']}</code>, <code>min_inliers={kw['min_inliers']}</code>,\n"
        f"    <code>min_inlier_frac={kw['min_inlier_frac']}</code>, <code>warmup_rescue</code>, <code>adaptive_relock</code> — plus\n"
        f"    <code>rescue_buffer={kw['rescue_buffer']}</code> (the retroactive index-only rescue the re-lock uses) and a naming\n"
        f"    <b>roster</b> of the two reference cells. The roster names a cell after the driver finds it; <b>it never hands the\n"
        "    indexer a cell</b>. Events on (<code>events=True</code>) — every record on this page is one of the driver's own\n"
        "    per-frame outcomes. <b>OFF:</b> retry cascade, geometry refinement, second-lattice detection, alias gate, lock probe.\n"
        "    <br><br>\n"
        f"    <b>What happened.</b> Consensus locked the first cell after <b>{c['locked_after']} frames</b> and the registry named it\n"
        f"    <b>{cells[0]['display']}</b>; the {cv['n_wu_resc']} frames spent on discovery were re-indexed against it. When the schedule\n"
        "    started blending Proteinase K in, its frames missed, the watchdog's misses voted a recurring cell the active set\n"
        + (f"    did not explain, and at frame <b>{rl['locked_at']}</b> the driver <b>added</b> a second cell — named <b>{rl['display']}</b>\n"
           f"    from the roster — and re-indexed the {cv['n_rl_resc']} misses still in its buffer against it. <b>One re-lock, no alias.</b>\n" if rl else
           "    did not explain — no re-lock in this run.\n") +
        f"    Over the run the watchdog rescued <b>{c['n_watchdog_rescued']}</b> individual frames the batch pass had missed. Final active\n"
        f"    set: {len(cells)} cells; " + "; ".join(f"{x['display']} {x['n_frames']} frames" for x in cells) + ".\n"
        "    <br><br>\n"
        f"    <b>Scored honestly.</b> Each frame is scored at the paper's strict bar (≥25% of spots <i>and</i> ≥10 reflections)\n"
        f"    against <b>its own species' reference lattice</b>, so a frame claimed by the wrong cell scores zero. The run indexes\n"
        f"    <b>{t['strict_ok']}/{t['n']}</b>: lysozyme <b>{lyso.get('ok','?')}/{lyso.get('n','?')}</b>, Proteinase K <b>{prok.get('ok','?')}/{prok.get('n','?')}</b>.\n"
        f"    Attribution, planted species → cell: lysozyme frames → {conf_lyso}; Proteinase K frames → {conf_prok}\n"
        "    (<code>-</code> = missed).\n"
        + ctrl_line +
        "    <br><br>\n"
        f"    Recorded on {p.get('gpu','?')} ({p.get('host','?')}), torch {p.get('torch','?')}, numpy {p.get('numpy','?')};\n"
        f"    {p.get('elapsed_s','?')} s of wall time for the whole replay ({t['n']} frames). Playback is looped and the frame rate is a\n"
        "    display choice; every counter, both cells, the lock and re-lock frames, the ring occupancy, the rescues and\n"
        "    the attribution are read out of that one recorded run. Cells are printed as measured (medians over the frames\n"
        "    the driver attributed to them) and are not re-jittered for the animation.\n"
        "  </p>\n"
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", default=DEFAULT_TRACE)
    ap.add_argument("--control", action="append", default=[], metavar="NAME=solo.json",
                    help="the same species replayed ALONE through the same arm; scored on the frames the mixture used")
    a = ap.parse_args(argv)
    tr = json.load(open(a.trace))
    h = tr["header"]
    # refuse-checks: this page's footer describes ONE recorded run
    got = dict(n=h["n"], strict_ok=h["totals"]["strict_ok"], n_relock=h["counters"]["n_relock"],
               n_watchdog_rescued=h["counters"]["n_watchdog_rescued"], n_rescued=h["counters"]["n_rescued"])
    if got != EXPECT:
        raise SystemExit(f"refusing to build: trace header {got} != expected {EXPECT}")
    if h.get("ingest", "q") != "q":
        raise SystemExit("refusing to build: this page is the q-level (index-only) replay; the trace says otherwise")
    if sorted(m["name"] for m in h["inputs"]) != ["lyso", "prok"]:
        raise SystemExit(f"refusing to build: expected the lyso+prok inputs, got {[m['name'] for m in h['inputs']]}")
    controls = dict(spec.split("=", 1) for spec in a.control)
    cv = convert(tr, controls)
    # the schedule must reproduce the recorded truth sequence (determinism of the planted order)
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    import record_stream_replay as rsr
    pools = {m["name"]: list(range(m["n"])) for m in h["inputs"]}
    order, _ = rsr.build_schedule(pools, h["schedule"])
    truth = [(r["truth"], r["src_index"]) for r in tr["records"]]
    if order != truth:
        raise SystemExit("refusing to build: build_schedule no longer reproduces the recorded frame order")

    lines = open(TEMPLATE).read().split("\n")
    start = next(i for i, l in enumerate(lines) if l.strip().startswith("const SPECIES=["))
    end = next(i for i, l in enumerate(lines) if "================= RENDER" in l)
    txt = "\n".join(lines[:start - 1]) + "\n" + engine(tr, cv) + "\n" + "\n".join(lines[end - 1:])

    txt = strip_panels(txt)
    txt = replace_function(txt, "drawPipeline", DRAW_PIPELINE)
    txt = replace_function(txt, "render", RENDER)
    txt = replace_function(txt, "renderCells", RENDER_CELLS)
    txt = replace_function(txt, "drawStream", DRAW_STREAM)
    txt = replace_function(txt, "buildLegend", BUILD_LEGEND)
    # a second particle accumulator for the re-lock burst
    txt = must_replace(txt, "  let pipeParts=[], pipeAcc={}, pipePrevF=0;", "  let pipeParts=[], pipeAcc={}, pipePrevF=0, pipePrevR=0;", "pipePrevR")
    txt = must_replace(txt,
                       "  window.addEventListener('resize',fitAll);",
                       "  window.addEventListener('resize',function(){\n"
                       "    fitAll();\n"
                       "    try{ render(); drawPipeline(); }catch(e){}\n"
                       "  });", "resize repaint")
    txt = must_replace(txt, "<title>GLINT Streaming Driver — Live Monitor</title>",
                       '<meta charset="utf-8">\n<title>GLINT streaming driver · two-species replay</title>', "title")
    txt = must_replace(txt, "GLINT · device-resident streaming indexer",
                       "GLINT · device-resident streaming driver · recorded replay", "eyebrow")
    txt = must_replace(txt, "Live Streaming Monitor", "cxidb-17 + cxidb-45 · two-species replay", "h1")
    txt = must_replace(txt, '<span class="note">LCLS → detector → PeakReducer → GLINT</span>',
                       '<span class="note">recorded peaks (two deposits) → ring → GLINT · first fit over the active cells</span>', "pipe note")
    txt = must_replace(txt, "  .foot{margin-top:16px;",
                       "  .app.printmode .controls .btn,\n"
                       "  .app.printmode .controls .speed{display:none;}\n"
                       "  .foot{margin-top:16px;", "print css")
    txt = must_replace(txt,
                       "  const _th=new URLSearchParams(location.search).get('theme'); "
                       "if(_th) document.documentElement.dataset.theme=_th;",
                       "  const _qs=new URLSearchParams(location.search);\n"
                       "  const _th=_qs.get('theme'); if(_th) document.documentElement.dataset.theme=_th;\n"
                       "  if(_qs.get('print')) document.querySelector('.app').classList.add('printmode');",
                       "print param")
    txt = must_replace(txt,
                       "  if(warm>0){ for(let acc=0;acc<warm;acc+=0.033) advance(0.033); fitAll(); render(); }",
                       "  if(warm>0){ for(let acc=0;acc<warm;acc+=0.033) advance(0.033); fitAll(); render(); }\n"
                       "  if(_qs.get('print')) playing=false;   // freeze the still exactly at ?warm=",
                       "print freeze")
    a0 = txt.index('  <p class="foot">')
    b0 = txt.index("</p>", a0) + len("</p>\n")
    txt = txt[:a0] + footer(tr, cv) + txt[b0:]

    with open(OUT, "w") as f:
        f.write(txt)
    print(f"wrote {OUT}")
    c = h["counters"]; t = h["totals"]
    print(f"  {h['n']} records · locked after {c['locked_after']} · re-locks {c['n_relock']} at {cv['relock_is']} · "
          f"watchdog rescues {c['n_watchdog_rescued']} · buffered rescues {c['n_rescued']} · strict {t['strict_ok']}/{t['n']}")
    for x in cv["cells"]:
        print(f"  cell {x['id']} {x['name']:6s} {x['source']:7s} locked_at {x['locked_at']:4d} n {x['n_frames']:4d} "
              f"strict {x['strict']:4d} cell {x['cell']} claimed {x['by_truth']}")
    if cv["controls"]:
        for s, d in cv["controls"].items():
            print(f"  control {s}: solo {d['strict']}/{d['n']} vs mixed {h['by_species'][s]['ok']}/{h['by_species'][s]['n']}")
    if cv["relock_is"]:
        print(f"  snapshot: ?warm={(cv['relock_is'][0] + 7) / 12.0:.2f}&print=1  (≈6 frames past the re-lock)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
