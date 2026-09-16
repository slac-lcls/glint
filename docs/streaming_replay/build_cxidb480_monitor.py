#!/usr/bin/env python3
"""Build `glint_streaming_cxidb480_real.html` -- the streaming monitor replaying a RECORDED run of
the SHIPPED `glint.stream_driver.StreamDriver` over the cxidb-17 **480-frame** extension, the
paper's primary streaming demonstration.

Parallel build to build_cxidb_monitor.py (the 120-frame subset control), NOT a replacement: that
builder, its trace and its two snapshots stay exactly as they are. Two things differ here beyond
the frame count, and they are the reason the 480 replay is the better Fig 9 candidate:

  * the arm is **both** published opt-ins -- `warmup_rescue=True` AND `adaptive_relock=True` --
    scoring 331/480 (69.0%), the top of the paper's printed 67-69% band;
  * at 480 the **watchdog actually fires**: six individual rescues and one re-lock, exactly the
    counts the paper's SI reports (`glint_rewrite_JAC_refined.tex` ~694 and ~2255). Across 120
    frames of the same rebuilt peak list it fires zero times, so the 120 build has no watchdog
    story to draw and this one does.

Same builder pattern as build_stream_monitor.py: take the sim template, swap its simulation engine
for a player over a recorded TRACE, and STRIP every panel the recorded run cannot honestly fill.
The stripping is the point. The mfx cut removed the injected-QA rows because that archival run has
no injections; this cut removes more, because the SHIPPED driver at its default configuration emits
less than drive_stream.py's bespoke two-path driver did:

    panel / lane                 why it goes
    detector geometry            geom_refine=False (default) -- the driver refines no geometry
    operational QA + spurious    no injections in a benchmark replay, and lock_probe=False
    sample mixture               one protein; a "mixture" of one is a lie by layout
    blind FLEET multiplier       the warm-up blind indexer is serial; there is no fleet
    second-lattice (2x) KPI      double_hit=False (default) -- never computed
    blank / veto lane            the benchmark peak lists are already a hit-selected subset

What is left is exactly what the shipped driver does: fill a device-resident ring of B frames,
batch-index them against the locked cell, accept or refuse each at the live gate, and -- before any
of that -- discover the cell blind from the first few frames by running consensus, then rescue the
frames that discovery spent.

Nothing is invented. Every number on screen is read out of the recorded trace, including the
recovered cell, which is printed VERBATIM (the sim template jitters displayed cell values through
updateMeas(); that function is deleted here rather than inherited).

  python3 build_cxidb480_monitor.py [trace.json]
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The SIM template this builder retrofits is NOT in the repository (it lives with the slides/animations
# work); set GLINT_REPLAY_TEMPLATE to its path to rebuild. The shipped glint_streaming_cxidb480_real.html
# next to this file is the finished, self-contained output and is what the README GIF was captured from.
TEMPLATE = os.environ.get("GLINT_REPLAY_TEMPLATE", "glint_streaming_live_demo.html")
OUT = os.path.join(HERE, "glint_streaming_cxidb480_real.html")
DEFAULT_TRACE = os.path.join(HERE, "cxidb480_strace_a100.json")
EXPECT_TOTAL, EXPECT_FRAMES = 331, 480


def must_replace(txt, a, b, what):
    """Fail-fast replacement -- a half-applied build must not be possible (build_halloween.py's rule)."""
    if a not in txt:
        raise SystemExit(f"anchor not found ({what}): {a[:70]!r}")
    return txt.replace(a, b, 1)


def replace_function(txt, name, new_src):
    """Swap a whole `function NAME(...){...}` by brace matching, so the replacement does not depend
    on the body staying byte-stable."""
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


# --------------------------------------------------------------------- the trace player ---------
def engine(tr):
    TRACE = json.dumps(tr["trace"], separators=(",", ":"))
    cell = tr["cell"]
    meta = {k: tr[k] for k in ("nframes", "locked_after", "consensus_support", "batch_B",
                               "n_accepted_live", "n_gate_rejected", "n_warmup",
                               "n_warmup_rescued", "strict_total", "strict_postlock",
                               "strict_warmup", "cell", "gate_live", "gate_strict",
                               "driver_kw", "same_lattice_vs_reference",
                               "n_watchdog_rescued", "n_relock", "relock_frame",
                               "published_band")}
    meta["provenance"] = tr["provenance"]
    META = json.dumps(meta, separators=(",", ":"))
    return f"""  // ---- RECORDED run of the SHIPPED glint.stream_driver.StreamDriver -------------------------
  // Input: the cxidb-17 480-frame extension of the same run (CXIDB entry 17, CC0). The peaks are
  // CrystFEL 0.12.0 peakfinder8's, read out of the blind indexamajig stream and converted to
  // q-vectors through the package's OWN bridge (glint.geom.read_crystfel_peaks + peaks_to_q).
  // No pixels are replayed and no peak finder runs here: the driver is fed q-vectors.
  // Config: constructor defaults + warmup_rescue=True + adaptive_relock=True -- both published
  // opt-ins, which is the 331/480 arm. Everything else opt-in is OFF.
  // Every quantity below is read out of the recorded trace. Nothing is simulated.
  const META={META};
  const SPECIES=[{{id:'lyso',name:'Lysozyme',sys:'tetragonal (metric) \\u00b7 recovered blind',
                  cc:'--s1',cell:{json.dumps(cell)}}}];
  const NS=1;
  const TRACE={TRACE};
  const NF_TRACE=TRACE.length;
  const B_RING=META.batch_B;                 // device-resident ring depth (StreamDriver B)
  const SUPPORT=META.consensus_support;      // consensus votes the lock actually fired on
  const BIN=8, PLAY_FPS=12;                  // frames per composition bar / playback rate

  let S;
  function freshBin(){{ return {{total:0,veto:0,gen:0,idx:new Array(NS).fill(0),buf:new Array(NS).fill(0)}}; }}
  function fresh(){{
    S={{ tf:0, t:0, frames:0, pushed:0, accepted:0, strict:0, refused:0, rescued:0, warmup:0,
      ring:0, wbuf:0, locked:false, support:0, lead:0, lockFrame:0,
      idxR:0, okR:0, phase:'warmup', phaseHold:0, rescueFlash:0, flushFlash:0, batchFlash:0,
      wdog:0, relocks:0, relockFrame:0, wdogFlash:0, relockFlash:0,
      bins:[], curBin:freshBin(), histIdx:[], histOk:[], histRing:[] }};
  }}
  fresh();
  function resetForLoop(){{ const t=S.t; fresh(); S.t=t; }}

  // Composition bins. The categories are the driver's OWN three outcomes per pushed frame:
  //   idx  = cleared the strict research bar (>=25% of spots AND >=10 reflections, right cell)
  //   buf  = accepted at the driver's live gate but below that bar
  //   gen  = refused at the live gate (min_inlier_frac 0.15) -- dropped, and counted
  function binAdd(cat){{ const b=S.curBin;
    if(cat==='gen')b.gen++; else if(cat==='idx')b.idx[0]++; else b.buf[0]++;
    if(++b.total>=BIN){{ S.bins.push(b); if(S.bins.length>200)S.bins.shift(); S.curBin=freshBin(); }} }}
  function ema(k,x){{ const a=0.06; S[k]+=(x-S[k])*a; }}

  function playFrame(){{
    if(S.tf>=NF_TRACE){{ S.tf=0; resetForLoop(); }}
    const r=TRACE[S.tf++]; S.frames++; S.pushed++;
    if(r.wu){{
      // Warm-up: the frame casts a blind consensus vote and is HELD. It is not indexed yet, so it
      // scores nothing yet -- the credit lands at the lock, when the rescue re-indexes it.
      S.warmup++; S.wbuf=r.buf; S.support=r.sup; S.lead=r.lead||0;
      S.phase='warmup'; binAdd('buf');
      ema('idxR',0); ema('okR',0);
    }} else {{
      S.ring=r.buf;
      if(r.o===1){{ S.accepted++; S.strict+=r.ok?1:0; binAdd(r.ok?'idx':'buf'); }}
      else {{ S.refused++; binAdd('gen'); }}
      ema('idxR', r.o===1?1:0); ema('okR', r.ok?1:0);
    }}
    if(r.lock>=0){{
      S.locked=true; S.lockFrame=S.frames; S.support=r.sup;
      S.rescued+=r.resc; S.strict+=r.resc_ok; S.wbuf=0;
      S.rescueFlash=1; S.phase='rescuing'; S.phaseHold=Math.round(PLAY_FPS*1.4);
    }}
    // The watchdog. Both of these are counters the 120-frame replay could not draw, because on
    // 120 frames of this same peak list the watchdog fires zero times. r.wresc marks a frame the
    // batch pass MISSED and the watchdog then indexed individually; r.relock marks the flush at
    // which the accumulated misses voted a recurring cell the active set did not explain, which
    // ADDS a second cell rather than replacing the locked one.
    if(r.wresc){{ S.wdog++; S.wdogFlash=1; }}
    if(r.relock){{
      S.relocks++; S.relockFrame=S.frames; S.relockFlash=1;
      S.phase='relock'; S.phaseHold=Math.round(PLAY_FPS*2.0);
    }}
    if(r.flush){{ S.batchFlash=1; S.ring=0; }}   // the ring drained: B frames indexed as one batch
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
    // The schematic scales with the panel: this build gives it a whole column, and node boxes sized
    // for the template's 200 px strip read as dust at 392. K is that scale, capped so it stays a
    // diagram rather than becoming a poster.
    const K=Math.min(1.3, Math.max(1, h/232));
    const R1=h*0.235, R2=h*0.715;
    // The SHIPPED driver's actual dataflow. Row 1 is the locked steady state, row 2 the blind
    // warm-up that gets it there. No lane here is decorative: each carries a counter from the trace.
    const SRC=[w*0.075,R1], PK=[w*0.235,R1], RING=[w*0.455,R1], GL=[w*0.665,R1], IDX=[w*0.885,R1],
          BLIND=[w*0.245,R2], CONS=[w*0.485,R2], DROP=[w*0.80,R2];
    const pipes={ i1:{a:SRC,b:PK,c:mut,wd:3}, feed:{a:PK,b:RING,c:acc,wd:4.5},
      batch:{a:RING,b:GL,c:acc,wd:5}, fast:{a:GL,b:IDX,c:good,wd:5},
      drop:{a:GL,b:DROP,c:mut,wd:2.6},
      warm:{a:PK,b:BLIND,c:gold,wd:3.2}, vote:{a:BLIND,b:CONS,c:gold,wd:3.2},
      lock:{a:CONS,b:GL,c:good,wd:3}, resc:{a:CONS,b:IDX,c:col('--rescue'),wd:2.8},
      // The watchdog lane. It reaches BACK into the miss box: a frame the batch pass could not
      // register is blind-solved again on its own, and if that solution is the same lattice as a
      // cell already in the active set it is indexed after all. Six frames take this path here.
      wdog:{a:DROP,b:IDX,c:coral,wd:2.4},
      // ...and the misses that are NOT the active cell vote among themselves. One re-lock.
      wvote:{a:DROP,b:CONS,c:coral,wd:2.0} };
    pcx.lineCap='round';
    for(const k in pipes){ const p=pipes[k]; pcx.strokeStyle=hair; pcx.globalAlpha=0.5; pcx.lineWidth=p.wd;
      pcx.setLineDash(k==='i1'?[4,3]:[]);        // the images themselves are not replayed here
      pcx.beginPath(); pcx.moveTo(p.a[0],p.a[1]); pcx.lineTo(p.b[0],p.b[1]); pcx.stroke(); }
    pcx.setLineDash([]); pcx.globalAlpha=1;
    const idx=clamp(S.idxR,0,1), ok=clamp(S.okR,0,1), warm=S.locked?0:1,
          rfill=clamp(S.ring/B_RING,0,1), wfill=clamp(S.wbuf/Math.max(META.n_warmup,1),0,1);
    const spawn=(k,rate,sz)=>{ if(rate<=0)return; pipeAcc[k]=(pipeAcc[k]||0)+rate;
      while(pipeAcc[k]>=1){ pipeAcc[k]-=1; const p=pipes[k];
        pipeParts.push({a:p.a,b:p.b,t:0,spd:0.02+rand()*0.012,c:p.c,r:sz}); } };
    spawn('i1',0.30,1.9); spawn('feed',0.55,2.3);
    spawn('batch',0.10+1.5*S.batchFlash,2.5); spawn('fast',0.06+0.85*idx,2.4);
    spawn('drop',0.02+0.9*(1-idx),1.6);
    spawn('warm',0.55*warm,2.1); spawn('vote',0.5*warm,2.1);
    // The watchdog lanes carry traffic ONLY when the recorded trace says they fired. They are not
    // idle decoration and they are not a steady trickle: six rescue events and one re-lock, in the
    // 480 frames of this run.
    spawn('wdog',1.6*S.wdogFlash,2.4); spawn('wvote',1.2*S.relockFlash,2.2);
    if(S.rescueFlash>pipePrevF+0.25){
      for(let i=0;i<9;i++) pipeParts.push({a:CONS,b:GL,t:i*0.05,spd:0.03,c:good,r:2.6});
      for(let i=0;i<7;i++) pipeParts.push({a:CONS,b:IDX,t:i*0.05,spd:0.028,c:col('--rescue'),r:2.7}); }
    pipePrevF=S.rescueFlash;
    for(let i=pipeParts.length-1;i>=0;i--){ const q=pipeParts[i]; q.t+=q.spd; if(q.t>=1){ pipeParts.splice(i,1); continue; }
      const x=q.a[0]+(q.b[0]-q.a[0])*q.t, y=q.a[1]+(q.b[1]-q.a[1])*q.t;
      pcx.globalAlpha=0.9; pcx.fillStyle=q.c; pcx.beginPath(); pcx.arc(x,y,q.r,0,7); pcx.fill(); }
    pcx.globalAlpha=1; if(pipeParts.length>500) pipeParts.splice(0,pipeParts.length-500);
    // lane labels
    pcx.fillStyle=mut; pcx.font='600 '+(7*K).toFixed(1)+'px '+FONT; pcx.textAlign='center'; pcx.textBaseline='middle';
    const mid=(p,dx,dy,t)=>pcx.fillText(t,(p.a[0]+p.b[0])/2+dx,(p.a[1]+p.b[1])/2+dy);
    // above the node band, not level with it: at K>1 a label on the centre line is eaten by the boxes
    mid(pipes.feed,0,-25,'spot positions'); mid(pipes.batch,0,-25,'batch of '+B_RING);
    mid(pipes.drop,-26,17,'missed'); mid(pipes.warm,-6,4,'blind'); mid(pipes.lock,-18,8,'lock');
    mid(pipes.resc,12,-9,'rescue');
    // Both watchdog labels are offset off their own mid-points: 'watchdog' on the line would sit on
    // top of the rescue counter under the indexed node, and 're-lock vote' on the line would sit on
    // top of the warm-up hold box. Placed clear of both, on the same side as the lane they name.
    pcx.fillStyle=S.wdogFlash>0.05?coral:mut;  mid(pipes.wdog,-24,16,'watchdog');
    pcx.fillStyle=S.relockFlash>0.05?coral:mut; mid(pipes.wvote,40,34,'re-lock vote');
    pcx.fillStyle=mut;
    const node=(P,rx,ry,lab,glow,gc,fs,dash)=>{ rx*=K; ry*=K; fs=(fs||9.5)*K;
      pcx.fillStyle=surf; pcx.strokeStyle=gc; pcx.lineWidth=1.5*K;
      if(glow>0.02){ pcx.shadowColor=gc; pcx.shadowBlur=3+11*glow; } rr(P[0]-rx,P[1]-ry,rx*2,ry*2,6); pcx.fill();
      pcx.shadowBlur=0; if(dash)pcx.setLineDash([3,2.5]); pcx.stroke(); pcx.setLineDash([]);
      pcx.fillStyle=dash?mut:ink; pcx.font='600 '+fs.toFixed(1)+'px '+FONT; pcx.fillText(lab,P[0],P[1]); };
    node(SRC,30,12,'cxidb-17',0.12,mut,8,true);
    node(PK,47,14,'recorded peaks',0.5,acc,8.5);
    node(GL,30,14,'GLINT',idx,good,10.5);
    node(IDX,29,13,'indexed',0.3+0.5*ok,good,9);
    node(BLIND,40,13,'blind warm-up',0.15+0.8*warm,gold,8.5);
    node(CONS,42,13,S.locked?'consensus ✓':'consensus',
         S.locked?(0.85+0.6*S.relockFlash):0.2+0.6*warm,
         S.relockFlash>0.05?coral:(S.locked?good:gold),8.5);
    node(DROP,26,11,'missed',0.1+0.7*S.wdogFlash,S.wdogFlash>0.05?coral:mut,8,true);
    pcx.fillStyle=mut; pcx.font='600 '+(6.8*K).toFixed(1)+'px '+FONT; pcx.textAlign='center'; pcx.textBaseline='middle';
    // ONE line, on the canvas floor: the provenance claim this whole build turns on. Kept out of
    // the node captions because two short captions under adjacent nodes collided at every width.
    pcx.fillText('peaks are CrystFEL peakfinder8’s, read from the recorded stream — no live peak-finding pass here',
                 w/2, h-7);
    // the device-resident ring, drawn as the tank it is: fills to B, drains when the batch indexes
    const twd=60*K,tht=30*K,tx=RING[0]-twd/2,ty=RING[1]-tht/2;
    pcx.fillStyle=surf; pcx.strokeStyle=bcol; pcx.lineWidth=1.5; rr(tx,ty,twd,tht,4); pcx.fill();
    pcx.save(); rr(tx,ty,twd,tht,4); pcx.clip(); pcx.globalAlpha=0.5; pcx.fillStyle=bcol;
    pcx.fillRect(tx,ty+tht*(1-rfill),twd,tht*rfill); pcx.globalAlpha=1; pcx.restore();
    if(S.batchFlash>0.05){ pcx.save(); pcx.globalAlpha=0.55*S.batchFlash; pcx.strokeStyle=acc;
      pcx.lineWidth=2.5; rr(tx-2,ty-2,twd+4,tht+4,5); pcx.stroke(); pcx.restore(); }
    rr(tx,ty,twd,tht,4); pcx.stroke();
    pcx.textAlign='center'; pcx.textBaseline='middle';
    pcx.fillStyle=ink; pcx.font='700 '+(12*K).toFixed(1)+'px '+FONT; pcx.fillText(S.ring,RING[0],RING[1]-3*K);
    pcx.fillStyle=mut; pcx.font='600 '+(7*K).toFixed(1)+'px '+FONT; pcx.fillText('device ring',RING[0],RING[1]+9*K);   // batch size is already on the pipe above ('batch of B')
    // the warm-up hold, beside the consensus node -- this is the buffer the rescue drains
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
    pcx.fillText('live gate: \u2265'+Math.round(META.gate_live.min_inlier_frac*100)+'% of spots fit', GL[0], R1+32*K);
    pcx.fillStyle=good; pcx.font='700 '+(9.5*K).toFixed(1)+'px '+FONT;
    pcx.fillText(S.strict+' / '+NF_TRACE, IDX[0], R1+23*K);
    pcx.fillStyle=mut;  pcx.font='700 '+(6.6*K).toFixed(1)+'px '+FONT;
    pcx.fillText('solved = correct lattice,', IDX[0], R1+33*K);
    pcx.fillText('≥25% of spots · ≥10 refl', IDX[0], R1+42*K);
    if(S.rescued>0){ pcx.fillStyle=col('--rescue'); pcx.font='700 '+(7.6*K).toFixed(1)+'px '+FONT;
      pcx.fillText('↑'+S.rescued+' rescued at lock', IDX[0], R1+53*K); }
    if(S.wdog>0){ pcx.fillStyle=coral; pcx.font='700 '+(7.6*K).toFixed(1)+'px '+FONT;
      pcx.fillText('↑'+S.wdog+' watchdog rescue'+(S.wdog>1?'s':''), IDX[0], R1+63*K); }
    pcx.fillStyle=mut; pcx.font='700 '+(7*K).toFixed(1)+'px '+FONT;
    pcx.fillText(S.refused+' of '+NF_TRACE, DROP[0], DROP[1]+16*K);
    pcx.fillStyle=S.locked?good:gold; pcx.font='700 '+(7.8*K).toFixed(1)+'px '+FONT;
    pcx.fillText(S.locked?('locked at frame '+S.lockFrame):('support '+S.support+'/'+SUPPORT),
                 CONS[0], CONS[1]+21*K);
    // The re-lock, once it has happened, stays on screen as a standing fact -- it changed the
    // active set, so it is not a flash that goes away.
    if(S.relocks>0){ pcx.fillStyle=coral; pcx.font='700 '+(7.4*K).toFixed(1)+'px '+FONT;
      pcx.fillText('re-lock ×'+S.relocks+' at frame '+S.relockFrame+' · 2nd cell',
                   CONS[0], CONS[1]+31*K); }
    pcx.textAlign='left'; pcx.textBaseline='alphabetic';
  }
"""

RENDER = r"""  function render(){
    $('frames').textContent=S.frames.toLocaleString();
    $('clk').textContent=S.t.toFixed(1);
    const ph=$('phase'),pt=$('phaseTxt');
    // The re-lock badge is AMBER, not the green 'rescuing' state. Green would read as an
    // endorsement, and on this run the cell the re-lock added is not the sample's lattice — it is
    // a recurring cell voted by frames nothing in the active set explained. Amber says "a
    // mechanism fired", which is all the trace supports.
    ph.className='phase '+(S.phase==='warmup'||S.phase==='relock'?'discovering':S.phase);
    pt.textContent = S.phase==='warmup'?'Blind warm-up — no cell yet'
                   : S.phase==='rescuing'?'Locked + rescuing warm-up frames'
                   : S.phase==='relock'?'Watchdog re-lock — second cell added'
                   : 'Steady — locked, known-cell batch';
    ph.title = S.phase==='warmup'
        ? 'Mc=None: the driver has no cell. It indexes the opening frames one at a time and pools their candidates in a running cross-frame consensus.'
      : S.phase==='rescuing'
        ? 'The consensus fired. The frames spent on discovery are re-indexed against the freshly locked cell (warmup_rescue).'
      : S.phase==='relock'
        ? 'adaptive_relock: the misses the batch pass could not explain voted a recurring cell of their own. It is ADDED to the active set — the first locked cell is not replaced, and the score on this page is against that first cell.'
        : 'The cell is locked: frames fill the device-resident ring and are indexed a batch at a time against it.';
    renderCells();
    drawStream();
  }
"""

RENDER_CELLS = r"""  function renderCells(){
    const host=$('cells'); if(!host) return;
    const sd=SPECIES[0], locked=S.locked;
    const c = locked ? col(sd.cc) : col('--muted');
    // The recovered cell, printed exactly as the driver reported it. The sim template jittered these
    // digits (updateMeas); that function is deleted in this build -- a recorded cell is a measurement.
    const cs = locked ? sd.cell.map((v,k)=> k<3? v.toFixed(2): v.toFixed(2)) : ['—','—','—','—','—','—'];
    const nm  = locked ? sd.name : 'Discovering…';
    const sys = locked ? sd.sys  : 'no cell yet · blind warm-up';
    const pill = locked?`<span class="pill locked">LOCKED ✓</span>`
                       :`<span class="pill disc">CONSENSUS ${Math.min(S.support,SUPPORT)}/${SUPPORT}</span>`;
    host.innerHTML=`<div class="cell${locked?'':' pend'}" style="--cc:${c}">
      <div class="rail"></div>
      <div class="chead"><span class="chip"></span>
        <div><div class="cname">${nm}</div><div class="csys">${sys}</div></div>${pill}</div>
      <div class="params">
        <div><span class="lbl">a b c&nbsp;</span>${cs[0]} · ${cs[1]} · ${cs[2]} <span class="lbl">Å</span></div>
        <div><span class="lbl">α β γ&nbsp;</span>${cs[3]} · ${cs[4]} · ${cs[5]} <span class="lbl">°</span></div>
      </div>
      <div class="foot"><span class="share">${locked?((100*S.strict/Math.max(S.pushed,1)).toFixed(0)+'%'):'—'}</span>
        <span class="fmeta">${S.accepted.toLocaleString()} accepted${locked?'':' · holding'}<br>pushed ${S.pushed.toLocaleString()}</span></div>
    </div>`;
  }
"""

DRAW_STREAM = r"""  function drawStream(){
    // One bar per BIN pushed frames, normalised to that bin's frame count. Bottom-up the stack is
    // the driver's own three outcomes: cleared the strict bar (solid), accepted at the live gate but
    // below it (faint), refused at the live gate (grey). The two envelopes are the two rates those
    // categories define, so the strict line always nests under the accept line by construction.
    const [w,h]=wf._d||fit(wf); wfx.clearRect(0,0,w,h);
    const arr = S.curBin.total>0 ? S.bins.concat([S.curBin]) : S.bins;
    const n=arr.length; if(!n) return;
    const VIS=Math.min(n,64), start=n-VIS, bw=w/VIS;
    const sc=col('--s1'), cg=col('--muted');
    const accOf=b=>b.total>0?(b.idx[0]+b.buf[0])/b.total:0;
    const okOf =b=>b.total>0? b.idx[0]/b.total:0;
    const Y=f=>h-2-(h-4)*clamp(f,0,1);
    for(let k=start;k<n;k++){
      const b=arr[k], T=b.total; if(T<=0) continue;
      const x=(k-start)*bw, bwid=Math.max(1,bw-0.7); let a=0;
      const seg=(cn,color,alpha)=>{ if(cn<=0)return; const y0=Y(a), y1=Y(a+cn/T);
        wfx.fillStyle=color; wfx.globalAlpha=alpha; wfx.fillRect(x,y1,bwid,y0-y1+0.4); a+=cn/T; };
      seg(b.idx[0], sc, 1); seg(b.buf[0], sc, 0.32); seg(b.gen, cg, 0.5);
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
    add(`<span class="sw" style="background:${col('--s1')}"></span>cleared the strict bar`);
    add(`<span class="sw" style="background:${col('--s1')};opacity:.4"></span>accepted, below it`);
    add(`<span class="sw" style="background:${col('--muted')};opacity:.55"></span>refused at the live gate`);
    add(`<span class="swl" style="background:${col('--accent')}"></span>strict rate`);
    add(`<span class="swl" style="background:${col('--ink2')};opacity:.8"></span>live accept rate`);
    const note=document.createElement('span'); note.className='k'; note.style.opacity='.7';
    note.textContent='one bar = 8 pushed frames'; L.appendChild(note);
  }
"""


def strip_panels(txt):
    """Remove the DOM of every panel the shipped driver cannot fill, plus the controls that mean
    nothing on a replay."""
    # 1. detector geometry panel
    a = txt.index('      <div class="panel">\n        <div class="ph"><h2>Detector geometry')
    b = txt.index('      <div class="panel">\n        <div class="ph"><h2>Operational QA')
    txt = txt[:a] + txt[b:]
    # 2. operational QA panel (injected checks + spurious meter)
    a = txt.index('      <div class="panel">\n        <div class="ph"><h2>Operational QA')
    b = txt.index('      <div class="panel">\n        <div class="ph"><h2>Active cells')
    txt = txt[:a] + txt[b:]
    # 3. sample-mixture panel
    a = txt.index('      <div class="panel">\n        <div class="ph"><h2>Sample mixture')
    b = txt.index('      <div class="panel">\n        <div class="ph"><div id="phase"')
    txt = txt[:a] + txt[b:]
    # 4. the canvas handles for the removed panels; drawGeoBox/drawSpurious go with them
    txt = must_replace(txt, "  const geobox=$('geobox'), gbx=geobox.getContext('2d');",
                       "  // geometry panel removed: geom_refine=False, the driver refines nothing",
                       "geobox handle")
    txt = replace_function(txt, "drawGeoBox", "  function drawGeoBox(){}   // geometry panel removed")
    txt = replace_function(txt, "renderQA", "  function renderQA(){}      // no injections on a benchmark replay")
    txt = replace_function(txt, "drawSpurious", "  function drawSpurious(){} // lock_probe=False -- no spurious meter is computed")
    txt = must_replace(txt, "  function buildGeo(){}", "  function buildGeo(){}  // no geometry box on this build",
                       "buildGeo")
    # the geometry box's own constants outlive the engine block (they sit in the RENDER section) and
    # read TRUE_GEO, which went with the simulation. Remove them or the script dies on load.
    a = txt.index("  const GEO_XC=TRUE_GEO.x")
    b = txt.index("  function hexa(", a)
    txt = txt[:a] + "  // geometry-box constants removed with the panel (they read the sim's TRUE_GEO)\n" + txt[b:]
    # 5. one cell slot, not four
    txt = must_replace(txt, '<h2>Active cells · at most 4 in play</h2><span class="note" id="cellNote">multi-cell active set</span>',
                       '<h2>Recovered cell</h2><span class="note" id="cellNote">one protein · discovered blind, no cell handed in</span>',
                       "cells header")
    # 6. relayout. Three of the template's six panels are gone, and the surviving one on the left
    #    (a single cell card in a 2-up grid) left half a column empty. Put the schematic first and
    #    wide, the cell card second and narrow, and let the single card fill its panel.
    txt = must_replace(txt, ".main{display:grid;grid-template-columns:1.32fr 1fr;",
                       ".main{display:grid;grid-template-columns:1.62fr 1fr;", "main grid")
    txt = must_replace(txt, ".cells{display:grid;grid-template-columns:1fr 1fr;",
                       ".cells{display:grid;grid-template-columns:1fr;", "cells grid")
    #    Rebuild the two columns explicitly from the three surviving panels rather than shuffling the
    #    template's own nesting: schematic alone on the wide side, cell card over the composition bars
    #    on the narrow side, which is what leaves the two columns roughly the same height.
    m = re.search(r'  <div class="main">\n(.*?)\n  </div>\n\n  <p class="foot">', txt, re.S)
    if not m:
        raise SystemExit("main block not found -- cannot rebuild the layout")
    # Panels sit at a 6-space indent and close with a line that is exactly `      </div>`; the column
    # wrappers are at 4. Scan lines rather than splitting on the open tag, so a column closer is never
    # mistaken for a panel closer.
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
        key = ("cells" if 'id="cells"' in p else "pipe" if 'id="pipe"' in p else
               "wf" if 'id="wf"' in p else None)
        if key:
            by[key] = p
    if set(by) != {"cells", "pipe", "wf"}:
        raise SystemExit(f"expected exactly the cells/pipe/wf panels, got {sorted(by)}")
    main = ('  <div class="main">\n'
            '    <div class="col">\n' + by["pipe"] + '    </div>\n\n'
            '    <div class="col">\n' + by["cells"] + by["wf"] + '    </div>\n'
            '  </div>\n')
    txt = txt[:m.start()] + main + '\n  <p class="foot">' + txt[m.end():]
    txt = must_replace(txt, '<canvas id="pipe" height="200" style="height:200px">',
                       '<canvas id="pipe" height="316" style="height:316px">', "pipe height")
    txt = must_replace(txt,
                       '<span class="note"><span style="color:var(--ink2)">hit</span> &amp; '
                       '<span style="color:var(--accent)">indexed</span> share · newest →</span>',
                       '<span class="note"><span style="color:var(--accent)">strict</span> &amp; '
                       '<span style="color:var(--ink2)">accepted</span> share · newest →</span>',
                       "composition note")
    txt = must_replace(txt, '<span class="note" id="cellNote">one protein · discovered blind, no cell handed in</span>',
                       '<span class="note" id="cellNote">discovered blind</span>', "cell note")
    # 7. the inject button is meaningless over a recorded trace (it repositions the playhead only)
    txt = must_replace(txt, '      <button id="inject" class="btn accent">＋ Inject sample</button>\n', "",
                       "inject button")
    txt = must_replace(txt, "  $('inject').addEventListener('click',()=>{ injectSample();",
                       "  const _inj=$('inject');   // absent on replay builds\n"
                       "  if(_inj) _inj.addEventListener('click',()=>{ injectSample();",
                       "inject handler")
    # injectSample() itself lived inside the sim engine block and went with it; the handler above is
    # already guarded by the null `_inj`, so the reference is never evaluated.
    return txt


def footer(tr):
    p = tr["provenance"]
    kw = tr["driver_kw"]
    band = tr["published_band"]
    ex = tr.get("extra_cells") or []
    relock_cell = (", ".join(f"{v:.1f}" for v in ex[0]["cell"])) if ex else "—"
    relock_is_ref = bool(ex and ex[0]["same_lattice_vs_reference"])
    return (
        "  <p class=\"foot\">\n"
        "    <b>Recorded replay — not a live beamline, and not a simulation.</b> Every frame pushed through this\n"
        "    monitor is a peak list from the <b>480-frame extension of the cxidb-17 sparse lysozyme run</b>\n"
        "    (CXIDB entry 17, CC0) — the paper's primary streaming set, four times the published 120-frame\n"
        "    subset, and the <i>same run and the same crystals</i>: 480 buys statistical power, not generality.\n"
        "    One protein. Nothing injected, nothing planted, no second species, no drift.\n"
        "    <br><br>\n"
        "    <b>Where the peaks come from.</b> They are <b>CrystFEL 0.12.0 peakfinder8</b> peaks, read out of the\n"
        "    blind <code>indexamajig</code> stream for these 480 images and converted to reciprocal-space\n"
        "    q-vectors through <b>the package's own bridge</b>, <code>glint.geom.read_crystfel_peaks</code> +\n"
        "    <code>peaks_to_q</code> — not a re-derivation, so panel membership, <code>coffset</code> and the\n"
        "    wavelength convention cannot drift. Following the published 120-frame benchmark, <b>one fixed\n"
        "    wavelength (1.322216 Å, 9377.00 eV — frame 0's photon energy) is used for every frame</b> rather\n"
        "    than the per-shot SASE energy; that is the convention the published q lists were built under, and\n"
        "    it is worth a few frames per arm, so it is stated rather than assumed. No pixels are replayed and\n"
        "    <b>no peak finder runs here</b> — the driver is fed q-vectors, so no peak-finding lane is claimed.\n"
        "    <br><br>\n"
        "    <b>The first 120 of this list are not the published 120-frame subset.</b> Sliced to its first 120\n"
        "    frames this input gives 76 / 79 at the strict bar where the published subset gives 73 / 78: the\n"
        "    rebuilt lists keep peaks in the last fractional pixel of each panel that the pre-fix geometry code\n"
        "    dropped. Different peak lists, expected, not a discrepancy — but it is why the number on this page\n"
        "    should be quoted with its input named.\n"
        "    <br><br>\n"
        f"    The indexer is the <b>shipped</b> <code>glint.stream_driver.StreamDriver</code> at commit\n"
        f"    <code>{p.get('git','')[:12]}</code>, cold-started with no cell (<code>Mc=None</code>), at its constructor\n"
        f"    defaults plus the two published opt-ins <code>warmup_rescue=True</code> and\n"
        f"    <code>adaptive_relock=True</code>: "
        f"<code>B={kw['B']}</code>, <code>dmin={kw['dmin']}</code>, <code>tol={kw['tol']}</code>, "
        f"<code>warmup_nbest={kw['warmup_nbest']}</code>, live accept gate "
        f"<code>min_inlier_frac={tr['gate_live']['min_inlier_frac']}</code>, "
        f"<code>min_inliers={tr['gate_live']['min_inliers']}</code>. <b>OFF:</b> retry cascade, live geometry\n"
        "    refinement, second-lattice detection, alias gate, lock probe, miss-buffer rescue. Panels those\n"
        "    options would have filled are removed rather than left idle, which is why there is no geometry box,\n"
        "    no injected-QA row and no spurious meter on this build.\n"
        "    <br><br>\n"
        f"    Consensus locked the cell after <b>{tr['locked_after']} frames</b> on {tr['consensus_support']} votes; the\n"
        f"    {tr['n_warmup_rescued']} frames spent on discovery were re-indexed against it and\n"
        f"    <b>{tr['strict_warmup']} of them cleared the strict bar</b>. The driver accepted\n"
        f"    <b>{tr['n_accepted_live']} of {tr['nframes']}</b> at its own live gate and refused {tr['n_gate_rejected']}. Scored at the paper's strict\n"
        f"    research bar (≥25% of spots <i>and</i> ≥10 reflections, correct lattice), the run indexes\n"
        f"    <b>{tr['strict_total']}/{tr['nframes']}</b> ({100.0*tr['strict_total']/tr['nframes']:.1f}%) — "
        f"{tr['strict_postlock']} post-lock plus the {tr['strict_warmup']} rescued. That is the\n"
        f"    <b>high end</b> of the range the paper quotes for the streaming driver on this set,\n"
        f"    <b>“{band['pct']} of the 480-frame set”</b> ({band['low']}–{band['high']} of {band['of']}): the low end is\n"
        f"    <b>{band['low_arm']}</b>, so the two options enabled here are published arms, not tuned ones. The\n"
        f"    offline known-cell reference on the same frames and the same gate is\n"
        f"    <b>{band['offline']}/{band['of']}</b> ({100.0*band['offline']/band['of']:.1f}%).\n"
        "    <br><br>\n"
        f"    <b>What 480 frames show that 120 cannot.</b> Across 120 frames of this same list the watchdog\n"
        f"    fires zero times. Across these 480 it performs <b>{tr['n_watchdog_rescued']} individual rescues</b> — frames the\n"
        f"    batch pass missed, blind-solved again one at a time and accepted after all — and <b>one re-lock</b>,\n"
        f"    at frame {tr['relock_frame']}. Read the re-lock honestly: it does not replace the locked cell, it\n"
        f"    <b>adds a second cell to the active set</b>, and that added cell "
        + ("<b>is</b> the reference lattice.\n" if relock_is_ref else
           f"is <b>not</b> the lysozyme lattice\n    ({relock_cell}) — it is a cell voted by frames the active set could not explain. Every number on\n"
           "    this page is scored against the <b>first</b> locked cell only, so the re-lock adds nothing to\n"
           "    the count; it is on screen because it is what the mechanism did.\n") +
        "    <br><br>\n"
        f"    Recorded on {p.get('gpu','?')} ({p.get('host','?')}), cupy {p.get('cupy','?')}, torch\n"
        f"    {p.get('torch','?')}, numpy {p.get('numpy','?')}; {p.get('elapsed_s','?')} s of wall time for the whole replay.\n"
        "    Playback is looped and the frame rate is a display choice; every counter, the recovered cell, the\n"
        "    lock frame, the ring occupancy, the rescues and the re-lock are read out of that one recorded run.\n"
        "    The cell is printed as measured and is not re-jittered for the animation.\n"
        "  </p>\n"
    )


def main(trace_path):
    tr = json.load(open(trace_path))
    # The build refuses anything that is not the measured 331/480 arm. The gate run that licensed
    # this animation reproduced all four published arms on current main (323 / 326 / 328 / 331,
    # offline 357); a trace that does not land on 331 is a different run and must not be published
    # under this page's footer.
    if tr["nframes"] != EXPECT_FRAMES:
        raise SystemExit(f"refusing to build: trace has {tr['nframes']} frames, expected {EXPECT_FRAMES}")
    if tr["strict_total"] != EXPECT_TOTAL:
        raise SystemExit(f"refusing to build: trace scores {tr['strict_total']}/{tr['nframes']}, "
                         f"expected {EXPECT_TOTAL}/{EXPECT_FRAMES}")
    if not tr["same_lattice_vs_reference"]:
        raise SystemExit("refusing to build: the driver's locked cell is not the reference lattice")
    if (tr["n_watchdog_rescued"], tr["n_relock"]) != (6, 1):
        raise SystemExit(f"refusing to build: watchdog fired {tr['n_watchdog_rescued']} rescues / "
                         f"{tr['n_relock']} re-locks, expected 6 / 1 (the paper's SI counts)")

    lines = open(TEMPLATE).read().split("\n")
    start = next(i for i, l in enumerate(lines) if l.strip().startswith("const SPECIES=["))
    end = next(i for i, l in enumerate(lines) if "================= RENDER" in l)
    txt = "\n".join(lines[:start - 1]) + "\n" + engine(tr) + "\n" + "\n".join(lines[end - 1:])

    txt = strip_panels(txt)
    txt = replace_function(txt, "drawPipeline", DRAW_PIPELINE)
    txt = replace_function(txt, "render", RENDER)
    txt = replace_function(txt, "renderCells", RENDER_CELLS)
    txt = replace_function(txt, "drawStream", DRAW_STREAM)
    txt = replace_function(txt, "buildLegend", BUILD_LEGEND)

    # A resize CLEARS the canvases: fit() assigns cv.width/cv.height, and that resets the backing
    # store. The stream chart is repainted only by render(), which the rAF loop throttles behind
    # `now-lastRender>90`, so a resize landing inside that ~90 ms window leaves the panel blank --
    # 1 of 3 headless captures for the wide Fig 1(e) panel came out with an empty stream chart.
    # Repaint synchronously in the handler instead of waiting for the next unthrottled tick.
    txt = must_replace(txt,
                       "  window.addEventListener('resize',fitAll);",
                       "  window.addEventListener('resize',function(){\n"
                       "    fitAll();                          // this CLEARS every canvas (cv.width= resets the backing store)\n"
                       "    try{ render(); drawPipeline(); }catch(e){}   // ...so repaint now: render() is rAF-throttled to ~90 ms\n"
                       "  });",
                       "resize repaint")

    # header / title / captions
    # The template carries no charset declaration -- it is a fragment, and over http:// without one
    # the browser falls back to windows-1252 and mangles every ·, ≥ and em-dash on the page. Declare it.
    txt = must_replace(txt, "<title>GLINT Streaming Driver — Live Monitor</title>",
                       '<meta charset="utf-8">\n<title>GLINT streaming driver · cxidb-17 480-frame replay</title>',
                       "title")
    txt = must_replace(txt, "GLINT · device-resident streaming indexer",
                       "GLINT · device-resident streaming driver · recorded replay", "eyebrow")
    txt = must_replace(txt, "Live Streaming Monitor",
                       "cxidb-17 · 480-frame replay", "h1")
    txt = must_replace(txt, '<span class="note">LCLS → detector → PeakReducer → GLINT</span>',
                       '<span class="note">recorded peaks (benchmark) → ring → GLINT</span>', "pipe note")
    # `?print=1` hides the transport chrome for a figure snapshot. The audit note on the CURRENT
    # Fig 9 asks for exactly this ("crop the Play/speed/Reset chrome"); doing it with a parameter
    # keeps one file that is both the interactive replay and the figure source, so the snapshot can
    # never drift from the page it claims to be of. The frame/clock readout STAYS -- it says which
    # moment of the replay the still is.
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
    # ...and freeze playback once the warm-up pre-advance has landed. Without this the still's frame
    # number is whatever the render loop reached before the screenshot fired, which differed between
    # the light and dark passes -- a figure pair has to be the SAME moment.
    txt = must_replace(txt,
                       "  if(warm>0){ for(let acc=0;acc<warm;acc+=0.033) advance(0.033); fitAll(); render(); }",
                       "  if(warm>0){ for(let acc=0;acc<warm;acc+=0.033) advance(0.033); fitAll(); render(); }\n"
                       "  if(_qs.get('print')) playing=false;   // freeze the still exactly at ?warm=",
                       "print freeze")
    # footer
    a = txt.index('  <p class="foot">')
    b = txt.index("</p>", a) + len("</p>\n")
    txt = txt[:a] + footer(tr) + txt[b:]

    with open(OUT, "w") as f:
        f.write(txt)
    print(f"wrote {OUT}")
    print(f"  {tr['nframes']} trace records · locked after {tr['locked_after']} · "
          f"{tr['n_warmup_rescued']} warm-up rescued ({tr['strict_warmup']} pass) · "
          f"{tr['strict_total']}/{tr['nframes']} at the strict bar")
    print(f"  watchdog: {tr['n_watchdog_rescued']} rescues, {tr['n_relock']} re-lock "
          f"at frame {tr['relock_frame']}")
    print(f"  cell {tr['cell']}")
    # The snapshot moment this build is FOR: the re-lock. PLAY_FPS is 12, so ?warm= is
    # frames/12 seconds of pre-advance.
    print(f"  snapshot: ?warm={(tr['relock_frame'] + 6) / 12.0:.2f}&print=1  "
          f"(≈6 frames past the re-lock, inside its 2 s phase hold)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TRACE)
