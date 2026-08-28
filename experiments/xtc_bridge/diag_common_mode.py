import os, sys
os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np
G="/sdf/home/s/smarches/glint_bench_pf8"
sys.path.insert(0,G); sys.path.insert(0,G+"/experiments/xtc_bridge")
import psana
psana.setOption("psana.calib-dir","/sdf/home/s/smarches/glint_gt_mfxx49820/calib")
ds=psana.DataSource("exp=mfxx49820:run=16"); det=psana.Detector("MfxEndstation.0:Epix10ka2M.0")
import cupy as cp, gpu_calib
gc=gpu_calib.GpuCalibrator(det,16)
for evt in ds.events():
    raw=det.raw(evt)
    if raw is None: continue
    r=cp.asarray(raw).reshape(gc.shape); hi=(r & gc.B14)!=0
    a=(r & gc.DATA_MASK).astype(cp.float32)-cp.where(hi,gc._p1,gc._p0)
    g=cp.where(hi,gc._h1,gc._h0)
    print("mask_total: None?",gc._mask is None,
          " frac good =",float(cp.mean(gc._mask))if gc._mask is not None else None)
    print("gmask (H/M and good) fraction:",float(cp.mean(g.astype(cp.float32))))
    print("gain-bit-14 set fraction     :",float(cp.mean(hi.astype(cp.float32))))
    nseg,H,W=a.shape; hr=H//2
    va=cp.ascontiguousarray(a.reshape(nseg,2,hr,W).transpose(0,1,3,2))
    ga=cp.ascontiguousarray(g.reshape(nseg,2,hr,W).transpose(0,1,3,2))
    s=cp.sort(cp.where(ga.astype(bool),va,cp.inf),axis=-1)
    k=ga.sum(axis=-1,dtype=cp.int32)
    lo=cp.maximum((k-1)//2,0)[...,None]; hiix=(k//2)[...,None]
    m=0.5*(cp.take_along_axis(s,lo,-1)+cp.take_along_axis(s,hiix,-1))[...,0]
    kk=cp.asnumpy(k).ravel(); mm=cp.asnumpy(m).ravel()
    fin=np.isfinite(mm)
    print(f"column groups: {kk.size}  with npix>10: {(kk>10).sum()}  "
          f"({100.0*(kk>10).sum()/kk.size:.1f}%)  median npix={np.median(kk):.0f}")
    v=mm[fin & (kk>10)]
    if v.size:
        print(f"raw offsets (npix>10): n={v.size} |m| median={np.median(np.abs(v)):.3f} "
              f"p90={np.percentile(np.abs(v),90):.3f} max={np.abs(v).max():.3f} ADU")
        print(f"  passing |m|<cormax={gc.cormax}: {(np.abs(v)<gc.cormax).sum()} "
              f"({100.0*(np.abs(v)<gc.cormax).sum()/v.size:.1f}%)")
    applied=gc._seg_offset(va,ga)
    ap=cp.asnumpy(applied).ravel()
    print(f"offsets AFTER both vetoes: nonzero={np.count_nonzero(ap)} / {ap.size}  "
          f"max|.|={np.abs(ap).max():.3f} ADU")
    # what psana itself does, same event, cm on vs off
    c_on=np.asarray(det.calib(evt),np.float32)
    c_off=np.asarray(det.calib(evt,cmpars=(7,0,0,0)),np.float32)
    d=np.abs(c_on-c_off)
    print(f"psana det.calib cm-on vs cm-off: changed px={np.count_nonzero(d)} "
          f"max|delta|={d.max():.5f} rms={np.sqrt((d*d).mean()):.6f}")
    break
