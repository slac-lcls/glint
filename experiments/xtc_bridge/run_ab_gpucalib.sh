#!/bin/bash
#SBATCH -p ampere -A lcls:prjdat21 --gpus=1 -c 8 --mem=96G -t 2:00:00
#SBATCH -o /sdf/home/s/smarches/glint_gt_mfxx49820/abcalib_%j.log
H=/sdf/home/s/smarches/glint_gt_mfxx49820
G=/sdf/home/s/smarches/glint_bench_pf8
GEOM=/sdf/data/lcls/ds/mfx/mfxx49820/results/btx/geom/r0016.geom
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
conda activate ana-4.0.58-py3-minipytorch
export PYTHONPATH=$G:$PYTHONPATH
cd $G/experiments/xtc_bridge
hostname; nvidia-smi --query-gpu=name --format=csv,noheader

COMMON="--exp mfxx49820 --run 16 --det MfxEndstation.0:Epix10ka2M.0 \
  --zdist 0.102973 --wavelength 1.290757 --psana 1 --min-peaks 6 --max-events 1500 \
  --calib-dir $H/calib --geom $GEOM"

echo "############ A: det.calib (CPU reference) ############"
time python -u glint_xtc.py $COMMON -o $H/ab_cpu.stream 2>&1 | tail -12

echo; echo "############ B: --gpu-calib ############"
time python -u glint_xtc.py $COMMON --gpu-calib -o $H/ab_gpu.stream 2>&1 | tail -12

echo; echo "############ STREAM COMPARISON ############"
python -u - <<PY
import re
def chunks(p):
    t=open(p).read()
    out={}
    for c in t.split("----- Begin chunk -----")[1:]:
        ev=re.search(r"Event:\s*//(\d+)",c)
        cell=re.findall(r"[abc]star = *([-\d.]+) *([-\d.]+) *([-\d.]+)",c)
        npk=re.search(r"num_peaks *= *(\d+)",c)
        out[int(ev.group(1)) if ev else -1]=(tuple(tuple(r) for r in cell),
                                             npk.group(1) if npk else None)
    return out
a=chunks("$H/ab_cpu.stream"); b=chunks("$H/ab_gpu.stream")
print(f"indexed chunks: CPU={len(a)}  GPU={len(b)}")
print(f"same event set: {set(a)==set(b)}")
if set(a)!=set(b):
    print("  only CPU:",sorted(set(a)-set(b))[:20])
    print("  only GPU:",sorted(set(b)-set(a))[:20])
common=sorted(set(a)&set(b))
diff=[e for e in common if a[e]!=b[e]]
print(f"common events: {len(common)}   differing astar/bstar/cstar or num_peaks: {len(diff)}")
for e in diff[:5]: print("   ev",e,"\n     CPU",a[e],"\n     GPU",b[e])
import hashlib
ha=hashlib.sha256(open("$H/ab_cpu.stream","rb").read()).hexdigest()[:16]
hb=hashlib.sha256(open("$H/ab_gpu.stream","rb").read()).hexdigest()[:16]
print(f"stream sha256: CPU={ha} GPU={hb}  IDENTICAL={ha==hb}")
PY
