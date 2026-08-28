"""Build EXACT assembled-image -> lab coordinate maps for the $GLINT_EXP r278 run from psana's
real calibrated geometry (real distance, panel tilts, exact beam center), so PeakNet
peaks in assembled (row,col) convert to q with no flat-detector guessing. Also grab the
real photon energy (CXI energy=0). Saves AX,AY,AZ (assembled HxW, meters) + lam.

The deployment assembled via psana (geom_file=null), so psana's det.raw.image layout =
the CXI's 1667x1668 grid; get_pixel_coord_indexes() gives each panel pixel's assembled
(iy,ix), get_pixel_coords() its lab (X,Y,Z) -> scatter into the assembled grid.

  source .../psconda.sh ; python extract_geom_1038.py
"""
import os, sys, numpy as np
from psana import DataSource

EXP = os.environ.get("GLINT_EXP")   # beamtime ID: proprietary, so not committed
if not EXP:
    sys.exit("set GLINT_EXP=<experiment id>; the ID is deliberately not in this file "
             "(proprietary beamtime data -- see experiments/README_beamtime.md)")
RUN = int(os.environ.get("GLINT_RUN", "278"))
ds = DataSource(exp=EXP, run=RUN)
myrun = next(ds.runs())

# find the epix10ka2M detector
detname = None
for nm in ("epix10k2M", "epix10ka2M", "epixquad", "epix10kaquad", "Epix10ka2M", "epix"):
    try:
        d = myrun.Detector(nm)
        detname = nm
        det = d
        break
    except Exception:
        continue
if detname is None:
    print("detector names:", myrun.detnames if hasattr(myrun, "detnames") else "?")
    raise SystemExit("no epix detector found")
print(f"detector = {detname}", flush=True)

from psana.pscalib.geometry.GeometryAccess import GeometryAccess
geo = GeometryAccess()
geo.load_pars_from_str(det.calibconst["geometry"][0])
X, Y, Z = geo.get_pixel_coords()                 # lab coords (um), panel-shaped
ix, iy = geo.get_pixel_coord_indexes()           # assembled-image indices, panel-shaped
X = X.ravel() * 1e-6; Y = Y.ravel() * 1e-6; Z = Z.ravel() * 1e-6   # -> meters
ix = ix.ravel().astype(int); iy = iy.ravel().astype(int)
H, W = int(iy.max()) + 1, int(ix.max()) + 1
print(f"assembled grid {H}x{W} (CXI is 1667x1668)", flush=True)
AX = np.full((H, W), np.nan, np.float32); AY = np.full((H, W), np.nan, np.float32)
AZ = np.full((H, W), np.nan, np.float32)
AX[iy, ix] = X; AY[iy, ix] = Y; AZ[iy, ix] = Z
print(f"lab ranges  X[{np.nanmin(AX):.3f},{np.nanmax(AX):.3f}]  "
      f"Y[{np.nanmin(AY):.3f},{np.nanmax(AY):.3f}]  Z[{np.nanmin(AZ):.4f},{np.nanmax(AZ):.4f}] m", flush=True)
# beam center (lab 0,0) in assembled coords
cy0 = float(np.interp(0.0, [np.nanmin(AY), np.nanmax(AY)], [0, H]))   # rough, just for log
print(f"approx beam row where Y=0 ~ {cy0:.0f}", flush=True)

# real photon energy
lam = None
for dn in ("ebeam", "ebeamh", "feespec", "gasdet"):
    try:
        e = myrun.Detector(dn)
        for evt in myrun.events():
            pe = None
            for meth in ("ebeamPhotonEnergy", "photonEnergy"):
                try:
                    pe = getattr(e.raw, meth)(evt)
                except Exception:
                    pass
                if pe:
                    break
            if pe and pe > 1000:
                lam = 12398.42 / pe
                print(f"photon energy {pe:.0f} eV (det {dn}) -> lam {lam:.4f} A", flush=True)
                break
        if lam:
            break
    except Exception:
        continue
if lam is None:
    lam = 12398.42 / 9800.0
    print(f"no energy found; nominal lam {lam:.4f} A", flush=True)

np.savez("/sdf/home/s/smarches/geom_1038.npz", AX=AX, AY=AY, AZ=AZ, lam=np.float32(lam), H=H, W=W)
print("wrote /sdf/home/s/smarches/geom_1038.npz", flush=True)
