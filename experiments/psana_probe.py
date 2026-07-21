"""Probe 2: get per-pixel lab coords (x,y,z) from psana geometry + photon energy.
These give exact q -- no distance calibration needed."""
import sys, os

import numpy as np
from psana import DataSource

EXP = os.environ.get("GLINT_EXP")   # beamtime ID: proprietary, so not committed
if not EXP:
    sys.exit("set GLINT_EXP=<experiment id>; the ID is deliberately not in this file "
             "(proprietary beamtime data -- see experiments/README_beamtime.md)")
ds = DataSource(exp=EXP, run=int(os.environ.get("GLINT_RUN", "51")))
myrun = next(ds.runs())
det = myrun.Detector("jungfrau")

# --- geometry: per-pixel coords ---
try:
    cc = det.calibconst
    print("calibconst keys:", list(cc.keys()) if hasattr(cc, "keys") else type(cc))
    geotxt = cc["geometry"][0]
    from psana.pscalib.geometry.GeometryAccess import GeometryAccess
    geo = GeometryAccess()
    geo.load_pars_from_str(geotxt if isinstance(geotxt, str) else geotxt.decode())
    X, Y, Z = geo.get_pixel_coords()
    print(f"pixel coords (um): X{X.shape} Z-range[{np.nanmin(Z):.0f},{np.nanmax(Z):.0f}] "
          f"(detector distance ~ {np.nanmean(Z)/1e4:.1f} cm)")
except Exception as e:
    print(f"geometry coords ERR: {type(e).__name__}: {e}")

# --- photon energy ---
for dn in ("ebeamh", "gasdet", "feespec", "epicsinfo"):
    try:
        d = myrun.Detector(dn)
        print(f"{dn}: raw methods {[m for m in dir(d.raw) if not m.startswith('_')][:8]}")
    except Exception as e:
        print(f"{dn}: ERR {e}")

for i, evt in enumerate(myrun.events()):
    if det.raw.calib(evt) is not None:
        try:
            eb = myrun.Detector("ebeamh")
            print("ebeam photon energy attempt:", eb.raw.ebeamPhotonEnergy(evt))
        except Exception as e:
            print("photon energy ERR:", e)
        break
    if i > 30:
        break
