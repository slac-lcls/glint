"""Verify all 4 CBXD datasets regenerate from their frozen selections, RESULT-identical to committed."""
import os, sys, glob
sys.path.insert(0, "..")
import numpy as np


def cmp_crystals(rec, comm):
    ok = 0
    for r, c in zip(rec, comm):
        if (np.allclose(r["Rt"], c["Rt"]) and r["kobs"].shape == c["kobs"].shape
                and np.allclose(r["kobs"], c["kobs"]) and np.array_equal(r["lab"], c["lab"])):
            ok += 1
    return ok


# ---- base (flat, own selection) ----
import generate_dataset as g
rec = g.build_dataset()
comm = [dict(Rt=(d := np.load(f"{g.OUTDIR}/crystal_{i:03d}.npz"))["Rt"], kobs=d["kobs"], lab=d["lab"]) for i in range(g.N_KEEP)]
print(f"base      : {cmp_crystals(rec, comm)}/{g.N_KEEP} reproduced")

# ---- lowna (flat, own selection) ----
import generate_dataset_lowna as gl
rec = gl.build_dataset()
comm = [dict(Rt=(d := np.load(f"{gl.OUTDIR}/crystal_{i:03d}.npz"))["Rt"], kobs=d["kobs"], lab=d["lab"]) for i in range(gl.N_KEEP)]
print(f"lowna     : {cmp_crystals(rec, comm)}/{gl.N_KEEP} reproduced")

# ---- grid (frozen via orientations_na*.npz) : regen a committed cell inline, compare ----
import generate_dataset_grid as gg
from cbxd_joint import simulate
cell = sorted(glob.glob(os.path.join(gg.GRID_DIR, "na0.028_noise*")))[0]
noise = float(os.path.basename(cell).split("noise")[1])
na = 0.028
Rts = gg.get_orientation_pool(na)
gg.set_NA(na)
ok = 0
for i, Rt in enumerate(Rts):
    rng = np.random.default_rng(200_000 + int(round(na * 1000)) + i)
    k, l, c = simulate(Rt, rng, noise)
    d = np.load(f"{cell}/crystal_{i:03d}.npz")
    ok += int(k.shape == d["kobs"].shape and np.allclose(k, d["kobs"]) and np.array_equal(l, d["lab"]))
print(f"grid      : {ok}/{len(Rts)} reproduced  (cell {os.path.basename(cell)})")

# ---- multishot (reuses grid orientations_na0.028) : regen a committed noise dir inline ----
import generate_dataset_multishot as gm
nd = sorted(glob.glob(os.path.join(gm.OUTDIR, "noise_*")))[0]
mnoise = float(os.path.basename(nd).split("noise_")[1])
gm.set_NA(gm.NA)
Rts = gm.generate_crystals()
ok = 0
for i, Rt in enumerate(Rts):
    rng = np.random.default_rng(200_000 + int(round(gm.NA * 1000)) + i)
    shots = gm.generate_shots(Rt, mnoise, gm.M_MAX, rng)
    z = np.load(f"{nd}/crystal_{i:03d}.npz")
    ok += int(all(shots[m][0].shape == z[f"kobs_{m}"].shape and np.allclose(shots[m][0], z[f"kobs_{m}"])
                  for m in range(gm.M_MAX)))
print(f"multishot : {ok}/{len(Rts)} reproduced  (dir {os.path.basename(nd)}, all {gm.M_MAX} shots)")
