"""cctbx/DIALS-python: dump rlp (lab-frame reciprocal vectors, 1/A) + frame id +
miller_index from a .refl to .npz, so fftindex (psana2 env) can index them."""
import sys

import numpy as np
from dials.array_family import flex

refl = flex.reflection_table.from_file(sys.argv[1])
print("N reflections:", len(refl))
print("columns:", list(refl.keys()))
if "rlp" not in refl:
    sys.exit("no 'rlp' column -- need spot-finding/indexing output")


def vec3(col):
    return np.array(col.as_double()).reshape(-1, 3)


out = {"rlp": vec3(refl["rlp"])}
try:
    out["id"] = refl["id"].as_numpy_array()
except Exception:
    out["id"] = np.array(list(refl["id"]))
if "miller_index" in refl:
    out["miller"] = np.array([list(m) for m in refl["miller_index"]])
np.savez(sys.argv[2], **out)
print(f"saved {sys.argv[2]}: rlp {out['rlp'].shape}, frames {len(set(out['id'].tolist()))}, "
      f"|rlp| range [{np.linalg.norm(out['rlp'],axis=1).min():.4f},"
      f"{np.linalg.norm(out['rlp'],axis=1).max():.4f}]")
