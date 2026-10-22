"""The xtc route's stream must carry the peaks each frame was indexed from, in the frame its
orientation is written in -- no psana, no torch, numpy only.

    python test_stream_peaks.py

Until this was fixed, glint/stream.py:write_stream wrote `num_peaks = 0` on every chunk, so a strict
score (correct lattice AND enough peaks near integer hkl) could not be computed from an xtc-route
stream and only same_lattice could be checked. And the orientation was written in pass 1's frame,
which puts the detector at Z_SIGN*zdist, so even with the peaks a reader of the stream and its .geom
would have tested CrystFEL-frame q against a mirrored basis.

The test builds a two-panel .geom, turns a set of pixels into pass-1 q exactly as xtc_core.frame_q
does, plants a basis M that indexes those q, writes the stream through glint_xtc.stream_records +
glint.stream.write_stream, and reads the peak list back through the .geom with GLINT's own
lute_bridge.peaks_to_q. It checks:
  1. every chunk carries its peaks (count, panel names, |q| column);
  2. the peaks read back are the same q, in CrystFEL's frame (q_CrystFEL = S q_pass1);
  3. q_CrystFEL . M_written equals the written hkl -- the invariant a strict score rests on.
Checks 1 and 3 fail on the old writer.
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))     # repo root, for glint

import geom_coords
import glint_xtc
import xtc_core
from glint.lute_bridge import HC_EV_A, peaks_to_q
from glint.stream import write_stream

H, W = 64, 96                       # two panels stacked in ss, as a 2-segment detector
ZDIST = 0.12                        # m
GEOM = f"""res = 13333.3
clen = /LCLS/detector_z
photon_energy = 9537
coffset = 0.025
p0/min_fs = 0
p0/max_fs = {W - 1}
p0/min_ss = 0
p0/max_ss = {H - 1}
p0/fs = +1.0x +0.0y +0.2z
p0/ss = +0.0x +1.0y -0.1z
p0/corner_x = -48.0
p0/corner_y = -70.0
p0/coffset = 0.012
p1/min_fs = 0
p1/max_fs = {W - 1}
p1/min_ss = {H}
p1/max_ss = {2 * H - 1}
p1/fs = -1.0x +0.0y -0.3z
p1/ss = +0.0x -1.0y +0.4z
p1/corner_x = 48.0
p1/corner_y = 70.0
p1/coffset = -0.017
"""


def pass1_q(geom_path, seg, ss, fs, wavelength):
    """Pixels -> q exactly as the reader builds it (geom_coords + xtc_core.frame_q's arithmetic)."""
    Xf, Yf, Zf = geom_coords.coords_from_geom(geom_path, (2, H, W), ZDIST)
    X = Xf.reshape(2, H, W) * 1e-6
    Y = Yf.reshape(2, H, W) * 1e-6
    Zc = np.sign(np.nanmean(Zf)) * ZDIST
    kin = np.array([0.0, 0.0, np.sign(Zc)])
    r = np.stack([X[seg, ss, fs], Y[seg, ss, fs], np.full(seg.size, Zc)], axis=1)
    s = r / np.linalg.norm(r, axis=1, keepdims=True)
    return (s - kin) / wavelength


def read_peaks(stream_text):
    """-> list per chunk of (n,3) fs, ss, invd[1/nm] and panel names, from the peak-list block."""
    chunks, cur, inpk = [], None, False
    for line in stream_text.splitlines():
        if line.startswith("----- Begin chunk"):
            cur = {"rows": [], "panels": [], "num_peaks": None, "energy": None, "clen": None}
        elif line.startswith("num_peaks = ") and cur is not None:
            cur["num_peaks"] = int(line.split("=")[1])
        elif line.startswith("photon_energy_eV = ") and cur is not None:
            cur["energy"] = float(line.split("=")[1])
        elif line.startswith("average_camera_length = ") and cur is not None:
            cur["clen"] = float(line.split("=")[1].split()[0])
        elif line.startswith("  fs/px   ss/px"):
            inpk = True
        elif line.startswith("End of peak list"):
            inpk = False
        elif inpk:
            v = line.split()
            cur["rows"].append([float(v[0]), float(v[1]), float(v[2])])
            cur["panels"].append(v[4])
        elif line.startswith("----- End chunk"):
            chunks.append(cur)
    return chunks


def serialized_geom(stream_text):
    return stream_text.split("----- Begin geometry file -----\n", 1)[1].split(
        "\n----- End geometry file -----", 1)[0]


def main():
    rng = np.random.default_rng(7)
    fails = 0
    normalized = glint_xtc.stream_geometry(
        "panel1z/fs = +1x +0y +z\npanel1z/ss = +0x +1y -0.25z\n", ZDIST)
    directions = {line.split("=", 1)[0].strip(): line.split("=", 1)[1].split()
                  for line in normalized.splitlines() if "=" in line}
    if (directions.get("panel1z/fs") != ["+1x", "+0y", "+0z"] or
            directions.get("panel1z/ss") != ["+0x", "+1y", "+0z"]):
        print("FAIL: geometry normalization must only flatten z direction values")
        fails += 1
    with tempfile.TemporaryDirectory() as td:
        gpath = os.path.join(td, "two.geom")
        open(gpath, "w").write(GEOM)

        out = {"qframes": [], "events": [], "peaks": [], "lams": []}
        results = []
        wavelengths = (1.3, 1.1)
        for ev, wavelength in zip((3, 11), wavelengths):
            n = 40
            seg = rng.integers(0, 2, n)
            ss = rng.integers(0, H, n)
            fs = rng.integers(0, W, n)
            q1 = pass1_q(gpath, seg, ss, fs, wavelength)
            # A basis that indexes these q to integers: M = inv(B) for a random real-space cell B
            # whose reciprocal rows we fit to... simpler and exact -- pick M, then hkl = q1 @ M is
            # whatever it is; the invariant tested is q_CF . M_written == q1 . M (+-), row for row.
            M = rng.normal(size=(3, 3)) * 30.0
            out["qframes"].append(q1)
            out["events"].append(ev)
            out["peaks"].append(xtc_core.slab_peaks(np.stack([seg, ss, fs], 1), rng.random(n) * 100, H))
            out["lams"].append(wavelength)
            results.append({"image": "xtc://x_r1", "event": ev, "M": M, "q": q1, "hkl": q1 @ M})
        export_geom = glint_xtc.stream_geometry(GEOM, ZDIST)
        recs = glint_xtc.stream_records(results, out, export_geom, ZDIST)
        spath = os.path.join(td, "x.stream")
        write_stream(recs, spath, geom_text=export_geom)
        stream_text = open(spath).read()
        chunks = read_peaks(stream_text)
        geom_path_out = os.path.join(td, "stream.geom")
        geom_text_out = serialized_geom(stream_text)
        open(geom_path_out, "w").write(geom_text_out)
        geom_keys = [(line.split("=", 1)[0].strip(), line.split("=", 1)[1].split(";")[0].strip())
                     for line in geom_text_out.splitlines() if "=" in line]
        clens = [value for key, value in geom_keys if key == "clen"]
        offsets = [float(value) for key, value in geom_keys if key.rsplit("/", 1)[-1] == "coffset"]
        if clens != [f"{ZDIST:.12g}"] or not offsets or any(offset != 0 for offset in offsets):
            print("FAIL: serialized geometry does not match pass-1 distance / zero offsets")
            fails += 1
        panels_out = list(geom_coords.parse_geom(geom_path_out).values())
        panels_in = list(geom_coords.parse_geom(gpath).values())
        for pin, pout in zip(panels_in, panels_out):
            if (not np.allclose(pout["fs"][:2], pin["fs"][:2]) or pout["fs"][2] != 0 or
                    not np.allclose(pout["ss"][:2], pin["ss"][:2]) or pout["ss"][2] != 0):
                print("FAIL: exported panel vectors must preserve x/y and flatten z")
                fails += 1
        for p in panels_out:
            p.update(cx=p["corner_x"], cy=p["corner_y"])

        for k, (ch, rec, q1, res) in enumerate(zip(chunks, recs, out["qframes"], results)):
            n = len(q1)
            if ch["num_peaks"] != n or len(ch["rows"]) != n:
                print(f"FAIL chunk {k}: num_peaks {ch['num_peaks']}, rows {len(ch['rows'])}, expected {n}")
                fails += 1
                continue
            rows = np.array(ch["rows"])
            want_panels = ["p0" if s < H else "p1" for s in rows[:, 1]]
            if ch["panels"] != want_panels:
                print(f"FAIL chunk {k}: panel names do not follow the .geom")
                fails += 1
            invd = rows[:, 2] / 10.0
            if not np.allclose(invd, np.linalg.norm(q1, axis=1), atol=2e-5):
                print(f"FAIL chunk {k}: (1/d) column is not |q|")
                fails += 1
            qcf = peaks_to_q(rows[:, 0], rows[:, 1], panels_out, ch["clen"], HC_EV_A / ch["energy"])
            S = np.diag([1.0, 1.0, geom_coords.Z_SIGN])
            if not np.allclose(qcf, q1 @ S, atol=1e-6):
                print(f"FAIL chunk {k}: peaks read back through the .geom are not S q_pass1 "
                      f"(max dev {np.nanmax(np.abs(qcf - q1 @ S)):.2e})")
                fails += 1
            Mw = np.asarray(rec["M"], float)
            hw = np.asarray(rec["hkl"], float)
            if not np.allclose(qcf @ Mw, hw, atol=1e-4):
                print(f"FAIL chunk {k}: q_CrystFEL . M_written != written hkl "
                      f"(max dev {np.abs(qcf @ Mw - hw).max():.2e})")
                fails += 1
            if np.linalg.det(Mw) < 0:
                print(f"FAIL chunk {k}: written basis is left-handed")
                fails += 1

        # An older reader (no `peaks`) still yields a valid stream with empty peak lists.
        old = {"qframes": out["qframes"], "events": out["events"]}
        write_stream(glint_xtc.stream_records(results, old, export_geom), spath, geom_text=export_geom)
        if any(c["num_peaks"] != 0 for c in read_peaks(open(spath).read())):
            print("FAIL: a reader without peaks should give num_peaks = 0")
            fails += 1

    print("PASS" if not fails else f"{fails} FAIL")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
