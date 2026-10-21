"""Minimal CrystFEL .stream writer for GLINT output -> drops into LUTE / CrystFEL
downstream (cell + per-frame orientation; partialator/CrystFEL predict+integrate from the
astar/bstar/cstar). GLINT indexes (assigns hkl to observed peaks); integration is downstream,
so reflection rows carry h k l and intensity placeholders (fs/ss need detector geometry).

  from glint.stream import write_stream
  write_stream([{ "image": "run.cxi", "event": 0, "M": M, "q": q, "hkl": hkl }, ...], "out.stream",
               geom_text=open("det.geom").read())

THE OUTPUT IS ORIENTATION-ONLY AND STILL NOT MERGEABLE -- every I/sigma(I) below is 0.00, so a
merge of it merges zeros. Use `predict.write_stream_integrated` (GLINT's `--integrate`) for real
intensities, or hand the orientations to `indexamajig --indexing=file`. What changed is that
CrystFEL can now *open* this file: it previously could not, for three separate reasons, each of
which failed with an error naming none of them (measured against CrystFEL 0.10.2):

  no geometry block      the reader scans for `----- Begin geometry file -----` and gives up with
                         "Too much audit information." having consumed the entire stream
  `Begin crystal`        CrystFEL's marker is `--- Begin crystal`; without the dashes no crystal
                         is ever found, and the reader instead trips on the line before it
  `indexed_by = glint`   validated against CrystFEL's indexer list -> "Bad list of indexing
                         methods: 'glint'". `file` is the honest value: the orientation is
                         externally supplied as far as CrystFEL is concerned.

So a stream that reported 2170 crystals merged to nothing, silently, on all three counts.
"""
import re

import numpy as np

from glint.geom import is_bad_region

_RCOL = "   h    k    l          I   sigma(I)   peak  background  fs/px  ss/px panel\n"

# Enough of a detector for the reader to accept the stream when the caller has no .geom -- the xtc
# route can run off psana's own pixel coordinates, which are positions and not a CrystFEL tiling.
_FALLBACK_GEOM = ("clen = 0.1\nres = 5000\nadu_per_photon = 1\nphoton_energy = 9400\n"
                  "p0/min_fs = 0\np0/max_fs = 1023\np0/min_ss = 0\np0/max_ss = 1023\n"
                  "p0/corner_x = -512.5\np0/corner_y = -512.5\np0/fs = x\np0/ss = y\n")

# A registration whose basis spans less than this volume (A^3) is degenerate, not a crystal. The stream writers
# write such a frame as unindexed; the solution-file writer, which inverts M, used to stop on the first singular
# one and lose the whole file.
MIN_CELL_VOLUME_A3 = 1.0

# 1 A^3 is five orders of magnitude below a protein cell, so it only catches the singular case. A basis whose three axes
# nearly lie in a plane is not a plausible cell at any size: |det M| / (|a||b||c|) is the cell's volume as a fraction of
# the box with the same edges, 1 for a rectangular cell, 0.87 for a hexagonal one and at least 0.71 for a reduced cell
# (the three shortest independent lattice vectors; Minkowski's bound). On a sparse run where the consensus declined to
# establish a cell, 9 of the 2,408 cells GLINT wrote had 27-123 A axes and 3-64 A^3 volumes (fractions below 2e-4);
# every cell CrystFEL returned in the same comparisons was at 0.52 or above. A conventional setting falls below 0.1 only
# at a monoclinic beta above 174 degrees or a rhombohedral alpha below 20 degrees (hexagonal c/a above 8.6).
MIN_CELL_VOLUME_FRACTION = 0.1

# How the run summary and the solution-file writer name the frames ``has_cell`` turns down.
NO_CELL = (f"non-finite, |det M| < {MIN_CELL_VOLUME_A3:g} A^3, or |det M| < {MIN_CELL_VOLUME_FRACTION:g} "
           f"|a||b||c|")


def has_cell(M):
    """True when ``M`` (columns a, b, c in A) is a usable basis: finite, |det M| >= MIN_CELL_VOLUME_A3, and
    |det M| >= MIN_CELL_VOLUME_FRACTION * |a||b||c|. The one rule shared by ``write_stream``,
    ``predict._write_chunk`` and ``predict.write_solution_file``, so the stream and the solution file hold the
    same frames.

    The singularity test comes first. With a zero-length axis both sides of the scaled test are 0, and with a
    non-finite entry both are NaN, so the scaled test alone cannot decide either case, and a basis it let through
    would raise in the inversion downstream."""
    if M is None:
        return False
    M = np.asarray(M, float)
    if not np.all(np.isfinite(M)):
        return False
    v = abs(np.linalg.det(M))
    return bool(v >= MIN_CELL_VOLUME_A3 and v >= MIN_CELL_VOLUME_FRACTION * np.prod(np.linalg.norm(M, axis=0)))


def written_note(stats):
    """The run-summary line for a stream that holds a different number of crystals (``stats["n_written"]``,
    from the writer) than ``stats["n_idx"]`` counts, or None. A registration with no usable cell is counted as
    indexed but written as unindexed, and nothing else in the summary says so."""
    w = stats.get("n_written")
    if w is None or w == stats["n_idx"]:
        return None
    return (f"  written to stream  : {w} crystals, not {stats['n_idx']} -- frames with no usable cell "
            f"({NO_CELL}) are written as unindexed")


def _header(geom_text, generator="GLINT (fftindex)"):
    """CrystFEL stream header. The reader needs a geometry-file block right after the format line
    (else it consumes the whole file as audit info and aborts with 'Too much audit information')."""
    g = geom_text if geom_text is not None else _FALLBACK_GEOM
    return (f"CrystFEL stream format 2.3\nGenerated by {generator}\n"
            "----- Begin geometry file -----\n" + g.rstrip("\n") + "\n----- End geometry file -----\n")


_PANEL_KEY = re.compile(r"^\s*([\w.]+)/(min_fs|max_fs|min_ss|max_ss)\s*=\s*([-\d]+)", re.M)


def panel_bounds(geom_text):
    """[(name, min_fs, max_fs, min_ss, max_ss), ...] from a .geom's text, or [] if there is none."""
    b = {}
    for name, key, val in _PANEL_KEY.findall(geom_text or ""):
        if is_bad_region(name):                 # `badregionA/min_fs ...` bounds a bad region, not a panel
            continue
        b.setdefault(name, {})[key] = int(val)
    return [(n, d["min_fs"], d["max_fs"], d["min_ss"], d["max_ss"])
            for n, d in b.items() if len(d) == 4]


def panel_at(bounds, fs, ss, default="p0"):
    """Name of the panel a data-array (fs, ss) lands in.

    OBSERVED peaks carry real positions, so labelling them with a fixed 'p0' is not merely a name
    CrystFEL rejects -- it is the wrong panel. Linear over panels because a frame has tens of peaks
    and a detector has tens of panels."""
    for name, f0, f1, s0, s1 in bounds:
        if f0 <= fs <= f1 and s0 <= ss <= s1:
            return name
    return default


def _origin_panel(geom_text, default="p0"):
    """Name of the panel containing data-array (0, 0), or `default`.

    The reflection rows below carry fs = ss = 0.00 because this writer has no detector positions,
    so the panel that covers the origin is the one consistent with them. It is NOT cosmetic:
    CrystFEL resolves the column by NAME and warns 'No such panel' once per row otherwise, and a
    reader that trusts the name would place every reflection on a panel that does not exist."""
    if not geom_text:
        return default
    bounds = {}
    for name, key, val in _PANEL_KEY.findall(geom_text):
        if is_bad_region(name):                 # a bad region is never the panel a row belongs to
            continue
        bounds.setdefault(name, {})[key] = int(val)
    for name, b in bounds.items():
        if len(b) == 4 and b["min_fs"] <= 0 <= b["max_fs"] and b["min_ss"] <= 0 <= b["max_ss"]:
            return name
    return next(iter(bounds), default)


def _cell_block(M):
    M = np.asarray(M, float)
    a, b, c = (np.linalg.norm(M[:, i]) for i in range(3))            # real axes, A
    def ang(i, j):
        u, v = M[:, i], M[:, j]
        return np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
    Minv = np.linalg.inv(M)                                          # rows = a*,b*,c* in A^-1
    out = ["--- Begin crystal",
           f"Cell parameters {a/10:.5f} {b/10:.5f} {c/10:.5f} nm, "
           f"{ang(1,2):.3f} {ang(0,2):.3f} {ang(0,1):.3f} deg"]
    for name, v in zip(("astar", "bstar", "cstar"), Minv):
        out.append(f"{name} = {v[0]*10:+.7f} {v[1]*10:+.7f} {v[2]*10:+.7f} nm^-1")
    out += ["lattice_type = triclinic", "centering = P", "unique_axis = *"]
    return "\n".join(out)


def escalation_lines(r):
    """`glint/` provenance lines for a frame hybrid_index(escalate=...) accepted through the deep search and
    its scrambled null (retry_cascade.arm_known_deep), so a merger reading the stream can tell escalated
    orientations apart: the flag, the null's p = 1/(k_null + 1), the fit's matched peaks and the best
    scrambled copy's. Empty for every other frame. Both stream writers call it inside the crystal block;
    CrystFEL skips unrecognised `key = value` lines (see predict._write_chunk)."""
    rec = r.get("escalation")
    if not r.get("escalated") or not rec:
        return []
    null = rec.get("null_m") or [0]
    return ["glint/escalated = 1",
            f"glint/escalation_p = {float(rec['p']):.6g}",
            f"glint/escalation_matched = {int(rec['m'])}",
            f"glint/escalation_null_max = {int(max(null))}"]


def write_stream(results, path, geom_text=None, panel=None, photon_eV=9392.7, clen_m=0.15):
    """results: list of dicts {image, event, M (3x3 or None), q (N,3), hkl (N,3 int) or None}, and
    optionally `peaks` (n,3) [fs, ss, intensity] in the .geom's slab coordinates with `peaks_invd`
    (n,) |q| in 1/A: the OBSERVED peaks the frame was indexed from, written as the chunk's peak list.

    geom_text: contents of the .geom this run was indexed against. Pass it whenever the caller has
    one -- it is what makes the stream readable at all, and it names the panels."""
    panel = panel or _origin_panel(geom_text)
    bounds = panel_bounds(geom_text)
    n_idx = 0
    with open(path, "w") as f:
        f.write(_header(geom_text))
        for serial, r in enumerate(results, 1):
            f.write("----- Begin chunk -----\n")
            f.write(f"Image filename: {r.get('image', 'glint.cxi')}\n")
            f.write(f"Event: //{r.get('event', 0)}\n")
            f.write(f"Image serial number: {serial}\n")
            f.write("hit = 1\n")
            M = r.get("M")
            valid = has_cell(M)
            # 'file', not 'glint': CrystFEL validates this against its own indexer list, and from
            # its side the orientation IS externally supplied.
            f.write(f"indexed_by = {'file' if valid else 'none'}\n")
            f.write(f"photon_energy_eV = {float(photon_eV):.2f}\n")
            f.write("beam_divergence = 0.00e+00 rad\nbeam_bandwidth = 1.00e-08 %\n")
            f.write(f"average_camera_length = {float(r.get('clen_m', clen_m)):.6f} m\n")
            # A chunk needs a peak-list block even when empty, or the reader reports the chunk
            # incomplete. The peaks, when the caller has them, are what a strict score needs: the
            # reflection rows below are only predicted labels.
            pks = r.get("peaks")
            npk = 0 if pks is None else len(pks)
            f.write(f"num_peaks = {npk}\nnum_saturated_peaks = 0\n")
            f.write("Peaks from peak search\n  fs/px   ss/px (1/d)/nm^-1   Intensity  Panel\n")
            if npk:
                pks = np.asarray(pks, float)
                invd = r.get("peaks_invd")
                invd = np.zeros(npk) if invd is None else np.asarray(invd, float) * 10.0   # 1/A -> 1/nm
                for (pfs, pss, pI), d in zip(pks[:, :3].tolist(), invd.tolist()):
                    f.write("%7.2f %7.2f %10.5f %10.2f %s\n"
                            % (pfs, pss, d, pI, panel_at(bounds, pfs, pss, panel) if bounds else panel))
            f.write("End of peak list\n")
            if valid:
                n_idx += 1
                f.write(_cell_block(M) + "\n")
                f.writelines(line + "\n" for line in escalation_lines(r))
                f.write("Reflections measured after indexing\n")
                f.write(_RCOL)
                hkl = r.get("hkl")
                if hkl is not None:
                    for h, k, l in np.asarray(hkl, int):
                        f.write(f"{h:4d} {k:4d} {l:4d}       0.00      0.00   0.00       0.00"
                                f"    0.0    0.0 {panel}\n")
                f.write("End of reflections\n")
                f.write("--- End crystal\n")
            f.write("----- End chunk -----\n")
    return n_idx
