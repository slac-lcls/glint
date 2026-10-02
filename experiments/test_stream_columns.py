"""Which column of a reflection row holds which value -- pinned for every GLINT stream writer.

`predict._write_chunk` is the one row formatter behind write_stream_integrated (the CLI/LUTE
`--integrate` route) and StreamWriter (StreamDriver stream_out). Nothing checked its column ORDER:
swapping I and sigma(I) in its `row % (...)` tuple passed every CI step (review s8-01), and
partialator, which reads the rows by position, would then have merged sigma as I without an error.

What is asserted, on one hand-built record with a distinct planted value in every column (a
negative I included, and two panels named per reflection):
  * GLINT's column header has the same column names, in the same order, as the one CrystFEL writes
    (CRYSTFEL_RCOL below), for both the integrated and the orientation-only writer;
  * read the way CrystFEL reads it -- by POSITION, `sscanf("%i %i %i %f %f %f %f %f %f %63s")`, the
    header line ignored -- each row gives back the planted h, k, l, I, sigma(I), peak, background,
    fs, ss and panel. A swap of any two columns fails here even if the header were swapped with it;
  * read by the NAMES in the file's own header line, the same values come back;
  * write_stream_integrated and StreamWriter write byte-identical files for the same record;
  * the orientation-only writer (glint/stream.py, I = 0 placeholders) puts h, k, l in the h, k, l
    columns and writes ten fields per row.

Mutation check done when this was written: swapping I/sigma, fs/ss, peak/background or h/k in
predict._write_chunk's row tuple each makes this script exit 1; so does swapping I and sigma(I) in
the rows AND the header together, and swapping fs/px and ss/px in glint/stream.py's header.

Run: `PYTHONPATH=. python experiments/test_stream_columns.py` (exit 0/1). numpy only.
"""
import os
import sys
import tempfile

import numpy as np

from glint.predict import StreamWriter, write_stream_integrated
from glint.predict import _RCOL as RCOL_INTEGRATED
from glint.stream import _RCOL as RCOL_ORIENTATION
from glint.stream import write_stream

# The column header CrystFEL itself writes for stream format 2.3: libcrystfel/src/stream.c,
# write_stream_reflections(), CrystFEL master (github.com/taw10/crystfel). Its reader,
# read_stream_reflections_2_3(), skips that line and parses each row by position with
# sscanf("%i %i %i %f %f %f %f %f %f %63s", &h, &k, &l, &intensity, &sigma, &pk, &bg, &fs, &ss, pname).
CRYSTFEL_RCOL = "   h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel\n"
CRYSTFEL_FIELDS = ("h", "k", "l", "I", "sigma", "peak", "bg", "fs", "ss", "panel")   # sscanf order

fails = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        fails.append(name)


# --------------------------------------------------------------------- one record, every column distinct
PANEL_NAMES = ["q0", "q1"]
PLANTED = {                                   # one entry per reflection row
    "h":     [3, -2, 0],
    "k":     [-7, 5, 1],
    "l":     [11, -13, 9],
    "I":     [1234.56, -8.25, -150.75],       # negative intensities must survive as written
    "sigma": [6.78, 3.12, 21.5],
    "peak":  [345.6, 77.7, 9.9],
    "bg":    [12.34, 4.56, 88.12],
    "fs":    [101.5, 250.7, 12.3],
    "ss":    [203.5, 40.3, 900.9],
    "panel": ["q0", "q1", "q1"],
}
# what each column is printed with in _write_chunk: %10.2f for I/sigma/bg, %6.1f for peak/fs/ss
TOL = {"I": 0.006, "sigma": 0.006, "bg": 0.006, "peak": 0.051, "fs": 0.051, "ss": 0.051}

pred = np.zeros(3, dtype=[("h", int), ("k", int), ("l", int), ("fs", float), ("ss", float),
                          ("panel", int), ("exc", float), ("res", float)])
for c in ("h", "k", "l", "fs", "ss"):
    pred[c] = PLANTED[c]
pred["panel"] = [PANEL_NAMES.index(p) for p in PLANTED["panel"]]
pred["res"] = [3.0, 2.5, 4.0]
REC = dict(image="cols.cxi", event=4, M=np.diag([79.0, 79.0, 38.0]), pred=pred,
           I=np.array(PLANTED["I"]), sigma=np.array(PLANTED["sigma"]),
           peak=np.array(PLANTED["peak"]), bg=np.array(PLANTED["bg"]))


def reflection_block(txt):
    """(header line, [row lines]) of the single crystal's reflection list."""
    body = txt.split("Reflections measured after indexing\n", 1)[1].split("End of reflections", 1)[0]
    lines = body.split("\n")
    return lines[0] + "\n", [l for l in lines[1:] if l.strip()]


def crystfel_parse(row):
    """A row read the way CrystFEL reads it: by position, ten fields, the header ignored."""
    t = row.split()
    if len(t) != 10:
        return None
    return dict(zip(CRYSTFEL_FIELDS, [int(x) for x in t[:3]] + [float(x) for x in t[3:9]] + [t[9]]))


def by_header(header, row):
    """A row read by the column NAMES of the file's own header line."""
    name = {"h": "h", "k": "k", "l": "l", "I": "I", "sigma(I)": "sigma", "peak": "peak",
            "background": "bg", "fs/px": "fs", "ss/px": "ss", "panel": "panel"}
    cols = [name.get(c, c) for c in header.split()]
    return dict(zip(cols, row.split()))


def matches(rowvals, j):
    """Does a parsed row carry reflection j's planted value in every named column?"""
    bad = []
    for c, vals in PLANTED.items():
        got, want = rowvals.get(c), vals[j]
        if got is None:
            bad.append(f"{c} missing")
        elif c in ("h", "k", "l"):
            if int(got) != want:
                bad.append(f"{c}={got} want {want}")
        elif c == "panel":
            if got != want:
                bad.append(f"{c}={got} want {want}")
        elif abs(float(got) - want) > TOL[c]:
            bad.append(f"{c}={got} want {want}")
    return bad


print("column header vs CrystFEL's")
check("integrated writer: same column names, same order as CrystFEL",
      RCOL_INTEGRATED.split() == CRYSTFEL_RCOL.split(), RCOL_INTEGRATED.strip())
check("orientation-only writer: same column names, same order as CrystFEL",
      RCOL_ORIENTATION.split() == CRYSTFEL_RCOL.split(), RCOL_ORIENTATION.strip())

with tempfile.TemporaryDirectory() as d:
    p_batch, p_live = os.path.join(d, "batch.stream"), os.path.join(d, "live.stream")
    n = write_stream_integrated([REC], p_batch, panel_names=PANEL_NAMES)
    with StreamWriter(p_live, panel_names=PANEL_NAMES) as w:
        w.write(REC)
    batch, live = open(p_batch).read(), open(p_live).read()
    check("one indexed crystal written", n == 1 and batch.count("--- Begin crystal") == 1, n)
    check("write_stream_integrated and StreamWriter are byte-identical", batch == live)

    for tag, txt in (("write_stream_integrated", batch), ("StreamWriter", live)):
        print(f"\n{tag}: reflection rows")
        header, rows = reflection_block(txt)
        check(f"{tag}: header line in the file is the writer's", header == RCOL_INTEGRATED, header)
        check(f"{tag}: one row per reflection", len(rows) == 3, rows)
        for j, row in enumerate(rows[:3]):
            cf = crystfel_parse(row)
            check(f"{tag}: row {j} read by position (CrystFEL) gives the planted values",
                  cf is not None and not matches(cf, j),
                  row.strip() if cf is None else f"{matches(cf, j)} in {row.strip()!r}")
            check(f"{tag}: row {j} read by header name gives the planted values",
                  not matches(by_header(header, row), j),
                  f"{matches(by_header(header, row), j)} in {row.strip()!r}")

    print("\nno panel_names: every row takes the scalar panel name")
    p_scalar = os.path.join(d, "scalar.stream")
    write_stream_integrated([REC], p_scalar, panel_name="pX")
    _, rows = reflection_block(open(p_scalar).read())
    cf = [crystfel_parse(r) for r in rows]
    check("panel column holds the scalar name; I and sigma(I) unchanged",
          len(cf) == 3 and all(c is not None and c["panel"] == "pX" for c in cf)
          and [c["I"] for c in cf] == PLANTED["I"] and [c["sigma"] for c in cf] == PLANTED["sigma"],
          rows)

    print("\norientation-only writer (glint/stream.py)")
    p_or = os.path.join(d, "orient.stream")
    hkl = np.array([PLANTED["h"], PLANTED["k"], PLANTED["l"]]).T
    write_stream([dict(image="o.cxi", event=0, M=np.diag([79.0, 79.0, 38.0]), hkl=hkl)], p_or)
    header, rows = reflection_block(open(p_or).read())
    cf = [crystfel_parse(r) for r in rows]
    check("header line is the writer's", header == RCOL_ORIENTATION, header)
    check("ten fields per row, h k l in the h k l columns",
          len(cf) == 3 and all(c is not None for c in cf)
          and [(c["h"], c["k"], c["l"]) for c in cf] == [tuple(x) for x in hkl.tolist()], rows)

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + str(len(fails)) + '  ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
