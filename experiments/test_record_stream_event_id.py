#!/usr/bin/env python3
"""record_stream_replay's event-id parsing on the forms CrystFEL actually writes.

indexamajig 0.10+ writes `Event: //` on every chunk of a one-image-per-file source (the cxidb-17
xg480_blind.stream has it on 480/480 chunks). The parser read that empty id as a frame index and
exited ("cannot read a frame index out of event id ' //\\n'"), so every pixel and --peaks-in replay of
that stream died at load time. CPU only, no data: tiny .stream/.lst files in a temp dir. Runs without torch
(glint.glint_fast, which only the replay itself uses, is stubbed then); the pixel reads need h5py.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

try:
    import torch  # noqa: F401
except ImportError:
    # The CPU CI job has no torch. record_stream_replay imports glint.glint_fast (torch) for the replay itself;
    # the parsers tested here never touch it, so stand in a module whose functions refuse to run.
    import types

    def _stubbed(*a, **k):
        raise RuntimeError("glint.glint_fast is stubbed in this test (no torch)")
    _gf = types.ModuleType("glint.glint_fast")
    _gf.GATE_FRAC = _gf.GATE_MIN = _gf.LYSO = None
    _gf.load = _gf.matched_strict = _stubbed
    sys.modules["glint.glint_fast"] = _gf

import record_stream_replay as R   # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def ev(tok):
    """_event_index, with its SystemExit turned into a value so one failing form does not hide the rest."""
    try:
        return R._event_index(tok)
    except SystemExit as e:
        return f"EXIT: {e}"


def entry(line):
    try:
        return R._list_entry(line)
    except SystemExit as e:
        return f"EXIT: {e}"


def load(*a, **k):
    try:
        return R.load_pixel_list(*a, **k)
    except SystemExit as e:
        return f"EXIT: {e}"


def refuses(tok):
    try:
        R._event_index(tok)
    except SystemExit:
        return True
    return False


def chunk(fname, ev_line, peaks):
    s = ["----- Begin chunk -----", f"Image filename: {fname}"]
    if ev_line is not None:
        s.append(ev_line)
    s += ["Image serial number: 1", "hit = 1", "Peaks from peak search",
          "  fs/px   ss/px (1/d)/nm^-1   Intensity  Panel"]
    s += [f"{fs:8.2f} {ss:8.2f}       1.00    {I:10.2f}   q0a0" for fs, ss, I in peaks]
    s += ["End of peak list", "----- End chunk -----"]
    return "\n".join(s) + "\n"


def main():
    # the id forms
    check("'//12' -> 12", ev("//12") == 12)
    check("'entry_1//7' -> 7", ev("entry_1//7") == 7)
    check("' //\\n' (indexamajig, one image per file) -> None", ev(" //\n") is None, repr(ev(" //\n")))
    check("'//' -> None", ev("//") is None, repr(ev("//")))
    check("'' -> None", ev("") is None, repr(ev("")))
    check("'entry_1//x' still refuses", refuses("entry_1//x"))
    check("'entry_1//' still refuses (a path part with no frame index is not the empty id)", refuses("entry_1//"))
    check("list line 'f.h5 //' -> (f.h5, None)", entry("f.h5 //") == ("f.h5", None), repr(entry("f.h5 //")))
    check("list line 'f.h5 //3' -> (f.h5, 3)", entry("f.h5 //3") == ("f.h5", 3))

    with tempfile.TemporaryDirectory() as d:
        st = os.path.join(d, "x.stream")
        with open(st, "w") as f:                       # completion order b, a (as indexamajig -j writes)
            f.write("CrystFEL stream format 2.3\n")
            f.write(chunk("run/b.h5", "Event: //", [(10, 20, 300), (11, 21, 400)]))
            f.write(chunk("run/a.h5", "Event: //", [(30, 40, 500)]))
        lst = os.path.join(d, "x.lst")
        with open(lst, "w") as f:
            f.write("run/a.h5\nrun/b.h5\n")
        pool = load(st, root=d, order=lst)
        if isinstance(pool, str):
            check("stream with `Event: //` loads", False, pool)
        else:
            check("stream with `Event: //` loads in --order order, event None",
                  pool.items == [("run/a.h5", None), ("run/b.h5", None)], str(pool.items))
            check("its peak lists follow the reorder", [len(p) for p in pool.peaks] == [1, 2],
                  str([len(p) for p in pool.peaks]))
            check("same digest as the .lst of the same files (both: no event)",
                  pool.digest() == R.load_pixel_list(lst, root=d).digest())
        st2 = os.path.join(d, "y.stream")
        with open(st2, "w") as f:
            f.write(chunk("run/s.h5", "Event: //3", [(1, 2, 3)]))
        check("a stacked file's `Event: //3` still reads 3", getattr(load(st2), "items", None) == [("run/s.h5", 3)])

        try:
            import h5py
        except ImportError:
            print("SKIP  pixel reads: no h5py")
        else:
            import numpy as np
            os.makedirs(os.path.join(d, "px"))
            with h5py.File(os.path.join(d, "px", "one.h5"), "w") as f:
                f["data/data"] = np.full((1, 4, 5), 7, np.int16)
            with h5py.File(os.path.join(d, "px", "stack.h5"), "w") as f:
                f["data/data"] = np.arange(3 * 4 * 5, dtype=np.int16).reshape(3, 4, 5)
            pool = R.PixelPool([("px/one.h5", None), ("px/stack.h5", 2), ("px/stack.h5", None)], root=d)

            def read(i):
                """pool.read, with a refusal (SystemExit) or any other error turned into a value, so one
                unexpected outcome does not end the checks after it."""
                try:
                    return pool.read(i, "data/data")
                except SystemExit as e:
                    return f"EXIT: {e}"
                except Exception as e:                                  # noqa: BLE001
                    return f"ERROR: {type(e).__name__}: {e}"
            one = read(0)
            check("an empty id reads a one-image file's only frame", not isinstance(one, str)
                  and one.shape in ((1, 4, 5), (4, 5)) and float(one.max()) == 7.0, str(getattr(one, "shape", one)))
            two = read(1)
            check("an event index reads that frame of a stack", not isinstance(two, str)
                  and two.shape == (4, 5) and float(two[0, 0]) == 40.0,
                  two if isinstance(two, str) else str(two.shape))
            msg = read(2)
            check("an empty id on a 3-frame stack refuses instead of reading the whole stack",
                  isinstance(msg, str) and msg.startswith("EXIT: "), str(msg) if isinstance(msg, str) else str(msg.shape))
            named = isinstance(msg, str) and all(w in msg for w in ("stack of events", "panel stack",
                                                                   "not supported by this replay"))
            check("the refusal names both readings: an event stack (give the index) or a panel stack (unsupported)",
                  named, "" if named else str(msg if isinstance(msg, str) else msg.shape))

    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
