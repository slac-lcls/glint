#!/usr/bin/env python3
"""The withdrawn "peakfinder_v4 agrees with psalgos peaks_droplet" claim stays withdrawn, everywhere in the repo.

glint/peakfinder_v4.py's docstring said the finder "agrees with ``psalgos.pypsalgos.peaks_droplet`` on peaks",
and peakfinder9.py's said the dual-threshold psalgos finder "is matched by peakfinder_v4.py". That rested on one
comparison with the psana1-era psalgos package (one synthetic frame, different thresholds, one direction), and it
cannot be repeated: psana2 ships ``peak_finder_v3r3_d2`` but not ``peak_finder_v4r3_d2``. So the claim is
withdrawn, and the docstrings say it is not verified. (pf9's sibling ``peak_finder_v3r3_d2`` IS in psana2, and a
true statement that peakfinder9 agrees with it must not trip this test.)

This scans every tracked text file for the withdrawn wording, whitespace and reST backticks normalised so a
re-wrapped docstring cannot slip past, and checks that peakfinder_v4.py still says it is not verified. It
first shows that its patterns match the sentences as they stood at d21de4c, so a pass means the text is gone
rather than the patterns being blind. Quoting the withdrawn sentence, even to retract it, trips the scan (this
test's own CI comment did, in draft): paraphrase it. Only this file may quote it.

    PYTHONPATH=. python experiments/test_pfv4_psalgos_claim.py
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SELF = os.path.relpath(os.path.abspath(__file__), ROOT)
TEXT_EXT = {".py", ".md", ".txt", ".rst", ".tex", ".yml", ".yaml", ".cff", ".toml", ".cfg", ".sh", ".ipynb"}

# The withdrawn claim, in each shape it has been or is likely to be written: an agreement verb before the v4r3
# oracle's name (in one sentence), the reverse order, and d21de4c's own phrase. Matched against normalised text
# (see _norm). "agreement" alone is not a verb here, so "agreement with peaks_droplet is NOT verified" is free.
_ORACLE = r"(?:peaks_droplet|peak_finder_v4r3|v4r3)"
_SAME_SENTENCE = r"(?:[^.]|\.(?=\S)){0,80}?"     # no sentence end; a dotted name (psalgos.pypsalgos.x) is fine
WITHDRAWN = [
    ("agrees with / matches / reproduces / validated ... the v4r3 oracle",
     re.compile(r"\b(?:agrees?|agreed|match(?:es|ed)?|reproduc(?:es?|ed)|validated)\b" + _SAME_SENTENCE + _ORACLE, re.I)),
    ("the v4r3 oracle ... is matched by / agrees with peakfinder_v4",
     re.compile(_ORACLE + _SAME_SENTENCE +
                r"\b(?:(?:matched|reproduced)\s+by|agree(?:s|d)?\s+with)\s+peakfinder_v4", re.I)),
    ("agrees with peaks_droplet on peaks", re.compile(r"peaks_droplet\s+on\s+peaks", re.I)),
]

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


def _norm(text):
    """Drop reST literal backticks and collapse every run of whitespace (line wraps included) to one space."""
    return re.sub(r"\s+", " ", text.replace("`", ""))


def _hits(text):
    t = _norm(text)
    return [name for name, rx in WITHDRAWN if rx.search(t)]


def _tracked_files():
    """Tracked files from git; a plain walk if this is not a git checkout (an sdist, a tarball)."""
    try:
        out = subprocess.run(["git", "-C", ROOT, "ls-files", "-z"], capture_output=True, check=True).stdout
        files = [f for f in out.decode("utf-8", "replace").split("\0") if f]
        if files:
            return files, "git ls-files"
    except (OSError, subprocess.CalledProcessError):
        pass
    files = []
    for d, sub, names in os.walk(ROOT):
        sub[:] = [s for s in sub if s not in (".git", "__pycache__")]
        files += [os.path.relpath(os.path.join(d, n), ROOT) for n in names]
    return files, "os.walk"


print("the patterns see the claim as it stood at d21de4c")
# Both sentences verbatim from d21de4c (glint/peakfinder_v4.py:17-20, glint/peakfinder9.py:31-32), line breaks
# and all. This file is excluded from the scan below, which is what lets it quote them.
OLD_V4 = ("NOTE this is *modelled on* the Dubrovin/psalgos DUAL-THRESHOLD finder ``peaks_droplet`` = "
          "``peak_finder_v4r3``\n(a GPU re-implementation of the idea), not a port of the psalgos C++; it agrees "
          "with\n``psalgos.pypsalgos.peaks_droplet`` on peaks but is not bit-identical.")
OLD_P9 = ("The psalgos DUAL-THRESHOLD finder\n``peaks_droplet`` = ``peak_finder_v4r3`` is matched by "
          "peakfinder_v4.py instead.")
check("d21de4c peakfinder_v4.py sentence is caught", len(_hits(OLD_V4)) >= 1, _hits(OLD_V4))
check("d21de4c peakfinder9.py sentence is caught", _hits(OLD_P9) != [], _hits(OLD_P9))
# shapes a re-introduction is likely to take (the 2026-10-11 checker's D1, D2) are caught too ...
for shape in ("validated: it agrees with `peaks_droplet` (27/27 of our peaks matched by psana)",
              "PeakFinderV4 reproduces psalgos peaks_droplet's peak list",
              "it matches psalgos.pypsalgos.peaks_droplet (not bit-identical)",
              "the dual-threshold peak_finder_v4r3 is matched by peakfinder_v4.py"):
    check(f"caught: {shape[:60]}", _hits(shape) != [], _hits(shape))
# ... and true or retracting sentences are not
CORRECT = ("peakfinder9 agrees with it on peak positions but isn't bit-identical. The psalgos DUAL-THRESHOLD "
           "finder ``peaks_droplet`` = ``peak_finder_v4r3`` is the model for peakfinder_v4.py, whose agreement with "
           "it is NOT verified. Until 2026-10 this docstring said the two agree on peaks. Agreement with "
           "``peaks_droplet`` is therefore NOT verified.")
check("the corrected wording is not caught", _hits(CORRECT) == [], _hits(CORRECT))
PF9_TRUE = "peakfinder9 agrees with psalgos `peak_finder_v3r3_d2` on peak positions."
check("a true pf9 / v3r3 sentence is not caught (checker's D3)", _hits(PF9_TRUE) == [], _hits(PF9_TRUE))

print("\nno tracked file carries the withdrawn claim")
files, how = _tracked_files()
scanned, bad = [], []
for rel in files:
    if rel == SELF or os.path.splitext(rel)[1].lower() not in TEXT_EXT:
        continue
    path = os.path.join(ROOT, rel)
    if not os.path.isfile(path):
        continue
    with open(path, encoding="utf-8", errors="replace") as fh:
        found = _hits(fh.read())
    scanned.append(rel)
    if found:
        bad.append(f"{rel}: {found}")
must = {os.path.join("glint", "peakfinder_v4.py"), os.path.join("glint", "peakfinder9.py"),
        "THIRD_PARTY_NOTICES.md", "README.md"}
check(f"the scan reached the finder modules and the notices ({len(scanned)} files via {how})",
      must <= set(scanned), sorted(must - set(scanned)))
check("no tracked text file says peakfinder_v4 agrees with / is matched to psalgos", not bad, bad)

print("\npeakfinder_v4.py says what is and is not verified")
with open(os.path.join(ROOT, "glint", "peakfinder_v4.py"), encoding="utf-8") as fh:
    head = _norm(fh.read().split('"""')[1])
check("the module docstring says agreement with peaks_droplet is NOT verified",
      re.search(r"Agreement with peaks_droplet is therefore NOT verified", head) is not None)
check("...names the missing psana2 method", "peak_finder_v4r3_d2" in head)
check("...and says what the old claim rested on (one psana1-era comparison)",
      re.search(r"one comparison with the psana1-era psalgos package", head) is not None)

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
