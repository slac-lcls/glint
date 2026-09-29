#!/usr/bin/env python3
"""docs/stream_driver_options.md names every StreamDriver constructor option, and nothing else.

The page is the option reference a new collaborator reads instead of the constructor. docs/onboarding.md
drifted for a month while the constructor grew from the Laue merge to the effort policy, so this pins the
page to the signature: every keyword of StreamDriver.__init__ and every setting of _EffortPolicy.KEYS is the
first cell of exactly one table row, no table row names an option that does not exist, and every recorder
flag the tables cite is one experiments/record_stream_replay.py defines. Text against the signature; nothing
is constructed, so it runs on the CPU job.

    PYTHONPATH=. python experiments/test_stream_driver_options_doc.py
"""
import inspect
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from glint.stream_driver import StreamDriver, _EffortPolicy      # noqa: E402

DOC = os.path.join(ROOT, "docs", "stream_driver_options.md")
RECORDER = os.path.join(HERE, "record_stream_replay.py")
ROW = re.compile(r"^\|\s*`([A-Za-z_][A-Za-z0-9_]*)`\s*\|")          # a table row whose first cell is one identifier
FLAG = re.compile(r"`(--[A-Za-z][A-Za-z0-9-]*)")                     # a cited command-line flag, inside backticks


def _rows():
    with open(DOC, encoding="utf-8") as fh:
        return [l for l in fh.read().splitlines() if ROW.match(l)]


def test_every_option_is_documented_once_and_nothing_else():
    named = [ROW.match(l).group(1) for l in _rows()]
    expected = [p for p in inspect.signature(StreamDriver.__init__).parameters if p != "self"]
    expected += list(_EffortPolicy.KEYS)
    dup = sorted({n for n in named if named.count(n) > 1})
    assert not dup, f"documented more than once: {dup}"
    missing = [p for p in expected if p not in named]
    extra = [n for n in named if n not in expected]
    assert not missing and not extra, (f"missing from the page: {missing}; "
                                       f"on the page but not in the code: {extra}")
    assert len(named) == len(expected) and len(named) > 60, len(named)


def test_cited_recorder_flags_exist():
    with open(RECORDER, encoding="utf-8") as fh:
        flags = set(re.findall(r"add_argument\(\s*\"(--[A-Za-z][A-Za-z0-9-]*)\"", fh.read()))
    assert "--effort" in flags and "--B" in flags, sorted(flags)      # the parser was read, not an empty match
    cited = sorted({f for l in _rows() for f in FLAG.findall(l)})
    unknown = [f for f in cited if f not in flags]
    assert cited and not unknown, f"flags cited that the recorder does not define: {unknown}"


TESTS = [test_every_option_is_documented_once_and_nothing_else,
         test_cited_recorder_flags_exist]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as exc:                                        # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(exc).__name__}({exc})")
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
