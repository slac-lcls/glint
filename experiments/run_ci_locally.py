"""Run the CPU CI job's test steps locally, in an environment as poor as CI's.

WHY THIS EXISTS. glint#122 added a test that passed on the author's Mac and failed CI with
`ModuleNotFoundError: No module named 'torch'`. The CPU job installs numpy, scipy, pytest and
pydantic and nothing else, while a development machine has torch (and often cupy and numba), so
"it passes locally" says nothing about a suite whose whole point is to run where those are absent.
Optional dependencies are guarded by try/except all over glint, which means importing them
SUCCEEDS locally and the guard is never taken -- the divergence is invisible until CI is red.

WHAT IT DOES. Blocks torch, cupy and numba at the import system, then runs each CI test step in a
subprocess and reports its exit code. ModuleNotFoundError subclasses ImportError, so code that
legitimately guards these imports still takes its fallback path -- only code that ASSUMES them
fails, which is the thing worth finding.

    python experiments/run_ci_locally.py             # run every CI step
    python experiments/run_ci_locally.py -v          # ...and show their output
    python experiments/run_ci_locally.py --check-sync  # only verify STEPS matches ci.yml

Keep STEPS in sync with .github/workflows/ci.yml; --check-sync verifies it and is the reason a
step added to CI but not here does not silently go unchecked.
"""
import re
import subprocess
import sys
from pathlib import Path

BLOCKED = ("torch", "cupy", "numba")
ROOT = Path(__file__).resolve().parent.parent
CI = ROOT / ".github" / "workflows" / "ci.yml"

# The `experiments/` steps of the CPU job, plus the one experiments/ step of the torch-CPU job
# (test_nbest_prefix.py, which self-skips here because torch is blocked -- the skip path is what
# this harness covers). The xtc_bridge ones run from their own directory and are left to CI; lute/
# needs pytest+pydantic and is likewise CI's business.
STEPS = [
    "experiments/test_alias_gate.py",
    "experiments/test_pf8_thr_adu.py",
    "experiments/test_asic_seam_mask.py",
    "experiments/test_negative_intensities.py",
    "experiments/test_integrate_event.py",
    "experiments/test_device_selection.py",
    "experiments/test_panel_stack_integrate.py",
    "experiments/test_integrate_cxi_layout.py",
    "experiments/test_geom_data_key.py",
    "experiments/test_consensus_degenerate.py",
    "experiments/test_consensus_order.py",
    "experiments/test_stream_gate_lock.py",
    "experiments/test_lock_probe.py",
    "experiments/test_running_consensus.py",
    "experiments/test_lock_gate_wiring.py",
    "experiments/test_laue_ops.py",
    "experiments/test_streamdriver_laue.py",
    "experiments/test_multilattice.py",
    "experiments/test_warmup_batch.py",
    "experiments/test_missbuf_rescue.py",
    "experiments/test_watchdog_fanout_guard.py",
    "experiments/test_retry_cascade.py",
    "experiments/test_cell_registry.py",
    "experiments/test_geom_bridge.py",
    "experiments/test_cli_smoke.py",
    "experiments/test_geom_refine.py",
    # Skips without a GPU and exits 0 (glint#123). Running it here is still worth the second it
    # costs: the module body resolves HKLGrid/_panel_geom/recip_from_M/cell_to_Ar at import, so a
    # rename in glint/ fails here rather than waiting for someone with a GPU.
    "experiments/check_numbers.py",
    "experiments/test_check_numbers_ci.py",
    "experiments/test_fp32_b32_033_rule.py",
    "experiments/test_gate_constants_dedup.py",
    "experiments/test_consensus_gate_constants.py",
    "experiments/test_seqstop_replay.py",
    "experiments/test_consensus_voter_set.py",
    "experiments/test_same_lattice_symmetry.py",
    "experiments/test_gate_project.py",
    "experiments/test_axis_standardizer.py",
    "experiments/bench_integrate_fused.py",
    # torch-CPU job: runs for real there (CPU torch), SKIPs here with torch blocked.
    "experiments/test_nbest_prefix.py",
]

# Runs in the child via `python -c`, with the target script passed as argv[1] -- embedding the
# path in the source instead would mean escaping it through two layers of %-formatting, which is
# how the first version of this file broke.
_HOOK = """
import sys, importlib.abc, runpy
BLOCKED = {blocked!r}


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in BLOCKED:
            raise ModuleNotFoundError("No module named " + repr(root), name=name)
        return None


sys.meta_path.insert(0, _Block())
for _m in [m for m in sys.modules if m.split(".")[0] in BLOCKED]:
    del sys.modules[_m]
_target = sys.argv[1]
sys.argv = [_target]        # a script that reads argv[1:] would otherwise see its own path as an
                            # argument -- check_numbers.py took it as a TARGET and checked itself,
                            # reporting 64 problems that were its own rule patterns
runpy.run_path(_target, run_name="__main__")
""".format(blocked=BLOCKED)


SELF = "experiments/" + Path(__file__).name


def ci_steps():
    """The experiments/*.py TEST steps the CI workflow runs, read out of the workflow itself.

    This file's own --check-sync step is excluded: it is a consistency check on STEPS, not a suite
    STEPS should contain, and including it would make the sync check permanently fail against
    itself."""
    if not CI.exists():
        return None
    return [s for s in re.findall(r"python (experiments/\S+\.py)", CI.read_text()) if s != SELF]


def check_sync():
    found = ci_steps()
    if found is None:
        print("!! .github/workflows/ci.yml not found"); return False
    missing = [s for s in found if s not in STEPS]
    extra = [s for s in STEPS if s not in found]
    for s in missing:
        print(f"!! in ci.yml but not in STEPS (it would go unchecked here): {s}")
    for s in extra:
        print(f"!! in STEPS but no longer in ci.yml: {s}")
    if not missing and not extra:
        print(f"STEPS matches ci.yml ({len(found)} steps)")
    return not (missing or extra)


def main(argv):
    verbose = "-v" in argv
    ok = check_sync()
    if "--check-sync" in argv:            # CI runs only this half: it has no reason to re-run the
        return 0 if ok else 2             # suites it is already running as its own steps
    if not ok:
        return 2
    print(f"blocking {', '.join(BLOCKED)}\n")
    bad = []
    for step in STEPS:
        r = subprocess.run([sys.executable, "-c", _HOOK, str(ROOT / step)],
                           cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
                           capture_output=True, text=True)
        tail = [l for l in r.stdout.strip().splitlines() if l.strip()]
        print(f"  {'ok  ' if r.returncode == 0 else 'FAIL'} {step:44s} exit {r.returncode}"
              f"   {tail[-1] if tail else ''}")
        if verbose or r.returncode:
            for line in (r.stdout + r.stderr).splitlines():
                print(f"        {line}")
        if r.returncode:
            bad.append(step)
    print()
    print("ALL PASS" if not bad else f"{len(bad)} FAILED: {', '.join(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
