#!/bin/bash
# Interactive checks on sdfiana027 (no Slurm; seconds of CPU): environment, inputs, the subset and null writers,
# one ~25-frame merge with each merger and one null merge, and whether the crystal order sets the half-split.
set -uo pipefail
source "${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/r51_merge_vs_frames/config.sh"
glint_env || exit 1
DR=$OUT/dryrun; rm -rf "$DR"; mkdir -p "$DR"; export CMDLOG=$DR/commands.log
STREAM=${A_INPUTS#*:}
echo "== env: $(python -c 'import numpy, sys; print(sys.version.split()[0], "numpy", numpy.__version__)')  GLINT $(cat $GLINT_TREE/COMMIT)"
echo "== CrystFEL: $($CFBIN/partialator --version | head -1)"
for f in "$STREAM" "$A_LUTE_HKL" "${A_LUTE_HKL}1" "${A_LUTE_HKL}2" "$CELLFILE"; do
    [ -s "$f" ] && echo "ok  $f" || echo "MISSING $f"
done
md5sum "$CELLFILE"
for IN in $B_INPUTS; do f=${IN#*:}; [ -s "$f" ] && echo "stage B input present: $f" || echo "stage B input not yet made: $f"; done
echo "== frames"; time python -B "$TOOLS_HOME/sweep_tools.py" frames "$STREAM"

echo "== subset writers (10%, seed 1): real and both nulls"
for P in none all shell; do python -B "$TOOLS_HOME/sweep_tools.py" subset "$STREAM" 0.10 1 $P "$DR/s1_$P.stream"; done
python -B - "$DR" <<'EOF'
import os, sys, collections
sys.path.insert(0, os.path.join(os.environ["VERIFY_DIR"], "r51_merge_vs_frames"))
from sweep_tools import read_stream
d = sys.argv[1]
def refl(path):
    out = []
    for _, lines, _ in read_stream(path)[1]:
        on = False
        for ln in lines:
            if ln.startswith("Reflections measured after indexing"): on = 1; continue
            if on == 1: on = 2; continue
            if ln.startswith("End of reflections"): on = False; continue
            if on == 2:
                p = ln.split(); out.append((tuple(p[:3]), p[3], p[4], tuple(p[5:])))
    return out
r = {p: refl(f"{d}/s1_{p}.stream") for p in ("none", "all", "shell")}
for p in ("all", "shell"):
    same_hkl = [a[0] for a in r["none"]] == [b[0] for b in r[p]]
    same_rest = [a[3] for a in r["none"]] == [b[3] for b in r[p]]
    same_multiset = collections.Counter((a[1], a[2]) for a in r["none"]) == collections.Counter((b[1], b[2]) for b in r[p])
    moved = sum(a[1] != b[1] for a, b in zip(r["none"], r[p])) / max(1, len(r[p]))
    print(f"null {p}: {len(r[p])} reflections; hkl and positions unchanged {same_hkl and same_rest}; "
          f"(I, sigma) multiset unchanged {same_multiset}; fraction of I moved {moved:.3f}")
EOF

echo "== merges of the 10% seed-1 subset (fixed range from LUTE's full merge: 79.170 2.171 A)"
export DRANGE="79.170 2.171"
time "$TOOLS_HOME/sweep_merge_one.sh" "$STREAM" r51CF 0.10 1 none partialator "$DR/m_partialator_none" 2
time "$TOOLS_HOME/sweep_merge_one.sh" "$STREAM" r51CF 0.10 1 none process_hkl "$DR/m_process_hkl_none" 2
time "$TOOLS_HOME/sweep_merge_one.sh" "$STREAM" r51CF 0.10 1 all partialator "$DR/m_partialator_all" 2
for m in m_partialator_none m_process_hkl_none m_partialator_all; do cat "$DR/$m/record.log"; done
echo "-- partialator log tail"; tail -n 25 "$DR/m_partialator_none/partialator.log"

echo "== does the order set the half-split? the same frames in stream order (seed 0) and in seed-2 order"
"$TOOLS_HOME/sweep_merge_one.sh" "$DR/s1_none.stream" order 1.00 0 none partialator "$DR/o_s0" 2
"$TOOLS_HOME/sweep_merge_one.sh" "$DR/s1_none.stream" order 1.00 2 none partialator "$DR/o_s2" 2
for x in merged.hkl merged.hkl1; do
    a=$(grep -E '^ +-?[0-9]+ +-?[0-9]+ +-?[0-9]+ ' "$DR/o_s0/$x" | md5sum | cut -c1-12)
    b=$(grep -E '^ +-?[0-9]+ +-?[0-9]+ +-?[0-9]+ ' "$DR/o_s2/$x" | md5sum | cut -c1-12)
    echo "$x: stream order $a, seed-2 order $b  $([ "$a" = "$b" ] && echo same || echo differ)"
done
cat "$DR/o_s0/record.log" "$DR/o_s2/record.log"
rm -f "$DR"/s1_*.stream
du -sh "$DR"
