#!/bin/bash
# GLINT launcher for the LUTE `IndexGLINT` ThirdPartyTask: activate the GLINT GPU (torch) env and run
# the right GLINT entry point. LUTE builds the flags from IndexGLINTParameters and invokes:
#   glint_launch.sh <flags>
# Point IndexGLINTParameters.executable at this script.
#
# TWO DESTINATIONS, because the three frame sources are not one program:
#   `peaks` / `images`  -> glint.glint_cli                       (peak stream, or raw .cxi)
#   `exp` (+ `run`)     -> experiments/xtc_bridge/glint_xtc.py   (raw xtc; psana1 in-process, or
#                                                                 psana2 in conda2 over envbridge)
#
# Flags are FILTERED per destination rather than forwarded wholesale. glint_xtc.py's argparse rejects
# unknown flags, and several glint_cli options carry non-empty defaults (e.g. --peakfinder stored) so
# LUTE emits them on every run -- forwarding blindly would kill the xtc route on a flag the user never
# set. Whitelisting keeps the failure loud and local: an xtc flag missing from the list below is
# rejected by argparse rather than silently ignored.
set -o pipefail
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh >/dev/null 2>&1
conda activate ana-4.0.58-py3-minipytorch >/dev/null 2>&1
cd "$(dirname "$0")/.." || exit 1                       # repo root (so `glint` imports)

for a in "$@"; do                                       # does this invocation name the xtc source?
    if [ "$a" = "--exp" ]; then XTC=1; fi
done

if [ -z "$XTC" ]; then
    exec python -m glint.glint_cli "$@"                 # unchanged: the peaks / images routes
fi

# Everything glint_xtc.py accepts. Each takes a value; LUTE emits no bare switches on this route.
XTC_FLAGS=" --exp --run --det --zdist --wavelength --psana --geom --calib-dir --cell --nbest \
--min-peaks --max-events --min-pix --son-min --thr-high --thr-low --reader-env --energy-det -o --out "
args=(); dropped=()
i=1
while [ $i -le $# ]; do
    a="${!i}"; j=$((i + 1)); v="${!j}"
    case "$XTC_FLAGS" in
        *" $a "*) args+=("$a" "$v") ;;
        *)        dropped+=("$a") ;;
    esac
    i=$((i + 2))
done
# Report what was dropped. These are glint_cli-only options with no meaning for raw xtc (--peaks,
# --images, --peakfinder, --top-peaks, --integrate, --tofile, --image-dir, -N ...). Staying silent
# would let a run look as though it had honoured a setting the indexer never received.
if [ ${#dropped[@]} -gt 0 ]; then
    echo "glint_launch: xtc route; dropped flags that do not apply to glint_xtc.py: ${dropped[*]}" >&2
fi
exec python experiments/xtc_bridge/glint_xtc.py "${args[@]}"
