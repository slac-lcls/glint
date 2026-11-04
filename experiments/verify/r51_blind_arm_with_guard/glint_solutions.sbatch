#!/bin/bash
#SBATCH --job-name=guard_glint
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=sdfampere002,sdfampere003,sdfampere038,sdfampere042
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.log
# working directory: submit from $VERIFY_OUT/r51_blind_arm_with_guard, or pass --chdir= to it (else indexamajig.<pid>/ lands in the submitter's cwd)
# Job 1 of 3 (GPU): GLINT's orientations for the tofile route, at the GLINT tree of config.sh (abe330c here).
#   GLb  blind: no --cell, shipped defaults (mode auto, nbest 3, min-peaks 6), the .cxi's stored peakfinder8
#        peaks, --gate strict (the LUTE tofile configuration), --lattice = the dataset's Bravais code
#   GLk  the reference cell given: the LUTE GLINT functional-test configuration (reproduction anchor)
# Both write a CrystFEL --indexing=file solution file (--tofile) and GLINT's orientation-only stream.
# Fallback if lcls:prjdat21 is refused on ampere: --account=lcls:default@ampere --qos=preemptable
set -uo pipefail
source "${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/r51_blind_arm_with_guard/config.sh"
glint_env || exit 1
echo "host $(hostname)  job ${SLURM_JOB_ID:-}  start $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python -c "import torch, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), 'numpy', numpy.__version__)"
echo "GLINT commit $(cat $GLINT_TREE/COMMIT)"

for D in $DATASETS; do
    v=${D}_LIST; LIST=${!v}; v=${D}_GEOM; GEOM=${!v}; v=${D}_CELL; CELL=${!v}; v=${D}_LATT; LATT=${!v}
    [ -s "$LIST" ] || { echo "== $D: no input list $LIST; skipping"; continue; }
    mkdir -p "$OUT/$D"
    echo "== $D  $(date -u +%FT%TZ)"
    aside "$OUT/$D/GLb.sol" "$OUT/$D/GLb_orient.stream" "$OUT/$D/GLk.sol" "$OUT/$D/GLk_orient.stream"
    SECONDS=0
    run python -m glint.glint_cli --images "$LIST" --geom "$GEOM" --peakfinder stored --gate strict --lattice "$LATT" \
        --tofile "$OUT/$D/GLb.sol" -o "$OUT/$D/GLb_orient.stream" > "$OUT/$D/GLb_glint.log" 2>&1
    rc=$?; echo "GLb exit $rc  ${SECONDS} s"; tail -12 "$OUT/$D/GLb_glint.log"
    # a failed run can leave a partial .sol; stop so the afterok indexing and merges never start on it
    [ $rc -eq 0 ] || { echo "$D: GLINT GLb failed (exit $rc), see GLb_glint.log"; exit 1; }
    SECONDS=0
    run python -m glint.glint_cli --images "$LIST" --geom "$GEOM" --peakfinder stored --gate strict --lattice "$LATT" \
        --cell $CELL --tofile "$OUT/$D/GLk.sol" -o "$OUT/$D/GLk_orient.stream" > "$OUT/$D/GLk_glint.log" 2>&1
    rc=$?; echo "GLk exit $rc  ${SECONDS} s"; tail -12 "$OUT/$D/GLk_glint.log"
    # a failed run can leave a partial .sol; stop so the afterok indexing and merges never start on it
    [ $rc -eq 0 ] || { echo "$D: GLINT GLk failed (exit $rc), see GLk_glint.log"; exit 1; }
    for A in GLb GLk; do
        [ -s "$OUT/$D/$A.sol" ] && python "$TOOLS_HOME/merge_tools.py" cellcheck "$OUT/$D/$A.sol" "$CELL" "$OUT/$D/${A}_cellcheck.json"
    done
    sha256sum "$OUT/$D"/*.sol > "$OUT/$D/sol.sha256"
done
echo "end $(date -u +%FT%TZ)"
