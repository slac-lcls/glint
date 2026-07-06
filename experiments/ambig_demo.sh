#!/bin/bash
# Indexing-ambiguity demo. Simulate a merohedral ambiguity (tetragonal 4/m structure on a 4/mmm lattice),
# scramble each frame's indexing mode at random, then resolve with CrystFEL ambigator and watch the merge
# recover. Needs CrystFEL (partialator, ambigator) + numpy. Set CF to your CrystFEL bin dir if not on PATH.
# Sim knobs (env, see sim_ambig.py): RHO (pseudo-symmetry 0..1), NREF (spots/frame), NOISE, NFRAMES.
# Frontier (measured): resolution succeeds when stills are reflection-rich (>~50-60 spots/frame) but fails
# on ultra-sparse stills regardless of frame count -- ambigator clusters on reflections common to crystal pairs.
set -e
CF=${CF:-}
if [ -n "$CF" ]; then PART="$CF/partialator"; AMBIG="$CF/ambigator"; else PART=partialator; AMBIG=ambigator; fi
echo "== simulate (random per-frame indexing modes) =="; python sim_ambig.py
cat > sim.cell <<EOF
CrystFEL unit cell file version 1.0

lattice_type = tetragonal
centering = P
unique_axis = c

a = 60.00 A
b = 60.00 A
c = 40.00 A
al = 90.00 deg
be = 90.00 deg
ga = 90.00 deg
EOF
echo "== merge scrambled (naive, as any indexer leaves it) =="
$PART -i sim_scrambled.stream -o scr.hkl -y 4/m --iterations=1 --model=unity -j 8 >/dev/null 2>&1
python correlate_truth.py scr.hkl
echo "== ambigator resolve (-y 4/m -w 4/mmm) =="
$AMBIG sim_scrambled.stream -o sim_resolved.stream -y 4/m -w 4/mmm -n 8 --end-assignments=asn.txt -j 8 2>&1 | grep -iE "mean f"
echo "== merge resolved =="
$PART -i sim_resolved.stream -o res.hkl -y 4/m --iterations=1 --model=unity -j 8 >/dev/null 2>&1
python correlate_truth.py res.hkl
python recover.py | awk '{print "  ambigator recovered the true mode for "$1"% of frames (up to global flip)"}'
