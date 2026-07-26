#!/bin/bash
# Exercise every branch of install_into_lute.sh's provenance guard against throwaway LUTE trees.
# A guard that never fires is worse than none, so the DIVERGED cases assert a NON-ZERO exit.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
pass=0; fail=0

mklute() {   # $1 = dir, $2 = file to install as the deployed glint_index.py (empty = none)
    rm -rf "$1"; mkdir -p "$1/lute/io/models"
    : > "$1/lute/io/models/__init__.py"
    : > "$1/lute/managed_tasks.py"
    [ -n "$2" ] && cp "$2" "$1/lute/io/models/glint_index.py"
    return 0
}

check() {    # $1 label, $2 expected exit, $3 grep pattern ("" = check exit code only), then the command
    local label="$1" want="$2" pat="$3"; shift 3
    local out; out="$("$@" 2>&1)"; local rc=$?
    if [ "$rc" -eq "$want" ] && { [ -z "$pat" ] || echo "$out" | grep -q "$pat"; }; then
        echo "  [PASS] $label (exit $rc)"; pass=$((pass+1))
    else
        echo "  [FAIL] $label -- exit $rc (want $want), pattern '$pat'"; echo "$out" | sed 's/^/         /'; fail=$((fail+1))
    fi
}

echo "1. fresh install (no deployed file)"
mklute "$T/fresh"
check "installs cleanly" 0 "installed IndexGLINT" bash "$REPO/lute/install_into_lute.sh" "$T/fresh"

echo "2. deployed copy IDENTICAL to repo"
mklute "$T/same" "$REPO/lute/glint_index.py"
check "reports nothing to install" 0 "already identical" bash "$REPO/lute/install_into_lute.sh" "$T/same"

echo "3. deployed copy is an OLDER COMMITTED version (plain upgrade)"
# Most recent committed version whose CONTENT differs from the working copy. Not "the 2nd commit":
# squash-merges leave several commits carrying byte-identical files, so an index-based pick lands on
# a duplicate of the current file and the fixture stops testing anything.
CUR_SHA="$(shasum -a 256 "$REPO/lute/glint_index.py" | cut -d' ' -f1)"
PREV=""
for c in $(git -C "$REPO" log --all --format=%H -- lute/glint_index.py); do
    if [ "$(git -C "$REPO" show "$c:lute/glint_index.py" 2>/dev/null | shasum -a 256 | cut -d' ' -f1)" != "$CUR_SHA" ]; then
        PREV="$c"; break
    fi
done
[ -n "$PREV" ] || { echo "  [SKIP] no committed version differs from the working copy"; exit 0; }
git -C "$REPO" show "$PREV:lute/glint_index.py" > "$T/old_committed.py"
mklute "$T/old" "$T/old_committed.py"
check "recognised as a repo version" 0 "plain upgrade" bash "$REPO/lute/install_into_lute.sh" "$T/old"

echo "4. deployed copy has LOCAL EDITS (the real case -- must BLOCK)"
cp "$REPO/lute/glint_index.py" "$T/edited.py"; echo "# local hand edit" >> "$T/edited.py"
mklute "$T/edited" "$T/edited.py"
check "refuses without --force" 2 "MATCHES NO COMMITTED VERSION" bash "$REPO/lute/install_into_lute.sh" "$T/edited"
check "local edits NOT overwritten" 0 "" grep -q "local hand edit" "$T/edited/lute/io/models/glint_index.py"
check "--force overrides" 0 "force given" bash "$REPO/lute/install_into_lute.sh" "$T/edited" --force
check "backup preserves the local edits" 0 "" bash -c "grep -q 'local hand edit' $T/edited/lute/io/models/glint_index.py.*.bak"
check "target now matches the repo copy" 0 "" bash -c "diff -q $REPO/lute/glint_index.py $T/edited/lute/io/models/glint_index.py"

echo "5. run from a NON-git directory (provenance unknowable -- must BLOCK)"
rm -rf "$T/copied"; mkdir -p "$T/copied"
cp "$REPO/lute/install_into_lute.sh" "$REPO/lute/glint_index.py" "$REPO/lute/glint_launch.sh" "$T/copied/"
mklute "$T/nogit" "$T/old_committed.py"
check "refuses without --force" 2 "CANNOT VERIFY" bash "$T/copied/install_into_lute.sh" "$T/nogit"
check "--force overrides" 0 "force given" bash "$T/copied/install_into_lute.sh" "$T/nogit" --force

echo "6. idempotence: export + executor lines appended exactly once"
mklute "$T/idem"
bash "$REPO/lute/install_into_lute.sh" "$T/idem" >/dev/null 2>&1
bash "$REPO/lute/install_into_lute.sh" "$T/idem" >/dev/null 2>&1
n1=$(grep -c "from .glint_index import" "$T/idem/lute/io/models/__init__.py")
n2=$(grep -c "IndexGLINT" "$T/idem/lute/managed_tasks.py")
if [ "$n1" = "1" ] && [ "$n2" = "1" ]; then echo "  [PASS] no duplicate appends after 2 runs"; pass=$((pass+1));
else echo "  [FAIL] duplicates: import=$n1 executor=$n2"; fail=$((fail+1)); fi

echo; echo "$pass passed, $fail failed"; [ "$fail" -eq 0 ]
