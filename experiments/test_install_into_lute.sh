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

echo "7. activation-trap check (glint#128)"
# A fake `python3` that reports a fixed version, first on PATH -- lets us control which
# lib/pythonX.Y tree the activation guard would pick without needing a real 3.9/3.12 interpreter.
fakepy() {   # $1 = dir to create, $2 = version string it should report
    mkdir -p "$1"
    printf '#!/bin/bash\n[ "$1" = "-c" ] && echo "%s"\n' "$2" > "$1/python3"
    chmod +x "$1/python3"
}
fakepy "$T/py312" 3.12
fakepy "$T/py39" 3.9

# The exact ambient-derivation snippet quoted in glint#128 / lute/upstream_activate_installation.patch
# -- both hallmarks _activation_is_vulnerable() greps for (sys.version_info.major, site-packages).
VULN_ACTIVATION='PY_VER=$(python3 -c '"'"'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'"'"')
LUTE_LIB_PATH=(${LUTE_INSTALL_PATH}lib/python${PY_VER}/site-packages)
export PYTHONPATH="${LUTE_LIB_PATH}:${PYTHONPATH}"'

mkactivation() {   # $1 = LUTE dir, $2 = "vuln" (matches the pattern) or "fixed" (does not)
    mkdir -p "$1/install/bin"
    if [ "$2" = vuln ]; then
        printf '%s\n' "$VULN_ACTIVATION" > "$1/install/bin/activate_installation"
    else
        printf '#!/bin/bash\necho "a hypothetical fixed activate_installation"\n' > "$1/install/bin/activate_installation"
    fi
}

echo "  7a. install/ not built yet -- silent no-op, installs cleanly"
mklute "$T/act_noinstall"
check "no install/ -> proceeds" 0 "installed IndexGLINT" \
    env PATH="$T/py312:$PATH" bash "$REPO/lute/install_into_lute.sh" "$T/act_noinstall"

echo "  7b. vulnerable pattern, ambient tree EMPTY, an alternate tree populated -- must BLOCK"
mklute "$T/act_bad"; mkactivation "$T/act_bad" vuln
mkdir -p "$T/act_bad/install/lib/python3.12/site-packages"                      # ambient (3.12): empty
mkdir -p "$T/act_bad/install/lib/python3.9/site-packages/launch_scripts"        # alternate: populated
check "refuses, exit 3" 3 "TRAP DETECTED" \
    env PATH="$T/py312:$PATH" bash "$REPO/lute/install_into_lute.sh" "$T/act_bad"
check "banner names the ambient version" 3 "ambient python3 is 3.12" \
    env PATH="$T/py312:$PATH" bash "$REPO/lute/install_into_lute.sh" "$T/act_bad"
check "banner names the empty path" 3 "python3.12/site-packages" \
    env PATH="$T/py312:$PATH" bash "$REPO/lute/install_into_lute.sh" "$T/act_bad"
check "banner names the populated alternative" 3 "python3.9: launch_scripts present" \
    env PATH="$T/py312:$PATH" bash "$REPO/lute/install_into_lute.sh" "$T/act_bad"
check "banner states the fix" 3 "put a python3 from a populated tree on PATH" \
    env PATH="$T/py312:$PATH" bash "$REPO/lute/install_into_lute.sh" "$T/act_bad"
check "not installed" 1 "" bash -c "[ -f '$T/act_bad/lute/io/models/glint_index.py' ]"

echo "  7c. vulnerable pattern, ambient tree POPULATED -- silent pass, installs"
mklute "$T/act_good"; mkactivation "$T/act_good" vuln
mkdir -p "$T/act_good/install/lib/python3.9/site-packages/launch_scripts"
check "installs cleanly, no TRAP banner" 0 "installed IndexGLINT" \
    env PATH="$T/py39:$PATH" bash "$REPO/lute/install_into_lute.sh" "$T/act_good"

echo "  7d. --skip-activation-check on the bad tree -- warns, proceeds anyway"
mklute "$T/act_skip"; mkactivation "$T/act_skip" vuln
mkdir -p "$T/act_skip/install/lib/python3.12/site-packages"
mkdir -p "$T/act_skip/install/lib/python3.9/site-packages/launch_scripts"
check "warns and proceeds" 0 "proceeding anyway" \
    env PATH="$T/py312:$PATH" bash "$REPO/lute/install_into_lute.sh" "$T/act_skip" --skip-activation-check
check "still installs" 0 "" bash -c "[ -f '$T/act_skip/lute/io/models/glint_index.py' ]"

echo "  7e. FIXED activate_installation (pattern absent), ambient tree empty -- pattern gate skips, installs"
mklute "$T/act_fixed"; mkactivation "$T/act_fixed" fixed
mkdir -p "$T/act_fixed/install/lib/python3.12/site-packages"   # empty, but the pattern is absent
check "pattern gate skips the check" 0 "does not match the known ambient-derivation pattern" \
    env PATH="$T/py312:$PATH" bash "$REPO/lute/install_into_lute.sh" "$T/act_fixed"

echo; echo "$pass passed, $fail failed"; [ "$fail" -eq 0 ]
