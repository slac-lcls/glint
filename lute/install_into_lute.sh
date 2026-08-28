#!/bin/bash
# Install the GLINT LUTE task (IndexGLINT / GLINTIndexer) into a LUTE clone.
#
#   ./install_into_lute.sh [/path/to/lute_new/lute] [--force] [--skip-activation-check]
#
# The repo is authoritative, but the DEPLOYED copy can drift. On 2026-07-20 the installed
# glint_index.py matched NO committed version: it was a hand-rolled variant carrying an `-N`
# field and a `peakfinder=stored` default that the repo copy did not have. Installing over it
# would have dropped both -- and because ThirdPartyParameters sets extra="allow", a config still
# setting `n:` would have passed validation and only failed at runtime.
#
# So this script will not clobber a target it cannot account for. Before overwriting it asks:
# does the deployed file match some commit in this repo's history?
#   yes, the current one -> nothing to install
#   yes, an older one    -> plain upgrade, proceed
#   no                   -> DIVERGED: stop and say what to diff. --force overrides, but fold the
#                           differences back into the repo first or the next install loses them again.
# A timestamped backup is always taken before the copy.
#
# ACTIVATION-TRAP CHECK (glint#128). Upstream LUTE's `install/bin/activate_installation` derives
# its PYTHONPATH from the AMBIENT python3, not a pinned one. A LUTE install can carry several
# lib/pythonX.Y trees side by side and leave some of them unpopulated, so activation exports a
# PYTHONPATH into an empty tree and reports nothing wrong -- every LUTE task, GLINT included, then
# dies later with `ModuleNotFoundError: No module named 'launch_scripts'` or a bare subprocess
# return code 127, far from the actual cause. This script cannot fix upstream LUTE (see
# lute/upstream_activate_installation.patch for a DRAFT that could), so instead it checks the
# target LUTE install right here, where $LUTE is already in hand, and refuses to proceed if the
# trap is present. Override with --skip-activation-check if you already have a workaround (e.g.
# the shim described in glint#128) or want to install anyway.
#
# GATED, not unconditional: the check only fires when activate_installation still contains the
# ambient-derivation pattern quoted in glint#128 (see _activation_is_vulnerable below). A future
# upstream fix that stops deriving the tree from the ambient interpreter would no longer match, and
# an unconditional "does the ambient tree contain launch_scripts" check would then false-positive
# on that fixed script whenever the ambient tree happened to be empty for an unrelated reason.
#
# BEST-EFFORT, not a runtime guarantee: this samples python3 from the shell running THIS script.
# The shell that actually launches a job sources psconda.sh first (see lute/README.md's Run
# section), which can select a different interpreter and therefore a different, possibly empty,
# tree -- our scripts are not in that launch chain (see glint#128's own analysis), so we cannot
# check it from here. The population map below covers every tree this install carries so a
# different runtime interpreter's risk is visible even when the install-time one is fine.
set -e

FORCE=0
SKIP_ACTIVATION_CHECK=0
POSITIONAL=()
for a in "$@"; do
    case "$a" in
        --force|-f) FORCE=1 ;;
        --skip-activation-check) SKIP_ACTIVATION_CHECK=1 ;;
        -h|--help) sed -n '2,42p' "$0"; exit 0 ;;
        *) POSITIONAL+=("$a") ;;
    esac
done

LUTE="${POSITIONAL[0]:-$HOME/git/lute_new/lute}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/glint_index.py"
TARGET="$LUTE/lute/io/models/glint_index.py"
[ -d "$LUTE/lute/io/models" ] || { echo "not a LUTE repo: $LUTE"; exit 1; }

_installed_content() {
    local here_sed
    here_sed="$(printf '%s' "$HERE" | sed 's/[\\&#]/\\&/g')"
    sed 's#"/sdf/home/s/smarches/git/glint/lute/glint_launch.sh"#"'"$here_sed"'/glint_launch.sh"#'
}

EXPECTED="$(mktemp)"
COMMITTED="$(mktemp)"
trap 'rm -f "$EXPECTED" "$COMMITTED"' EXIT
_installed_content < "$SRC" > "$EXPECTED"

# --- activation-trap check (glint#128) -----------------------------------------------------------
_activation_is_vulnerable() {
    # $1 = path to an activate_installation. True only if it still shows BOTH hallmarks of the
    # ambient-derivation bug quoted in glint#128 (and reconstructed in
    # lute/upstream_activate_installation.patch): deriving a version from the AMBIENT python3's
    # sys.version_info, and interpolating that into a lib/pythonX.Y/site-packages path. A future
    # upstream fix that pins the interpreter, or checks launch_scripts itself, would drop one of
    # these -- so gating on both keeps a patched activate_installation from being false-flagged
    # just because the ambient tree happens to be empty for an unrelated reason.
    grep -q "sys.version_info.major" "$1" 2>/dev/null && grep -q "site-packages" "$1" 2>/dev/null
}

_check_lute_activation() {
    # $1 = LUTE root. Prints a loud diagnostic and returns nonzero if upstream LUTE's
    # activate_installation would silently select a python tree with no launch_scripts in it.
    # Install-time and best-effort only -- see the header comment above set -e for why this cannot
    # be a runtime guarantee.
    local lute="$1" activation py_ver site d ver populated=() empty=()
    activation="$lute/install/bin/activate_installation"
    [ -f "$activation" ] || return 0   # install/ not built yet here -- nothing to check

    if ! _activation_is_vulnerable "$activation"; then
        echo "note: $activation does not match the known ambient-derivation pattern (glint#128)" >&2
        echo "  -- skipping the trap check. If a newer upstream still has this trap in a different" >&2
        echo "  shape, update _activation_is_vulnerable() in install_into_lute.sh." >&2
        return 0
    fi

    py_ver="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)" || true
    if [ -z "$py_ver" ]; then
        echo "warning: could not determine the ambient python3 version -- skipping the" >&2
        echo "  activate_installation sanity check (glint#128). Verify by hand before relying on it." >&2
        return 0
    fi

    # Population map for EVERY lib/pythonX.Y tree this install carries, not just the ambient one:
    # sampled from this (install-time) shell's python3, so it cannot see what a differently-set-up
    # RUN shell (after sourcing psconda.sh) would pick -- but it at least surfaces every tree that
    # COULD be picked, empty or not.
    for d in "$lute"/install/lib/python*/site-packages; do
        [ -d "$d" ] || continue
        ver="$(basename "$(dirname "$d")")"
        if [ -d "$d/launch_scripts" ]; then populated+=("$ver"); else empty+=("$ver"); fi
    done

    site="$lute/install/lib/python${py_ver}/site-packages"
    if [ -d "$site/launch_scripts" ]; then
        if [ ${#empty[@]} -gt 0 ]; then
            echo "note: activate_installation would work right now -- ambient python3 (${py_ver})" >&2
            echo "  resolves to a populated tree -- but $lute/install also carries EMPTY tree(s):" >&2
            echo "  ${empty[*]}. If the shell that actually runs launch_slurm/submit_slurm (after" >&2
            echo "  sourcing psconda.sh, per lute/README.md's Run section) ends up on a different" >&2
            echo "  python3, the trap still applies there and this install-time check cannot see it." >&2
        fi
        return 0   # ambient tree is populated -- activation would work from here
    fi

    echo "*** activate_installation TRAP DETECTED (glint#128) ***" >&2
    echo "    ambient python3 is ${py_ver} -> $activation would export" >&2
    echo "        PYTHONPATH=${site}" >&2
    echo "    but that tree has no launch_scripts -- it was never populated when $lute/install" >&2
    echo "    was built. Every LUTE task submitted this way (GLINTIndexer included) will die" >&2
    echo "    later with ModuleNotFoundError or a bare subprocess return code 127, far from" >&2
    echo "    this cause." >&2
    echo "    population map for $lute/install (this shell's python3 only -- best effort, see" >&2
    echo "    lute/README.md's Run section for why the RUN shell can differ):" >&2
    for ver in "${populated[@]}"; do echo "        ${ver}: launch_scripts present" >&2; done
    for ver in "${empty[@]}"; do echo "        ${ver}: EMPTY" >&2; done
    if [ ${#populated[@]} -gt 0 ]; then
        echo "    FIX: put a python3 from a populated tree on PATH (e.g. activate a conda env" >&2
        echo "    pinned to it) BEFORE sourcing install/bin/activate_installation or calling" >&2
        echo "    launch_slurm / submit_slurm, then retry." >&2
    else
        echo "    no lib/python*/site-packages tree under $lute/install contains launch_scripts --" >&2
        echo "    this LUTE install looks broken beyond the version mismatch; rebuild it." >&2
    fi
    echo "    see lute/upstream_activate_installation.patch for a draft fix to propose upstream." >&2
    return 1
}

if ! _check_lute_activation "$LUTE"; then
    if [ "$SKIP_ACTIVATION_CHECK" -eq 1 ]; then
        echo "    --skip-activation-check given: proceeding anyway." >&2
    else
        echo "    refusing to install without --skip-activation-check" >&2
        exit 3
    fi
fi

_sha() { shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1; }

# --- provenance guard ---------------------------------------------------------------------------
if [ -f "$TARGET" ]; then
    SRC_SHA="$(_sha "$EXPECTED")"
    TGT_SHA="$(_sha "$TARGET")"
    if [ "$SRC_SHA" = "$TGT_SHA" ]; then
        echo "glint_index.py already identical to the expected installed copy -- nothing to install."
        echo "  (export + executor lines are re-checked below; both are idempotent)"
        SKIP_COPY=1          # nothing would change: no copy, and no .bak litter per no-op run
    elif ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null)" && [ -n "$ROOT" ]; then
        # Pathspecs resolve against the CWD, and $HERE is the lute/ subdir -- so run git from the
        # repo ROOT, or `lute/glint_index.py` silently becomes lute/lute/... and matches nothing,
        # which would make every deployed copy look diverged and the guard cry wolf every run.
        MATCH=""
        for c in $(git -C "$ROOT" log --all --format=%H -- lute/glint_index.py 2>/dev/null); do
            git -C "$ROOT" show "$c:lute/glint_index.py" > "$COMMITTED" 2>/dev/null
            COMMIT_SHA="$(_sha "$COMMITTED")"
            INSTALLED_COMMIT_SHA="$(_installed_content < "$COMMITTED" | shasum -a 256 | cut -d' ' -f1)"
            if [ "$COMMIT_SHA" = "$TGT_SHA" ] || [ "$INSTALLED_COMMIT_SHA" = "$TGT_SHA" ]; then
                MATCH="$c"; break
            fi
        done
        if [ -n "$MATCH" ]; then
            echo "deployed copy is repo version $(git -C "$ROOT" log -1 --format='%h (%ad) %s' --date=short "$MATCH")"
            echo "  -> plain upgrade, proceeding."
        else
            echo "*** DEPLOYED glint_index.py MATCHES NO COMMITTED VERSION ***"
            echo "    deployed  $TARGET"
            echo "        sha   $TGT_SHA"
            echo "    repo      $SRC"
            echo "        sha   $SRC_SHA"
            echo
            echo "    It carries local edits. Overwriting would silently discard them -- the exact"
            echo "    failure this guard exists for (see the header). Inspect first:"
            echo "        diff \"$TARGET\" \"$SRC\""
            echo "    then fold anything worth keeping into the repo copy and re-run."
            [ "$FORCE" -eq 1 ] || { echo; echo "    refusing without --force"; exit 2; }
            echo "    --force given: overwriting anyway."
        fi
    else
        echo "*** CANNOT VERIFY the deployed copy: $HERE is not a git checkout ***"
        echo "    (running from a copied-out directory -- provenance is unknowable here)"
        echo "    deployed sha $TGT_SHA differs from repo sha $SRC_SHA."
        echo "    Re-run from a real GLINT checkout so history can be searched, or inspect:"
        echo "        diff \"$TARGET\" \"$SRC\""
        [ "$FORCE" -eq 1 ] || { echo; echo "    refusing without --force"; exit 2; }
        echo "    --force given: overwriting anyway."
    fi
    if [ "${SKIP_COPY:-0}" -eq 0 ]; then
        BACKUP="$TARGET.$(date +%Y%m%d-%H%M%S).bak"
        cp "$TARGET" "$BACKUP"
        echo "backed up deployed copy -> $BACKUP"
    fi
fi

# --- install ------------------------------------------------------------------------------------
[ "${SKIP_COPY:-0}" -eq 1 ] || cp "$EXPECTED" "$TARGET"
grep -q "from .glint_index import" "$LUTE/lute/io/models/__init__.py" || \
  echo "from .glint_index import *" >> "$LUTE/lute/io/models/__init__.py"
grep -q "IndexGLINT" "$LUTE/lute/managed_tasks.py" || \
  echo 'GLINTIndexer: Executor = Executor("IndexGLINT")' >> "$LUTE/lute/managed_tasks.py"
chmod +x "$HERE/glint_launch.sh"
echo "installed IndexGLINT into $LUTE (model + export + executor). Edit executable path in glint_index.py if the GLINT repo moved."
