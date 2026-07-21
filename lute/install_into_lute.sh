#!/bin/bash
# Install the GLINT LUTE task (IndexGLINT / GLINTIndexer) into a LUTE clone.
#
#   ./install_into_lute.sh [/path/to/lute_new/lute] [--force]
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
set -e

FORCE=0
POSITIONAL=()
for a in "$@"; do
    case "$a" in
        --force|-f) FORCE=1 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) POSITIONAL+=("$a") ;;
    esac
done

LUTE="${POSITIONAL[0]:-$HOME/git/lute_new/lute}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/glint_index.py"
TARGET="$LUTE/lute/io/models/glint_index.py"
[ -d "$LUTE/lute/io/models" ] || { echo "not a LUTE repo: $LUTE"; exit 1; }

_sha() { shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1; }

# --- provenance guard ---------------------------------------------------------------------------
if [ -f "$TARGET" ]; then
    SRC_SHA="$(_sha "$SRC")"
    TGT_SHA="$(_sha "$TARGET")"
    if [ "$SRC_SHA" = "$TGT_SHA" ]; then
        echo "glint_index.py already identical to the repo copy -- nothing to install."
        echo "  (export + executor lines are re-checked below; both are idempotent)"
        SKIP_COPY=1          # nothing would change: no copy, and no .bak litter per no-op run
    elif ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null)" && [ -n "$ROOT" ]; then
        # Pathspecs resolve against the CWD, and $HERE is the lute/ subdir -- so run git from the
        # repo ROOT, or `lute/glint_index.py` silently becomes lute/lute/... and matches nothing,
        # which would make every deployed copy look diverged and the guard cry wolf every run.
        MATCH=""
        for c in $(git -C "$ROOT" log --all --format=%H -- lute/glint_index.py 2>/dev/null); do
            if [ "$(git -C "$ROOT" show "$c:lute/glint_index.py" 2>/dev/null | shasum -a 256 | cut -d' ' -f1)" = "$TGT_SHA" ]; then
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
[ "${SKIP_COPY:-0}" -eq 1 ] || cp "$SRC" "$TARGET"
grep -q "from .glint_index import" "$LUTE/lute/io/models/__init__.py" || \
  echo "from .glint_index import *" >> "$LUTE/lute/io/models/__init__.py"
grep -q "IndexGLINT" "$LUTE/lute/managed_tasks.py" || \
  echo 'GLINTIndexer: Executor = Executor("IndexGLINT")' >> "$LUTE/lute/managed_tasks.py"
chmod +x "$HERE/glint_launch.sh"
echo "installed IndexGLINT into $LUTE (model + export + executor). Edit executable path in glint_index.py if the GLINT repo moved."
