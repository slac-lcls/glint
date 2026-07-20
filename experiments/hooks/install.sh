#!/bin/sh
# Install the number-check pre-push hook.
#
#   ./install.sh                 # papers/glint = strict (blocks), glint = warn (does not block)
#   ./install.sh <repo> <mode>   # install into one repo explicitly; mode = strict|warn
#
# Git hooks are not version-controlled, so this has to be run once per clone. It is idempotent and
# refuses to clobber an unrelated existing hook.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
SRC="$HERE/pre-push"

install_one() {
    repo="$1"; mode="$2"
    if [ ! -d "$repo/.git" ]; then
        echo "  skip $repo (not a git repo)"
        return 0
    fi
    dest="$repo/.git/hooks/pre-push"
    if [ -e "$dest" ] && ! grep -q "check_numbers.py" "$dest" 2>/dev/null; then
        echo "  REFUSING $dest -- an unrelated pre-push hook is already installed" >&2
        return 1
    fi
    sed "s|@MODE@|$mode|" "$SRC" > "$dest"
    chmod +x "$dest"
    echo "  installed $dest  (mode=$mode)"
}

if [ $# -eq 2 ]; then
    install_one "$1" "$2"
else
    echo "installing number-check pre-push hooks:"
    # The paper goes to Overleaf and onward to a journal -- block there.
    install_one "$HOME/git/papers/glint" strict
    # The engine repo does not contain the deliverables -- warn only.
    install_one "$HOME/git/glint" warn
    # ~/git/slides is not a git repo, so it has no push to hook; it is covered by the checks the
    # other two hooks run (check_numbers.py always scans all targets, wherever it is invoked from).
    echo "note: ~/git/slides is not a git repo -- decks are covered by the above hooks' full scan."
fi
