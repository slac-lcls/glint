#!/bin/bash
# Install the GLINT LUTE task (IndexGLINT / GLINTIndexer) into a LUTE clone. Idempotent-ish; back up first.
set -e
LUTE="${1:-$HOME/git/lute_new/lute}"
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -d "$LUTE/lute/io/models" ] || { echo "not a LUTE repo: $LUTE"; exit 1; }
cp "$HERE/glint_index.py" "$LUTE/lute/io/models/glint_index.py"
grep -q "from .glint_index import" "$LUTE/lute/io/models/__init__.py" || \
  echo "from .glint_index import *" >> "$LUTE/lute/io/models/__init__.py"
grep -q "IndexGLINT" "$LUTE/lute/managed_tasks.py" || \
  echo 'GLINTIndexer: Executor = Executor("IndexGLINT")' >> "$LUTE/lute/managed_tasks.py"
chmod +x "$HERE/glint_launch.sh"
echo "installed IndexGLINT into $LUTE (model + export + executor). Edit executable path in glint_index.py if the GLINT repo moved."
