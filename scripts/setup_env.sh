#!/bin/bash
# setup_env.sh — Install Python deps for SFT/RL training on AML Singularity (ptca env)
#
# Designed to run on BOTH:
#   - AML Singularity job nodes (Python 3.8 in /opt/conda/envs/ptca)
#   - Interactive debug nodes (Python 3.10 in /opt/conda/envs/ptca)
#
# Strategy for transformers==4.51.3 on Python 3.8:
#   - tokenizers>=0.21 has no cp38 wheel → pin tokenizers==0.20.3 (last cp38 wheel)
#   - install huggingface-hub and other deps normally first
#   - install transformers==4.51.3 with --no-deps so pip doesn't try to upgrade tokenizers
#   - apply from __future__ import annotations patch to transformers/utils/*.py (ABCMeta fix)
#
# Usage:
#   bash /tmp/gpu_code/scripts/setup_env.sh
#   bash /tmp/gpu_code/scripts/setup_env.sh --skip-if-ok   # no-op if transformers 4.51.3 already present

set -e

SKIP_IF_OK=0
for arg in "$@"; do
  [ "$arg" = "--skip-if-ok" ] && SKIP_IF_OK=1
done

PTCA_PIP=/opt/conda/envs/ptca/bin/pip
PTCA_PY=/opt/conda/envs/ptca/bin/python

PTCA_PYVER=$($PTCA_PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  ptca Python version: $PTCA_PYVER"

export PYTHONUSERBASE=/tmp/ptca_user
mkdir -p /tmp/ptca_user

USER_SITE=/tmp/ptca_user/lib/python${PTCA_PYVER}/site-packages

# --skip-if-ok: bail early if transformers 4.51.3 is already installed
if [ "$SKIP_IF_OK" = "1" ]; then
  TV=$($PTCA_PY -c "
import sys
sys.path.insert(0, '$USER_SITE')
try:
    import transformers
    print(transformers.__version__)
except Exception:
    print('missing')
" 2>/dev/null)
  if [ "$TV" = "4.51.3" ]; then
    echo "  transformers 4.51.3 already installed — skipping setup_env.sh"
    # Still export PYTHONPATH so the caller can use the env
    export PYTHONPATH=$USER_SITE:${PYTHONPATH:-}
    exit 0
  fi
fi

echo "[env] Step 1/5: tokenizers==0.20.3 (last cp38 binary wheel)"
$PTCA_PIP install --user --only-binary :all: --no-deps 'tokenizers==0.20.3' 2>&1 | tail -3

echo "[env] Step 2/5: huggingface-hub + safetensors + regex (transformers deps)"
# Install WITH deps so transitive deps (filelock, requests, tqdm, ...) come in too.
# --only-binary: safetensors has a Rust build — use the pre-built wheel.
# --ignore-requires-python: some wheel metadata says >=3.9 but works fine on 3.8.
$PTCA_PIP install --user --only-binary :all: --ignore-requires-python \
    'huggingface-hub>=0.30.0,<1.0' \
    'safetensors>=0.4.3' \
    'regex!=2019.12.17' \
    2>&1 | tail -3

echo "[env] Step 3/5: transformers==4.51.3 (--no-deps to keep tokenizers==0.20.3)"
# Pure Python wheel (py3-none-any). --no-deps prevents pip from upgrading tokenizers.
$PTCA_PIP install --user --no-deps --ignore-requires-python 'transformers==4.51.3' 2>&1 | tail -3

echo "[env] Step 4/5: backbone_utils Python 3.8 patch"
# On Python 3.8, Optional[Iterable[str]] in a function signature raises
# "TypeError: 'ABCMeta' object is not subscriptable" at class-definition time.
# Adding 'from __future__ import annotations' defers annotation evaluation — fixed.
PATCHED=0
for F in $USER_SITE/transformers/utils/*.py; do
  [ -f "$F" ] || continue
  grep -q 'Iterable\[\|Sequence\[\|Callable\[\|Iterator\[\|Generator\[' "$F" 2>/dev/null || continue
  head -1 "$F" | grep -q '__future__' && continue
  sed -i '1s/^/from __future__ import annotations\n/' "$F"
  echo "    Patched: $(basename $F)"
  PATCHED=$((PATCHED+1))
done
[ "$PATCHED" -eq 0 ] && echo "    No patches needed (Python 3.10+ or already patched)"

echo "[env] Step 4b: verify transformers import"
$PTCA_PY -c "
import sys
sys.path.insert(0, '$USER_SITE')
import transformers
v = transformers.__version__
print('  transformers', v, transformers.__file__)
assert v == '4.51.3', f'FATAL: transformers {v} != 4.51.3 — install failed'
print('  OK')
"

echo "[env] Step 5/5: datasets + peft"
$PTCA_PIP install --user --only-binary :all: --ignore-requires-python datasets peft 2>&1 | tail -3

export PYTHONPATH=$USER_SITE:${PYTHONPATH:-}
echo "[env] PYTHONPATH prepended: $USER_SITE"

echo "[env] Final check:"
$PTCA_PY -c "
import sys
sys.path.insert(0, '$USER_SITE'.replace('\${PTCA_PYVER}', sys.version[:3]))
import transformers, datasets, peft
print('  transformers', transformers.__version__)
print('  datasets    ', datasets.__version__)
print('  peft        ', peft.__version__)
"
echo "[env] setup_env.sh complete."
