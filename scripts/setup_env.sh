#!/bin/bash
# setup_env.sh — Install Python deps for SFT/RL training on AML Singularity (ptca env)
#
# Works on BOTH:
#   - Python 3.10 interactive nodes  → normal install (tokenizers 0.21.x available)
#   - Python 3.8 job nodes           → pin tokenizers==0.20.3 + patch runtime version gate
#
# Usage:
#   bash /tmp/gpu_code/scripts/setup_env.sh
#   bash /tmp/gpu_code/scripts/setup_env.sh --skip-if-ok

set -e

SKIP_IF_OK=0
for arg in "$@"; do
  [ "$arg" = "--skip-if-ok" ] && SKIP_IF_OK=1
done

PTCA_PIP=/opt/conda/envs/ptca/bin/pip
PTCA_PY=/opt/conda/envs/ptca/bin/python

PTCA_PYVER=$($PTCA_PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PTCA_PYMINOR=$($PTCA_PY -c 'import sys; print(sys.version_info.minor)')
echo "  ptca Python version: $PTCA_PYVER"

export PYTHONUSERBASE=/tmp/ptca_user
mkdir -p /tmp/ptca_user
USER_SITE=/tmp/ptca_user/lib/python${PTCA_PYVER}/site-packages

# --skip-if-ok: bail early if transformers 4.51.3 already imports cleanly
if [ "$SKIP_IF_OK" = "1" ]; then
  TV=$($PTCA_PY -c "
import sys
sys.path.insert(0, '$USER_SITE')
try:
    import transformers, accelerate
    print(transformers.__version__)
except Exception:
    print('missing')
" 2>/dev/null)
  if [ "$TV" = "4.51.3" ]; then
    echo "  transformers 4.51.3 + accelerate already installed — skipping"
    export PYTHONPATH=$USER_SITE:${PYTHONPATH:-}
    exit 0
  fi
fi

# -----------------------------------------------------------------------
# Branch on Python version:
#   Python >=3.9 (e.g. 3.10 interactive nodes): tokenizers 0.21.x has wheels
#   Python 3.8 (AML Singularity job nodes):     no cp38 wheel for tokenizers>=0.21
# -----------------------------------------------------------------------
if [ "$PTCA_PYMINOR" -ge 9 ]; then
  # =====================================================================
  # Python 3.10+ path — straightforward
  # =====================================================================
  echo "[env] Python $PTCA_PYVER: normal install (tokenizers 0.21.x available)"

  echo "[env] Step 1a: transformers==4.51.3"
  $PTCA_PIP install --user --ignore-requires-python 'transformers==4.51.3'

  echo "[env] Step 1b: accelerate"
  $PTCA_PIP install --user --ignore-requires-python 'accelerate'

  echo "[env] Step 1c: deepspeed (DS_BUILD_OPS=0 = no CUDA compile, uses JIT)"
  DS_BUILD_OPS=0 $PTCA_PIP install --user --ignore-requires-python 'deepspeed'

  echo "[env] Step 1d: datasets + peft (pin pandas<3 to avoid Python 3.11 requirement)"
  $PTCA_PIP install --user --ignore-requires-python \
      'pandas<3.0' \
      'datasets<3.0' \
      'peft'

  echo "[env] Step 2/2: verify"
  $PTCA_PY -c "
import sys
sys.path.insert(0, '$USER_SITE')
import transformers, datasets, peft, accelerate
v = transformers.__version__
assert v == '4.51.3', f'FATAL: transformers {v}'
print('  transformers', v)
print('  accelerate  ', accelerate.__version__)
print('  datasets    ', datasets.__version__)
print('  peft        ', peft.__version__)
print('  OK')
"

else
  # =====================================================================
  # Python 3.8 path — tokenizers pinned, version gate patched
  # =====================================================================
  echo "[env] Python $PTCA_PYVER: Python 3.8 compatibility path"

  echo "[env] Step 1/5: tokenizers==0.20.3 (last cp38 binary wheel)"
  $PTCA_PIP install --user --only-binary :all: --no-deps 'tokenizers==0.20.3' 2>&1 | tail -3

  echo "[env] Step 2/5: huggingface-hub + safetensors + regex"
  $PTCA_PIP install --user --only-binary :all: --ignore-requires-python \
      'huggingface-hub>=0.30.0,<1.0' \
      'safetensors>=0.4.3' \
      'regex!=2019.12.17' \
      2>&1 | tail -3

  echo "[env] Step 3/5: transformers==4.51.3 (--no-deps keeps tokenizers==0.20.3)"
  $PTCA_PIP install --user --no-deps --ignore-requires-python 'transformers==4.51.3' 2>&1 | tail -3

  echo "[env] Step 4/5: patch runtime tokenizers version gate in dependency_versions_check.py"
  # transformers 4.51.3 enforces tokenizers>=0.21 at import time via require_version_core().
  # On Python 3.8 we can't get tokenizers>=0.21 (no cp38 wheel), but 0.20.3 works at runtime.
  # Patch: skip the tokenizers check when running on Python < 3.9.
  $PTCA_PY - <<'PYEOF'
import sys, pathlib

dep_check = pathlib.Path(
    f"/tmp/ptca_user/lib/python{sys.version_info.major}.{sys.version_info.minor}"
    "/site-packages/transformers/dependency_versions_check.py"
)
if not dep_check.exists():
    print(f"  File not found: {dep_check} — skipping patch")
    sys.exit(0)

content = dep_check.read_text(encoding="utf-8")
MARKER = "# py38-tokenizers-patch"
if MARKER in content:
    print("  Already patched")
    sys.exit(0)

old = "require_version_core(deps[pkg])"
new = (
    'if pkg == "tokenizers" and sys.version_info < (3, 9): '
    'continue  # py38-tokenizers-patch\n        require_version_core(deps[pkg])'
)
if old not in content:
    print(f"  WARNING: expected pattern not found in {dep_check}")
    sys.exit(0)

dep_check.write_text(content.replace(old, new), encoding="utf-8")
print(f"  Patched: {dep_check}")
PYEOF

  echo "[env] Step 4b: fix Python 3.8 ABCMeta subscript in transformers/utils/*.py"
  PATCHED=0
  for F in $USER_SITE/transformers/utils/*.py; do
    [ -f "$F" ] || continue
    grep -q 'Iterable\[\|Sequence\[\|Callable\[\|Iterator\[\|Generator\[' "$F" 2>/dev/null || continue
    head -1 "$F" | grep -q '__future__' && continue
    sed -i '1s/^/from __future__ import annotations\n/' "$F"
    echo "    Patched: $(basename $F)"
    PATCHED=$((PATCHED+1))
  done
  [ "$PATCHED" -eq 0 ] && echo "    No patches needed"

  echo "[env] Step 4c: verify transformers import"
  $PTCA_PY -c "
import sys
sys.path.insert(0, '$USER_SITE')
import transformers
v = transformers.__version__
print('  transformers', v, transformers.__file__)
assert v == '4.51.3', f'FATAL: transformers {v} != 4.51.3'
print('  OK')
"

  echo "[env] Step 5/5: datasets + peft"
  $PTCA_PIP install --user --only-binary :all: --ignore-requires-python datasets peft 2>&1 | tail -3

fi

# -----------------------------------------------------------------------
# Prepend user site-packages (both paths)
# -----------------------------------------------------------------------
export PYTHONPATH=$USER_SITE:${PYTHONPATH:-}
echo "[env] PYTHONPATH: $USER_SITE"
echo "[env] setup_env.sh complete."
