#!/usr/bin/env bash
set -euo pipefail

: "${PYQED_ROOT:?set PYQED_ROOT to the uploaded pyqed source root}"
python=${PYTHON:-python}

cd "$PYQED_ROOT"
PYQED_BUILD_EXTENSIONS=1 \
PYQED_EXTENSION_GROUPS=letta \
    "$python" setup.py build_ext --inplace

PYTHONPATH="$PYQED_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$python" -c \
    'import importlib; [importlib.import_module("pyqed.letta." + name) for name in ("_conditional_gauge_cpp", "_copy_einsum_cpp", "_physical_blocks_cpp", "_support_kernels_cpp")]; print("all four native LETTA extensions loaded")'
