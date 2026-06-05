"""Optional Cython kernels for Abelian qchem NARG setup loops."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


CYTHON_AVAILABLE = False
collect_integral_terms = None
precompute_integral_terms = None


def _set_kernels(module):
    global CYTHON_AVAILABLE, collect_integral_terms, precompute_integral_terms
    collect_integral_terms = module.collect_integral_terms
    precompute_integral_terms = module.precompute_integral_terms
    CYTHON_AVAILABLE = True


if os.environ.get("NARG_DISABLE_CYTHON", "0") != "1":
    try:
        from . import abelian_cython_kernels as _kernels

        _set_kernels(_kernels)
    except Exception:
        if os.environ.get("NARG_USE_CYTHON", "0") == "1":
            try:
                import pyximport

                build_dir = Path(os.environ.get("NARG_CYTHON_BUILD", "/private/tmp/narg-cython"))
                build_dir.mkdir(parents=True, exist_ok=True)
                pyximport.install(
                    build_dir=str(build_dir),
                    language_level=3,
                    setup_args={"include_dirs": np.get_include()},
                )
                from . import abelian_cython_kernels as _kernels

                _set_kernels(_kernels)
            except Exception:
                CYTHON_AVAILABLE = False
                collect_integral_terms = None
                precompute_integral_terms = None
