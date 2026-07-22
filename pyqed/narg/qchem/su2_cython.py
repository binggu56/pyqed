"""Optional Cython kernels for SU(2)-NARG numeric inner loops.

The kernels are opt-in while the reduced SU(2) path is still experimental; the
angular contractions prepack reusable index arrays when this backend is enabled.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


CYTHON_AVAILABLE = False
accumulate_bilinear = None
product_tensor_estimate_entries = None
product_tensor_pair_entries = None
product_tensor_group_indices = None
scalar_product_pair_entries = None


if (
    os.environ.get("SU2_NARG_USE_CYTHON", "0") == "1"
    and os.environ.get("SU2_NARG_DISABLE_CYTHON", "0") != "1"
):
    try:
        import pyximport

        build_dir = Path(os.environ.get("SU2_NARG_CYTHON_BUILD", "/private/tmp/su2-narg-cython"))
        build_dir.mkdir(parents=True, exist_ok=True)
        pyximport.install(
            build_dir=str(build_dir),
            language_level=3,
            setup_args={"include_dirs": np.get_include()},
        )
        from .su2_cython_kernels import accumulate_bilinear as _accumulate_bilinear
        from .su2_cython_kernels import (
            product_tensor_estimate_entries as _product_tensor_estimate_entries,
        )
        from .su2_cython_kernels import (
            product_tensor_pair_entries as _product_tensor_pair_entries,
        )
        from .su2_cython_kernels import (
            product_tensor_group_indices as _product_tensor_group_indices,
        )
        from .su2_cython_kernels import (
            scalar_product_pair_entries as _scalar_product_pair_entries,
        )

        accumulate_bilinear = _accumulate_bilinear
        product_tensor_estimate_entries = _product_tensor_estimate_entries
        product_tensor_pair_entries = _product_tensor_pair_entries
        product_tensor_group_indices = _product_tensor_group_indices
        scalar_product_pair_entries = _scalar_product_pair_entries
        CYTHON_AVAILABLE = True
    except Exception:
        CYTHON_AVAILABLE = False
        accumulate_bilinear = None
        product_tensor_estimate_entries = None
        product_tensor_pair_entries = None
        product_tensor_group_indices = None
        scalar_product_pair_entries = None
