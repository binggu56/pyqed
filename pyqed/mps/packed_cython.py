"""Optional Cython kernels for packed Abelian DMRG matvecs."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


CYTHON_AVAILABLE = False
batched_matrix_chain_e_a_accum = None
batched_matrix_chain_r_w_accum = None
batched_matrix_chain_t2_w_accum = None
batched_matrix_chain_t3_f_accum = None
packed_left_boundary_block = None
packed_right_boundary_block = None
packed_left_identity_boundary_block = None
packed_right_identity_boundary_block = None
run_batched_matrix_chain = None
run_batched_matrix_chain_arenas = None
sparse_coo_matvec = None
sparse_csr_matvec = None
direct_operator_entry_coo = None
direct_operator_entry_sparse_product_coo = None
direct_operator_entries_coo = None
direct_operator_entries_csr = None
direct_operator_entries_csr_np_extract = None
csr_dense_lookup = None
direct_operator_entries_csr_refill = None
direct_operator_entries_matvec = None
direct_operator_groups_matvec = None
direct_operator_groups_dense_blocks = None
direct_operator_block_matrices_matvec = None
direct_operator_block_sparse_matvec = None


def _set_kernels(module):
    global CYTHON_AVAILABLE
    global batched_matrix_chain_e_a_accum
    global batched_matrix_chain_r_w_accum
    global batched_matrix_chain_t2_w_accum
    global batched_matrix_chain_t3_f_accum
    global packed_left_boundary_block
    global packed_right_boundary_block
    global packed_left_identity_boundary_block
    global packed_right_identity_boundary_block
    global run_batched_matrix_chain
    global run_batched_matrix_chain_arenas
    global sparse_coo_matvec
    global sparse_csr_matvec
    global direct_operator_entry_coo
    global direct_operator_entry_sparse_product_coo
    global direct_operator_entries_coo
    global direct_operator_entries_csr
    global direct_operator_entries_csr_np_extract
    global csr_dense_lookup
    global direct_operator_entries_csr_refill
    global direct_operator_entries_matvec
    global direct_operator_groups_matvec
    global direct_operator_groups_dense_blocks
    global direct_operator_block_matrices_matvec
    global direct_operator_block_sparse_matvec

    batched_matrix_chain_e_a_accum = module.batched_matrix_chain_e_a_accum
    batched_matrix_chain_r_w_accum = module.batched_matrix_chain_r_w_accum
    batched_matrix_chain_t2_w_accum = module.batched_matrix_chain_t2_w_accum
    batched_matrix_chain_t3_f_accum = module.batched_matrix_chain_t3_f_accum
    packed_left_boundary_block = module.packed_left_boundary_block
    packed_right_boundary_block = module.packed_right_boundary_block
    packed_left_identity_boundary_block = module.packed_left_identity_boundary_block
    packed_right_identity_boundary_block = module.packed_right_identity_boundary_block
    run_batched_matrix_chain = module.run_batched_matrix_chain
    run_batched_matrix_chain_arenas = module.run_batched_matrix_chain_arenas
    sparse_coo_matvec = module.sparse_coo_matvec
    sparse_csr_matvec = module.sparse_csr_matvec
    direct_operator_entry_coo = module.direct_operator_entry_coo
    direct_operator_entry_sparse_product_coo = module.direct_operator_entry_sparse_product_coo
    direct_operator_entries_coo = module.direct_operator_entries_coo
    direct_operator_entries_csr = module.direct_operator_entries_csr
    direct_operator_entries_csr_np_extract = module.direct_operator_entries_csr_np_extract
    csr_dense_lookup = module.csr_dense_lookup
    direct_operator_entries_csr_refill = module.direct_operator_entries_csr_refill
    direct_operator_entries_matvec = module.direct_operator_entries_matvec
    direct_operator_groups_matvec = module.direct_operator_groups_matvec
    direct_operator_groups_dense_blocks = module.direct_operator_groups_dense_blocks
    direct_operator_block_matrices_matvec = module.direct_operator_block_matrices_matvec
    direct_operator_block_sparse_matvec = module.direct_operator_block_sparse_matvec
    CYTHON_AVAILABLE = True


if os.environ.get("PYQED_MPS_DISABLE_CYTHON", "0") != "1":
    try:
        from . import packed_cython_kernels as _kernels

        _set_kernels(_kernels)
    except Exception:
        use_pyximport = os.environ.get(
            "PYQED_MPS_USE_CYTHON",
            os.environ.get("PYQED_MPS_AUTO_CYTHON", "0"),
        )
        if str(use_pyximport).strip().lower() not in {"0", "false", "no", "off"}:
            try:
                import pyximport

                build_dir = Path(
                    os.environ.get(
                        "PYQED_MPS_CYTHON_BUILD",
                        "/private/tmp/pyqed-mps-cython",
                    )
                )
                build_dir.mkdir(parents=True, exist_ok=True)
                pyximport.install(
                    build_dir=str(build_dir),
                    language_level=3,
                    setup_args={"include_dirs": np.get_include()},
                )
                from . import packed_cython_kernels as _kernels

                _set_kernels(_kernels)
            except Exception:
                CYTHON_AVAILABLE = False
                batched_matrix_chain_e_a_accum = None
                batched_matrix_chain_r_w_accum = None
                batched_matrix_chain_t2_w_accum = None
                batched_matrix_chain_t3_f_accum = None
                packed_left_boundary_block = None
                packed_right_boundary_block = None
                packed_left_identity_boundary_block = None
                packed_right_identity_boundary_block = None
                run_batched_matrix_chain = None
                run_batched_matrix_chain_arenas = None
                sparse_coo_matvec = None
                sparse_csr_matvec = None
                direct_operator_entry_coo = None
                direct_operator_entry_sparse_product_coo = None
                direct_operator_entries_coo = None
                direct_operator_entries_csr = None
                direct_operator_entries_csr_np_extract = None
                csr_dense_lookup = None
                direct_operator_entries_csr_refill = None
                direct_operator_entries_matvec = None
                direct_operator_groups_matvec = None
                direct_operator_groups_dense_blocks = None
                direct_operator_block_matrices_matvec = None
                direct_operator_block_sparse_matvec = None
