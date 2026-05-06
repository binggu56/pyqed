#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-Abelian tensor foundations for future symmetry-adapted MPS/DMRG.

This package intentionally keeps a small, explicit surface:

- :mod:`tensor` stores reduced block tensors and fusion-edge metadata
- :mod:`contraction` implements fixed-layout reduced contractions
- :mod:`linalg` stores reusable reduced projection/SVD/truncation helpers
- :mod:`decompose` provides the first reduced two-site SVD/truncation helper

Short names are preferred inside this package.
"""

from .tensor import FusionLeg, FusionEdge, FusionPipe, FusionPipeEntry, NonabelianTensor
from .mps import MPS
from .coupling import (
    CouplingChannel,
    ReducedBondSpace,
    two_m_values,
    clebsch_gordan,
    clebsch_gordan_tensor,
    couple_two_sectors_matrix,
    enumerate_sector_couplings,
    fuse_charge_spin_sector_sequence,
    normalize_coupling_scheme,
    reduced_bond_space,
    recoupling_matrix,
)
from .mpo import (
    PhysicalLeg,
    SiteOperator,
    MPO,
    IrreducibleChannelTerm,
    IrreducibleMPO,
    RankCoupledChannelTerm,
    RankCoupledMPO,
)
from .builder import AutoMPO, identity_operator
from .operators import (
    AdjointReducedTensorOperator,
    ReducedTensorOperator,
    compose_site_operators,
    physical_leg_from_spatial_orbital,
    reduced_physical_leg_from_spatial_orbital,
    spatial_identity,
    reduced_spatial_identity,
    spatial_number,
    reduced_spatial_number,
    spatial_number_up,
    spatial_number_down,
    spatial_double_occupancy,
    reduced_spatial_double_occupancy,
    spatial_spin_square,
    spatial_projector,
    spatial_parity,
    reduced_spatial_parity,
    reduced_spatial_fermion_annihilation,
    spatial_annihilate_up,
    spatial_create_up,
    spatial_annihilate_down,
    spatial_create_down,
)
from .models import (
    add_spatial_density_terms,
    build_spatial_density_mpo,
    add_spatial_hubbard_terms,
    build_hubbard_mpo,
    build_spatial_hubbard_mpo,
    add_spatial_qchem_terms,
    build_spatial_qchem_mpo,
)
from .states import (
    spatial_target_sector,
    half_filled_singlet_sector,
    build_random_spatial_mps,
    build_random_reduced_spatial_mps,
    build_product_spatial_mps,
    build_reduced_product_spatial_mps,
    build_product_state,
    build_spin_spatial_mps,
)
from .contraction import (
    tensordot,
    merge_mps_sites,
    combine_legs,
    split_legs,
    recouple_fused_leg,
)
from .linalg import (
    ReducedProjectedSector,
    ReducedProjectedSVD,
    ReducedTruncation,
    normalize_max_bond_mode,
    sector_state_weight,
    select_kept_singular_values,
    project_reduced_sector,
    truncate_reduced_svds,
)
from .decompose import (
    svd_two_site,
)
from .canonical import (
    left_canonical_error,
    right_canonical_error,
    left_cg_canonical_error,
    right_cg_canonical_error,
    mixed_cg_canonical_errors,
    left_canonicalize_sites,
    right_canonicalize_sites,
    mixed_canonicalize_sites,
)
from .solver import (
    LocalOperator,
    TwoSiteEffectiveH,
    ReducedStateLayout,
    ReducedStateVector,
    ReducedDiagonalPreconditioner,
    PackedBlockPreconditioner,
    pack_two_site_state,
    unpack_two_site_state,
    solve_local_two_site,
)
from .environment import (
    DenseEnvironmentChain,
    DenseEnvironmentSweep,
    BlockSparseEnvironmentChain,
    BlockSparseEnvironmentSweep,
    build_dense_bond_operator,
    build_block_sparse_bond_operator,
    contract_chain_expectation,
)
from .update import (
    two_site_update,
)
from .sweep import (
    sweep_once,
    run_sweeps,
)
from .driver import (
    SweepDriver,
    Driver,
)
