"""
Convenience re-exports for :mod:`pyqed.mps`.

This package contains both legacy MPS/DMRG code and newer symmetry-adapted
non-Abelian prototypes.  Some of the legacy modules depend on optional heavy
dependencies (notably SciPy).  To make ``import pyqed.mps.nonabelian`` usable in
minimal environments (and in CI jobs that only exercise SU(2) code), we guard
those imports here.
"""

from __future__ import annotations

# Lightweight symmetry utilities are safe to export.
from .su2 import (  # noqa: F401
    SU2Irrep,
    SpinChargeSector,
    SpatialOrbitalSite,
    SpinOrbitalSite,
    fuse_irreps,
    fuse_charge_spin_sectors,
)
from .symmetry import (  # noqa: F401
    Sector,
    AbelianSector,
    QN,
    SymmetryManager,
    is_sector_like,
    zero_like_sector,
)
from .umps import UniformCanonicalForm, UniformMPS, UMPS  # noqa: F401

# Non-Abelian prototype exports (kept available even without SciPy).
from .nonabelian import (  # noqa: F401
    NonabelianTensor,
    PhysicalLeg,
    SiteOperator,
    MPO,
    AutoMPO,
    identity_operator,
    compose_site_operators,
    physical_leg_from_spatial_orbital,
    spatial_identity,
    spatial_number,
    spatial_number_up,
    spatial_number_down,
    spatial_double_occupancy,
    spatial_spin_square,
    spatial_projector,
    spatial_parity,
    spatial_annihilate_up,
    spatial_create_up,
    spatial_annihilate_down,
    spatial_create_down,
    add_spatial_one_body_terms,
    build_spatial_one_body_reduced_mpo,
    add_spatial_spinfree_eri_terms,
    build_spatial_spinfree_eri_mpo,
    add_spatial_density_terms,
    build_spatial_density_mpo,
    add_spatial_hubbard_terms,
    build_hubbard_mpo,
    build_spatial_hubbard_mpo,
    spatial_target_sector,
    half_filled_singlet_sector,
    build_random_spatial_mps,
    build_product_spatial_mps,
    FusionLeg,
    FusionEdge,
    FusionPipe,
    FusionPipeEntry,
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
    tensordot,
    merge_mps_sites,
    combine_legs,
    split_legs,
    recouple_fused_leg,
    svd_two_site,
    left_canonical_error,
    right_canonical_error,
    left_canonicalize_sites,
    right_canonicalize_sites,
    mixed_canonicalize_sites,
    LocalOperator,
    TwoSiteEffectiveH,
    pack_two_site_state,
    unpack_two_site_state,
    solve_local_two_site,
    DenseEnvironmentChain,
    DenseEnvironmentSweep,
    BlockSparseEnvironmentChain,
    BlockSparseEnvironmentSweep,
    build_dense_bond_operator,
    build_block_sparse_bond_operator,
    two_site_update,
    sweep_once,
    run_sweeps,
    SweepDriver,
    Driver,
)

# Legacy exports: available only when optional deps exist.
try:  # pragma: no cover
    from .mps import *  # noqa: F401,F403
    from .dmrg import (  # noqa: F401
        DMRG,
        dmrg_matvec_options,
        resolve_abelian_matvec_options,
    )
    from .tdmps import TDMPS  # noqa: F401
    from .tdvp import one_site_tdvp_step, two_site_tdvp_step  # noqa: F401
    from .first_quantization import Chain, FiniteDimLocalBasis  # noqa: F401
except (ModuleNotFoundError, ImportError, OSError, TimeoutError):
    pass

# Keep the public uniform-MPS name pinned to the NumPy-only implementation even
# when legacy wildcard exports are available.
from .umps import UniformCanonicalForm, UniformMPS, UMPS  # noqa: F401,E402
