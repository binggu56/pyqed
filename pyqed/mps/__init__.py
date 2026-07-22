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
from .idmrg import (  # noqa: F401
    InfiniteDMRG,
    NearestNeighborTerms,
    factorize_nearest_neighbor_hamiltonian,
    idmrg_nearest_neighbor,
    iDMRG,
    iDMRGBlock,
    iDMRGStep,
)
from .cmps import (  # noqa: F401
    CMPS,
    ContinuousMPS,
    canonical_parameter_size,
    pack_canonical_parameters,
    skew_pairs,
    unpack_canonical_parameters,
)

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
    from .tdvp import (  # noqa: F401
        SymmetricTDVP,
        block_sparse_one_site_tdvp_step,
        one_site_tdvp_step,
        spatial_fermion_number_sz_sectors,
        two_site_tdvp_step,
    )
    from .first_quantization import Chain, FiniteDimLocalBasis  # noqa: F401
except (ModuleNotFoundError, ImportError, OSError, TimeoutError):
    pass

# Keep the public uniform-MPS name pinned to the NumPy-only implementation even
# when legacy wildcard exports are available.
from .umps import UniformCanonicalForm, UniformMPS, UMPS  # noqa: F401,E402
from .idmrg import (  # noqa: F401,E402
    InfiniteDMRG,
    NearestNeighborTerms,
    factorize_nearest_neighbor_hamiltonian,
    idmrg_nearest_neighbor,
    iDMRG,
    iDMRGBlock,
    iDMRGStep,
)
from .cmps import (  # noqa: F401,E402
    CMPS,
    ContinuousMPS,
    canonical_parameter_size,
    fit_exponential_kernel_nonlinear,
    fit_exponential_kernel_prony,
    pack_canonical_parameters,
    skew_pairs,
    softened_yukawa_kernel,
    unpack_canonical_parameters,
)
from .cletta import (  # noqa: F401,E402
    apply_cletta_bra_insertion,
    apply_cletta_ket_insertion,
    apply_cletta_memory_hierarchy,
    apply_cletta_multimode_bra_insertion,
    apply_cletta_multimode_ket_insertion,
    apply_cletta_multimode_memory_hierarchy,
    apply_cletta_multimode_memory_hierarchy_adjoint,
    cletta_bra_insertion_matrix,
    cletta_ket_insertion_matrix,
    cletta_memory_fock_keys,
    cletta_memory_hierarchy_generator,
    cletta_memory_matrices,
    cletta_multimode_bra_insertion_matrix,
    cletta_multimode_hierarchy_generator,
    cletta_multimode_hierarchy_sparse_generator,
    cletta_multimode_ket_insertion_matrix,
    cletta_multimode_memory_matrices,
    cletta_multifield_memory_matrices,
    hierarchy_blocks_to_matrix,
    matrix_to_hierarchy_blocks,
)
from .cylinder import (  # noqa: F401,E402
    commuting_cylinder_parameter_size,
    cylinder_density_mode_correlation,
    cylinder_fixed_density_observables,
    cylinder_static_structure_factor,
    optimize_cylinder_cletta,
    optimize_cylinder_cmps,
    pack_commuting_cylinder_parameters,
    softened_yukawa_cylinder_fourier,
    unpack_commuting_cylinder_parameters,
)
from .luttinger import (  # noqa: F401,E402
    ExponentialLuttingerModel,
    GaussianLuttingerCLETTA,
    cmps_luttinger_energy_shift_density,
    cmps_luttinger_parameter,
    cmps_luttinger_spectra,
    optimize_luttinger_cletta,
)
from .pip_pairing import (  # noqa: F401,E402
    ContinuumPipPairingModel,
    ThermodynamicPipBCS,
    ThermodynamicPipCLETTA,
    ExactOnePairPipState,
    OneScalePipCLETTA,
    TwoPairPipCLETTA,
    TwoPairPipD3CLETTA,
)
from .bose_gas_2d import (  # noqa: F401,E402
    D2M1HierarchicalCLETTA2D,
    D2M1NestedCLETTA2D,
    DiluteBoseGas2D,
    GaussianPotentialBoseGas2D,
    HierarchicalShellContraction,
    RankOneDensityTransferChannel2D,
    fixed_density_gns_nested_hletta_state,
    fixed_density_nested_hletta_state,
    optimize_condensate_gns_hletta_fixed_density,
    optimize_condensate_nested_hletta_fixed_density,
    optimize_nested_hletta_fixed_density,
)
