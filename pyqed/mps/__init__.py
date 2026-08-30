"""Matrix-product-state algorithms over the canonical :mod:`pyqed.tn` types.

The package root is intentionally a small, explicit, lazily loaded façade.
Backend-specific representations live in their own namespaces, notably
``pyqed.mps.nonabelian``.
"""

from __future__ import annotations

from importlib import import_module

from pyqed.tn import MPO


_EXPORTS = {
    # Finite dense MPS and conversions.
    "MPS": ("pyqed.mps.mps", "MPS"),
    "dense_to_symmetric": ("pyqed.mps.mps", "dense_to_symmetric"),
    "dense_to_symmetric_mpo": ("pyqed.mps.mps", "dense_to_symmetric_mpo"),
    "expect_mps": ("pyqed.mps.mps", "expect_mps"),
    "fDMRG_1site_GS_OBC": ("pyqed.mps.mps", "fDMRG_1site_GS_OBC"),
    "symmetric_to_dense": ("pyqed.mps.mps", "symmetric_to_dense"),
    "two_site_dmrg": ("pyqed.mps.mps", "two_site_dmrg"),
    "one_site_dmrg3s": ("pyqed.mps._dmrg3s", "one_site_dmrg3s"),
    # Ground-state and time-evolution algorithms.
    "DMRG": ("pyqed.mps.dmrg", "DMRG"),
    "resolve_abelian_matvec_options": (
        "pyqed.mps.dmrg",
        "resolve_abelian_matvec_options",
    ),
    "TDMPS": ("pyqed.mps.tdmps", "TDMPS"),
    "TEBD": ("pyqed.mps.tebd", "TEBD"),
    "tebd": ("pyqed.mps.tebd", "tebd"),
    "SymmetricTDVP": ("pyqed.mps.tdvp", "SymmetricTDVP"),
    "block_sparse_one_site_tdvp_step": (
        "pyqed.mps.tdvp",
        "block_sparse_one_site_tdvp_step",
    ),
    "block_sparse_two_site_tdvp_step": (
        "pyqed.mps.tdvp",
        "block_sparse_two_site_tdvp_step",
    ),
    "one_site_tdvp_step": ("pyqed.mps.tdvp", "one_site_tdvp_step"),
    "two_site_tdvp_step": ("pyqed.mps.tdvp", "two_site_tdvp_step"),
    # Purification-based finite-temperature states.
    "PurifiedMPS": ("pyqed.mps.thermal", "PurifiedMPS"),
    "infinite_temperature_mps": (
        "pyqed.mps.thermal",
        "infinite_temperature_mps",
    ),
    "lift_physical_mpo": ("pyqed.mps.thermal", "lift_physical_mpo"),
    # Infinite, uniform, and continuous states.
    "UMPS": ("pyqed.mps.umps", "UMPS"),
    "UniformMPS": ("pyqed.mps.umps", "UniformMPS"),
    "InfiniteDMRG": ("pyqed.mps.idmrg", "InfiniteDMRG"),
    "iDMRG": ("pyqed.mps.idmrg", "iDMRG"),
    "factorize_nearest_neighbor_hamiltonian": (
        "pyqed.mps.idmrg",
        "factorize_nearest_neighbor_hamiltonian",
    ),
    "idmrg_nearest_neighbor": ("pyqed.mps.idmrg", "idmrg_nearest_neighbor"),
    "CMPS": ("pyqed.mps.cmps", "CMPS"),
    "ContinuousMPS": ("pyqed.mps.cmps", "ContinuousMPS"),
    "canonical_parameter_size": (
        "pyqed.mps.cmps",
        "canonical_parameter_size",
    ),
    "fit_exponential_kernel_nonlinear": (
        "pyqed.mps.cmps",
        "fit_exponential_kernel_nonlinear",
    ),
    "fit_exponential_kernel_prony": (
        "pyqed.mps.cmps",
        "fit_exponential_kernel_prony",
    ),
    "pack_canonical_parameters": (
        "pyqed.mps.cmps",
        "pack_canonical_parameters",
    ),
    "softened_yukawa_kernel": ("pyqed.mps.cmps", "softened_yukawa_kernel"),
    "unpack_canonical_parameters": (
        "pyqed.mps.cmps",
        "unpack_canonical_parameters",
    ),
    # CLETTA/continuum helpers retained as explicit algorithms.
    "apply_cletta_bra_insertion": (
        "pyqed.mps.cletta",
        "apply_cletta_bra_insertion",
    ),
    "apply_cletta_ket_insertion": (
        "pyqed.mps.cletta",
        "apply_cletta_ket_insertion",
    ),
    "apply_cletta_memory_hierarchy": (
        "pyqed.mps.cletta",
        "apply_cletta_memory_hierarchy",
    ),
    "apply_cletta_multimode_bra_insertion": (
        "pyqed.mps.cletta",
        "apply_cletta_multimode_bra_insertion",
    ),
    "apply_cletta_multimode_ket_insertion": (
        "pyqed.mps.cletta",
        "apply_cletta_multimode_ket_insertion",
    ),
    "apply_cletta_multimode_memory_hierarchy": (
        "pyqed.mps.cletta",
        "apply_cletta_multimode_memory_hierarchy",
    ),
    "apply_cletta_multimode_memory_hierarchy_adjoint": (
        "pyqed.mps.cletta",
        "apply_cletta_multimode_memory_hierarchy_adjoint",
    ),
    "cletta_bra_insertion_matrix": (
        "pyqed.mps.cletta",
        "cletta_bra_insertion_matrix",
    ),
    "cletta_ket_insertion_matrix": (
        "pyqed.mps.cletta",
        "cletta_ket_insertion_matrix",
    ),
    "cletta_memory_fock_keys": (
        "pyqed.mps.cletta",
        "cletta_memory_fock_keys",
    ),
    "cletta_memory_hierarchy_generator": (
        "pyqed.mps.cletta",
        "cletta_memory_hierarchy_generator",
    ),
    "cletta_memory_matrices": (
        "pyqed.mps.cletta",
        "cletta_memory_matrices",
    ),
    "cletta_multifield_memory_matrices": (
        "pyqed.mps.cletta",
        "cletta_multifield_memory_matrices",
    ),
    "cletta_multimode_bra_insertion_matrix": (
        "pyqed.mps.cletta",
        "cletta_multimode_bra_insertion_matrix",
    ),
    "cletta_multimode_hierarchy_generator": (
        "pyqed.mps.cletta",
        "cletta_multimode_hierarchy_generator",
    ),
    "cletta_multimode_hierarchy_sparse_generator": (
        "pyqed.mps.cletta",
        "cletta_multimode_hierarchy_sparse_generator",
    ),
    "cletta_multimode_ket_insertion_matrix": (
        "pyqed.mps.cletta",
        "cletta_multimode_ket_insertion_matrix",
    ),
    "cletta_multimode_memory_matrices": (
        "pyqed.mps.cletta",
        "cletta_multimode_memory_matrices",
    ),
    "hierarchy_blocks_to_matrix": (
        "pyqed.mps.cletta",
        "hierarchy_blocks_to_matrix",
    ),
    "commuting_cylinder_parameter_size": (
        "pyqed.mps.cylinder",
        "commuting_cylinder_parameter_size",
    ),
    "cylinder_density_mode_correlation": (
        "pyqed.mps.cylinder",
        "cylinder_density_mode_correlation",
    ),
    "cylinder_fixed_density_observables": (
        "pyqed.mps.cylinder",
        "cylinder_fixed_density_observables",
    ),
    "cylinder_static_structure_factor": (
        "pyqed.mps.cylinder",
        "cylinder_static_structure_factor",
    ),
    "optimize_cylinder_cletta": (
        "pyqed.mps.cylinder",
        "optimize_cylinder_cletta",
    ),
    "optimize_cylinder_cmps": (
        "pyqed.mps.cylinder",
        "optimize_cylinder_cmps",
    ),
    "pack_commuting_cylinder_parameters": (
        "pyqed.mps.cylinder",
        "pack_commuting_cylinder_parameters",
    ),
    "softened_yukawa_cylinder_fourier": (
        "pyqed.mps.cylinder",
        "softened_yukawa_cylinder_fourier",
    ),
    "unpack_commuting_cylinder_parameters": (
        "pyqed.mps.cylinder",
        "unpack_commuting_cylinder_parameters",
    ),
    "ExponentialLuttingerModel": (
        "pyqed.mps.luttinger",
        "ExponentialLuttingerModel",
    ),
    "GaussianLuttingerCLETTA": (
        "pyqed.mps.luttinger",
        "GaussianLuttingerCLETTA",
    ),
    "cmps_luttinger_energy_shift_density": (
        "pyqed.mps.luttinger",
        "cmps_luttinger_energy_shift_density",
    ),
    "cmps_luttinger_parameter": (
        "pyqed.mps.luttinger",
        "cmps_luttinger_parameter",
    ),
    "cmps_luttinger_spectra": (
        "pyqed.mps.luttinger",
        "cmps_luttinger_spectra",
    ),
    "optimize_luttinger_cletta": (
        "pyqed.mps.luttinger",
        "optimize_luttinger_cletta",
    ),
}

_MODULE_EXPORTS = {
    "cpp_davidson": "pyqed.mps.cpp_davidson",
    "packed_cython": "pyqed.mps.packed_cython",
    "tdvp_cpp": "pyqed.mps.tdvp_cpp",
}


def __getattr__(name):
    if name in _MODULE_EXPORTS:
        value = import_module(_MODULE_EXPORTS[name])
    else:
        try:
            module_name, attribute = _EXPORTS[name]
        except KeyError as error:
            raise AttributeError(name) from error
        value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)


__all__ = sorted({"MPO", *_EXPORTS, *_MODULE_EXPORTS})
