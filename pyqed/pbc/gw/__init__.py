"""Periodic GW/BSE entry points."""

from importlib import import_module

_LAZY_IMPORTS = {
    "GammaPBCSCFAdapter": ("pyqed.pbc.gw.adapter", "GammaPBCSCFAdapter"),
    "KPointSCFAdapter": ("pyqed.pbc.gw.adapter", "KPointSCFAdapter"),
    "PeriodicBSEBlock": ("pyqed.pbc.gw.bse", "PeriodicBSEBlock"),
    "PeriodicBSEResult": ("pyqed.pbc.gw.bse", "PeriodicBSEResult"),
    "PeriodicBSESpectrum": ("pyqed.pbc.gw.bse", "PeriodicBSESpectrum"),
    "periodic_bse": ("pyqed.pbc.gw.bse", "periodic_bse"),
    "periodic_bse_matrices": ("pyqed.pbc.gw.bse", "periodic_bse_matrices"),
    "periodic_bse_spectrum": ("pyqed.pbc.gw.bse", "periodic_bse_spectrum"),
    "periodic_tda": ("pyqed.pbc.gw.bse", "periodic_tda"),
    "periodic_tda_spectrum": ("pyqed.pbc.gw.bse", "periodic_tda_spectrum"),
    "PeriodicTDAChannel": (
        "pyqed.pbc.gw.bse_operator",
        "PeriodicTDAChannel",
    ),
    "PeriodicTDABlockGroup": (
        "pyqed.pbc.gw.bse_operator",
        "PeriodicTDABlockGroup",
    ),
    "PeriodicTDAOperator": (
        "pyqed.pbc.gw.bse_operator",
        "PeriodicTDAOperator",
    ),
    "periodic_tda_operator": (
        "pyqed.pbc.gw.bse_operator",
        "periodic_tda_operator",
    ),
    "ProjectedTDAContinuum": (
        "pyqed.pbc.gw.embedding",
        "ProjectedTDAContinuum",
    ),
    "ExcitonPhononChannel": (
        "pyqed.pbc.gw.embedding",
        "ExcitonPhononChannel",
    ),
    "ExcitonPhononContinuum": (
        "pyqed.pbc.gw.embedding",
        "ExcitonPhononContinuum",
    ),
    "ExcitonPhononCoupling": (
        "pyqed.pbc.gw.embedding",
        "ExcitonPhononCoupling",
    ),
    "TotalMomentumSector": (
        "pyqed.pbc.gw.embedding",
        "TotalMomentumSector",
    ),
    "bose_occupation": (
        "pyqed.pbc.gw.embedding",
        "bose_occupation",
    ),
    "PeriodicTDAElectronPhononDerivative": (
        "pyqed.pbc.gw.electron_phonon",
        "PeriodicTDAElectronPhononDerivative",
    ),
    "analytic_tda_electron_phonon_coupling": (
        "pyqed.pbc.gw.electron_phonon",
        "analytic_tda_electron_phonon_coupling",
    ),
    "commensurate_tda_electron_phonon_coupling": (
        "pyqed.pbc.gw.electron_phonon",
        "commensurate_tda_electron_phonon_coupling",
    ),
    "commensurate_gdf_bare_tda_kernel_derivative": (
        "pyqed.pbc.gw.electron_phonon",
        "commensurate_gdf_bare_tda_kernel_derivative",
    ),
    "CommensurateGDFScreenedInteractionDerivative": (
        "pyqed.pbc.gw.electron_phonon",
        "CommensurateGDFScreenedInteractionDerivative",
    ),
    "GDFQDerivativeFactors": (
        "pyqed.pbc.gw.electron_phonon",
        "GDFQDerivativeFactors",
    ),
    "gdf_q_derivative_factors": (
        "pyqed.pbc.gw.electron_phonon",
        "gdf_q_derivative_factors",
    ),
    "commensurate_gdf_screened_tda_kernel_derivative": (
        "pyqed.pbc.gw.electron_phonon",
        "commensurate_gdf_screened_tda_kernel_derivative",
    ),
    "validate_commensurate_gdf_screened_tda_kernel_derivative": (
        "pyqed.pbc.gw.derivative_validation",
        "validate_commensurate_gdf_screened_tda_kernel_derivative",
    ),
    "electron_phonon_mo_couplings": (
        "pyqed.pbc.gw.electron_phonon",
        "electron_phonon_mo_couplings",
    ),
    "gamma_tda_electron_phonon_coupling": (
        "pyqed.pbc.gw.electron_phonon",
        "gamma_tda_electron_phonon_coupling",
    ),
    "gamma_gdf_bare_tda_kernel_derivative": (
        "pyqed.pbc.gw.electron_phonon",
        "gamma_gdf_bare_tda_kernel_derivative",
    ),
    "GammaGDFScreenedInteractionDerivative": (
        "pyqed.pbc.gw.electron_phonon",
        "GammaGDFScreenedInteractionDerivative",
    ),
    "gamma_gdf_screened_interaction_derivative": (
        "pyqed.pbc.gw.electron_phonon",
        "gamma_gdf_screened_interaction_derivative",
    ),
    "gamma_gdf_screened_tda_kernel_derivative": (
        "pyqed.pbc.gw.electron_phonon",
        "gamma_gdf_screened_tda_kernel_derivative",
    ),
    "gamma_gdf_diagonal_self_energy_derivative": (
        "pyqed.pbc.gw.electron_phonon",
        "gamma_gdf_diagonal_self_energy_derivative",
    ),
    "gamma_gdf_g0w0_energy_derivative": (
        "pyqed.pbc.gw.electron_phonon",
        "gamma_gdf_g0w0_energy_derivative",
    ),
    "COULOMB_BACKGROUND": ("pyqed.pbc.gw.coulomb", "COULOMB_BACKGROUND"),
    "FULL_EWALD": ("pyqed.pbc.gw.coulomb", "FULL_EWALD"),
    "GDF": ("pyqed.pbc.gw.coulomb", "GDF"),
    "PYSCF_GDF": ("pyqed.pbc.gw.coulomb", "PYSCF_GDF"),
    "RECIPROCAL_EWALD_LR": ("pyqed.pbc.gw.coulomb", "RECIPROCAL_EWALD_LR"),
    "SHORT_RANGE_EWALD": ("pyqed.pbc.gw.coulomb", "SHORT_RANGE_EWALD"),
    "SUPPORTED_DENSE_GAMMA_COULOMB_COMPONENTS": (
        "pyqed.pbc.gw.coulomb",
        "SUPPORTED_DENSE_GAMMA_COULOMB_COMPONENTS",
    ),
    "SUPPORTED_PERIODIC_COULOMB_COMPONENTS": (
        "pyqed.pbc.gw.coulomb",
        "SUPPORTED_PERIODIC_COULOMB_COMPONENTS",
    ),
    "is_full_ewald_component": ("pyqed.pbc.gw.coulomb", "is_full_ewald_component"),
    "is_gdf_component": ("pyqed.pbc.gw.coulomb", "is_gdf_component"),
    "is_pyscf_gdf_component": ("pyqed.pbc.gw.coulomb", "is_pyscf_gdf_component"),
    "normalize_coulomb_component": ("pyqed.pbc.gw.coulomb", "normalize_coulomb_component"),
    "DiagonalFiniteSizeCorrection": (
        "pyqed.pbc.gw.finite_size",
        "DiagonalFiniteSizeCorrection",
    ),
    "cell_volume": ("pyqed.pbc.gw.finite_size", "cell_volume"),
    "bloch_ao_gradient_matrices": (
        "pyqed.pbc.gw.finite_size",
        "bloch_ao_gradient_matrices",
    ),
    "diagonal_finite_size_correction": (
        "pyqed.pbc.gw.finite_size",
        "diagonal_finite_size_correction",
    ),
    "finite_size_q_vector": ("pyqed.pbc.gw.finite_size", "finite_size_q_vector"),
    "ReciprocalOrbitalPairFactors": (
        "pyqed.pbc.gw.integrals",
        "ReciprocalOrbitalPairFactors",
    ),
    "ReciprocalTransitionFactors": (
        "pyqed.pbc.gw.integrals",
        "ReciprocalTransitionFactors",
    ),
    "GDFTransitionFactors": ("pyqed.pbc.gw.integrals", "GDFTransitionFactors"),
    "PySCFGDFTransitionFactors": (
        "pyqed.pbc.gw.integrals",
        "PySCFGDFTransitionFactors",
    ),
    "dense_gamma_orbital_pair_coupling": (
        "pyqed.pbc.gw.integrals",
        "dense_gamma_orbital_pair_coupling",
    ),
    "dense_gamma_orbital_pair_metric": (
        "pyqed.pbc.gw.integrals",
        "dense_gamma_orbital_pair_metric",
    ),
    "dense_gamma_transition_metric": (
        "pyqed.pbc.gw.integrals",
        "dense_gamma_transition_metric",
    ),
    "full_ewald_orbital_pair_coupling": (
        "pyqed.pbc.gw.integrals",
        "full_ewald_orbital_pair_coupling",
    ),
    "full_ewald_orbital_pair_metric": (
        "pyqed.pbc.gw.integrals",
        "full_ewald_orbital_pair_metric",
    ),
    "full_ewald_transition_metric": (
        "pyqed.pbc.gw.integrals",
        "full_ewald_transition_metric",
    ),
    "gdf_orbital_pair_coupling": (
        "pyqed.pbc.gw.integrals",
        "gdf_orbital_pair_coupling",
    ),
    "gdf_orbital_pair_metric": ("pyqed.pbc.gw.integrals", "gdf_orbital_pair_metric"),
    "gdf_mo_jk": ("pyqed.pbc.gw.integrals", "gdf_mo_jk"),
    "gdf_transition_factors": ("pyqed.pbc.gw.integrals", "gdf_transition_factors"),
    "attach_pyscf_gdf_context": (
        "pyqed.pbc.gw.integrals",
        "attach_pyscf_gdf_context",
    ),
    "gdf_transition_metric": ("pyqed.pbc.gw.integrals", "gdf_transition_metric"),
    "pyscf_gdf_orbital_pair_coupling": (
        "pyqed.pbc.gw.integrals",
        "pyscf_gdf_orbital_pair_coupling",
    ),
    "pyscf_gdf_orbital_pair_metric": (
        "pyqed.pbc.gw.integrals",
        "pyscf_gdf_orbital_pair_metric",
    ),
    "pyscf_gdf_transition_factors": (
        "pyqed.pbc.gw.integrals",
        "pyscf_gdf_transition_factors",
    ),
    "pyscf_gdf_transition_metric": (
        "pyqed.pbc.gw.integrals",
        "pyscf_gdf_transition_metric",
    ),
    "prebuild_gdf_q_ao_stores": (
        "pyqed.pbc.gw.integrals",
        "prebuild_gdf_q_ao_stores",
    ),
    "reciprocal_orbital_pair_factors": (
        "pyqed.pbc.gw.integrals",
        "reciprocal_orbital_pair_factors",
    ),
    "reciprocal_transition_factors": (
        "pyqed.pbc.gw.integrals",
        "reciprocal_transition_factors",
    ),
    "KBSE": ("pyqed.pbc.gw.kbse", "KBSE"),
    "KTDA": ("pyqed.pbc.gw.kbse", "KTDA"),
    "KGW": ("pyqed.pbc.gw.kgw", "KGW"),
    "PeriodicBSEOpticalResult": (
        "pyqed.pbc.gw.optics",
        "PeriodicBSEOpticalResult",
    ),
    "PeriodicBSEHaydockResult": (
        "pyqed.pbc.gw.optics",
        "PeriodicBSEHaydockResult",
    ),
    "periodic_bse_absorption": (
        "pyqed.pbc.gw.optics",
        "periodic_bse_absorption",
    ),
    "periodic_tda_haydock": (
        "pyqed.pbc.gw.optics",
        "periodic_tda_haydock",
    ),
    "periodic_transition_velocity_matrix_elements": (
        "pyqed.pbc.gw.optics",
        "periodic_transition_velocity_matrix_elements",
    ),
    "PeriodicPESPeakResult": (
        "pyqed.pbc.gw.pes",
        "PeriodicPESPeakResult",
    ),
    "PeriodicPhotoemissionResult": (
        "pyqed.pbc.gw.pes",
        "PeriodicPhotoemissionResult",
    ),
    "PeriodicSpectralFunctionResult": (
        "pyqed.pbc.gw.pes",
        "PeriodicSpectralFunctionResult",
    ),
    "periodic_spectral_function": (
        "pyqed.pbc.gw.pes",
        "periodic_spectral_function",
    ),
    "periodic_spectral_peaks": (
        "pyqed.pbc.gw.pes",
        "periodic_spectral_peaks",
    ),
    "periodic_photoemission_peaks": (
        "pyqed.pbc.gw.pes",
        "periodic_photoemission_peaks",
    ),
    "periodic_photoemission_spectrum": (
        "pyqed.pbc.gw.pes",
        "periodic_photoemission_spectrum",
    ),
    "periodic_plane_wave_orbital_ft": (
        "pyqed.pbc.gw.pes",
        "periodic_plane_wave_orbital_ft",
    ),
    "periodic_plane_wave_velocity_matrix_elements": (
        "pyqed.pbc.gw.pes",
        "periodic_plane_wave_velocity_matrix_elements",
    ),
    "KPointTransitionSpace": ("pyqed.pbc.gw.response", "KPointTransitionSpace"),
    "KTransition": ("pyqed.pbc.gw.response", "KTransition"),
    "QBlockResponse": ("pyqed.pbc.gw.response", "QBlockResponse"),
    "ScreenedInteractionPoles": ("pyqed.pbc.gw.response", "ScreenedInteractionPoles"),
    "build_transition_space": ("pyqed.pbc.gw.response", "build_transition_space"),
    "direct_rpa": ("pyqed.pbc.gw.response", "direct_rpa"),
    "direct_tdh_matrices": ("pyqed.pbc.gw.response", "direct_tdh_matrices"),
    "screened_interaction_poles": (
        "pyqed.pbc.gw.response",
        "screened_interaction_poles",
    ),
    "DiagonalEVGWResult": ("pyqed.pbc.gw.self_energy", "DiagonalEVGWResult"),
    "DiagonalG0W0Result": ("pyqed.pbc.gw.self_energy", "DiagonalG0W0Result"),
    "DiagonalSelfEnergyCache": (
        "pyqed.pbc.gw.self_energy",
        "DiagonalSelfEnergyCache",
    ),
    "DiagonalSelfEnergy": ("pyqed.pbc.gw.self_energy", "DiagonalSelfEnergy"),
    "diagonal_correlation_self_energy": (
        "pyqed.pbc.gw.self_energy",
        "diagonal_correlation_self_energy",
    ),
    "diagonal_evgw": ("pyqed.pbc.gw.self_energy", "diagonal_evgw"),
    "diagonal_g0w0": ("pyqed.pbc.gw.self_energy", "diagonal_g0w0"),
}

__all__ = sorted(_LAZY_IMPORTS)


def __getattr__(name):
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
