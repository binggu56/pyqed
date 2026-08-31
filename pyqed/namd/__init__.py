import os


if os.environ.get("PYQED_LIGHTWEIGHT_IMPORTS") != "1":
    from .bh import BornHuang2, BornHuang
    from .ehrenfest import (
        AbInitioEhrenfest,
        CoupledOscillatorModel,
        Ehrenfest,
        EhrenfestTrajectory,
        GeometricEhrenfest,
        TDDFTDriver,
        TDDFTEhrenfest,
        TDDFTTrajectory,
    )
    from .ldrfg import (
        AbInitioLDRFGAdapter,
        LDRFG,
        LDRFGRHS,
        grad_overlap_from_derivative_couplings,
    )
    from .psgldr import PSGLDR, PSGLDRRHS
    from .tdscf import TDSCF, TDSCFTrajectory
    from .rtldr import (
        RTLDR,
        RTLDRTrajectory,
        RTTDHFFrame,
        RetainedStateRTLDR,
        RetainedStateTrajectory,
        det_overlap,
        frames_from_overlap,
    )
    from .liquid_ldr import (
        LiquidAvoidedCrossingLDRModel,
        EmbeddedLDRFGTDVPModel,
        MethanolFGCoordinateFrame,
        PhaseGaugedLiquidLDRModel,
        SolventEmbeddedLDRSnapshot,
        SolventEmbeddedLDRTrajectory,
        XYZFrame,
        build_embedded_casci_ldr_trajectory,
        build_embedded_h2_casci_ldr_trajectory,
        build_solvent_embedded_ldr_trajectory,
        compare_embedded_geometric_contribution,
        compare_embedded_ldr_to_static,
        compare_liquid_geometric_contribution,
        compare_liquid_to_static_ldr,
        embedded_casci_ldr_snapshot,
        embedded_ldr_hamiltonian,
        embedded_ldr_comparison_metrics,
        embedded_ldr_frame_overlap_diagnostics,
        embedded_ldr_geometric_hotspots,
        embedded_ldr_geometric_population_hotspots,
        embedded_ldr_geometric_population_quality,
        embedded_ldr_geometric_population_signal_summary,
        embedded_ldr_geometric_population_stride_convergence,
        embedded_ldr_geometric_quality,
        embedded_ldr_geometric_readiness,
        embedded_ldr_geometric_signal_summary,
        embedded_ldr_geometric_state_convergence,
        embedded_ldr_geometric_step_diagnostics,
        embedded_ldr_transport_holonomy,
        embedded_ldr_trajectory_diagnostics,
        embedded_h2_casci_ldr_snapshot,
        h2_bond_geometry,
        initial_ldr_packet,
        embedded_ldr_substep_convergence,
        liquid_ldr_diagnostics,
        liquid_ldr_geometric_driver_correlations,
        liquid_ldr_geometric_gauge_invariance,
        liquid_ldr_geometric_gauge_substep_convergence,
        liquid_ldr_geometric_hotspots,
        liquid_ldr_geometric_quality,
        liquid_ldr_geometric_readiness,
        liquid_ldr_geometric_signal_summary,
        liquid_ldr_geometric_stride_convergence,
        liquid_ldr_geometric_step_diagnostics,
        liquid_ldr_hotspot_driver_summary,
        liquid_ldr_substep_convergence,
        embedded_ldrfg_path_linearized_model,
        methanol_fg_path_diagnostics,
        methanol_fg_path_classical_forces,
        methanol_fg_path_force_callback,
        methanol_full_fg_coordinate_path,
        propagate_embedded_ldr_snapshots,
        propagate_liquid_ldrfg_tdvp,
        propagate_liquid_ldr,
        read_xyz_trajectory,
        second_derivative_kinetic,
        solvent_electric_field_coordinate,
        solvent_embedded_ldr_snapshot,
        solvent_point_charges_from_frame,
        solute_bond_distance_geometry_builder,
    )


def __getattr__(name):
    if name == "WavepacketScattering":
        from .scattering import WavepacketScattering

        globals()[name] = WavepacketScattering
        return WavepacketScattering
    if name in {"TNLDR", "TTLDR"}:
        from .ttldr import TNLDR, TTLDR

        globals().update(TNLDR=TNLDR, TTLDR=TTLDR)
        return globals()[name]
    if name in {"phenol_metric_evaluators", "build_phenol_5d_keo_mpo"}:
        from .phenol import build_phenol_5d_keo_mpo, phenol_metric_evaluators

        value = {
            "phenol_metric_evaluators": phenol_metric_evaluators,
            "build_phenol_5d_keo_mpo": build_phenol_5d_keo_mpo,
        }[name]
        globals()[name] = value
        return value
    if name in {"Triatom", "Triatomic"}:
        from .triatomic import Triatom, Triatomic

        value = {"Triatom": Triatom, "Triatomic": Triatomic}[name]
        globals()[name] = value
        return value
    if name in {
        "PolysphericalTree",
        "build_keo",
        "build_analytic_keo_mpo",
        "build_keo_mpo",
        "build_keo_mpo_cross",
    }:
        from .polyspherical import (
            PolysphericalTree,
            build_keo,
            build_analytic_keo_mpo,
            build_keo_mpo,
            build_keo_mpo_cross,
        )

        value = {
            "PolysphericalTree": PolysphericalTree,
            "build_keo": build_keo,
            "build_analytic_keo_mpo": build_analytic_keo_mpo,
            "build_keo_mpo": build_keo_mpo,
            "build_keo_mpo_cross": build_keo_mpo_cross,
        }[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AbInitioEhrenfest",
    "BornHuang",
    "BornHuang2",
    "CoupledOscillatorModel",
    "Ehrenfest",
    "EhrenfestTrajectory",
    "GeometricEhrenfest",
    "AbInitioLDRFGAdapter",
    "LDRFG",
    "LDRFGRHS",
    "PSGLDR",
    "PSGLDRRHS",
    "RTLDR",
    "RTLDRTrajectory",
    "RTTDHFFrame",
    "RetainedStateRTLDR",
    "RetainedStateTrajectory",
    "TDSCF",
    "TDSCFTrajectory",
    "frames_from_overlap",
    "det_overlap",
    "LiquidAvoidedCrossingLDRModel",
    "EmbeddedLDRFGTDVPModel",
    "MethanolFGCoordinateFrame",
    "PhaseGaugedLiquidLDRModel",
    "SolventEmbeddedLDRSnapshot",
    "SolventEmbeddedLDRTrajectory",
    "XYZFrame",
    "build_embedded_casci_ldr_trajectory",
    "build_embedded_h2_casci_ldr_trajectory",
    "build_solvent_embedded_ldr_trajectory",
    "compare_embedded_geometric_contribution",
    "compare_embedded_ldr_to_static",
    "compare_liquid_geometric_contribution",
    "compare_liquid_to_static_ldr",
    "embedded_casci_ldr_snapshot",
    "embedded_ldr_hamiltonian",
    "embedded_ldr_comparison_metrics",
    "embedded_ldr_frame_overlap_diagnostics",
    "embedded_ldr_geometric_hotspots",
    "embedded_ldr_geometric_population_hotspots",
    "embedded_ldr_geometric_population_quality",
    "embedded_ldr_geometric_population_signal_summary",
    "embedded_ldr_geometric_population_stride_convergence",
    "embedded_ldr_geometric_quality",
    "embedded_ldr_geometric_readiness",
    "embedded_ldr_geometric_signal_summary",
    "embedded_ldr_geometric_state_convergence",
    "embedded_ldr_geometric_step_diagnostics",
    "embedded_ldr_transport_holonomy",
    "embedded_ldr_trajectory_diagnostics",
    "embedded_h2_casci_ldr_snapshot",
    "embedded_ldr_substep_convergence",
    "grad_overlap_from_derivative_couplings",
    "h2_bond_geometry",
    "initial_ldr_packet",
    "liquid_ldr_diagnostics",
    "liquid_ldr_geometric_driver_correlations",
    "liquid_ldr_geometric_gauge_invariance",
    "liquid_ldr_geometric_gauge_substep_convergence",
    "liquid_ldr_geometric_hotspots",
    "liquid_ldr_geometric_quality",
    "liquid_ldr_geometric_readiness",
    "liquid_ldr_geometric_signal_summary",
    "liquid_ldr_geometric_stride_convergence",
    "liquid_ldr_geometric_step_diagnostics",
    "liquid_ldr_hotspot_driver_summary",
    "liquid_ldr_substep_convergence",
    "embedded_ldrfg_path_linearized_model",
    "methanol_fg_path_diagnostics",
    "methanol_fg_path_classical_forces",
    "methanol_fg_path_force_callback",
    "methanol_full_fg_coordinate_path",
    "propagate_embedded_ldr_snapshots",
    "propagate_liquid_ldrfg_tdvp",
    "propagate_liquid_ldr",
    "read_xyz_trajectory",
    "second_derivative_kinetic",
    "solvent_electric_field_coordinate",
    "solvent_embedded_ldr_snapshot",
    "solvent_point_charges_from_frame",
    "solute_bond_distance_geometry_builder",
    "TDDFTDriver",
    "TDDFTEhrenfest",
    "TDDFTTrajectory",
    "WavepacketScattering",
    "TNLDR",
    "TTLDR",
    "Triatom",
    "Triatomic",
    "PolysphericalTree",
    "build_keo",
    "build_analytic_keo_mpo",
    "build_keo_mpo",
    "build_keo_mpo_cross",
    "phenol_metric_evaluators",
    "build_phenol_5d_keo_mpo",
]
