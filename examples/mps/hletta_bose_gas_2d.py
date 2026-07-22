"""Direct-continuum starting benchmark for a hierarchical 2D Bose gas."""

from __future__ import annotations

import argparse

import numpy as np

from pyqed.mps.bose_gas_2d import (
    D2M1HierarchicalCLETTA2D,
    D2M1NestedCLETTA2D,
    DiluteBoseGas2D,
    GaussianPotentialBoseGas2D,
    HierarchicalShellContraction,
    RankOneDensityTransferChannel2D,
    optimize_condensate_gns_hletta_fixed_density,
)


def run(args):
    dilute = DiluteBoseGas2D(
        density=args.density,
        scattering_length=args.scattering_length,
        kinetic_prefactor=args.kinetic_prefactor,
    )
    print("universal dilute 2D Bose gas")
    print(f"n a_2D^2                 = {dilute.gas_parameter:.12g}")
    print(f"Y                        = {dilute.expansion_parameter:.12g}")
    print(f"leading energy/area      = {dilute.leading_energy_density:.12g}")
    print(f"with logarithmic term    = {dilute.logarithmic_energy_density:.12g}")
    print(f"through constant Y term  = {dilute.energy_density:.12g}")
    print(
        "constant-order contribution = "
        f"{dilute.constant_order_energy_correction:.12g}"
    )

    gaussian = GaussianPotentialBoseGas2D(
        density=args.density,
        interaction_strength=args.interaction_strength,
        interaction_range=args.interaction_range,
        kinetic_prefactor=args.kinetic_prefactor,
    )
    print("\nsmooth finite-range Bogoliubov reference")
    print("(an independent bare Gaussian potential, not scattering-length matched)")
    print(f"mean-field energy/area   = {gaussian.mean_field_energy_density:.12g}")
    print(
        "zero-point correction    = "
        f"{gaussian.bogoliubov_energy_correction_density():.12g}"
    )
    print(f"Bogoliubov energy/area   = {gaussian.bogoliubov_energy_density:.12g}")
    print(f"depletion density        = {gaussian.depletion_density():.12g}")
    gaussian.optimize_full_gaussian(
        quadrature_points=args.gaussian_points,
        maxiter=args.gaussian_maxiter,
    )
    print("\nfull interacting Hamiltonian, optimized Gaussian state")
    print(f"energy/area              = {gaussian.full_gaussian_energy_density:.12g}")
    print(f"squeezing amplitude      = {gaussian.squeezing_amplitude:.12g}")
    print(f"momentum rescaling       = {gaussian.squeezing_momentum_scale:.12g}")
    print(f"optimizer success        = {gaussian.success}")
    def density_validation_squeezing(momentum):
        return 0.22 * np.exp(-0.6 * np.asarray(momentum) ** 2)

    density_direct_energy = gaussian.full_gaussian_energy_density_for_squeezing(
        density_validation_squeezing,
        quadrature_points=args.gaussian_points,
    )
    density_transfer_energy = (
        gaussian.density_transfer_energy_density_for_squeezing(
            density_validation_squeezing,
            radial_points=args.density_radial_points,
            angular_points=args.density_angular_points,
        )
    )
    print("\nmomentum-conserving density-transfer check")
    print("single radial profile u(q), with V(q) = |u(q)|^2")
    print(f"direct Wick energy/area  = {density_direct_energy:.12g}")
    print(f"density-mode energy/area = {density_transfer_energy:.12g}")
    print(
        "absolute difference       = "
        f"{abs(density_transfer_energy - density_direct_energy):.3e}"
    )

    contraction = HierarchicalShellContraction(
        energy_cutoff=args.energy_cutoff,
        radial_points=args.radial_points,
        angular_points=args.angular_points,
    )
    demo_generator = np.diag([-0.2, 0.07])

    def angular_generator(_energy, _theta, radial_width):
        return radial_width * demo_generator

    value, _ = contraction.contract(
        angular_generator,
        left_boundary=np.ones(2),
        right_boundary=np.array([0.6, 0.4]),
    )
    print("\nhierarchical shell-contraction check")
    print(f"radial quadrature points = {args.radial_points}")
    print(f"angular quadrature points= {args.angular_points}")
    print(f"contracted scalar        = {value.real:.12g}")
    print("nodes are continuum quadrature points, not physical orbitals")

    hletta = D2M1HierarchicalCLETTA2D(
        contraction=contraction,
        q_matrix=np.diag([-0.2, -0.5]),
        r_matrix=np.array([[0.0, 0.12], [0.08, 0.0]]),
        tie_matrix=np.array([[0.0, 0.04], [0.03, 0.0]]),
        memory_decay=0.7,
        radial_decay=0.9,
        angular_momentum=0,
    )
    print("\nflattened finite cLETTA diagnostic")
    print(f"outer bond dimension D = {hletta.bond_dim}")
    print(f"memory channels M       = {hletta.num_memory_modes}")
    print(f"memory cutoff dimension = {hletta.memory_dim}")
    print(f"ket auxiliary dimension = {hletta.effective_bond_dim}")
    print(f"double-layer dimension  = {hletta.transfer_dim}")
    print(f"vacuum-boundary norm     = {hletta.norm():.12g}")
    print(f"particle number          = {hletta.particle_number():.12g}")
    print(f"kinetic expectation      = {hletta.kinetic_energy():.12g}")
    pair = hletta.antipodal_pair_expectation()
    print(f"antipodal pair amplitude = {pair.real:.12g} {pair.imag:+.12g}i")
    print(
        "shell-normalized quadratic functional = "
        f"{hletta.bogoliubov_shell_functional(gaussian):.12g}"
    )
    print("this is not yet a thermodynamic energy per real-space area")
    print("the full quartic 2D interaction is not included")

    nested = D2M1NestedCLETTA2D(
        contraction=contraction,
        q_matrix=np.diag([-0.2, -0.5]),
        r_matrix=np.array([[0.0, 0.12], [0.08, 0.0]]),
        tie_matrix=np.array([[0.0, 0.04], [0.03, 0.0]]),
        angular_memory_decay=0.7,
        radial_decay=0.9,
    )
    print("\ngenuinely nested D=2, M=1 hLETTA")
    print(f"inner ket dimension      = {nested.inner_bond_dim}")
    print(f"inner transfer dimension = {nested.inner_transfer_dim}")
    print(f"outer transfer dimension = {nested.outer_transfer_dim}")
    print(f"nested vacuum norm        = {nested.norm():.12g}")
    print(f"nested particle number    = {nested.particle_number():.12g}")
    print(f"nested kinetic expectation= {nested.kinetic_energy():.12g}")
    print("angular memory is projected to vacuum before radial composition")
    hletta_channel = RankOneDensityTransferChannel2D(
        radial_profile=gaussian.density_transfer_profile,
        momentum_cutoff=2.0
        * np.sqrt(args.energy_cutoff / args.kinetic_prefactor),
        radial_points=args.hletta_transfer_points,
    )
    nested.evaluate_rank_one_density_transfer(
        hletta_channel,
        kinetic_prefactor=args.kinetic_prefactor,
        structure_radial_points=args.hletta_structure_radial_points,
        structure_angular_points=args.hletta_structure_angular_points,
        propagation_points=args.hletta_propagation_points,
    )
    print("\nnested hLETTA rank-one interacting cutoff theory")
    print(f"particle density          = {nested.particle_density:.12g}")
    print(f"kinetic energy/area       = {nested.kinetic_energy_density:.12g}")
    print(f"interaction energy/area   = {nested.interaction_energy_density:.12g}")
    print(f"total energy/area         = {nested.energy_density:.12g}")
    print("this value is evaluated, not yet fixed-density optimized")

    optimized = optimize_condensate_gns_hletta_fixed_density(
        gaussian,
        target_density=args.density,
        energy_cutoff=args.energy_cutoff,
        radial_points=args.fixed_density_radial_points,
        angular_points=args.angular_points,
        channel_points=args.hletta_transfer_points,
        structure_radial_points=args.hletta_structure_radial_points,
        structure_angular_points=args.hletta_structure_angular_points,
        pair_angular_points=args.hletta_pair_angular_points,
        maxiter=args.fixed_density_maxiter,
    )
    print("\nthermodynamic GNS condensate plus D=2, M=1 hLETTA")
    print(f"target density            = {args.density:.12g}")
    print(f"contracted density        = {optimized.particle_density:.12g}")
    print(f"condensate density n0     = {optimized.condensate_density:.12g}")
    print(f"fluctuation density       = {optimized.fluctuation_density:.12g}")
    print(f"condensate fraction       = {optimized.condensate_fraction:.12g}")
    print(f"initial energy/area       = {optimized.initial_energy_density:.12g}")
    print(f"GNS energy/area           = {optimized.energy_density:.12g}")
    print(
        "pure-condensate energy   = "
        f"{optimized.pure_condensate_energy_density:.12g}"
    )
    print(
        "condensate mean field    = "
        f"{optimized.condensate_mean_field_energy_density:.12g}"
    )
    print(
        "condensate normal term   = "
        f"{optimized.condensate_normal_interaction_density:.12g}"
    )
    print(
        "condensate anomalous term= "
        f"{optimized.condensate_anomalous_interaction_density:.12g}"
    )
    print(
        "four-fluctuation term    = "
        f"{optimized.fluctuation_quartic_interaction_density:.12g}"
    )
    print(f"real-space area parameter = none (GNS fixed point)")
    print(f"area drift                = {optimized.area_drift:.3e}")
    print(
        "minimum transfer gap     = "
        f"{optimized.minimum_gns_transfer_gap:.12g}"
    )
    print(f"boundary limited          = {optimized.boundary_limited}")
    print(f"thermodynamic valid       = {optimized.thermodynamic_valid}")
    print(f"optimizer accepted        = {optimized.success}")
    for name, value in optimized.optimized_parameters.items():
        print(f"{name:26s}= {value:.12g}")
    print(f"audit message             = {optimized.message}")
    return dilute, gaussian, nested


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--scattering-length", type=float, default=1.0e-3)
    parser.add_argument("--kinetic-prefactor", type=float, default=1.0)
    parser.add_argument("--interaction-strength", type=float, default=0.5)
    parser.add_argument("--interaction-range", type=float, default=0.7)
    parser.add_argument("--energy-cutoff", type=float, default=1.0)
    parser.add_argument("--radial-points", type=int, default=16)
    parser.add_argument("--angular-points", type=int, default=24)
    parser.add_argument("--gaussian-points", type=int, default=64)
    parser.add_argument("--gaussian-maxiter", type=int, default=160)
    parser.add_argument("--density-radial-points", type=int, default=48)
    parser.add_argument("--density-angular-points", type=int, default=40)
    parser.add_argument("--hletta-transfer-points", type=int, default=8)
    parser.add_argument("--hletta-structure-radial-points", type=int, default=2)
    parser.add_argument("--hletta-structure-angular-points", type=int, default=3)
    parser.add_argument("--hletta-pair-angular-points", type=int, default=3)
    parser.add_argument("--hletta-propagation-points", type=int, default=2)
    parser.add_argument("--fixed-density-radial-points", type=int, default=6)
    parser.add_argument("--fixed-density-maxiter", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
