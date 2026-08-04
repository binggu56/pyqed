"""Infinite-system HNC/0 contraction for a finite-range 2D Bose gas."""

from __future__ import annotations

import argparse

from pyqed.mps.bose_gas_2d import (
    FunctionalD2HNC2D,
    GaussianPotentialBoseGas2D,
    HNCELBoseGas2D,
    optimize_d2_triplet_hnc,
    optimize_gaussian_jastrow_hnc,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--interaction-strength", type=float, default=0.5)
    parser.add_argument("--interaction-range", type=float, default=0.7)
    parser.add_argument("--quadrature-points", type=int, default=192)
    parser.add_argument("--maxiter", type=int, default=80)
    args = parser.parse_args()

    model = GaussianPotentialBoseGas2D(
        density=args.density,
        interaction_strength=args.interaction_strength,
        interaction_range=args.interaction_range,
    )
    state = optimize_gaussian_jastrow_hnc(
        model,
        quadrature_points=args.quadrature_points,
        maxiter=args.maxiter,
    )
    d2_state = optimize_d2_triplet_hnc(
        model,
        pair_state=state,
        maxiter=args.maxiter,
    )
    hncel = HNCELBoseGas2D(
        model,
        quadrature_points=args.quadrature_points,
    ).solve()
    functional_d2 = FunctionalD2HNC2D(
        model,
        quadrature_points=128,
        angular_points=12,
        transverse_coefficient_bound=0.02,
        initial_transverse_amplitude=0.01,
    ).optimize(maxiter=args.maxiter)

    print("2D continuum Bose gas: thermodynamic HNC/0")
    print(f"density                    = {model.density:.12g}")
    print(f"interaction strength       = {model.interaction_strength:.12g}")
    print(f"interaction range          = {model.interaction_range:.12g}")
    print(f"mean-field energy/area     = {model.mean_field_energy_density:.12g}")
    print(f"Bogoliubov energy/area     = {model.bogoliubov_energy_density:.12g}")
    print(f"HNC/0 energy/area          = {state.energy_density:.12g}")
    print(f"  kinetic                  = {state.kinetic_energy_density:.12g}")
    print(f"  potential                = {state.potential_energy_density:.12g}")
    print(
        "optimized Jastrow amplitude = "
        f"{state.optimized_parameters['jastrow_amplitude']:.12g}"
    )
    print(
        "optimized Jastrow range     = "
        f"{state.optimized_parameters['jastrow_range']:.12g}"
    )
    print(f"HNC iterations             = {state.iterations}")
    print(f"HNC fixed-point residual   = {state.fixed_point_residual:.6e}")
    print(f"optimizer evaluations      = {state.optimization_evaluations}")
    print(f"success                    = {state.success}")
    print(f"message                    = {state.message}")
    print("\nleading connected D=2 triplet channel")
    print(f"HNC/3 energy/area          = {d2_state.energy_density:.12g}")
    print(f"  pair kinetic             = {d2_state.pair_kinetic_energy_density:.12g}")
    print(
        f"  triplet kinetic          = "
        f"{d2_state.triplet_kinetic_energy_density:.12g}"
    )
    print(f"  potential                = {d2_state.potential_energy_density:.12g}")
    print(f"D=2 energy gain/area       = {d2_state.d2_energy_gain_density:.12g}")
    print(
        "transverse amplitude       = "
        f"{d2_state.optimized_parameters['transverse_amplitude']:.12g}"
    )
    print(
        "transverse range           = "
        f"{d2_state.optimized_parameters['transverse_range']:.12g}"
    )
    print(f"HNC/3 fixed-point residual = {d2_state.fixed_point_residual:.6e}")
    print(f"HNC/3 success              = {d2_state.success}")
    print("\nunrestricted scalar HNC-EL/0")
    print(f"HNC-EL energy/area         = {hncel.energy_density:.12g}")
    print(f"  kinetic                  = {hncel.kinetic_energy_density:.12g}")
    print(f"  potential                = {hncel.potential_energy_density:.12g}")
    print(f"infrared exponent          = {hncel.infrared_exponent:.12g}")
    print(f"lim S(k)/k                 = {hncel.infrared_slope:.12g}")
    print(f"Jastrow 1/r amplitude      = {hncel.jastrow_tail_amplitude:.12g}")
    print(f"Euler residual             = {hncel.euler_residual:.6e}")
    print(f"HNC-EL success             = {hncel.success}")
    print("\nfree D=2 transverse-functional stability test")
    print(
        "diagnostic energy/area     = "
        f"{functional_d2.energy_density:.12g}"
    )
    print(
        "coefficient bound reached = "
        f"{functional_d2.transverse_boundary_limited}"
    )
    print(
        "controlled stationary D=2 = "
        f"{functional_d2.controlled_d2_stationary_point}"
    )
    print(f"accepted D=2 result         = {functional_d2.success}")
    print(
        "The result is already at infinite area and fixed density. "
        "HNC/0 and the triplet HNC/3 closure are diagram resummations, "
        "not exact contractions."
    )
    if functional_d2.transverse_boundary_limited:
        print(
            "The free D=2 channel is a runaway of the kappa_2-only "
            "functional; its diagnostic energy is not a physical estimate."
        )


if __name__ == "__main__":
    main()
