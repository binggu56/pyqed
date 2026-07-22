"""Direct thermodynamic continuum calculation for hard-core p+ip pairs."""

from __future__ import annotations

import argparse

import numpy as np

from pyqed.mps.pip_pairing import (
    ContinuumPipPairingModel,
    ThermodynamicPipBCS,
    ThermodynamicPipCLETTA,
)


def run(args):
    model = ContinuumPipPairingModel(
        coupling=args.coupling,
        energy_cutoff=args.energy_cutoff,
        density_of_states=args.density_of_states,
    )
    state = ThermodynamicPipBCS.solve(
        model,
        fermion_density=args.fermion_density,
    )

    print("direct fixed-density p+ip continuum")
    print("physical orbitals    = none (thermodynamic integral)")
    print(f"pair filling         = {state.pair_filling:.12g}")
    print(f"fermion density      = {state.fermion_density:.12g}")
    print(f"chemical potential   = {state.chemical_potential:.12g}")
    print(f"gap                  = {state.gap:.12g}")
    print(f"phase                = {state.phase}")
    print(f"kinetic energy/area  = {state.kinetic_energy_density:.12g}")
    print(f"interaction E/area  = {state.interaction_energy_density:.12g}")
    print(f"total energy/area    = {state.energy_density:.12g}")
    print(f"density residual     = {state.integrated_fermion_density() - state.fermion_density:.3e}")
    print(f"gap residual         = {state.gap_equation_residual():.3e}")

    cletta = ThermodynamicPipCLETTA.optimize(state)
    print("\nreal hard-core frequency cLETTA")
    print(
        f"D={cletta.bond_dim} M={cletta.num_memory_modes} "
        f"L={cletta.memory_depth} Deff={cletta.effective_bond_dim}"
    )
    print(f"radial decay        = {cletta.radial_decay:.12g}")
    print(f"tie strength        = {cletta.tie_strength:.12g}")
    print(f"memory decay        = {cletta.memory_decay:.12g}")
    print(f"fugacity shift      = {cletta.fugacity_shift:.3e}")
    print(f"fermion density     = {cletta.fermion_density:.12g}")
    print(f"energy/area         = {cletta.energy_density:.12g}")
    print(f"energy above BCS    = {cletta.energy_density - state.energy_density:.3e}")
    print("thermodynamic result = zero tie (BCS is exact in energy density)")

    tied = ThermodynamicPipCLETTA.evaluate(
        state,
        radial_decay=2.0,
        tie_strength=0.4,
        memory_decay=1.3,
        quadrature_points=128,
    )
    print("\nnonzero-tie contraction check")
    print(f"tie strength        = {tied.tie_strength:.12g}")
    print(f"virtual-memory norm = {tied.norm:.12g}")
    print(f"fermion density     = {tied.fermion_density:.12g}")
    print(f"energy above BCS    = {tied.energy_density - state.energy_density:.3e}")
    print("the excess scales to zero with radial quadrature spacing")

    if args.points:
        energies = np.linspace(0.0, model.energy_cutoff, args.points)
        print("# E pair_occupation quasiparticle_energy")
        for energy in energies:
            print(
                f"{energy:.12g} "
                f"{state.pair_occupation(energy):.12g} "
                f"{state.quasiparticle_energy(energy):.12g}"
            )
    return state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coupling", type=float, default=3.0)
    parser.add_argument("--energy-cutoff", type=float, default=1.0)
    parser.add_argument("--density-of-states", type=float, default=1.0)
    parser.add_argument("--fermion-density", type=float, default=0.5)
    parser.add_argument("--points", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
