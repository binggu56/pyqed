"""Continuous p+ip benchmarks, including a genuine D=2, M=1 cLETTA."""

from __future__ import annotations

import argparse

import numpy as np

from pyqed.mps.pip_pairing import (
    ContinuumPipPairingModel,
    ExactOnePairPipState,
    OneScalePipCLETTA,
    TwoPairPipCLETTA,
    TwoPairPipD3CLETTA,
)


def run(args):
    model = ContinuumPipPairingModel(
        coupling=args.coupling,
        energy_cutoff=args.energy_cutoff,
        density_of_states=args.density_of_states,
    )
    reference = ExactOnePairPipState.from_model(model)
    state = OneScalePipCLETTA.optimize(model)

    print("continuous p+ip one-pair frequency cLETTA")
    print(f"D={state.bond_dim} M={state.num_tie_channels}")
    print(f"optimized decay   ={state.decay_rate:.12g}")
    print(f"exact energy      ={reference.energy:.12g}")
    print(f"variational energy={state.energy:.12g}")
    print(f"energy error      ={state.energy - reference.energy:.3e}")
    print(f"continuum norm    ={state.norm():.12g}")

    genuine = TwoPairPipCLETTA.optimize(model)
    print("\ngenuine two-pair cLETTA with explicit memory")
    print(
        f"D={genuine.bond_dim} M={genuine.num_memory_modes} "
        f"L={genuine.memory_depth} Deff=4"
    )
    print(f"radial decay      ={genuine.radial_decay:.12g}")
    print(f"memory decay      ={genuine.memory_decay:.12g}")
    print(f"exact dilute E    ={genuine.exact_dilute_pair_energy:.12g}")
    print(f"variational energy={genuine.energy:.12g}")
    print(f"energy error      ={genuine.energy - genuine.exact_dilute_pair_energy:.3e}")

    d3 = TwoPairPipD3CLETTA.optimize(model)
    print("\nfixed-two-pair D=3 comparison")
    print(
        f"D={d3.bond_dim} M={d3.num_memory_modes} "
        f"L={d3.memory_depth} Deff=6"
    )
    print(
        "radial decays    ="
        + ";".join(f"{value:.12g}" for value in d3.radial_decays)
    )
    print(f"mixing angle      ={d3.mixing_angle:.12g}")
    print(f"memory decay      ={d3.memory_decay:.12g}")
    print(f"variational energy={d3.energy:.12g}")
    print(f"energy error      ={d3.energy - d3.exact_dilute_pair_energy:.3e}")

    if args.points:
        energies = np.linspace(0.0, model.energy_cutoff, args.points)
        amplitudes = state.radial_amplitude(energies)
        print("# E radial_amplitude")
        for energy, amplitude in zip(energies, amplitudes):
            print(f"{energy:.12g} {amplitude:.12g}")
    return state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coupling", type=float, default=3.0)
    parser.add_argument("--energy-cutoff", type=float, default=1.0)
    parser.add_argument("--density-of-states", type=float, default=1.0)
    parser.add_argument("--points", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
