"""Factor-native all-future LETTA for the square-lattice J1-J2 model."""

import argparse

import numpy as np

from pyqed.lattice import SpinHalfSite
from pyqed.letta import FutureLETTA, GraphLETTA, VMC
from pyqed.tn import Hamiltonian


def snake_sites(rows, cols):
    coordinates = []
    for row in range(rows):
        columns = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)
        coordinates.extend((row, column) for column in columns)
    return coordinates


def j1j2_hamiltonian(rows, cols, j2):
    coordinates = snake_sites(rows, cols)
    index = {coordinate: site for site, coordinate in enumerate(coordinates)}
    hamiltonian = Hamiltonian((SpinHalfSite(),) * (rows * cols))
    couplings = []
    for row in range(rows):
        for column in range(cols):
            if column + 1 < cols:
                couplings.append((index[row, column], index[row, column + 1], 1.0))
            if row + 1 < rows:
                couplings.append((index[row, column], index[row + 1, column], 1.0))
            if row + 1 < rows and column + 1 < cols:
                couplings.append((index[row, column], index[row + 1, column + 1], j2))
                couplings.append((index[row, column + 1], index[row + 1, column], j2))
    for left, right, coupling in couplings:
        for operator in ("Sx", "Sy", "Sz"):
            hamiltonian.add_product(
                coupling,
                (left, operator),
                (right, operator),
            )
    return hamiltonian, coordinates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--j2", type=float, default=0.5)
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--chi", type=int, default=2)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--autoregressive", action="store_true")
    parser.add_argument("--exact", action="store_true")
    parser.add_argument(
        "--sparse-graph",
        action="store_true",
        help="start from only the J1/J2 Hamiltonian edges",
    )
    parser.add_argument("--adaptive-ties", type=int, default=0)
    args = parser.parse_args()

    hamiltonian, coordinates = j1j2_hamiltonian(args.rows, args.cols, args.j2)
    autoregressive = args.autoregressive or args.adaptive_ties > 0
    state_type = GraphLETTA if args.sparse_graph else FutureLETTA
    state = state_type(
        hamiltonian,
        target_charge={"Sz": 0},
        D=args.D,
        chi=args.chi,
        init="random" if autoregressive else "mps",
        autoregressive=autoregressive,
        seed=args.seed,
    )
    initial = np.asarray(
        [(row + column) % 2 for row, column in coordinates],
        dtype=np.intp,
    )
    print(
        f"sites={len(state.dims)} ties={len(state.graph)} "
        f"parameters={state.nparameters} D={state.D} chi={max(state.chi)}"
    )
    if args.adaptive_ties:
        state = state.adapt_ties(
            n_ties=args.adaptive_ties,
            nsamples=args.samples,
            seed=args.seed + 1,
        )
        print(f"adaptive={state.adaptation_history[-1]}")
    if args.exact:
        print(f"exact E={state.expectation(): .10f} norm={state.norm():.12f}")
    if autoregressive:
        samples = state.sample(args.samples, seed=args.seed + 1)
        print(
            f"independent_samples={len(samples)} "
            f"target_sector={bool(np.all(np.sum(samples, axis=1) == len(state.dims) // 2))}"
        )
        return

    vmc = VMC(
        state,
        seed=args.seed + 1,
        initial_configuration=initial,
        proposal="exchange",
    )
    for step in range(args.steps):
        samples = vmc.sample(
            args.samples,
            burn_in=20 if step == 0 else 5,
            sweeps_between=1,
        )
        estimate = vmc.estimate_from_samples(samples)
        print(
            f"step={step:2d} E={estimate.energy.real: .8f} "
            f"stderr={estimate.autocorrelation_standard_error:.3e} "
            f"accept={estimate.diagnostics.acceptance_rate:.3f}"
        )
        proposal = vmc.propose_sr(
            samples,
            step_size=0.02,
            diagonal_shift=1.0e-2,
            derivative_backend="sparse",
            max_iterations=80,
        )
        vmc.apply_sr(proposal)
    vmc.sync_to_state()


if __name__ == "__main__":
    main()
