#!/usr/bin/env python3
"""U(1)-blocked real-time PEPS evolution on a 2x2 Heisenberg plaquette."""

from pyqed.lattice import SpinHalfSite
from pyqed.peps import U1PEPS
from pyqed.tn import Hamiltonian


def main():
    shape = (2, 2)
    sites = tuple(SpinHalfSite() for _ in range(4))
    hamiltonian = Hamiltonian(sites)
    for first, second in ((0, 1), (0, 2), (1, 3), (2, 3)):
        for operator in ("X", "Y", "Z"):
            hamiltonian.add_product(
                0.25,
                (first, operator),
                (second, operator),
            )

    state = U1PEPS.product_state(sites, [0, 1, 1, 0], shape=shape)
    initial_energy = state.expectation(hamiltonian)
    evolution = state.evolve(
        hamiltonian,
        0.2,
        step=0.05,
        max_D=4,
        verbose=True,
    )

    print(f"initial energy = {initial_energy:.12f}")
    print(f"final energy = {evolution.energy:.12f}")
    print(f"bond dimensions = {state.bond_dims}")
    print(f"stored blocks = {state.block_count}")
    print(f"storage fraction = {state.storage_fraction:.6f}")
    print(f"norm = {state.norm_squared():.12f}")


if __name__ == "__main__":
    main()
