#!/usr/bin/env python3
"""Compare 1D Holstein NARG variants with a small real-space MPS baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.narg import TensorTrainLETTA
from pyqed.narg.holstein import (
    HolsteinChainAdiabaticNARG,
    HolsteinChainCouplingNARG,
    HolsteinChainNARG,
    HolsteinElectronicFirstNARG,
    holstein_chain_exact_energies,
)


def full_real_space_holstein_hamiltonian(
    nsites: int,
    nphonon: int,
    *,
    t: float,
    omega: float,
    g: float,
    number_penalty: float,
) -> np.ndarray:
    """Dense spinless Holstein Hamiltonian on local sites.

    Each site has basis ``|e, n_ph>`` with ``e = 0, 1``.  The number penalty
    selects the one-electron sector while preserving a product-site layout for
    the real-space MPS sweep.  This is intentionally a small benchmark helper.
    """
    nsites = int(nsites)
    nphonon = int(nphonon)
    local_dim = 2 * nphonon
    dim = local_dim**nsites
    hamiltonian = np.zeros((dim, dim), dtype=float)
    powers = np.asarray([local_dim**site for site in range(nsites)], dtype=int)

    def unpack(index: int) -> list[tuple[int, int]]:
        value = int(index)
        states = []
        for _site in range(nsites):
            local = value % local_dim
            value //= local_dim
            states.append((local // nphonon, local % nphonon))
        return states

    def pack(states: list[tuple[int, int]]) -> int:
        index = 0
        for site, (electron, phonon) in enumerate(states):
            index += (int(electron) * nphonon + int(phonon)) * powers[site]
        return int(index)

    for col in range(dim):
        states = unpack(col)
        nelec = sum(electron for electron, _phonon in states)
        diagonal = float(number_penalty) * (nelec - 1) ** 2

        for site, (electron, phonon) in enumerate(states):
            diagonal += float(omega) * phonon
            if not electron:
                continue
            if phonon + 1 < nphonon:
                row_state = list(states)
                row_state[site] = (electron, phonon + 1)
                hamiltonian[pack(row_state), col] += float(g) * np.sqrt(phonon + 1)
            if phonon > 0:
                row_state = list(states)
                row_state[site] = (electron, phonon - 1)
                hamiltonian[pack(row_state), col] += float(g) * np.sqrt(phonon)

        hamiltonian[col, col] += diagonal

        for site in range(nsites - 1):
            left_e, left_ph = states[site]
            right_e, right_ph = states[site + 1]
            if left_e == 1 and right_e == 0:
                row_state = list(states)
                row_state[site] = (0, left_ph)
                row_state[site + 1] = (1, right_ph)
                hamiltonian[pack(row_state), col] += -float(t)
            if left_e == 0 and right_e == 1:
                row_state = list(states)
                row_state[site] = (1, left_ph)
                row_state[site + 1] = (0, right_ph)
                hamiltonian[pack(row_state), col] += -float(t)

    return 0.5 * (hamiltonian + hamiltonian.T)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-L", "--nsites", type=int, default=4)
    parser.add_argument("--nphonon", type=int, default=4)
    parser.add_argument("-t", "--hopping", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("-g", "--coupling", type=float, default=1.0)
    parser.add_argument("-D", "--bond-dim", type=int, default=16)
    parser.add_argument("--local-dim", type=int, default=None)
    parser.add_argument(
        "--mode-order",
        type=int,
        nargs="+",
        default=None,
        help="Zero-based active-mode order for electronic-first NARG.",
    )
    parser.add_argument("--states-per-branch", type=int, nargs="+", default=[3, 16])
    parser.add_argument("--dmrg-bond-dims", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--dmrg-sweeps", type=int, default=6)
    parser.add_argument("--number-penalty", type=float, default=100.0)
    parser.add_argument("--skip-dmrg", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    common = dict(
        nsites=args.nsites,
        t=args.hopping,
        omega=args.omega,
        g=args.coupling,
        nphonon=args.nphonon,
    )
    exact = holstein_chain_exact_energies(**common, nroots=1)[0]
    print(
        f"1D spinless Holstein: L={args.nsites}, nphonon={args.nphonon}, "
        f"t={args.hopping:g}, omega={args.omega:g}, g={args.coupling:g}"
    )
    print(f"ED one-electron E0 = {exact:.12f}")
    print("method                         energy          error       seconds   details")

    def emit(name: str, energy: float, seconds: float, details: str = "") -> None:
        print(
            f"{name:28s} {energy: .12f} {energy - exact: .3e} "
            f"{seconds:9.3f}   {details}"
        )

    narg_common = dict(common, local_dim=args.local_dim, bond_dim=args.bond_dim)
    for name, cls in (
        ("NARG plain/polaron", HolsteinChainNARG),
        ("NARG conditional-electronic", HolsteinChainAdiabaticNARG),
    ):
        start = perf_counter()
        result = cls(**narg_common).run(nroots=1)
        emit(name, result.energies[0], perf_counter() - start)

    start = perf_counter()
    result = HolsteinElectronicFirstNARG(
        **narg_common,
        mode_order=None if args.mode_order is None else tuple(args.mode_order),
    ).run(nroots=1)
    emit(
        "NARG electronic-first",
        result.energies[0],
        perf_counter() - start,
        f"last={result.steps[-1].kept}/{result.steps[-1].product_dim}",
    )

    for states_per_branch in args.states_per_branch:
        start = perf_counter()
        result = HolsteinChainCouplingNARG(
            **narg_common,
            states_per_branch=int(states_per_branch),
        ).run(nroots=1)
        step = result.steps[-1]
        emit(
            f"NARG active-mode spb={states_per_branch}",
            result.energies[0],
            perf_counter() - start,
            f"Q={step.orthonormal_dim}/{step.raw_dim}",
        )

    if args.skip_dmrg:
        return

    start = perf_counter()
    hamiltonian = full_real_space_holstein_hamiltonian(
        args.nsites,
        args.nphonon,
        t=args.hopping,
        omega=args.omega,
        g=args.coupling,
        number_penalty=args.number_penalty,
    )
    print(f"built real-space local-product Hamiltonian in {perf_counter() - start:.3f} s")

    dims = (2 * args.nphonon,) * args.nsites
    for bond_dim in args.dmrg_bond_dims:
        start = perf_counter()
        dmrg = TensorTrainLETTA(
            hamiltonian,
            dims,
            bond_dim=int(bond_dim),
            seed=1,
        )
        result = dmrg.run(nsweeps=args.dmrg_sweeps, tol=1e-9)
        emit(
            f"real-space MPS D={bond_dim}",
            result.energy,
            perf_counter() - start,
            f"sweeps={result.ncompleted}",
        )


if __name__ == "__main__":
    main()
