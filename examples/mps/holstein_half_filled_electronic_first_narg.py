#!/usr/bin/env python3
"""Half-filled spinful Holstein electronic-first NARG benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.narg import SpinfulHolsteinElectronicFirstNARG, TensorTrainLETTA
from pyqed.narg.holstein import spinful_holstein_hubbard_exact_energies


def _bit_count(value: int) -> int:
    return int(value).bit_count()


def _spin_orbital_bits(up_bits: int, down_bits: int, nsites: int) -> int:
    bits = 0
    for site in range(int(nsites)):
        if (int(up_bits) >> site) & 1:
            bits |= 1 << (2 * site)
        if (int(down_bits) >> site) & 1:
            bits |= 1 << (2 * site + 1)
    return bits


def _split_spin_orbital_bits(bits: int, nsites: int) -> tuple[int, int]:
    up_bits = 0
    down_bits = 0
    for site in range(int(nsites)):
        if (int(bits) >> (2 * site)) & 1:
            up_bits |= 1 << site
        if (int(bits) >> (2 * site + 1)) & 1:
            down_bits |= 1 << site
    return up_bits, down_bits


def _apply_cdag_c(bits: int, create_orbital: int, annihilate_orbital: int):
    bits = int(bits)
    if ((bits >> int(annihilate_orbital)) & 1) == 0:
        return None
    sign = -1 if _bit_count(bits & ((1 << int(annihilate_orbital)) - 1)) % 2 else 1
    after_annihilate = bits & ~(1 << int(annihilate_orbital))
    if (after_annihilate >> int(create_orbital)) & 1:
        return None
    sign *= -1 if _bit_count(after_annihilate & ((1 << int(create_orbital)) - 1)) % 2 else 1
    return after_annihilate | (1 << int(create_orbital)), sign


def full_real_space_spinful_holstein_hamiltonian(
    nsites: int,
    nphonon: int,
    *,
    t: float,
    omega: float,
    g: float,
    hubbard_u: float,
    nup: int,
    ndown: int,
    number_penalty: float,
) -> np.ndarray:
    """Dense local-site Hamiltonian for tiny real-space MPS checks."""
    nsites = int(nsites)
    nphonon = int(nphonon)
    local_dim = 4 * nphonon
    dim = local_dim**nsites
    powers = np.asarray([local_dim**site for site in range(nsites)], dtype=int)
    hamiltonian = np.zeros((dim, dim), dtype=float)

    def unpack(index: int):
        value = int(index)
        states = []
        for _site in range(nsites):
            local = value % local_dim
            value //= local_dim
            electronic = local // nphonon
            states.append((electronic & 1, (electronic >> 1) & 1, local % nphonon))
        return states

    def pack(states) -> int:
        index = 0
        for site, (up, down, phonon) in enumerate(states):
            electronic = int(up) + 2 * int(down)
            index += (electronic * nphonon + int(phonon)) * powers[site]
        return int(index)

    for col in range(dim):
        states = unpack(col)
        up_bits = sum(int(up) << site for site, (up, _down, _phonon) in enumerate(states))
        down_bits = sum(int(down) << site for site, (_up, down, _phonon) in enumerate(states))
        current_nup = _bit_count(up_bits)
        current_ndown = _bit_count(down_bits)
        diagonal = float(number_penalty) * (
            (current_nup - int(nup)) ** 2 + (current_ndown - int(ndown)) ** 2
        )

        for site, (up, down, phonon) in enumerate(states):
            charge = int(up) + int(down)
            diagonal += float(omega) * phonon + float(hubbard_u) * int(up) * int(down)
            if charge and phonon + 1 < nphonon:
                row_state = list(states)
                row_state[site] = (up, down, phonon + 1)
                hamiltonian[pack(row_state), col] += float(g) * charge * np.sqrt(phonon + 1)
            if charge and phonon > 0:
                row_state = list(states)
                row_state[site] = (up, down, phonon - 1)
                hamiltonian[pack(row_state), col] += float(g) * charge * np.sqrt(phonon)

        hamiltonian[col, col] += diagonal

        bits = _spin_orbital_bits(up_bits, down_bits, nsites)
        for site in range(nsites - 1):
            for spin_offset in (0, 1):
                left = 2 * site + spin_offset
                right = 2 * (site + 1) + spin_offset
                for create_orbital, annihilate_orbital in ((left, right), (right, left)):
                    applied = _apply_cdag_c(bits, create_orbital, annihilate_orbital)
                    if applied is None:
                        continue
                    new_bits, sign = applied
                    new_up_bits, new_down_bits = _split_spin_orbital_bits(new_bits, nsites)
                    row_state = []
                    for mode, (_up, _down, phonon) in enumerate(states):
                        row_state.append(
                            (
                                (new_up_bits >> mode) & 1,
                                (new_down_bits >> mode) & 1,
                                phonon,
                            )
                        )
                    hamiltonian[pack(row_state), col] += -float(t) * sign

    return 0.5 * (hamiltonian + hamiltonian.T)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-L", "--nsites", type=int, default=4)
    parser.add_argument("--nphonon", type=int, default=2)
    parser.add_argument("-t", "--hopping", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("-g", "--coupling", type=float, default=1.0)
    parser.add_argument("-U", "--hubbard-u", type=float, default=0.0)
    parser.add_argument("-D", "--bond-dim", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--mode-order", type=int, nargs="+", default=None)
    parser.add_argument("--skip-mps", action="store_true")
    parser.add_argument("--mps-bond-dim", type=int, default=4)
    parser.add_argument("--mps-sweeps", type=int, default=4)
    parser.add_argument("--number-penalty", type=float, default=100.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.nsites % 2:
        raise ValueError("default half filling needs even L.")
    nup = ndown = args.nsites // 2
    common = dict(
        nsites=args.nsites,
        t=args.hopping,
        omega=args.omega,
        g=args.coupling,
        hubbard_u=args.hubbard_u,
        nphonon=args.nphonon,
        nup=nup,
        ndown=ndown,
    )
    exact = spinful_holstein_hubbard_exact_energies(**common, nroots=1)[0]
    print(
        f"Half-filled spinful Holstein: L={args.nsites}, "
        f"Nup=Ndown={nup}, nphonon={args.nphonon}, "
        f"t={args.hopping:g}, omega={args.omega:g}, "
        f"g={args.coupling:g}, U={args.hubbard_u:g}"
    )
    print(f"ED fixed-sector E0 = {exact:.12f}")
    print("method                         energy          error       seconds   details")

    def emit(name: str, energy: float, seconds: float, details: str = ""):
        print(
            f"{name:28s} {energy: .12f} {energy - exact: .3e} "
            f"{seconds:9.3f}   {details}"
        )

    for bond_dim in args.bond_dim:
        start = perf_counter()
        result = SpinfulHolsteinElectronicFirstNARG(
            **common,
            bond_dim=int(bond_dim),
            mode_order=None if args.mode_order is None else tuple(args.mode_order),
        ).run(nroots=1)
        emit(
            f"NARG electronic-first D={bond_dim}",
            result.energies[0],
            perf_counter() - start,
            f"electronic_dim={result.electronic_dim}, last={result.steps[-1].kept}/{result.steps[-1].product_dim}",
        )

    if args.skip_mps:
        return

    start = perf_counter()
    hamiltonian = full_real_space_spinful_holstein_hamiltonian(
        args.nsites,
        args.nphonon,
        t=args.hopping,
        omega=args.omega,
        g=args.coupling,
        hubbard_u=args.hubbard_u,
        nup=nup,
        ndown=ndown,
        number_penalty=args.number_penalty,
    )
    print(f"built real-space local-product Hamiltonian in {perf_counter() - start:.3f} s")
    start = perf_counter()
    mps = TensorTrainLETTA(
        hamiltonian,
        (4 * args.nphonon,) * args.nsites,
        bond_dim=args.mps_bond_dim,
        seed=1,
    )
    result = mps.run(nsweeps=args.mps_sweeps, tol=1e-9)
    emit(
        f"real-space MPS D={args.mps_bond_dim}",
        result.energy,
        perf_counter() - start,
        f"sweeps={result.ncompleted}",
    )


if __name__ == "__main__":
    main()
