"""Propagate the full 24-mode quadratic pyrazine model with TD-DMRG."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pyqed.models.pyrazine import qvc
from pyqed.mps import fock_state
from pyqed.mps.lvc import (
    electronic_populations,
    product_terms_mpo,
)
from pyqed.units import au2fs


def _basis_counts(value: str, nmodes: int) -> list[int]:
    counts = [int(item) for item in value.split(",")]
    if len(counts) == 1:
        counts *= nmodes
    if len(counts) != nmodes or any(count <= 0 for count in counts):
        raise ValueError(
            f"nbas must be one positive integer or {nmodes} comma-separated integers."
        )
    return counts


def electronic_projectors(nstates: int, nbas: list[int]):
    dimensions = [nstates, *nbas]
    projectors = []
    for state in range(nstates):
        matrix = np.zeros((nstates, nstates))
        matrix[state, state] = 1.0
        projectors.append(
            product_terms_mpo(dimensions, [(1.0, {0: matrix})])
        )
    return projectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nbas", default="4")
    parser.add_argument("--bond", type=int, default=16)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--state", type=int, default=1)
    parser.add_argument("--cutoff", type=float, default=0.0)
    parser.add_argument(
        "--integrator",
        choices=("tdvp2", "hybrid"),
        default="tdvp2",
    )
    parser.add_argument("--hybrid-warmup", type=int, default=5)
    parser.add_argument("--hybrid-interval", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pyrazine_24mode_qvc_tddmrg.npz"),
    )
    args = parser.parse_args()

    model = qvc()
    nbas = _basis_counts(args.nbas, model.nmodes)
    psi0 = fock_state(
        nbas,
        state=args.state,
        nstates=model.nstates,
    )
    projectors = electronic_projectors(model.nstates, nbas)
    driver = model.TDDMRG(nbas=nbas, D=args.bond).run(
        psi0,
        dt=args.dt_fs / au2fs,
        steps=args.steps,
        interval=args.interval,
        e_ops=projectors,
        integrator=args.integrator,
        cutoff=args.cutoff,
        hybrid_warmup_steps=args.hybrid_warmup,
        hybrid_tdvp2_interval=args.hybrid_interval,
    )

    times_fs = np.concatenate(([0.0], driver.times * au2fs))
    populations = np.vstack(
        (
            electronic_populations(psi0),
            np.real(driver.observables),
        )
    )
    np.savez(
        args.output,
        time_fs=times_fs,
        populations=populations,
        energy_times_fs=driver.energy_times * au2fs,
        static_energies=driver.static_energies,
        energy_drift=driver.energy_drift,
        tdvp_truncation_errors=driver.tdvp_truncation_errors,
        bond_dimensions=driver.bond_dimensions,
        integrator_history=driver.integrator_history,
        nbas=np.asarray(nbas),
        bond_dimension=args.bond,
        cutoff=args.cutoff,
    )

    print(f"saved: {args.output}")
    print(f"final populations: {populations[-1]}")
    print(f"final norm: {np.real(driver.final_state.norm()):.16f}")
    print(f"max |energy drift|: {np.max(np.abs(driver.energy_drift)):.6e}")
    print(
        "max TDVP truncation error: "
        f"{np.max(driver.tdvp_truncation_errors):.6e}"
    )


if __name__ == "__main__":
    main()
