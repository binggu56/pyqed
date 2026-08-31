#!/usr/bin/env python3
"""Compute and plot a finite-displacement periodic phonon spectrum."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.pbc import FiniteDisplacementPhonon, KRHFForceCalculator
from pyqed.qchem.pbc import Cell


class HarmonicChainForces:
    """Periodic nearest-neighbor force model used for a fast reproducible run."""

    def __init__(self, lattice_constant, ncell, spring_constant=0.3):
        self.equilibrium = np.zeros((ncell, 3), dtype=float)
        self.equilibrium[:, 0] = np.arange(ncell) * lattice_constant
        self.spring_constant = float(spring_constant)

    def forces(self, symbols, positions, lattice):
        del symbols, lattice
        displacement = np.asarray(positions) - self.equilibrium
        return -self.spring_constant * (
            2.0 * displacement
            - np.roll(displacement, 1, axis=0)
            - np.roll(displacement, -1, axis=0)
        )


def _model_phonon(displacement, spring_constant):
    cell = Cell(
        atom="He 0 0 0",
        a=np.diag([2.0, 7.0, 7.0]),
        basis="sto-3g",
        unit="bohr",
        integral_options={"eri_representation": "direct"},
    ).build()
    calculator = HarmonicChainForces(2.0, 3, spring_constant=spring_constant)
    return FiniteDisplacementPhonon(
        cell,
        calculator,
        supercell=(3, 1, 1),
        displacement=displacement,
        masses=[4.002602],
    )


def _krhf_phonon(displacement, supercell_x, recip_cut, *, gth_pade=False):
    pseudo = None
    if gth_pade:
        pseudo = {
            "H": [[1], 0.2, 2, [-4.1802368, 0.72507482], 0],
        }
    cell = Cell(
        atom="H 2.3 3.0 3.0; H 3.7 3.0 3.0",
        a=np.eye(3) * 6.0,
        basis="sto-3g",
        pseudo=pseudo,
        unit="bohr",
        integral_options={"eri_representation": "direct"},
    ).build()
    calculator = KRHFForceCalculator(
        "sto-3g",
        pseudo=pseudo,
        scf_options={
            "eta": 0.7,
            "real_cut": 0,
            "pair_cut": 2 if gth_pade else 0,
            "recip_cut": recip_cut,
            "pseudo_cut": 1,
            "one_body_nuclear_cut": 1,
            "eri_screen_tol": 0.0,
            "pair_ft_screen_tol": 0.0,
            "one_body_screen_tol": 0.0,
        },
        run_options={
            "max_cycle": 60,
            "conv_tol": 1.0e-10,
            "conv_tol_dm": 1.0e-8,
        },
    )
    return FiniteDisplacementPhonon(
        cell,
        calculator,
        supercell=(supercell_x, 1, 1),
        displacement=displacement,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("model", "krhf"), default="model")
    parser.add_argument("--displacement", type=float, default=0.01)
    parser.add_argument("--spring-constant", type=float, default=0.3)
    parser.add_argument("--supercell-x", type=int, default=2)
    parser.add_argument("--recip-cut", type=int, default=2)
    parser.add_argument(
        "--gth-pade",
        action="store_true",
        help="Use an explicit dependency-free GTH-Pade H pseudopotential.",
    )
    parser.add_argument("--points", type=int, default=61)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_phonon_spectrum.pdf"),
    )
    args = parser.parse_args()

    if args.backend == "model":
        phonon = _model_phonon(args.displacement, args.spring_constant)
    else:
        phonon = _krhf_phonon(
            args.displacement,
            args.supercell_x,
            args.recip_cut,
            gth_pade=args.gth_pade,
        )
    phonon.run()
    bands = phonon.band_structure(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
        labels=(r"$\Gamma$", "X", r"$\Gamma$"),
        points_per_segment=args.points,
    )

    fig, axis = plt.subplots(figsize=(4.8, 3.4))
    for branch in bands["frequencies"].T:
        axis.plot(bands["distances"], branch, color="#28666E", linewidth=1.35)
    for tick in bands["ticks"]:
        axis.axvline(bands["distances"][tick], color="#B8B8B8", linewidth=0.7)
    axis.axhline(0.0, color="#444444", linewidth=0.7)
    axis.set_xticks(
        bands["distances"][bands["ticks"]],
        bands["labels"],
    )
    axis.set_xlim(bands["distances"][0], bands["distances"][-1])
    axis.set_ylabel(r"Frequency (cm$^{-1}$)")
    title = (
        "Periodic harmonic-chain phonons"
        if args.backend == "model"
        else (
            r"Native GTH-KRHF H$_2$ phonons"
            if args.gth_pade
            else r"Native KRHF H$_2$ phonons"
        )
    )
    axis.set_title(title)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    png = args.output.with_suffix(".png")
    fig.savefig(png, dpi=300)
    data = args.output.with_suffix(".npz")
    np.savez(
        data,
        qpoints=bands["qpoints"],
        distances=bands["distances"],
        frequencies_cm1=bands["frequencies"],
        force_constants=phonon.force_constants,
    )
    print(f"acoustic sum-rule residual: {phonon.acoustic_sum_rule_residual:.3e}")
    calculator = phonon.force_calculator
    if getattr(calculator, "history", None):
        total_seconds = sum(record["seconds"] for record in calculator.history)
        print(f"native force evaluations: {len(calculator.history)} in {total_seconds:.3f} s")
    print(f"wrote {args.output}")
    print(f"wrote {png}")
    print(f"wrote {data}")


if __name__ == "__main__":
    main()
