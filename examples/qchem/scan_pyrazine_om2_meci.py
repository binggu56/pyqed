#!/usr/bin/env python3
"""Diagnostic native OM2-style/MECI scan along pyrazine Hessian modes."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.units import angstrom2au, au2ev
from pyqed.qchem.semiempirical import OM2


ANGSTROM_TO_BOHR = angstrom2au
HARTREE_TO_EV = au2ev
DEFAULT_HESSIAN = (
    Path.home()
    / "Library/CloudStorage/OneDrive-西湖大学"
    / "manuscripts/SD/calculations/real_smolyak_20260803"
    / "pyrazine_casci44_6-31g_hess-6-31g_1x1.npz"
)


def atom_string(symbols, coordinates_bohr):
    coordinates = np.asarray(coordinates_bohr) / ANGSTROM_TO_BOHR
    return "; ".join(
        f"{symbol} {x:.12f} {y:.12f} {z:.12f}"
        for symbol, (x, y, z) in zip(symbols, coordinates)
    )


def main():
    data = np.load(DEFAULT_HESSIAN)
    reference = np.asarray(data["reference_bohr"])
    modes = np.asarray(data["modes"])
    symbols = ("N", "C", "C", "N", "C", "C", "H", "H", "H", "H")
    coordinates = np.linspace(-0.35, 0.35, 9)
    output = Path("/private/tmp/pyrazine_om2_meci_scan")

    energies = np.empty((2, len(coordinates), 4))
    overlaps = np.empty((2, len(coordinates) - 1))
    residuals = np.empty((2, len(coordinates)))
    for axis in range(2):
        states = []
        for point, coordinate in enumerate(coordinates):
            geometry = reference + coordinate * modes[axis]
            om2 = OM2(atom=atom_string(symbols, geometry), unit="angstrom").run()
            if not om2.reference.converged:
                raise RuntimeError(f"OM2 SCF failed for mode {axis}, Q={coordinate:g}")
            ci = om2.MECI(
                nstates=4,
                ncas=6,
                spin_penalty=10.0,
                target_spin=0.0,
            ).run()
            energies[axis, point] = ci.e
            residuals[axis, point] = om2.reference.residual_norm
            states.append(ci)
        for point in range(len(coordinates) - 1):
            overlap = states[point].wavefunction_overlap(states[point + 1])
            overlaps[axis, point] = np.linalg.norm(
                overlap - np.diag(np.diag(overlap))
            )

    gaps = (energies - energies[..., :1]) * HARTREE_TO_EV
    midpoints = 0.5 * (coordinates[:-1] + coordinates[1:])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73")
    labels = (r"$S_1$", r"$S_2$", r"$S_3$")
    for mode, linestyle in enumerate(("-", "--")):
        for state, (color, label) in enumerate(zip(colors, labels), start=1):
            axes[0].plot(
                coordinates,
                gaps[mode, :, state],
                color=color,
                linestyle=linestyle,
                label=label if mode == 0 else None,
            )
        axes[1].plot(
            midpoints,
            overlaps[mode],
            linestyle=linestyle,
            color=("#0072B2", "#D55E00")[mode],
            label=rf"Mode {mode + 1}",
        )
    axes[0].set(xlabel=r"Mode displacement $Q$", ylabel="Vertical gap / eV")
    axes[1].set(
        xlabel=r"Link midpoint $Q$",
        ylabel=r"Off-diagonal $|S_{i,i+1}|_F$",
    )
    axes[0].legend(frameon=False, ncol=3)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(False)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    np.savez(
        output.with_suffix(".npz"),
        coordinates=coordinates,
        energies_Eh=energies,
        gaps_eV=gaps,
        adjacent_offdiagonal_overlap=overlaps,
        residual_norm=residuals,
        mode_ids=data["mode_ids"],
        frequencies_cm1=data["frequencies_cm1"],
    )
    print(f"maximum SCF residual: {residuals.max():.3e}")
    print(f"minimum S1/S2 gap: {np.min(gaps[..., 2] - gaps[..., 1]):.6f} eV")
    print(f"maximum adjacent off-diagonal overlap: {overlaps.max():.6f}")
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
