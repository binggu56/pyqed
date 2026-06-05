"""Molecular CD in membrane snapshots with explicit point-charge embedding.

This example uses two tiny toy snapshots so it can run as a smoke test.  The
QM chromophore is formamide, a minimal peptide-bond model for protein far-UV
CD.  In a production calculation, replace ``toy_membrane_snapshots()`` with
snapshots imported from an OpenMM/CHARMM-GUI membrane trajectory, keeping the
same charge array and ``qm_indices``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.md import Atoms
from pyqed.qchem import MembraneCD
from pyqed.units import au2angstrom, au2ev


FORMAMIDE_SYMBOLS = ("C", "O", "N", "H", "H", "H")
FORMAMIDE_LOCAL = np.array(
    [
        [0.000, 0.000, 0.000],
        [1.230, 0.000, 0.000],
        [-1.330, 0.000, 0.000],
        [0.000, -1.090, 0.100],
        [-1.850, 0.850, 0.150],
        [-1.850, -0.850, -0.150],
    ],
    dtype=float,
) / au2angstrom


def toy_membrane_snapshots():
    """Return two formamide peptide-bond snapshots in toy membrane charges."""

    snapshots = []
    for shift in (0.0, 0.2):
        qm_records = [
            [symbol, tuple(coord)]
            for symbol, coord in zip(FORMAMIDE_SYMBOLS, FORMAMIDE_LOCAL)
        ]
        atoms = Atoms(
            qm_records + [
                ["He", (7.0 + shift, 0.0, 0.0)],
                ["He", (-5.0, 2.0, 1.0)],
                ["He", (2.0, -6.5, -1.0)],
            ],
            cell=[20.0, 20.0, 20.0],
            pbc=True,
        )
        charges = np.concatenate([np.zeros(len(FORMAMIDE_SYMBOLS)), [-0.2, 0.1, 0.05]])
        leaflets = np.concatenate([np.zeros(len(FORMAMIDE_SYMBOLS), dtype=int), [1, -1, -1]])
        atoms.set_array("charges", charges, float, ())
        atoms.set_array("leaflets", leaflets, int, ())
        snapshots.append(atoms)
    return snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("tda", "tddft"), default="tda")
    parser.add_argument("--basis", default="sto3g")
    parser.add_argument("--nstates", type=int, default=2)
    parser.add_argument("--width-ev", type=float, default=0.4)
    parser.add_argument("--output-prefix", default="membrane_cd_workflow")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    cd = MembraneCD(
        toy_membrane_snapshots(),
        qm_indices=list(range(len(FORMAMIDE_SYMBOLS))),
        method=args.method,
        nstates=args.nstates,
        basis=args.basis,
        cutoff=9.0,
        embedding_pbc="nearest",
        cap_charge_distance=1.0,
        mf_run_kwargs={"verbose": 0, "max_cycle": 100},
    )
    result = cd.run()
    x, signal = result.spectrum(width=args.width_ev, units="ev")

    table_path = Path(f"{args.output_prefix}.txt")
    with table_path.open("w") as handle:
        handle.write("# frame state excitation_eV rotatory_au depth_bohr ncharges\n")
        for iframe, frame in enumerate(result.frames):
            energies = frame.cd_result.excitation_energies * au2ev
            strengths = frame.cd_result.rotatory_strengths
            for istate, (energy, strength) in enumerate(zip(energies, strengths), start=1):
                handle.write(
                    f"{iframe:5d} {istate:5d} {energy:16.8f} {strength:16.8e} "
                    f"{frame.snapshot.depth:16.8f} {len(frame.snapshot.charges):8d}\n"
                )

    spectrum_path = Path(f"{args.output_prefix}_spectrum.txt")
    np.savetxt(
        spectrum_path,
        np.column_stack([x, signal]),
        header="energy_eV averaged_cd_arb",
    )

    print(f"frames: {len(result.frames)}")
    print(f"method: {result.method}")
    print(f"table: {table_path}")
    print(f"spectrum: {spectrum_path}")

    if not args.no_plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.axhline(0.0, color="0.75", linewidth=0.8)
        ax.plot(x, signal, color="#1f6f78", linewidth=2.0)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Averaged CD intensity (arb.)")
        ax.set_title("Membrane-Embedded Peptide-Bond CD")
        fig.tight_layout()
        figure_path = Path(f"{args.output_prefix}.png")
        fig.savefig(figure_path, dpi=200)
        print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
