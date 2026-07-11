"""Membrane-embedded CD from an equilibrated OpenMM PDB system.

This is the production-style entry point for membrane CD:

1. Read an equilibrated PDB with an OpenMM topology and periodic box.
2. Build an OpenMM ``System`` from user-supplied force-field XML files.
3. Extract MM point charges from the OpenMM ``NonbondedForce``.
4. Select a QM chromophore by residue metadata or explicit atom indices.
5. Run PyQED membrane-embedded CD, or write a setup figure for inspection.

Example
-------
python examples/qchem/membrane_cd_from_openmm_pdb.py \\
    --pdb charmm_gui/step5_equilibrated.pdb \\
    --forcefield charmm36.xml --forcefield charmm36/water.xml \\
    --qm-resname IND --setup-figure /tmp/membrane_cd_setup.png --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.md import atoms_from_openmm_pdb
from pyqed.qchem import MembraneCD
from pyqed.units import au2angstrom, au2ev


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", required=True, help="Equilibrated OpenMM-readable PDB.")
    parser.add_argument(
        "--forcefield",
        action="append",
        required=True,
        help="OpenMM force-field XML file. Repeat for multiple files.",
    )
    parser.add_argument("--frame", action="append", type=int, default=None, help="PDB model/frame index. Repeatable.")
    parser.add_argument("--qm-indices", help="Comma-separated zero-based QM atom indices.")
    parser.add_argument("--qm-resname", help="Select QM atoms by residue name, e.g. IND or TRP.")
    parser.add_argument("--qm-resid", help="Select QM atoms by residue id. Comma-separated values allowed.")
    parser.add_argument("--qm-chain", help="Select QM atoms by chain id. Comma-separated values allowed.")
    parser.add_argument("--qm-atom-names", help="Optional comma-separated atom-name filter inside the selected residue.")
    parser.add_argument("--method", choices=("tda", "tddft", "casci"), default="tda")
    parser.add_argument("--basis", default="sto3g")
    parser.add_argument("--nstates", type=int, default=2)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--cutoff-A", type=float, default=12.0)
    parser.add_argument("--cap-charge-distance-A", type=float, default=1.5)
    parser.add_argument("--width-ev", type=float, default=0.35)
    parser.add_argument("--output-prefix", default="/private/tmp/pyqed_openmm_pdb_membrane_cd")
    parser.add_argument("--setup-figure", help="Write a side/top setup figure before running CD.")
    parser.add_argument("--list-residues", action="store_true", help="Print imported residues and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Only import/select/write setup figure; do not run quantum CD.")
    parser.add_argument("--no-plot", action="store_true", help="Skip final CD spectrum plot.")
    return parser.parse_args()


def main():
    args = parse_args()
    frames = args.frame if args.frame is not None else [0]
    qm_indices = _parse_indices(args.qm_indices)

    imported = [
        atoms_from_openmm_pdb(
            args.pdb,
            forcefield_files=args.forcefield,
            frame=frame,
            qm_indices=qm_indices,
            qm_resname=args.qm_resname,
            qm_resid=args.qm_resid,
            qm_chain=args.qm_chain,
            qm_atom_names=args.qm_atom_names,
        )
        for frame in frames
    ]
    if args.list_residues:
        print_residue_table(imported[0])
        return

    selected = imported[0].qm_indices
    if selected is None:
        raise ValueError("Specify --qm-indices or QM metadata filters such as --qm-resname.")
    for item in imported[1:]:
        if not np.array_equal(item.qm_indices, selected):
            raise ValueError("QM selection changed across frames; use explicit --qm-indices.")

    if args.setup_figure:
        write_setup_figure(imported[0], args.setup_figure)

    print(f"frames: {len(imported)}")
    print(f"atoms: {len(imported[0].atoms)}")
    print(f"qm_atoms: {len(selected)}")
    print(f"qm_indices: {','.join(str(int(i)) for i in selected)}")
    print(f"total_mm_charge_e: {float(np.sum(imported[0].atoms.get_array('charges'))):.8f}")

    if args.dry_run:
        return

    workflow = MembraneCD(
        [item.atoms for item in imported],
        qm_indices=selected,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        method=args.method,
        nstates=args.nstates,
        cutoff=args.cutoff_A / au2angstrom,
        embedding_pbc="nearest",
        cap_charge_distance=args.cap_charge_distance_A / au2angstrom,
        build_kwargs={"eri": "s8"},
        mf_run_kwargs={"verbose": 0, "max_cycle": 100},
    )
    result = workflow.run()
    x, signal = result.spectrum(width=args.width_ev, units="ev")

    prefix = Path(args.output_prefix)
    table = prefix.with_suffix(".txt")
    with table.open("w") as handle:
        handle.write("# frame state excitation_eV rotatory_au depth_A ncharges\n")
        for iframe, frame in enumerate(result.frames):
            energies = frame.cd_result.excitation_energies * au2ev
            strengths = frame.cd_result.rotatory_strengths
            for istate, (energy, strength) in enumerate(zip(energies, strengths), start=1):
                handle.write(
                    f"{frames[iframe]:5d} {istate:5d} {energy:16.8f} "
                    f"{strength:16.8e} {frame.snapshot.depth * au2angstrom:12.6f} "
                    f"{len(frame.snapshot.charges):8d}\n"
                )

    spectrum = prefix.with_name(prefix.name + "_spectrum.txt")
    np.savetxt(spectrum, np.column_stack([x, signal]), header="energy_eV averaged_cd_arb")
    print(f"table: {table}")
    print(f"spectrum: {spectrum}")

    if not args.no_plot:
        import matplotlib.pyplot as plt

        figure = prefix.with_suffix(".png")
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.axhline(0.0, color="0.75", linewidth=0.8)
        ax.plot(x, signal, color="#246c74", linewidth=2.0)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Averaged CD intensity (arb.)")
        ax.set_title("OpenMM PDB Membrane-Embedded CD")
        fig.tight_layout()
        fig.savefig(figure, dpi=200)
        print(f"figure: {figure}")


def write_setup_figure(imported, path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    atoms = imported.atoms
    positions = atoms.get_positions() * au2angstrom
    charges = atoms.get_array("charges")
    qm = np.zeros(len(atoms), dtype=bool)
    qm[imported.qm_indices] = True
    mm = ~qm

    box = np.asarray(atoms.get_cell().lengths(), dtype=float) * au2angstrom
    if not np.all(box > 0.0):
        span = np.ptp(positions, axis=0)
        box = np.maximum(span, 1.0)

    charge_scale = np.clip(np.abs(charges), 0.05, 1.0)
    colors = np.where(charges >= 0.0, "#2f7f91", "#8b7540")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    fig.patch.set_facecolor("#f7f3e8")
    for ax in axes:
        ax.set_facecolor("#fffdf8")
        ax.grid(True, color="#d9d0bf", linewidth=0.65, alpha=0.75)

    axes[0].scatter(
        positions[mm, 0],
        positions[mm, 2],
        s=18.0 * charge_scale[mm],
        c=colors[mm],
        alpha=0.45,
        label="MM point charges",
    )
    axes[0].scatter(
        positions[qm, 0],
        positions[qm, 2],
        s=64,
        c="#d44a32",
        edgecolors="#4b1710",
        linewidths=0.7,
        label="QM chromophore",
        zorder=5,
    )
    axes[0].axhline(0.0, color="#33332f", linewidth=0.9, linestyle="--", alpha=0.5)
    axes[0].add_patch(Rectangle((0, -0.5 * box[2]), box[0], box[2], fill=False, linewidth=1.3, edgecolor="#33332f"))
    axes[0].set_xlabel("x (Angstrom)")
    axes[0].set_ylabel("z / membrane normal (Angstrom)")
    axes[0].set_title("Side view")

    axes[1].scatter(
        positions[mm, 0],
        positions[mm, 1],
        s=18.0 * charge_scale[mm],
        c=colors[mm],
        alpha=0.40,
    )
    axes[1].scatter(
        positions[qm, 0],
        positions[qm, 1],
        s=64,
        c="#d44a32",
        edgecolors="#4b1710",
        linewidths=0.7,
        zorder=5,
    )
    axes[1].add_patch(Rectangle((0, 0), box[0], box[1], fill=False, linewidth=1.3, edgecolor="#33332f"))
    axes[1].set_xlabel("x (Angstrom)")
    axes[1].set_ylabel("y (Angstrom)")
    axes[1].set_title("Top view")
    axes[1].set_aspect("equal", adjustable="box")

    fig.suptitle("Imported OpenMM Membrane-CD Setup", fontsize=16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.text(
        0.5,
        0.09,
        f"Imported frame {imported.frame}: {len(atoms)} atoms, {int(qm.sum())} QM atoms; MM charge size/color from OpenMM NonbondedForce",
        ha="center",
        fontsize=10.2,
        color="#504a42",
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.20, wspace=0.23)
    fig.savefig(path, dpi=220)


def print_residue_table(imported):
    rows = {}
    for record in imported.atom_records:
        key = (record.chain_id, record.residue_id, record.residue_name)
        rows.setdefault(key, []).append(record)
    print("# chain resid resname natoms atom_names")
    for (chain, resid, resname), records in rows.items():
        names = ",".join(record.name for record in records[:12])
        if len(records) > 12:
            names += ",..."
        print(f"{chain:>5s} {resid:>5s} {resname:>7s} {len(records):6d} {names}")


def _parse_indices(text):
    if text is None or not str(text).strip():
        return None
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


if __name__ == "__main__":
    main()
