"""Run a small native-MD membrane CD calculation.

This is a local end-to-end demonstration for environments where OpenMM is not
installed.  It propagates a tiny all-atom toy membrane with PyQED native MD,
inserts a fixed chiral H2O2 chromophore into each saved frame, and runs
membrane-embedded molecular CD with explicit point charges.

For a production calculation, replace ``native_membrane_frames(...)`` with
OpenMM/CHARMM-GUI trajectory frames converted to :class:`pyqed.md.Atoms`.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.md.all_atom_lipid_membrane import build_membrane
from pyqed.md import Atoms, Langevin, set_maxwell_boltzmann_velocities
from pyqed.qchem import MembraneCD
from pyqed.units import au2angstrom, au2ev, au2fs, fs


H2O2_LOCAL = np.array(
    [
        [0.000, 0.000, 0.000],
        [2.740, 0.000, 0.000],
        [-0.850, 1.436, 0.000],
        [3.590, 1.436, 1.134],
    ],
    dtype=float,
)
H2O2_SYMBOLS = ("O", "O", "H", "H")


def native_membrane_frames(
    nframes=3,
    md_steps_per_frame=2,
    timestep_fs=0.002,
    temperature_K=20.0,
    seed=4,
):
    membrane, _ = build_membrane(
        nx=1,
        ny=1,
        waters_per_lipid=1,
        salt_pairs=0,
        coulomb_method="cutoff",
        pme_mesh=12,
    )
    set_maxwell_boltzmann_velocities(membrane, temperature=temperature_K, seed=seed)
    dynamics = Langevin(
        membrane,
        timestep=timestep_fs * fs,
        temperature_K=temperature_K,
        friction=1.0e-3,
    )

    frames = []
    energies = []
    for iframe in range(int(nframes)):
        if iframe:
            dynamics.run(int(md_steps_per_frame))
        energies.append(float(membrane.get_potential_energy()))
        frames.append(_cd_snapshot_from_membrane(membrane))
    return frames, np.asarray(energies, dtype=float), dynamics.get_time()


def _cd_snapshot_from_membrane(membrane):
    membrane_positions = membrane.get_positions()
    membrane_symbols = membrane.atom_symbols()
    lengths = np.asarray(membrane.get_cell().lengths(), dtype=float)
    chrom_center = np.array([0.5 * lengths[0], 0.5 * lengths[1], 0.0])
    local_center = np.mean(H2O2_LOCAL, axis=0)
    qm_positions = H2O2_LOCAL - local_center + chrom_center

    atom_records = [
        [symbol, tuple(coord)]
        for symbol, coord in zip(H2O2_SYMBOLS, qm_positions)
    ]
    atom_records.extend(
        [symbol, tuple(coord)]
        for symbol, coord in zip(membrane_symbols, membrane_positions)
    )
    snapshot = Atoms(atom_records, cell=membrane.get_cell(), pbc=membrane.get_pbc())

    charges = np.concatenate([
        np.zeros(len(H2O2_SYMBOLS), dtype=float),
        membrane.get_array("charges"),
    ])
    snapshot.set_array("charges", charges, float, ())

    leaflets = np.zeros(len(snapshot), dtype=int)
    z = membrane_positions[:, 2]
    leaflets[len(H2O2_SYMBOLS):] = np.where(z >= np.median(z), 1, -1)
    snapshot.set_array("leaflets", leaflets, int, ())
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nframes", type=int, default=3)
    parser.add_argument("--md-steps-per-frame", type=int, default=2)
    parser.add_argument("--timestep-fs", type=float, default=0.002)
    parser.add_argument("--temperature-K", type=float, default=20.0)
    parser.add_argument("--nstates", type=int, default=2)
    parser.add_argument("--basis", default="sto3g")
    parser.add_argument("--method", choices=("tda", "tddft"), default="tda")
    parser.add_argument("--output-prefix", default="/private/tmp/pyqed_real_membrane_cd")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    frames, md_energies, final_time = native_membrane_frames(
        nframes=args.nframes,
        md_steps_per_frame=args.md_steps_per_frame,
        timestep_fs=args.timestep_fs,
        temperature_K=args.temperature_K,
    )
    workflow = MembraneCD(
        frames,
        qm_indices=[0, 1, 2, 3],
        basis=args.basis,
        method=args.method,
        nstates=args.nstates,
        cutoff=12.0 / au2angstrom,
        embedding_pbc="nearest",
        cap_charge_distance=1.5 / au2angstrom,
        mf_run_kwargs={"verbose": 0, "max_cycle": 100},
    )
    result = workflow.run()
    x, signal = result.spectrum(width=0.35, units="ev")

    prefix = Path(args.output_prefix)
    table = prefix.with_suffix(".txt")
    with table.open("w") as handle:
        handle.write("# frame state excitation_eV rotatory_au depth_A ncharges md_energy_Ha\n")
        for iframe, frame in enumerate(result.frames):
            for istate, (energy, strength) in enumerate(
                zip(frame.cd_result.excitation_energies * au2ev, frame.cd_result.rotatory_strengths),
                start=1,
            ):
                handle.write(
                    f"{iframe:5d} {istate:5d} {energy:16.8f} {strength:16.8e} "
                    f"{frame.snapshot.depth * au2angstrom:12.6f} "
                    f"{len(frame.snapshot.charges):8d} {md_energies[iframe]:16.8f}\n"
                )

    spectrum = prefix.with_name(prefix.name + "_spectrum.txt")
    np.savetxt(spectrum, np.column_stack([x, signal]), header="energy_eV averaged_cd_arb")

    print(f"frames: {len(result.frames)}")
    print(f"md_time_fs: {final_time * au2fs:.8f}")
    print(f"md_energy_Ha_minmax: {md_energies.min():.8f} {md_energies.max():.8f}")
    print(f"table: {table}")
    print(f"spectrum: {spectrum}")
    for iframe, frame in enumerate(result.frames):
        energies = frame.cd_result.excitation_energies * au2ev
        strengths = frame.cd_result.rotatory_strengths
        print(
            f"frame {iframe}: charges={len(frame.snapshot.charges)} "
            f"E_eV={np.array2string(energies, precision=4)} "
            f"R={np.array2string(strengths, precision=4)}"
        )

    if not args.no_plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.axhline(0.0, color="0.75", linewidth=0.8)
        ax.plot(x, signal, color="#1f6f78", linewidth=2.0)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Averaged CD intensity (arb.)")
        ax.set_title("Native-MD Membrane CD")
        fig.tight_layout()
        figure = prefix.with_suffix(".png")
        fig.savefig(figure, dpi=200)
        print(f"figure: {figure}")


if __name__ == "__main__":
    main()
