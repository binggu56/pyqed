"""B3LYP/aug-cc-pVDZ normal modes at the CASSCF(4,4) CHD geometry."""

import csv
from pathlib import Path

import numpy as np
from pyscf import dft, gto
from pyscf.hessian import thermo


GEOMETRY = Path("chd_c2_casscf44_augccpvdz.xyz")
OUTPUT_CSV = Path("chd_c2_b3lyp_augccpvdz_frequencies.csv")
OUTPUT_NPZ = Path("chd_c2_b3lyp_augccpvdz_normal_modes.npz")


def read_xyz(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[2 : 2 + int(lines[0])])


def main():
    mol = gto.M(
        atom=read_xyz(GEOMETRY),
        basis="aug-cc-pvdz",
        unit="Angstrom",
        charge=0,
        spin=0,
        symmetry="C2",
        verbose=4,
        output="chd_c2_b3lyp_augccpvdz_frequencies.log",
        max_memory=6000,
    )
    mf = dft.RKS(mol, xc="b3lyp")
    mf.grids.level = 4
    mf.conv_tol = 1.0e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("B3LYP SCF did not converge")

    gradient = mf.nuc_grad_method().kernel()
    hessian = mf.Hessian().kernel()
    modes = thermo.harmonic_analysis(
        mol, hessian, exclude_trans=True, exclude_rot=True, imaginary_freq=False
    )
    frequencies = np.asarray(modes["freq_wavenumber"], dtype=float)
    periods_fs = np.divide(
        33356.40952,
        np.abs(frequencies),
        out=np.full_like(frequencies, np.inf),
        where=frequencies != 0,
    )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["mode", "frequency_cm-1", "period_fs", "reduced_mass_amu"]
        )
        for index, (frequency, period, reduced_mass) in enumerate(
            zip(frequencies, periods_fs, modes["reduced_mass"]), start=1
        ):
            writer.writerow([index, frequency, period, reduced_mass])

    np.savez(
        OUTPUT_NPZ,
        symbols=np.asarray(mol.elements),
        coordinates_angstrom=mol.atom_coords(unit="Angstrom"),
        frequencies_cm1=frequencies,
        periods_fs=periods_fs,
        reduced_masses_amu=modes["reduced_mass"],
        normal_modes=modes["norm_mode"],
        hessian_au=hessian,
        energy_hartree=mf.e_tot,
        gradient_au=gradient,
    )
    print(f"E(B3LYP) = {mf.e_tot:.12f} Eh")
    print(
        "Gradient RMS/max = "
        f"{np.sqrt(np.mean(gradient**2)):.6e}/{np.max(np.abs(gradient)):.6e} Eh/Bohr"
    )
    print(f"Imaginary modes: {np.count_nonzero(frequencies < 0)}")
    for index, (frequency, period) in enumerate(
        zip(frequencies, periods_fs), start=1
    ):
        print(f"{index:2d} {frequency:12.4f} cm^-1 {period:10.3f} fs")


if __name__ == "__main__":
    main()
