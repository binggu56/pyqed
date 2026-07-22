"""Optimize neutral 1,3-cyclohexadiene at CASSCF(4,4)/aug-cc-pVDZ.

The starting coordinates are the PubChem 3D conformer for CID 11605.  The
active orbitals are the two occupied pi and two virtual pi* RHF orbitals;
diffuse aug-cc-pVDZ virtual orbitals make a simple HOMO/LUMO window invalid.
"""

from pathlib import Path
import sys

import numpy as np
from pyscf import gto, mcscf, scf


STARTING_GEOMETRY = """
C  -0.7887  -1.2644  -0.0907
C   0.7409  -1.2929   0.0907
C  -1.4174   0.0949   0.0247
C   1.4200   0.0421  -0.0247
C  -0.6959   1.2236   0.0478
C   0.7411   1.1968  -0.0479
H  -1.2452  -1.9353   0.6468
H  -1.0422  -1.6726  -1.0772
H   1.1722  -1.9804  -0.6468
H   0.9791  -1.7102   1.0772
H  -2.5007   0.1509   0.0771
H   2.5045   0.0576  -0.0771
H  -1.1819   2.1901   0.1228
H   1.2628   2.1446  -0.1228
"""


def write_xyz(mol, filename, comment):
    coords = mol.atom_coords(unit="Angstrom")
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write(f"{mol.natm}\n{comment}\n")
        for symbol, coord in zip(mol.elements, coords):
            handle.write(
                f"{symbol:2s} {coord[0]: .12f} {coord[1]: .12f} {coord[2]: .12f}\n"
            )


def main():
    # geomeTRIC was installed separately for this calculation. Appending its
    # directory avoids shadowing PySCF's NumPy/SciPy with temporary copies.
    geometric_path = Path("/private/tmp/pyqed-geometric")
    if geometric_path.exists():
        sys.path.append(str(geometric_path))
    from pyscf.geomopt.geometric_solver import optimize

    mol = gto.M(
        atom=STARTING_GEOMETRY,
        basis="aug-cc-pvdz",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=4,
        output="chd_casscf44_augccpvdz_opt.log",
        max_memory=6000,
    )
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1.0e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Initial RHF calculation did not converge")

    mc = mcscf.CASSCF(mf, 4, 4).density_fit()
    # PySCF uses one-based orbital indices in sort_mo. These are the two
    # occupied pi orbitals followed by the two valence pi* orbitals.
    mo = mc.sort_mo([21, 22, 31, 53])
    mc.conv_tol = 1.0e-8
    mc.max_cycle_macro = 80
    mc.kernel(mo)
    if not mc.converged:
        raise RuntimeError("Initial CASSCF calculation did not converge")

    mol_eq = optimize(
        mc,
        maxsteps=100,
        convergence_energy=1.0e-6,
        convergence_grms=3.0e-4,
        convergence_gmax=4.5e-4,
        convergence_drms=1.2e-3,
        convergence_dmax=1.8e-3,
    )

    # Re-evaluate at the reported geometry so the final energy and gradient
    # are explicitly recorded rather than inferred from the optimizer log.
    mf_final = scf.RHF(mol_eq).density_fit().run(conv_tol=1.0e-10)
    mc_final = mcscf.CASSCF(mf_final, 4, 4).density_fit()
    mo_final = mc_final.sort_mo([21, 22, 31, 53])
    mc_final.conv_tol = 1.0e-8
    mc_final.max_cycle_macro = 80
    mc_final.kernel(mo_final)
    gradient = mc_final.nuc_grad_method().kernel()

    energy = float(mc_final.e_tot)
    grad_rms = float(np.sqrt(np.mean(gradient**2)))
    grad_max = float(np.max(np.abs(gradient)))
    comment = (
        "CASSCF(4,4)/aug-cc-pVDZ neutral singlet; "
        f"E={energy:.12f} Eh; gradient RMS={grad_rms:.3e}, max={grad_max:.3e} Eh/Bohr"
    )
    write_xyz(mol_eq, "chd_casscf44_augccpvdz.xyz", comment)
    print(comment)


if __name__ == "__main__":
    main()
