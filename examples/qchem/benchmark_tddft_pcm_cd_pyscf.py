"""Benchmark native TDHF/TDDFT + PCM CD against PySCF.

This example uses a small gauche H2O2 geometry.  The molecule is non-planar, so
rotatory strengths are non-zero, while the calculation remains fast enough for
routine checks.
"""

import sys
from pathlib import Path

import numpy as np
from pyscf import gto, scf, solvent  # noqa: F401 - imports PCM method hooks

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.qchem import CD, Molecule, RHF, TDDFT
from pyqed.qchem.solvent.pcm import PCM
from pyqed.units import au2ev


ATOM = """
O 0.000000 0.000000 0.000000
O 1.450000 0.000000 0.000000
H -0.450000 0.760000 0.000000
H 1.900000 0.760000 0.600000
"""


def pyqed_reference_from_pyscf(pmf, mol, pmol):
    """Use the same PCM-relaxed orbitals in native pyqed TDDFT."""
    mf = RHF(mol)
    mf.mo_energy = np.array(pmf.mo_energy)
    mf.mo_coeff = np.array(pmf.mo_coeff)
    mf.mo_occ = np.array(pmf.mo_occ)
    mf.dm = np.array(pmf.make_rdm1())
    mf.hcore = np.array(mol.hcore)
    mf.vhf = np.array(pmf.get_veff(dm=pmf.make_rdm1()))
    mf.e_tot = float(pmf.e_tot)
    mf.e_nuc = float(pmol.energy_nuc())
    return mf


def main():
    nstates = 3

    pmol = gto.M(atom=ATOM, unit="Angstrom", basis="sto-3g", verbose=0)
    pmf = scf.RHF(pmol).PCM()
    pmf.with_solvent.lebedev_order = 3
    pmf.with_solvent.verbose = 0
    pmf.run(verbose=0)

    ptd = pmf.TDHF(equilibrium_solvation=False)
    ptd.nstates = nstates
    ptd.kernel()
    pyscf_rotatory = -np.einsum(
        "nx,nx->n",
        -ptd.transition_dipole(),
        0.5 * ptd.transition_magnetic_dipole(),
    )

    mol = Molecule(atom=ATOM, unit="angstrom", basis="sto-3g")
    mol.build()
    mf = pyqed_reference_from_pyscf(pmf, mol, pmol)

    pcm = PCM(mol)
    pcm.lebedev_order = 3
    pcm.verbose = 0
    td = TDDFT(mf).PCM(solvent_obj=pcm).run(nstates=nstates)
    cd = CD(td)
    result = cd.run()

    print("state  pyqed_e/eV    pyscf_e/eV    dE/meV      pyqed_R        pyscf_R")
    for istate, (e_qed, e_ref, r_qed, r_ref) in enumerate(
        zip(td.e, ptd.e, result.rotatory_strengths, pyscf_rotatory),
        start=1,
    ):
        print(
            f"{istate:5d}  {e_qed * au2ev:11.6f}  {e_ref * au2ev:11.6f}  "
            f"{(e_qed - e_ref) * au2ev * 1e3:8.4f}  {r_qed:12.8f}  {r_ref:12.8f}"
        )

    print(f"max |dE| = {np.max(np.abs(td.e - ptd.e)) * au2ev * 1e3:.4f} meV")
    print(
        "max |dR| = "
        f"{np.max(np.abs(result.rotatory_strengths - pyscf_rotatory)):.3e} a.u."
    )


if __name__ == "__main__":
    main()
