"""Clamped H2 with an axial static field and a single-mode dipole self-energy."""

import numpy as np
from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.casscf import CASSCF
from pyqed.polariton.cavity import dress_cavity_integrals
from pyqed.units import eV


OMEGA_C = 6.0 * eV
COUPLING = 0.05


def calculate(q, coupling=COUPLING):
    mol = Molecule(atom="H 0 0 -0.7; H 0 0 0.7", basis="6-31g", unit="bohr")
    mol.build(eri="s8")
    z = mol.position_integral()[2]
    pmol = mol.topyscf()
    ao = mol.pyscf_ao_permutation(pmol)
    z2 = pmol.intor("int1e_rr", comp=9)[8][np.ix_(ao, ao)]

    mol, offset, _ = dress_cavity_integrals(mol, OMEGA_C, coupling, q)

    mc = CASSCF(mol.RHF(verbose=0).run(), 2, 2, verbose=0).run()
    if not mc.converged:
        raise RuntimeError("CASSCF did not converge")
    c = mc.mo_coeff[:, :2]  # two active orbitals and no frozen core
    d1, d2 = mc.casci.make_rdm12(0)
    z_active = c.T @ z @ c
    dse = 0.5 * coupling**2 * (
        np.einsum("pq,qp", c.T @ z2 @ c, d1)
        + np.einsum("pq,rs,pqrs", z_active, z_active, d2)
    )
    return float(np.ravel(mc.e_tot)[0] + offset), float(dse)


if __name__ == "__main__":
    q = 0.01 / (OMEGA_C * COUPLING)
    energy, dse = calculate(q)
    photon = 0.5 * (OMEGA_C * q)**2
    print(f"omega_c = {OMEGA_C:.10f} Ha, q = {q:.10f} a.u.")
    print(f"Molecular energy = {energy:.10f} Ha, DSE expectation = {dse:.10f} Ha")
    print(f"Photon potential = {photon:.10f} Ha, fixed-q energy = {energy + photon:.10f} Ha")
