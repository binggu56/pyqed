from copy import copy

import numpy as np
import pytest
from pyscf import ao2mo

from pyqed.qchem import Molecule
from pyqed.polariton.cavity import Cavity, Polariton


@pytest.mark.parametrize('packed', [False, True])
def test_ri_dressing_matches_same_integrals_packed(packed):
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='bohr')
    mol.build(eri='ri')
    factors = np.asarray(mol.eri_factors)
    if factors.ndim == 2:
        from pyscf import lib
        factors = lib.unpack_tril(factors)
    reference = copy(mol)
    reference.eri_factors = None
    reference.eri_s8 = ao2mo.restore(8, np.einsum('Pij,Pkl->ijkl', factors, factors), mol.nao)
    if packed:
        mol.eri_factors = factors[:, *np.tril_indices(mol.nao)]
    else:
        mol.eri_factors = factors
    original = mol.eri_factors.copy()
    point = Polariton(mol, Cavity(.24)).electronic_hamiltonian(coupling=-.15)(-1.9)
    dressed, offset, photon = point
    exact, offset_ref, photon_ref = Polariton(reference, Cavity(.24)).electronic_hamiltonian(coupling=-.15)(-1.9)
    out = dressed.eri_factors
    if packed:
        from pyscf import lib
        out = lib.unpack_tril(out)
    eri = np.einsum('Pij,Pkl->ijkl', out, out)
    np.testing.assert_allclose(eri, ao2mo.restore(1, exact.eri_s8, mol.nao), atol=1e-12)
    np.testing.assert_array_equal(mol.eri_factors, original)
    np.testing.assert_allclose(dressed.hcore, exact.hcore)
    assert (offset, photon) == (offset_ref, photon_ref)
    assert dressed.eri_s8 is None
    e_ri = point.RHF(verbose=0).run().e_tot
    e_ref = exact.RHF(verbose=0).run().e_tot
    np.testing.assert_allclose(e_ri, e_ref, atol=1e-9)
    from pyqed.qchem.mcscf.casscf import CASSCF
    energies = []
    for molecule, factorized in ((dressed, True), (exact, False)):
        mf = molecule.RHF(verbose=0).run()
        mc = CASSCF(mf, ncas=2, nelecas=2, use_cholesky=factorized, verbose=0)
        mc.state_average([0.5, 0.5])
        mc.run(nstates=2)
        assert mc.converged
        energies.append(mc.e_tot)
    np.testing.assert_allclose(*energies, atol=1e-8)


def test_scan_reuses_dressing_without_changing_coordinates():
    from pyqed.polariton.cavity import _prepare_cavity_integrals, _cavity_at_coordinate
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='bohr')
    mol.build(eri='ri')
    prepared, position, nuclear = _prepare_cavity_integrals(mol, .24, -.15, (0, 0, 1))
    hcore = prepared.hcore.copy()
    for q in (-2., 0., 3.):
        dressed, offset, photon = _cavity_at_coordinate(prepared, position, nuclear, .24, -.15, q)
        reference, ref_offset, ref_photon = Polariton(mol, Cavity(.24)).electronic_hamiltonian(coupling=-.15)(q)
        assert dressed.eri_factors is prepared.eri_factors
        np.testing.assert_allclose(dressed.hcore, reference.hcore)
        assert (offset, photon) == (ref_offset, ref_photon)
    np.testing.assert_array_equal(prepared.hcore, hcore)
