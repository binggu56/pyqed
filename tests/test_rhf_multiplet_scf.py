"""Small integral references for symmetry-preserving RHF occupations."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyqed.qchem import Molecule, RHF

ATOM = 'H 1 0 0; H -.5 .8660254037844386 0; H -.5 -.8660254037844386 0'

@pytest.fixture(scope='module')
def reference():
    mol = Molecule(atom=ATOM, unit='bohr', basis='6-31g', charge=1, symmetry='D3h')
    mol.build(eri='s8', options={'eri_backend': 'python'})
    return RHF(mol).run(verbose=0, tol=1e-11)


def test_rhf_selects_complete_partner_occupation(reference):
    from pyqed.qchem.symmetry import SymmetryEigensolver
    from pyqed.qchem.hf.occupations import rhf_occupations
    solver = SymmetryEigensolver(reference.mol, reference.mo_coeff.T @ reference.mol.overlap)
    operator = np.zeros((6, 6))
    for basis, dimension in solver.blocks:
        values = [-3, 3] if dimension == 1 else [-2, 2]
        operator += basis @ np.diag(np.repeat(values, dimension)) @ basis.T
    energies, vectors = solver.eigh(operator)
    assert_allclose(operator @ vectors, vectors * energies, atol=1e-10)
    assert_allclose(rhf_occupations(energies, 4, solver.orbital_multiplet_ids), [0, 2, 2, 0, 0, 0])
    assert_allclose(rhf_occupations(energies, 6, solver.orbital_multiplet_ids), [2, 2, 2, 0, 0, 0])


@pytest.mark.parametrize('level_shift', [0., .2])
def test_non_aufbau_rhf_density_and_posthf_order(reference, monkeypatch, level_shift):
    import copy
    import importlib
    rhf = importlib.import_module('pyqed.qchem.hf.rhf')
    from pyqed.qchem.symmetry import SymmetryEigensolver
    mol = copy.copy(reference.mol)
    mol._nelec = 4
    overlap = mol.overlap
    solver = SymmetryEigensolver(mol, reference.mo_coeff.T @ overlap)
    operator = np.zeros((6, 6))
    for basis, dimension in solver.blocks:
        values = [-3, 3] if dimension == 1 else [-2, 2]
        operator += basis @ np.diag(np.repeat(values, dimension)) @ basis.T
    mol.hcore = overlap @ reference.mo_coeff @ operator @ reference.mo_coeff.T @ overlap
    monkeypatch.setattr(rhf, 'get_veff', lambda *a, **k: np.zeros((6, 6)))
    monkeypatch.setattr(rhf, 'get_veff_mo', lambda *a, **k: np.zeros((6, 6)))
    mf = RHF(mol); mf.chkfile = None
    mf.run(init_guess='hcore', level_shift=level_shift, stability_analysis=False)
    assert mf.scf_info['non_aufbau']
    assert mf.scf_info['occupation_policy'] == 'whole_multiplets'
    assert_allclose(mf.mo_energy, [-2, -2, -3, 2, 2, 3], atol=1e-9)
    assert_allclose(mf.mo_occ, [2, 2, 0, 0, 0, 0])
    assert_allclose(mf.dm, rhf.make_rdm1(mf.mo_coeff, mf.mo_occ), atol=1e-10)
    assert_allclose(np.trace(mf.dm @ overlap), 4, atol=1e-10)
    assert_allclose(mf.dm @ overlap @ mf.dm, 2 * mf.dm, atol=1e-10)
    for rep in mol.symmetry_info.ao_representations.values():
        assert_allclose(rep @ mf.dm @ rep.T, mf.dm, atol=1e-10)
    assert_allclose(mf.to_uhf().dm.sum(axis=0), mf.dm, atol=1e-10)
    # An external permutation must not change density or spin conversion.
    permutation = [2, 0, 3, 1, 4, 5]
    mf.mo_energy = mf.mo_energy[permutation]
    mf.mo_coeff = mf.mo_coeff[:, permutation]
    mf.mo_occ = mf.mo_occ[permutation]
    assert_allclose(mf.make_rdm1(), mf.dm, atol=1e-10)
    assert_allclose(mf.to_uhf().dm.sum(axis=0), mf.dm, atol=1e-10)
