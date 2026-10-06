import numpy as np
import pytest
from scipy.linalg import expm

from pyqed.qchem import Molecule, CASSCF
from pyqed.qchem.mcscf import direct_ci
from pyqed.qchem.mcscf.direct_ci import spin_square_from_rdm


def test_casscf_reuses_dense_slater_condon_tables(monkeypatch):
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g').build()
    mf = mol.RHF().run()
    original = direct_ci.SlaterCondon
    builds = []
    def counted(binary):
        builds.append(1)
        return original(binary)
    monkeypatch.setattr(direct_ci, 'SlaterCondon', counted)
    kappa = np.zeros((mf.nmo, mf.nmo))
    kappa[1, 3], kappa[3, 1] = .2, -.2
    mc = CASSCF(mf, ncas=2, nelecas=2, conv_tol_grad=1e-4,
                conv_tol_grad_relaxed=1e-4).state_average([.5, .5])
    mc.run(nstates=2, mo_coeff=mf.mo_coeff@expm(kappa))
    assert mc.converged and len(mc.micro_history) > 1
    assert len(builds) == 1


@pytest.mark.parametrize('complex_ci', [False, True])
def test_dense_spin_action_matches_rdm_and_invalidates_basis(monkeypatch, complex_ci):
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g').build()
    mc = direct_ci.CASCI(mol.RHF().run(), ncas=2, nelecas=2).run(
        nstates=2, method='direct_spin0_symm')
    assert mc.solver_backend == 'direct_spin0_symm_dense'
    rng = np.random.default_rng(131)
    vector = rng.normal(size=mc.binary.shape[0])
    if complex_ci:
        vector = vector+1j*rng.normal(size=vector.size)
    vector /= np.linalg.norm(vector)
    mc.ci = [vector]
    expected = spin_square_from_rdm(*mc.make_rdm12(0))
    monkeypatch.setattr(mc, 'make_rdm12', lambda *args: pytest.fail('Unexpected RDM construction'))
    np.testing.assert_allclose(mc.spin_square(), expected, atol=1e-12)
    operator = mc._s2_dense_cache[1]
    np.testing.assert_allclose(mc.spin_square(), expected, atol=1e-12)
    assert mc._s2_dense_cache[1] is operator
    mc.binary = mc.binary[::-1].copy()
    mc.ci = [vector[::-1].copy()]
    np.testing.assert_allclose(mc.spin_square(), expected, atol=1e-12)
    assert mc._s2_dense_cache[1] is not operator
