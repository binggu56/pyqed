"""Bounded RI rotation and current/frozen QN RDM ownership."""
import weakref

import numpy as np
import pytest
from scipy.linalg import expm

from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF


@pytest.mark.parametrize('naux', [0, 1, 11])
@pytest.mark.parametrize('complex_values', [False, True])
def test_factor_rotation_rectangular_spaces(naux, complex_values):
    rng = np.random.default_rng(105)
    b = rng.normal(size=(naux, 7, 7))
    u = rng.normal(size=(7, 4))
    if complex_values:
        b = b + 1j*rng.normal(size=b.shape)
        u = u + 1j*rng.normal(size=u.shape)
    solver = object.__new__(SecondOrderCASSCF)
    h, rotated = solver._transform_frozen_factor_integrals(np.eye(7), b, u)
    np.testing.assert_allclose(h, u.conj().T @ u, atol=1e-12)
    np.testing.assert_allclose(rotated, u.conj().T @ (b @ u), atol=1e-12)


def test_factor_rotation_bounds_scratch(monkeypatch):
    rng = np.random.default_rng(204)
    b = rng.normal(size=(507, 129, 129))
    u, _ = np.linalg.qr(rng.normal(size=(129, 129)))
    original = np.matmul
    calls = []
    def checked(left, right, *, out):
        calls.append((left.shape, out.shape, out.nbytes))
        return original(left, right, out=out)
    monkeypatch.setattr(np, 'matmul', checked)
    solver = object.__new__(SecondOrderCASSCF)
    _, actual = solver._transform_frozen_factor_integrals(np.eye(129), b, u)
    assert len(calls) == 4  # two products in each of two auxiliary blocks
    assert max(size for _, _, size in calls) <= 64 << 20
    for start in range(0, len(b), 17):
        np.testing.assert_allclose(actual[start:start+17],
            u.T @ (b[start:start+17] @ u), atol=1e-12, rtol=0)


@pytest.mark.parametrize('factorized', [False, True])
def test_qn_releases_current_rdms_before_trials(monkeypatch, factorized):
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(eri='factors' if factorized else 's8')
    mf = mol.RHF().run()
    solver = SecondOrderCASSCF(mf, ncas=2, nelecas=2, coupling='qn',
        conv_tol_grad=1e-4, conv_tol_grad_relaxed=1e-4, max_cycle=20)
    solver.state_average([.5, .5])
    monkeypatch.setattr(solver, '_factor_block_source', lambda *args: None)
    monkeypatch.setattr(solver, '_block_integral_source', lambda *args: None)
    name = '_effective_factor_rdms' if factorized else '_effective_rdms'
    original_rdms = getattr(solver, name)
    refs, trials = [], []
    def tracked(*args):
        assert all(ref() is None for ref in refs)
        values = original_rdms(*args)
        refs[:] = [weakref.ref(value) for value in values]
        return values
    monkeypatch.setattr(solver, name, tracked)
    name = '_factor_micro_line_search' if factorized else '_micro_line_search'
    original_search = getattr(solver, name)
    def checked(*args, **kwargs):
        assert all(ref() is None for ref in refs)
        trials.append(1)
        return original_search(*args, **kwargs)
    monkeypatch.setattr(solver, name, checked)
    k = np.zeros((mf.nmo, mf.nmo))
    k[1, 3], k[3, 1] = .2, -.2
    solver.run(nstates=2, mo_coeff=mf.mo_coeff @ expm(k))
    assert solver.converged and trials
