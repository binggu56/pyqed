import numpy as np
import pytest
from scipy.linalg import expm

from pyqed.qchem.basis import unpack_eri_s8
from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF
from pyqed.qchem.mcscf.orbopt import generalized_fock, create_integral_hessian_action
from pyqed.qchem.mcscf.integral_blocks import IntegralBlocks


def test_trial_blocks_preserve_integrals_gradient_and_hessian():
    rng = np.random.default_rng(24)
    n, no = 9, 4
    pairs = n * (n + 1) // 2
    eri = unpack_eri_s8(rng.normal(size=pairs * (pairs + 1) // 2), n)
    h = rng.normal(size=(n, n))
    h += h.T
    k = rng.normal(size=(n, n)) * .03
    u = expm(k - k.T)
    solver = object.__new__(SecondOrderCASSCF)
    solver.coupling, solver.ah_hessian = 'qn', 'analytic'
    solver.micro_ci_mode, solver.nstates, solver.ncas = 'full', 3, 3
    solver._default_ncore = lambda: 1
    hf, full = solver._transform_frozen_integrals(h, eri, u)
    hb, blocks = solver._transform_trial_integrals(h, eri, u)
    idx = np.indices((n,) * 4)
    required = np.sum(idx < no, axis=0) >= 2
    np.testing.assert_allclose(blocks[required], full[required], atol=1e-12)
    assert np.all(blocks[~required] == 0)
    np.testing.assert_allclose(hf, hb)
    d1, d2 = np.zeros((n, n)), np.zeros((n,) * 4)
    d1[:no, :no] = rng.normal(size=(no, no))
    d2[:no, :no, :no, :no] = rng.normal(size=(no,) * 4)
    np.testing.assert_allclose(generalized_fock(hb, blocks, d1, d2, nocc=no),
                               generalized_fock(hf, full, d1, d2), atol=1e-11)
    a = create_integral_hessian_action(hf, full, d1, d2, no)
    b = create_integral_hessian_action(hb, blocks, d1, d2, no)
    np.testing.assert_allclose(a(k-k.T), b(k-k.T), atol=1e-11)
    compact = IntegralBlocks(eri, u, no)
    np.testing.assert_allclose(compact.ppoo, full[:, :, :no, :no], atol=1e-12)
    np.testing.assert_allclose(compact.popo, full[:, :no, :, :no], atol=1e-12)
    np.testing.assert_allclose(generalized_fock(hb, compact, d1[:no, :no],
        d2[:no, :no, :no, :no]), generalized_fock(hf, full, d1, d2), atol=1e-11)
    action = create_integral_hessian_action(hb, compact, d1[:no, :no],
        d2[:no, :no, :no, :no], no)
    np.testing.assert_allclose(a(k-k.T), action(k-k.T), atol=1e-11)


def test_packed_blocks_and_rotations_match_dense_reference():
    rng = np.random.default_rng(82)
    n, no = 8, 3
    pairs = n * (n+1) // 2
    packed = rng.normal(size=pairs*(pairs+1)//2)
    c, _ = np.linalg.qr(rng.normal(size=(n, n)))
    dense = unpack_eri_s8(packed, n)
    blocks = IntegralBlocks(packed, c, no)
    k = rng.normal(size=(n, n)) * .05
    u = expm(k-k.T)
    rotated = blocks.rotate(u)
    ref = np.einsum('pi,qj,pqrs,rk,sl->ijkl', c@u, c@u, dense,
                    c@u, c@u, optimize=True)
    np.testing.assert_allclose(rotated.ppoo, ref[:, :, :no, :no], atol=1e-12)
    np.testing.assert_allclose(rotated.popo, ref[:, :no, :, :no], atol=1e-12)
    assert rotated.ppoo.size + rotated.popo.size == 2*n*n*no*no
    assert blocks.source is rotated.source
    dm = np.zeros((n, n))
    dm[0, 0] = 2
    np.testing.assert_allclose(rotated.veff(dm),
        np.einsum('rs,pqrs->pq', dm, ref)-.5*np.einsum('rs,prqs->pq', dm, ref), atol=1e-12)
    with pytest.raises(ValueError, match='core\\+active'):
        rotated.transform(*([np.eye(n)]*4))


@pytest.mark.parametrize('option,value', [('coupling', 'full'),
    ('ah_hessian', 'finite_difference'), ('micro_ci_mode', 'keyframe'),
    ('nstates', 1)])
def test_other_paths_keep_full_transform(option, value):
    solver = object.__new__(SecondOrderCASSCF)
    solver.coupling, solver.ah_hessian = 'qn', 'analytic'
    solver.micro_ci_mode, solver.nstates = 'full', 3
    setattr(solver, option, value)
    solver._transform_frozen_integrals = lambda *args: 'full'
    assert solver._transform_trial_integrals(None, np.zeros((2,)*4), np.eye(2)) == 'full'


def test_block_trial_casscf_matches_full_transform(monkeypatch):
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g', unit='angstrom').build()
    mf = mol.RHF().run()
    runs = []
    for blocks in (False, True):
        solver = SecondOrderCASSCF(mf, ncas=2, nelecas=2, max_cycle=40,
                                  conv_tol=1e-10, conv_tol_grad=1e-7,
                                  conv_tol_grad_relaxed=1e-7, verbose=0)
        solver.state_average([.5, .5])
        if not blocks:
            monkeypatch.setattr(solver, '_block_integral_source', lambda *args: None)
            monkeypatch.setattr(solver, '_transform_trial_integrals',
                                solver._transform_frozen_integrals)
        else:
            def forbidden(*args, **kwargs):
                raise AssertionError('Full MO integrals or padded RDMs requested')
            monkeypatch.setattr(mf, 'get_eri_mo', forbidden)
            monkeypatch.setattr(solver, '_effective_rdms', forbidden)
        solver.run(nstates=2)
        assert solver.converged
        runs.append(solver)
    np.testing.assert_allclose(runs[0].e_tot, runs[1].e_tot, atol=1e-7, rtol=0)
