"""Compact core/active contractions against explicit embedded RDMs."""
import numpy as np
import pytest
from scipy.linalg import expm

from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF
from pyqed.qchem.mcscf import orbopt


def embedded(core, d1, d2):
    active, occupied = len(d1), core+len(d1)
    one = np.zeros((occupied,)*2, dtype=d1.dtype)
    two = np.zeros((occupied,)*4, dtype=d2.dtype)
    one[:core, :core] = 2*np.eye(core)
    one[core:, core:] = d1
    identity = np.eye(core)
    two[:core, :core, :core, :core] = (
        4*np.einsum('ij,kl->ijkl', identity, identity)
        -2*np.einsum('il,kj->ijkl', identity, identity))
    for i in range(core):
        two[i, i, core:, core:] = two[core:, core:, i, i] = 2*d1
        two[i, core:, i, core:] = two[core:, i, core:, i] = -d1
    two[core:, core:, core:, core:] = d2
    return one, two


@pytest.mark.parametrize('core,active', [(0,3), (1,2), (3,2), (2,0)])
@pytest.mark.parametrize('complex_values', [False, True])
def test_contraction_matches_embedding(core, active, complex_values):
    rng = np.random.default_rng(127)
    d1, d2 = rng.normal(size=(active,)*2), rng.normal(size=(active,)*4)
    b = rng.normal(size=(5, core+active, core+active))
    if complex_values:
        d1 = d1+1j*rng.normal(size=d1.shape)
        d2 = d2+1j*rng.normal(size=d2.shape)
        b = b+1j*rng.normal(size=b.shape)
    density = orbopt.CoreActiveRDM(core, d1, d2)
    _, full = embedded(core, d1, d2)
    np.testing.assert_allclose(density.contract(b),
        np.einsum('Pst,rqst->Prq', b, full), atol=2e-12)
    frozen = density.copy()
    assert not np.shares_memory(frozen.dm2, density.dm2)


@pytest.mark.parametrize('core,active', [(0,3), (1,2), (3,2)])
@pytest.mark.parametrize('workspace', [False, True])
def test_gradient_hessian_match_dense(core, active, workspace, monkeypatch):
    rng = np.random.default_rng(218)
    n, occupied = core+active+2, core+active
    d1 = rng.normal(size=(active,)*2)
    d2 = rng.normal(size=(active,)*4)
    one, full = embedded(core, d1, d2)
    density = orbopt.CoreActiveRDM(core, d1, d2)
    b = rng.normal(size=(7,n,n)); b += b.transpose(0,2,1).copy()
    h = rng.normal(size=(n,n)); h += h.T.copy()
    k = rng.normal(size=(3,n,n)); k -= k.transpose(0,2,1).copy()
    f = orbopt.generalized_fock_from_factors(h, b[:,:,:occupied], one, full)
    np.testing.assert_allclose(orbopt.generalized_fock_from_factors(
        h,b[:,:,:occupied],one,density), f, atol=2e-11)
    solver = object.__new__(SecondOrderCASSCF)
    dense = solver._orbital_hessian_action_from_factors_batch(h,b,one,full,k)
    if not workspace:
        monkeypatch.setattr(orbopt, '_cpp_attr', lambda *args: None)
    else:
        from pyqed.qchem import _casscf_cpp
        if not _casscf_cpp.native_capabilities().get('core_active_hessian'):
            pytest.skip('Rebuild CBLAS extension for compact workspace validation')
    cache = {}
    compact = solver._orbital_hessian_action_from_factors_batch(h,b,one,density,k,
                                                              factor_cache=cache)
    if workspace:
        assert cache['native_workspace'] is not None
    np.testing.assert_allclose(compact, dense, atol=4e-11, rtol=2e-12)
    scalar = solver._orbital_hessian_action_from_factors(h,b,one,density,k[0],
                                                       factor_cache=cache)
    np.testing.assert_allclose(scalar, dense[0], atol=4e-11, rtol=2e-12)
    eps = 1e-5
    gradients = []
    for sign in (-1, 1):
        u = expm(sign*eps*k[0])
        rotated = u.T @ (b @ u)
        gradients.append(orbopt.orbital_gradient(orbopt.generalized_fock_from_factors(
            u.T@h@u, rotated[:,:,:occupied], one, density)))
    np.testing.assert_allclose(scalar, (gradients[1]-gradients[0])/(2*eps),
                               atol=2e-6, rtol=2e-7)


def test_six_singlet_workflow_matches_embedded_reference(monkeypatch):
    from pyqed.qchem import Molecule
    atom = '; '.join(f'H {i*.9} {(.07 if i % 2 else 0)} 0' for i in range(6))
    mf = Molecule(atom=atom, basis='sto-3g', unit='angstrom').build(
        eri='ri', auxbasis='def2-universal-jkfit').RHF().run()
    results = []
    for compact in (False, True):
        np.random.seed(136)
        solver = SecondOrderCASSCF(mf, 4, 4, multiplicity=1, ci_method='ci',
            conv_tol_grad=1e-4, conv_tol_grad_relaxed=1e-4, verbose=0)
        solver.state_average(np.ones(6)/6)
        if compact:
            def forbidden(*args, **kwargs):
                raise AssertionError('Core-inclusive RDM allocation')
            monkeypatch.setattr(solver, '_effective_rdms_occ', forbidden)
            monkeypatch.setattr(solver, '_effective_rdms', forbidden)
        else:
            monkeypatch.setattr(solver, '_effective_factor_rdms', solver._effective_rdms_occ)
        solver.run(nstates=6)
        assert solver.converged
        np.testing.assert_allclose(solver.spin_square(), 0, atol=1e-9)
        results.append(solver)
    np.testing.assert_allclose(results[0].e_tot, results[1].e_tot, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        [r['gradient_norm'] for r in results[0].micro_history],
        [r['gradient_norm'] for r in results[1].micro_history], atol=1e-10, rtol=0)


def test_complex_active_density_uses_blocked_fallback():
    rng = np.random.default_rng(401)
    d1 = np.eye(2)
    d2 = rng.normal(size=(2,)*4)+1j*rng.normal(size=(2,)*4)
    one, full = embedded(1,d1,d2)
    density = orbopt.CoreActiveRDM(1,d1,d2)
    b = rng.normal(size=(11,5,5)); b += b.transpose(0,2,1).copy()
    h = np.eye(5)
    k = rng.normal(size=(2,5,5)); k -= k.transpose(0,2,1).copy()
    solver = object.__new__(SecondOrderCASSCF)
    expected = solver._orbital_hessian_action_from_factors_batch(h,b,one,full,k)
    cache = {}
    actual = solver._orbital_hessian_action_from_factors_batch(h,b,one,density,k,cache)
    assert cache['native_workspace'] is None
    np.testing.assert_allclose(actual,expected,atol=1e-11)
    expected_fock = h[:,:3]@one + np.einsum('Ppr,Pst,rqst->pq',b[:,:,:3],b[:,:3,:3],full,optimize=True)
    np.testing.assert_allclose(orbopt.generalized_fock_from_factors(h,b[:,:,:3],one,density)[:,:3],
                               expected_fock,atol=1e-11)


def test_auxiliary_blocks_bound_workspace_and_preserve_tail(monkeypatch):
    from pyqed.qchem import _casscf_cpp
    if not hasattr(_casscf_cpp, 'factor_hessian_workspace_info'):
        pytest.skip('Rebuild extension for blocked workspace validation')
    rng = np.random.default_rng(2048)
    naux, n, core, active = 600, 80, 56, 4
    occupied = core+active
    one = 2*np.eye(occupied)
    one[core:,core:] = np.eye(active)
    density = orbopt.CoreActiveRDM(core, np.eye(active), np.ones((active,)*4))
    b = rng.normal(size=(naux,n,n)); b += b.transpose(0,2,1).copy()
    h = np.eye(n)
    k = rng.normal(size=(n,n))*.01; k -= k.T.copy()
    solver = object.__new__(SecondOrderCASSCF)
    cache = {}
    actual = solver._orbital_hessian_action_from_factors(h,b,one,density,k,cache)
    info = _casscf_cpp.factor_hessian_workspace_info(cache['native_workspace'])
    assert 0 < info['auxiliary_block_size'] < naux
    assert info['scratch_bytes'] <= (64 << 20)+8*(n*n+n*occupied)
    assert info['packed_factor_bytes'] == info['contracted_base_bytes'] == 0
    assert 'pair_full_occ_matrix' not in cache and 'contracted_dm2' not in cache
    with monkeypatch.context() as context:
        context.setattr(orbopt,'_cpp_attr',lambda *args: None)
        expected = solver._orbital_hessian_action_from_factors(h,b,one,density,k,{})
    np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=2e-12)
    full_contraction = density.contract(b[:,:occupied,:occupied])
    expected_fock = h[:,:occupied]@one + np.einsum('Ppr,Prq->pq',
        b[:,:,:occupied],full_contraction,optimize=True)
    lengths = []
    original = density.contract
    def tracked(factors):
        lengths.append(len(factors))
        return original(factors)
    monkeypatch.setattr(density,'contract',tracked)
    fock = orbopt.generalized_fock_from_factors(h,b[:,:,:occupied],one,density)
    assert len(lengths) > 1 and max(lengths) < naux
    np.testing.assert_allclose(fock[:,:occupied],expected_fock,atol=2e-9,rtol=2e-12)
    from types import SimpleNamespace
    dm = rng.normal(size=(n,n)); dm += dm.T.copy()
    potential = (np.einsum('Pkl,lk,Pij->ij',b,dm,b,optimize=True)
                 -.5*np.einsum('Pil,lk,Pkj->ij',b,dm,b,optimize=True))
    parent = SimpleNamespace(mol=None,mo_occ=np.zeros(n),energy_nuc=lambda: 0.)
    frozen = solver._FrozenFactorRHF(parent,h,b,h)
    original_einsum = np.einsum
    batches = []
    def checked(expression,*operands,**kwargs):
        if expression == 'Pkl,lk->P':
            batches.append(len(operands[0]))
        return original_einsum(expression,*operands,**kwargs)
    monkeypatch.setattr(np,'einsum',checked)
    np.testing.assert_allclose(frozen.get_veff(dm),potential,atol=2e-9,rtol=2e-12)
    assert len(batches) > 1 and max(batches) < naux
