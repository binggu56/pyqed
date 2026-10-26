"""Analytic molecular response against independently displaced GW/BSE solves."""
from copy import copy
from types import SimpleNamespace

import numpy as np
import pytest

from pyqed.gw import GW, BSE, TDA
from pyqed.qchem import Molecule


def _point(coords, cls=TDA, screening='hf', eri='dense', charge=0):
    mol = Molecule(atom=[('H',tuple(r)) for r in coords],unit='bohr',
                   basis='sto-3g',charge=charge).build(eri=eri, auxbasis='sto-3g', options={
                       'eri_screen_tol': 0., 'ri_screen_tol': 0., 'ri_cache': False,
                       'low_rank_tol': 1e-12})
    mf = mol.RHF().run(tol=1e-13)
    gw = GW(mf,freq_int='exact',eta=0.).run()
    no = int(np.count_nonzero(mf.mo_occ))
    return cls(gw,screening_energies=screening).run(
        nroots=no*(len(mf.mo_energy)-no),low_rank=False)


@pytest.mark.parametrize('cls',[TDA,BSE])
@pytest.mark.parametrize('screening',['hf','qp'])
def test_relaxed_surface_derivatives(cls,screening):
    coords = np.array([[0.,0.,-.7],[0.,0.,.7]])
    mode = np.array([[0.,0.,-.5],[0.,0.,.5]])
    ref = _point(coords,cls,screening)
    f,g = ref.vibronic_couplings(modes=mode[None])
    step = 2e-4
    plus,minus = [_point(coords+s*step*mode,cls,screening) for s in (1,-1)]
    fd1 = (plus.e_tot-minus.e_tot)/(2*step)
    fd2 = (plus.e_tot-2*ref.e_tot+minus.e_tot)/step**2
    np.testing.assert_allclose(ref.gradient(method="forward", modes=mode[None])[:, 0], fd1, atol=2e-7)
    np.testing.assert_allclose(ref.hessian(modes=mode[None], method='polarization')[:, 0, 0], fd2, atol=3e-6)
    assert callable(ref.gradient) and callable(ref.hessian)
    np.testing.assert_allclose(f[0,0],fd1,atol=2e-7)
    np.testing.assert_allclose(g[0,0,0],fd2,atol=3e-6)
    np.testing.assert_allclose(ref.energy_gradient[:,0],fd1,atol=2e-7)
    np.testing.assert_allclose(ref.energy_hessian[:,0,0],fd2,atol=3e-6)


@pytest.mark.parametrize('cls',[TDA,BSE])
def test_interstate_mixed_response_and_surface_hessian(cls):
    # Asymmetric H3+ has two nondegenerate excitations and nonzero off-diagonal F.
    coords = np.array([[0.,0.,0.],[0.,0.,1.5],[1.8,0.,.3]])
    modes = np.zeros((2,3,3))
    modes[0,1,2] = 1
    modes[1,2,0] = 1
    ref = _point(coords,cls,charge=1)
    f,g = ref.vibronic_couplings(modes=modes)
    analytic_surface = ref.energy_hessian.copy()
    from pyqed.gw.bse import bse_AB_matrices
    def root(matrix):
        e,u = np.linalg.eigh(matrix)
        return (u*np.sqrt(e))@u.T
    if cls is TDA:
        vectors = ref.x
    else:
        a,b = bse_AB_matrices(ref)
        vectors = np.linalg.solve(root(a-b),ref.x+ref.y)*np.sqrt(ref.e)[None,:]
    def local_matrix(point):
        # Remove only arbitrary canonical-MO signs, not electronic mixing.
        signs = np.sign(np.sum(ref.mo_coeff*point.mo_coeff,axis=0))
        signs = (signs[:point.nocc,None]*signs[None,point.nocc:]).ravel()
        a,b = bse_AB_matrices(point)
        h = a if cls is TDA else root(root(a-b)@(a+b)@root(a-b))
        return vectors.T@(signs[:,None]*h*signs[None,:])@vectors + np.eye(len(ref.e))*point._scf.e_tot
    assert abs(f[0,1,0]) > 1e-5
    np.testing.assert_allclose(f,f.swapaxes(0,1),atol=1e-12)
    np.testing.assert_allclose(g,g.swapaxes(0,1),atol=1e-12)
    np.testing.assert_allclose(g,g.swapaxes(2,3),atol=1e-12)
    step = 3e-4
    shifted = {}
    for x in range(2):
        for sign in (-1,1):
            shifted[x,sign] = _point(coords+sign*step*modes[x],cls,charge=1)
    for x in range(2):
        plus,minus = shifted[x,1],shifted[x,-1]
        fd1 = (plus.e_tot-minus.e_tot)/(2*step)
        fd2 = (plus.e_tot-2*ref.e_tot+minus.e_tot)/step**2
        np.testing.assert_allclose(np.diagonal(f[:,:,x]),fd1,atol=2e-6)
        np.testing.assert_allclose(analytic_surface[:,x,x],fd2,atol=2e-5)
        np.testing.assert_allclose(f[:,:,x],(local_matrix(plus)-local_matrix(minus))/(2*step),atol=2e-6)
        np.testing.assert_allclose(g[:,:,x,x],(local_matrix(plus)-2*local_matrix(ref)+local_matrix(minus))/step**2,atol=2e-5)
    corners = {(s,t):_point(coords+step*(s*modes[0]+t*modes[1]),cls,charge=1)
               for s in (-1,1) for t in (-1,1)}
    mixed = sum(s*t*point.e_tot for (s,t),point in corners.items())/(4*step**2)
    np.testing.assert_allclose(analytic_surface[:,0,1],mixed,atol=2e-5)
    mixed_matrix = sum(s*t*local_matrix(point) for (s,t),point in corners.items())/(4*step**2)
    np.testing.assert_allclose(g[:,:,0,1],mixed_matrix,atol=2e-5)
    # Subset Hessians must still include the omitted state's response.
    ref.vibronic_couplings(state_ids=[0],modes=modes)
    np.testing.assert_allclose(ref.energy_hessian[0],analytic_surface[0],atol=1e-11)


def test_translation_and_cartesian_projection():
    ref = _point(np.array([[0.,0.,-.7],[0.,0.,.7]]))
    f,g = ref.vibronic_couplings()
    modes = np.zeros((2,2,3))
    modes[0,:,2] = [-.5,.5]
    modes[1,:,0] = 1
    projected = ref.vibronic_couplings(modes=modes)
    flat = modes.reshape(2,6)
    np.testing.assert_allclose(projected[0],np.einsum('ijc,xc->ijx',f.reshape(1,1,6),flat),atol=1e-10)
    np.testing.assert_allclose(projected[1],np.einsum('ijcd,xc,yd->ijxy',g.reshape(1,1,6,6),flat,flat),atol=1e-10)
    np.testing.assert_allclose(projected[0][:,:,1],0,atol=1e-10)
    np.testing.assert_allclose(projected[1][:,:,1,:],0,atol=1e-10)


def test_cgldr_relaxed_frame_and_scanner(tmp_path):
    from pyqed.ldr.coarse_grained import CGLDR
    coords = np.array([[0.,0.,-.7],[0.,0.,.7]])
    ref = _point(coords)
    modes = np.array([[[0.,0.,-.5],[0.,0.,.5]]])
    cg = SimpleNamespace(state_ids=[0],expansion_modes=modes,nstates=1,nexpanded=1,
                         derivative_frame='relaxed')
    actual = CGLDR._electronic_derivatives(cg,ref)
    expected = ref.vibronic_couplings(modes=modes)
    np.testing.assert_allclose(actual[0],np.moveaxis(expected[0],-1,0))
    np.testing.assert_allclose(actual[1],np.moveaxis(expected[1],(-2,-1),(0,1)))
    scanner = ref.as_scanner()
    displaced = scanner.point(coords+.01*modes[0])
    assert displaced is not ref
    np.testing.assert_array_equal(ref.mol.atom_coords(),coords)
    np.testing.assert_allclose(displaced.e_tot,_point(coords+.01*modes[0]).e_tot,atol=1e-10)
    from pyqed.dvr import DVR
    from pyqed.ldr import ElectronicPartition
    grid = DVR(domains=[(1.35,1.45),(-.01,.01)],npts=[3,3],mass=[1.,1.],names=('R','Q'))
    def geometry(q):
        return np.array([[0.,0.,0.],[q['Q'],0.,q['R']]])
    transverse = np.array([[[0.,0.,0.],[1.,0.,0.]]])
    cache = tmp_path/'response.npz'
    model = CGLDR(grid,ElectronicPartition(sampled=('R',),expanded=('Q',),center=(0.,)),
                  state_ids=(0,),solver=ref,to_geometry=geometry,
                  expansion_modes=transverse,derivative_frame='relaxed',electronic_cache=cache)
    model.prepare_electronic_data()
    assert model.electronic_data.metadata['derivative_frame'] == 'relaxed'
    assert cache.exists()
    model.derivative_frame = 'fixed'
    model.H_matrices = None
    with pytest.raises(ValueError,match="derivative_frame='fixed'"):
        model.prepare_electronic_data()


def test_response_guards():
    ref = _point(np.array([[0.,0.,-.7],[0.,0.,.7]]))
    with pytest.raises(ValueError,match="frame='relaxed'"):
        ref.vibronic_couplings(frame='fixed')
    with pytest.raises(MemoryError):
        ref.vibronic_couplings(max_memory=.0001)
    with pytest.raises(ValueError,match='state_ids'):
        ref.vibronic_couplings(state_ids=[0,0])
    altered = copy(ref)
    altered.gw = copy(ref.gw)
    altered.gw.method = 'evgw'
    with pytest.raises(NotImplementedError,match='G0W0'):
        altered.vibronic_couplings()
    altered = copy(ref)
    altered.info = {'converged':False}
    with pytest.raises(ValueError,match='BSE/TDA did not converge'):
        altered.vibronic_couplings()


def test_cd_response_matches_dense(monkeypatch):
    coords = np.array([[0.,0.,-.7],[0.,0.,.7]])
    modes = np.array([[[0.,0.,-.5],[0.,0.,.5]]])
    dense = _point(coords)
    cd = _point(coords,eri='cd')
    expected = dense.vibronic_couplings(modes=modes)
    def forbidden(*args,**kwargs):
        raise AssertionError('Factor response must not expand a four-index ERI.')
    monkeypatch.setattr('pyqed.qchem.tddft._dense_eri',forbidden)
    monkeypatch.setattr('pyqed.qchem.basis_derivatives.directional_eri_derivatives',forbidden)
    actual = cd.vibronic_couplings(modes=modes)
    for value,reference in zip(actual,expected):
        np.testing.assert_allclose(value,reference,atol=2e-8)


def test_ri_response_matches_its_own_displaced_model(monkeypatch):
    coords = np.array([[0.,0.,-.7],[0.,0.,.7]])
    mode = np.array([[0.,0.,-.5],[0.,0.,.5]])
    ref = _point(coords,eri='ri')
    step = 3e-4
    plus,minus = [_point(coords+s*step*mode,eri='ri') for s in (1,-1)]
    def forbidden(*args,**kwargs):
        raise AssertionError('RI response must retain the fitted integral model.')
    monkeypatch.setattr('pyqed.qchem.tddft._dense_eri',forbidden)
    ref.vibronic_couplings(modes=mode[None])
    np.testing.assert_allclose(ref.energy_gradient[:,0],(plus.e_tot-minus.e_tot)/(2*step),atol=2e-7)
    np.testing.assert_allclose(ref.energy_hessian[:,0,0],(plus.e_tot-2*ref.e_tot+minus.e_tot)/step**2,atol=3e-6)


@pytest.mark.parametrize('eri',['dense','cd','ri'])
def test_multiple_occupied_orbital_response(eri):
    coords = np.array([[0.,0.,0.],[0.,1.4,1.1],[.2,-1.6,1.]])
    mode = np.zeros_like(coords)
    mode[1,1] = 1
    def point(q):
        mol = Molecule(atom=list(zip(('O','H','H'),map(tuple,q))),
                       unit='bohr',basis='sto-3g').build(eri=eri,auxbasis='sto-3g',options={
                           'eri_screen_tol':0.,'ri_screen_tol':0.,'ri_cache':False})
        mf = mol.RHF().run(tol=1e-13)
        return TDA(GW(mf,freq_int='exact').run(),screening_energies='hf').run(
            nroots=3,low_rank=False)
    ref = point(coords)
    ref.vibronic_couplings(modes=mode[None])
    step = 3e-4
    plus,minus = [point(coords+s*step*mode) for s in (1,-1)]
    np.testing.assert_allclose(ref.energy_gradient[:,0],(plus.e_tot-minus.e_tot)/(2*step),atol=3e-6)
    np.testing.assert_allclose(ref.energy_hessian[:,0,0],(plus.e_tot-2*ref.e_tot+minus.e_tot)/step**2,atol=3e-5)


@pytest.mark.parametrize('eri', ['dense', 'cd', 'ri'])
def test_gradient_never_requests_second_response(monkeypatch, eri):
    from pyqed.qchem import basis_derivatives as bd
    from pyqed.qchem.hf.hessian import RHFHessian
    from pyqed.qchem.hf.factor_curvature import FactorCurvature
    ref = _point(np.array([[0.,0.,-.7],[0.,0.,.7]]), BSE, eri=eri)
    mode = np.array([[[0.,0.,-.5],[0.,0.,.5]]])
    ref.hessian(modes=mode, method='polarization')
    expected = ref.energy_gradient.copy()
    ref._first_response_cache = None
    def forbidden(*args, **kwargs):
        raise AssertionError('Gradient requested second-order response')
    monkeypatch.setattr(RHFHessian, '_second_order_mo_response_data', forbidden)
    for name in ('directional_one_electron_derivatives', 'directional_eri_derivatives'):
        original = getattr(bd, name)
        def guarded(*args, _original=original, **kwargs):
            assert kwargs.get('order', 1) == 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(bd, name, guarded)
    original = FactorCurvature.integrals
    def factors(self, directions):
        assert len(directions) == 1
        return original(self, directions)
    monkeypatch.setattr(FactorCurvature, 'integrals', factors)
    np.testing.assert_allclose(ref.gradient(method="forward", modes=mode), expected, atol=1e-10)
    assert ref.energy_hessian is None


@pytest.mark.parametrize('eri', ['dense', 'cd', 'ri'])
def test_hessian_reuses_first_response(monkeypatch, eri):
    from pyqed.qchem.hf.hessian import RHFHessian
    from pyqed.qchem import basis_derivatives as bd
    from pyqed.qchem.hf.factor_curvature import FactorCurvature
    coords = np.array([[0.,0.,-.7],[0.,0.,.7]])
    mode = np.array([[[0.,0.,-.5],[0.,0.,.5]]])
    ref = _point(coords, BSE, eri=eri)
    expected = ref.hessian(modes=mode, method='polarization').copy()
    ref._first_response_cache = None
    grad = ref.gradient(method="forward", modes=mode).copy()
    cache = ref._first_response_cache
    with monkeypatch.context() as patch:
        def forbidden(*args, **kwargs):
            raise AssertionError('Repeated first-order orbital response')
        patch.setattr(RHFHessian, '_first_order_mo_response_data', forbidden)
        for name in ('directional_one_electron_derivatives', 'directional_eri_derivatives'):
            original = getattr(bd, name)
            def guarded(*args, _original=original, **kwargs):
                assert kwargs['order'] == 2
                return _original(*args, **kwargs)
            patch.setattr(bd, name, guarded)
        original = FactorCurvature.integrals
        def factors(self, directions):
            assert len(directions) == 2
            return original(self, directions)
        patch.setattr(FactorCurvature, 'integrals', factors)
        np.testing.assert_allclose(ref.hessian(modes=mode, method='polarization'), expected, atol=1e-10)
        np.testing.assert_allclose(ref.energy_gradient, grad, atol=1e-10)
        assert ref._first_response_cache is cache
    mode *= 1.1
    ref.gradient(method="forward", modes=mode)
    assert ref._first_response_cache is not cache
    cache = ref._first_response_cache
    ref.gradient(method="forward", modes=mode)
    assert ref._first_response_cache is cache
    ref._scf.mo_coeff[:, 0] *= -1
    ref.mo_coeff = ref._scf.mo_coeff.copy()
    ref.gradient(method="forward", modes=mode)
    assert ref._first_response_cache is not cache


@pytest.mark.parametrize('cls', [TDA, BSE])
@pytest.mark.parametrize('screening', ['hf', 'qp'])
def test_gw_bse_first_response_reuse(monkeypatch, cls, screening):
    from pyqed.gw import response, subspace_response
    coords = np.array([[0.,0.,0.],[0.,0.,1.5],[1.8,0.,.3]])
    modes = np.zeros((2,3,3))
    modes[0,1,2] = 1
    modes[1,2,0] = 1
    ref = _point(coords, cls, screening, charge=1)
    expected = ref.hessian(modes=modes, method='polarization').copy()
    ref._first_response_cache = None
    gradient = ref.gradient(method="forward", modes=modes).copy()
    cache = ref._first_response_cache
    paths = {}
    def index(node, path=()):
        paths[id(node)] = path
        for key, value in node.items():
            if isinstance(value, dict):
                index(value, path+(key,))
    index(cache)
    original = response._cached_first
    computed = []
    def guarded(name, compute):
        context = response._response_context.get()
        assert context is not None
        node, counts = context
        key = (name, counts.get(name, 0))
        def checked():
            # Only Hessian-only mixed nuclear/frequency response may be new.
            index(cache)
            path = paths[id(node)]+(key,)
            assert any(isinstance(k, tuple) and k[0] == '_self_energy_region' for k in path), path
            assert name in ('resolvent', 'contract:rLp,rLq->pq', 'mul'), path
            assert key not in node or 'data' not in node[key], path
            computed.append(path)
            return compute()
        return original(name, checked)
    monkeypatch.setattr(response, '_cached_first', guarded)
    monkeypatch.setattr(subspace_response, '_cached_first', guarded)
    np.testing.assert_allclose(ref.hessian(modes=modes, method='polarization'), expected, atol=1e-10)
    np.testing.assert_allclose(ref.energy_gradient, gradient, atol=1e-10)
    assert computed
    computed.clear()
    np.testing.assert_allclose(ref.gradient(method="forward", modes=modes), gradient, atol=1e-10)
    np.testing.assert_allclose(ref.hessian(modes=modes, method='polarization'), expected, atol=1e-10)
    assert not computed
    assert response._response_context.get() is None


def test_response_cache_key_includes_gw_bse_settings():
    from pyqed.gw.response import _response_key
    from pyqed.gw.subspace_response import orbital_groups
    ref = _point(np.array([[0.,0.,-.7],[0.,0.,.7]]), BSE)
    modes = np.eye(6)
    groups = orbital_groups(ref._scf.mo_energy, ref.nocc, 1e-7)
    def key():
        return _response_key(ref, modes, 'auto', groups)
    before = key()
    ref.e_qp[0] += 1e-6
    assert key() != before
    before = key()
    ref.screening_energies = 'qp'
    assert key() != before
    before = key()
    ref.bse_metric = 'tda'
    assert key() != before
    ref.x = np.ones((1, 1))
    before = key()
    ref.x.flat[0] += 1e-6
    assert key() != before
    before = key()
    ref.e[0] += 1e-6
    assert key() != before


@pytest.mark.parametrize('cls', [TDA, BSE])
@pytest.mark.parametrize('screening', ['hf', 'qp'])
@pytest.mark.parametrize('eri', ['dense', 'cd', 'ri'])
@pytest.mark.parametrize('method', ['mixed', 'polarization'])
def test_disk_directional_response_matches_full(monkeypatch, cls, screening, eri, method):
    from pyqed.gw import response_storage
    from pyqed.gw.response_storage import directional_derivatives
    monkeypatch.setattr(response_storage, '_ARRAY_MEMORY_BYTES', 0)
    from pyqed.gw.subspace_response import orbital_groups
    from pyqed.qchem.hf.hessian import RHFHessian
    coords = np.array([[0.,0.,0.],[0.,0.,1.5],[1.8,0.,.3]])
    modes = np.zeros((2,3,3))
    modes[0,1,2] = 1
    modes[1,2,0] = 1
    ref = _point(coords, cls, screening, eri=eri, charge=1)
    ids = np.arange(2)
    groups = orbital_groups(ref._scf.mo_energy, ref.nocc, 1e-7)
    expected = ref.hessian(modes=modes, method='polarization').copy()
    expected_f, expected_g = ref.F.copy(), ref.G.copy()
    ref._first_response_cache = None
    directional_derivatives(ref, ids, modes, 'auto', groups, 4000, 1e-7, 1)
    grad = ref.energy_gradient.copy()
    def forbidden(*args, **kwargs):
        raise AssertionError('Directional Hessian repeated first-order CPHF')
    monkeypatch.setattr(RHFHessian, '_first_order_mo_response_data', forbidden)
    # Public dispatch must recognize the existing directional cache.
    result = ref.hessian(modes=modes, method=method)
    np.testing.assert_allclose(result, expected, atol=1e-10)
    np.testing.assert_allclose(ref.energy_gradient, grad, atol=1e-10)
    np.testing.assert_allclose(ref.F, expected_f, atol=1e-10)
    np.testing.assert_allclose(ref.G, expected_g, atol=1e-10)
    assert 'directions' in ref._first_response_cache


def test_response_memory_budget_dispatches_one_direction(monkeypatch):
    from pyqed.gw import response
    ref = _point(np.array([[0.,0.,0.],[0.,0.,1.5],[1.8,0.,.3]]), BSE, charge=1)
    modes = np.zeros((2,3,3))
    modes[0,1,2], modes[1,2,0] = 1, 1
    expected = ref.hessian(modes=modes, method='polarization').copy()
    ref._first_response_cache = None
    original = response._reference_response
    def guarded(bse, directions, *args, **kwargs):
        assert len(directions) == (2 if response._mixed_response.get() else 1)
        return original(bse, directions, *args, **kwargs)
    monkeypatch.setattr(response, '_reference_response', guarded)
    ref.gradient(method="forward", modes=modes, max_memory=.015)
    store = ref._first_response_cache['store']
    np.testing.assert_allclose(ref.hessian(modes=modes, max_memory=.015, method='mixed'), expected, atol=1e-10)
    assert ref._first_response_cache['store'] is store
    ref.gradient(method="forward", modes=1.1*modes, max_memory=.015)
    assert ref._first_response_cache['store'] is not store


def test_directional_reference_shared_across_modes(monkeypatch):
    from pyqed.gw import response, response_storage
    from pyqed.gw.subspace_response import orbital_groups
    coords = np.array([[0.,0.,0.],[0.,0.,1.5],[1.8,0.,.3]])
    modes = np.zeros((2,3,3))
    modes[0,1,2], modes[1,2,0] = 1, 1
    ref = _point(coords, BSE, 'hf', eri='cd', charge=1)
    expected = ref.hessian(modes=modes, method='polarization').copy()
    ref._first_response_cache = None
    groups = orbital_groups(ref._scf.mo_energy, ref.nocc, 1e-7)
    counts = []
    original_eigh, original_derivatives = np.linalg.eigh, response._derivatives
    calls = []
    def eigh(*args, **kwargs):
        calls.append(args[0].shape)
        return original_eigh(*args, **kwargs)
    def derivatives(*args, **kwargs):
        start = len(calls)
        result = original_derivatives(*args, **kwargs)
        counts.append(len(calls)-start)
        return result
    monkeypatch.setattr(np.linalg, 'eigh', eigh)
    monkeypatch.setattr(response, '_derivatives', derivatives)
    monkeypatch.setattr(response_storage, '_ARRAY_MEMORY_BYTES', 0)
    response_storage.directional_derivatives(ref, np.arange(2), modes, 'auto', groups, 4000, 1e-7, 1)
    assert counts[0] > 0
    assert counts[1] == 0
    shared = ref._first_response_cache['reference']
    np.testing.assert_allclose(ref.hessian(modes=modes, method='polarization'), expected, atol=1e-10)
    assert all(count == 0 for count in counts[1:])
    assert ref._first_response_cache['reference'] is shared
    ref.gradient(method="forward", modes=modes*1.1, max_memory=.1)
    assert ref._first_response_cache.get('reference') is not shared


@pytest.mark.parametrize('screening', ['hf', 'qp'])
def test_direct_mixed_reuses_all_first_response(monkeypatch, screening):
    from pyqed.gw import response, subspace_response, response_storage
    from pyqed.qchem.hf.hessian import RHFHessian
    from pyqed.gw.subspace_response import orbital_groups
    coords = np.array([[0.,0.,0.],[0.,0.,1.5],[1.8,0.,.3]])
    modes = np.zeros((3,3,3))
    modes[0,1,2], modes[1,2,0], modes[2,2,2] = 1, 1, .7
    ref = _point(coords, BSE, screening, eri='cd', charge=1)
    groups = orbital_groups(ref._scf.mo_energy, ref.nocc, 1e-7)
    args = (ref, np.arange(2), modes, 'auto', groups, 4000, 1e-7)
    monkeypatch.setattr(response_storage, '_ARRAY_MEMORY_BYTES', 0)
    response_storage.directional_derivatives(*args, 1)
    build = RHFHessian._build_cphf_matrix
    builds = []
    def counted_build(self, *args, **kwargs):
        builds.append(1)
        return build(self, *args, **kwargs)
    monkeypatch.setattr(RHFHessian, '_build_cphf_matrix', counted_build)
    response_storage.directional_derivatives(*args, 2, method='polarization')
    assert len(builds) == 1
    expected = ref.energy_hessian.copy()
    original = response._cached_first
    count = []
    def guarded(name, compute):
        def checked():
            if response._mixed_response.get():
                raise AssertionError('Mixed pair recomputed first response: '+name)
            return compute()
        return original(name, checked)
    monkeypatch.setattr(response, '_cached_first', guarded)
    monkeypatch.setattr(subspace_response, '_cached_first', guarded)
    initialize = response._Jet.__init__
    def initialize_checked(self, value, first, second):
        if response._mixed_response.get():
            assert np.shape(first)[0] == 2
            assert np.shape(second)[:2] == (1,1)
            count.append(1)
        initialize(self,value,first,second)
    monkeypatch.setattr(response._Jet, '__init__', initialize_checked)
    response_storage.directional_derivatives(*args, 2, method='mixed')
    assert len(builds) == 1
    assert count
    np.testing.assert_allclose(ref.energy_hessian, expected, atol=2e-10)
    # Enriched first responses remain readable after pair scratch files close.
    np.testing.assert_allclose(ref.hessian(modes=modes, method='mixed'), expected, atol=2e-10)
    assert response._mixed_response.get() is False
