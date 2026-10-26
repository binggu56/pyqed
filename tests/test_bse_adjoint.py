"""GW/BSE reverse gradients against independently tested forward response."""
import numpy as np
import pytest
from test_bse_vibronic_response import _point
from pyqed.gw import BSE, TDA

pytest.importorskip('torch')

@pytest.mark.parametrize('cls', [BSE,TDA])
@pytest.mark.parametrize('screening', ['hf','qp'])
@pytest.mark.parametrize('eri', ['dense','cd','ri'])
def test_adjoint_matches_forward(cls, screening, eri, monkeypatch):
    from pyqed.qchem.hf.hessian import RHFHessian
    ref = _point(np.array([[0.,0.,0.],[0.,0.,1.5],[1.8,0.,.3]]),cls,screening,eri,charge=1)
    modes = np.random.default_rng(9).normal(size=(2,9))
    expected = ref.gradient(method="forward", modes=modes)
    def forbidden(*args, **kwargs):
        raise AssertionError('Adjoint gradient used nuclear forward CPHF')
    monkeypatch.setattr(RHFHessian, '_first_order_mo_response_data', forbidden)
    actual = ref.gradient(modes=modes,method='adjoint')
    np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=1e-8)
    assert max(ref.derivative_info['orbital_adjoint_residuals']) < 1e-10
    assert ref.energy_hessian is None


def test_reject_degenerate_orbitals():
    from test_bse_subspace_response import _h3
    ref = _h3()
    with pytest.raises(NotImplementedError,match='Degenerate HF'):
        ref.gradient(method='adjoint')


def test_multiple_occupied_and_mode_projection():
    from pyqed.qchem import Molecule
    from pyqed.gw import GW
    mol = Molecule(atom=[('O',(0,0,0)),('H',(0,1.4,1.1)),('H',(.2,-1.6,1.))],
                   unit='bohr',basis='sto-3g').build(eri='cd',options={'eri_screen_tol':0.,'low_rank_tol':1e-12})
    ref = BSE(GW(mol.RHF().run(tol=1e-13),freq_int='exact').run()).run(nroots=2,low_rank=False)
    expected = ref.gradient(method="forward")
    actual = ref.gradient(method='adjoint')
    np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=1e-8)
    np.testing.assert_allclose(actual.reshape(2,3,3).sum(axis=1),0,atol=1e-8)
    modes = np.random.default_rng(3).normal(size=(2,9))
    selected = ref.gradient(state_ids=[1],modes=modes,method='adjoint')
    np.testing.assert_allclose(selected,actual[1:2]@modes.T,atol=1e-10)


def test_default_disk_adjoint_matches_memory_and_cleans_up(monkeypatch):
    from pyqed.gw.response_storage import ResponseStore
    ref = _point(np.array([[0.,0.,0.],[0.,0.,1.5],[1.8,0.,.3]]),BSE,charge=1)
    modes = np.random.default_rng(7).normal(size=(2,9))
    expected = ref.gradient(modes=modes,storage='memory')
    paths=[]
    original = ResponseStore.__init__
    def tracked(self):
        original(self)
        paths.append(self.directory.name)
    monkeypatch.setattr(ResponseStore,'__init__',tracked)
    from pyqed.gw import response_storage
    monkeypatch.setattr(response_storage,'_ARRAY_MEMORY_BYTES',0)
    actual=ref.gradient(modes=modes)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
    assert ref.derivative_info['method']=='adjoint'
    assert ref.derivative_info['storage']=='disk'
    from pathlib import Path
    assert all(not Path(path).exists() for path in paths)


def test_disk_storage_cleanup_on_error(monkeypatch):
    from pyqed.gw import adjoint
    from pyqed.gw.response_storage import ResponseStore
    from pathlib import Path
    paths=[]
    original=ResponseStore.__init__
    def tracked(self):
        original(self)
        paths.append(self.directory.name)
    def fail(*args, **kwargs):
        import torch
        x=torch.ones((200,200),dtype=torch.float64,requires_grad=True)
        y=(x*x).sum()
        raise ValueError('deliberate failure after saving reverse tensors')
    monkeypatch.setattr(ResponseStore,'__init__',tracked)
    monkeypatch.setattr(adjoint,'_gradients',fail)
    with pytest.raises(ValueError,match='deliberate failure'):
        adjoint.gradients(_point(np.array([[0.,0.,0.],[0.,0.,1.5],[1.8,0.,.3]]),BSE,charge=1))
    assert paths and all(not Path(path).exists() for path in paths)


@pytest.mark.parametrize('cls', [BSE, TDA])
@pytest.mark.parametrize('screening', ['hf', 'qp'])
@pytest.mark.parametrize('eri', ['dense', 'cd', 'ri'])
def test_adjoint_hessian_matches_forward(cls, screening, eri, monkeypatch):
    from pyqed.qchem.hf.hessian import RHFHessian
    ref = _point(np.array([[0., 0., 0.], [0., 0., 1.5], [1.8, 0., .3]]),
                 cls, screening, eri, charge=1)
    modes = np.random.default_rng(9).normal(size=(2, 9))
    expected = ref.hessian(modes=modes, method='polarization')
    gradient = ref.energy_gradient.copy()
    def forbidden(*args, **kwargs):
        raise AssertionError('Adjoint Hessian used second-order orbital response')
    monkeypatch.setattr(RHFHessian, '_second_order_mo_response_data', forbidden)
    actual = ref.hessian(modes=modes, method='adjoint')
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=1e-8)
    np.testing.assert_allclose(ref.energy_gradient, gradient, atol=2e-9, rtol=1e-8)
    assert ref.derivative_info['max_asymmetry'] < 2e-8


def test_adjoint_hessian_finite_difference(monkeypatch):
    from pyqed.gw import response_storage
    monkeypatch.setattr(response_storage, '_ARRAY_MEMORY_BYTES', 0)
    coords = np.array([[0., 0., 0.], [0., 0., 1.5], [1.8, 0., .3]])
    modes = np.random.default_rng(4).normal(size=(2, 9))
    ref = _point(coords, BSE, 'qp', 'cd', charge=1)
    actual = ref.hessian(modes=modes, method='adjoint')
    step = 2e-5
    for y, mode in enumerate(modes):
        plus, minus = [_point(coords+sign*step*mode.reshape(3, 3), BSE, 'qp', 'cd', charge=1)
                       for sign in (1, -1)]
        expected = (plus.gradient(modes=modes)-minus.gradient(modes=modes))/(2*step)
        np.testing.assert_allclose(actual[:, :, y], expected, atol=2e-7, rtol=1e-7)
    memory = ref.hessian(modes=modes, method='adjoint', storage='memory')
    np.testing.assert_allclose(memory, actual, atol=1e-12)


def test_adjoint_cartesian_hessian_water():
    from pyqed.qchem import Molecule
    from pyqed.gw import GW
    mol = Molecule(atom=[('O', (0, 0, 0)), ('H', (0, 1.4, 1.1)), ('H', (.2, -1.6, 1.))],
                   unit='bohr', basis='sto-3g').build(eri='cd', options={
                       'eri_screen_tol': 0., 'low_rank_tol': 1e-12})
    ref = BSE(GW(mol.RHF().run(tol=1e-13), freq_int='exact', eta=0.).run()).run(nroots=2, low_rank=False)
    expected = ref.hessian(state_ids=[1], method='polarization')
    actual = ref.hessian(state_ids=[1], method='adjoint')
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=1e-7)
    # Translation must be a null displacement even away from equilibrium.
    np.testing.assert_allclose(actual.reshape(1, 9, 3, 3).sum(axis=2), 0., atol=2e-8)



def test_adjoint_hessian_cleanup_on_failure(monkeypatch):
    from pathlib import Path
    from pyqed.gw import adjoint
    from pyqed.gw.response_storage import ResponseStore
    ref = _point(np.array([[0., 0., 0.], [0., 0., 1.5], [1.8, 0., .3]]), BSE, charge=1)
    with ref.derivative_workspace():
        modes = np.eye(9)[[5, 6]]
        ref.gradient(method='forward', modes=modes)
        previous = ref._first_response_cache
        paths = []
        original = ResponseStore.__init__
        def tracked(self):
            original(self)
            paths.append(self.directory.name)
        def fail(*args, **kwargs):
            raise RuntimeError('deliberate Hessian pullback failure')
        monkeypatch.setattr(ResponseStore, '__init__', tracked)
        monkeypatch.setattr(adjoint, '_gradients', fail)
        with pytest.raises(RuntimeError, match='deliberate Hessian'):
            ref.hessian(modes=modes, method='adjoint')
        assert ref._first_response_cache is None
        assert paths and all(not Path(path).exists() for path in paths)


def test_cd_hessian_avoids_second_factor_tensors(monkeypatch):
    from pyqed.gw import response
    ref = _point(np.array([[0., 0., 0.], [0., 0., 1.5], [1.8, 0., .3]]),
                 BSE, 'qp', 'cd', charge=1)
    modes = np.random.default_rng(17).normal(size=(3, 9))
    expected = ref.hessian(modes=modes, method='polarization')
    original = response._factor_response
    def first_only(mol, modes, order=2, cache=None):
        if order != 1:
            raise AssertionError('Adjoint Hessian formed second factor tensors')
        return original(mol, modes, order=order, cache=cache)
    monkeypatch.setattr(response, '_factor_response', first_only)
    actual = ref.hessian(modes=modes, method='adjoint')
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=1e-8)


def test_blocked_pullbacks_preserve_hessian(monkeypatch):
    from pyqed.gw import adjoint
    ref = _point(np.array([[0., 0., 0.], [0., 0., 1.5], [1.8, 0., .3]]),
                 BSE, 'qp', 'cd', charge=1)
    modes = np.random.default_rng(15).normal(size=(2, 9))
    expected = ref.hessian(modes=modes, method='adjoint')
    monkeypatch.setattr(adjoint, '_QP_ORBITAL_BLOCK', 1)
    monkeypatch.setattr(adjoint, '_FACTOR_BLOCK', 1)
    actual = ref.hessian(modes=modes, method='adjoint')
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


def test_hessian_defaults_to_adjoint(monkeypatch):
    from pyqed.gw import response
    ref = _point(np.array([[0., 0., 0.], [0., 0., 1.5], [1.8, 0., .3]]),
                 BSE, 'hf', 'cd', charge=1)
    modes = np.eye(9)[[5, 6]]
    expected = ref.hessian(modes=modes, method='adjoint')
    def forbidden(*args, **kwargs):
        raise AssertionError('Default Hessian dispatched to forward response')
    monkeypatch.setattr(response, 'energy_derivatives', forbidden)
    actual = ref.hessian(modes=modes)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert ref.derivative_info['method'] == 'adjoint'


def test_hessian_releases_direction_graphs_between_columns(monkeypatch):
    import gc
    import weakref
    from pyqed.gw import adjoint
    ref = _point(np.array([[0., 0., 0.], [0., 0., 1.5], [1.8, 0., .3]]),
                 BSE, 'hf', 'cd', charge=1)
    original = adjoint._gradients
    directions = []
    class Direction(dict):
        pass
    def tracked(*args, **kwargs):
        assert all(value() is None for value in directions)
        kwargs['_tangent'] = Direction(kwargs['_tangent'])
        directions.append(weakref.ref(kwargs['_tangent']))
        return original(*args, **kwargs)
    monkeypatch.setattr(adjoint, '_gradients', tracked)
    enabled = gc.isenabled()
    gc.disable()
    try:
        ref.hessian(modes=np.eye(9)[[5, 6]])
        assert all(value() is None for value in directions)
    finally:
        if enabled:
            gc.enable()


def test_hessian_reuses_only_geometry_local_spectral_data(monkeypatch):
    from collections import Counter
    from pyqed.gw import response_storage
    ref = _point(np.array([[0., 0., 0.], [0., 0., 1.5], [1.8, 0., .3]]),
                 BSE, 'qp', 'cd', charge=1)
    with ref.derivative_workspace():
        modes = np.eye(9)[[5, 6]]
        expected = ref.hessian(modes=modes, method='polarization')
        original = response_storage.reference_value
        counts = Counter()
        def tracked(name, compute):
            def counted():
                if isinstance(name, tuple) and name[0] == 'adjoint':
                    counts[name] += 1
                return compute()
            return original(name, counted)
        monkeypatch.setattr(response_storage, 'reference_value', tracked)
        actual = ref.hessian(modes=modes)
        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=1e-8)
        assert counts and set(counts.values()) == {1}
        assert ('adjoint', 'qp_basis') in counts
        assert ('adjoint', 'excitation_basis') in counts
        assert any(isinstance(key[1], tuple) and key[1][0] == 'qp_vectors' for key in counts)
        # A new public call reuses the same geometry, including a mode subset.
        ref.hessian(modes=modes[:1])
        assert set(counts.values()) == {1}
        ref.clear_derivative_cache()
        ref.hessian(modes=modes[:1])
        assert set(counts.values()) == {2}


@pytest.mark.parametrize('mapped', [False, True])
def test_stored_tensor_views_preserve_disk_data(mapped):
    from pyqed.gw.adjoint import _stored_tensor
    from pyqed.gw.response_storage import ResponseStore, read
    store = ResponseStore()
    try:
        store.put(np.zeros(10000))
        expected = np.arange(40000., dtype=float).reshape(200, 200)
        saved = store.put(expected)
        view = _stored_tensor(read(saved) if mapped else saved)
        view.add_(3.)
        np.testing.assert_array_equal(view.numpy(), expected+3.)
        np.testing.assert_array_equal(read(saved), expected)
    finally:
        store.directory.cleanup()


@pytest.mark.parametrize("excitation_response", ["dense", "projected", "physical"])
@pytest.mark.parametrize('gw_response', ['dense', 'spectral'])
def test_adjoint_ccpvdz_hessian_displaced_gradients(excitation_response, gw_response):
    from pyqed.qchem import Molecule
    from pyqed.gw import GW
    coords = np.array([[0., 0., 0.], [0., .7409, .5821], [.1058, -.8467, .5292]])
    reference = Molecule(atom=list(zip(['O', 'H', 'H'], coords)),
                         unit='angstrom', basis='cc-pvdz')
    coords = reference.atom_coords()
    modes = np.eye(9)[[2, 4]]
    def point(displacement):
        mol = Molecule(atom=list(zip(['O', 'H', 'H'], coords+displacement)),
                       unit='bohr', basis='cc-pvdz').build(eri='cd', options={
                           'eri_screen_tol': 0., 'low_rank_tol': 1e-10})
        mf = mol.RHF().run(tol=1e-13, conv_tol_grad=1e-11)
        return BSE(GW(mf, freq_int='exact', eta=0.).run(),
                   screening_energies='hf').run(nroots=2, low_rank=True, tol=1e-9)
    ref = point(np.zeros((3, 3)))
    actual = ref.hessian(state_ids=[0], modes=modes, excitation_response=excitation_response,
                         gw_response=gw_response)
    if excitation_response == 'physical':
        workers = ref.mol.workers
        ref.mol.workers = 4
        try:
            parallel = ref.hessian(state_ids=[0], modes=modes, excitation_response=excitation_response,
                                   gw_response=gw_response)
            np.testing.assert_allclose(parallel, actual, atol=1e-10, rtol=0)
        finally:
            ref.mol.workers = workers
    step = 2e-4
    for column, mode in enumerate(modes):
        gradients = [point(sign*step*mode.reshape(3, 3)).gradient(
            state_ids=[0], modes=modes) for sign in (1, -1)]
        expected = (gradients[0]-gradients[1])/(2*step)
        np.testing.assert_allclose(actual[:, :, column], expected, atol=2e-6, rtol=0)


@pytest.mark.parametrize('eri', ['dense', 'cd', 'ri'])
def test_adjoint_selected_columns_keep_all_rows(eri, monkeypatch):
    from pyqed.qchem.hf.hessian import RHFHessian
    ref = _point(np.array([[0.,0.,0.],[0.,0.,1.5],[1.8,0.,.3]]), BSE, eri=eri, charge=1)
    modes = np.random.default_rng(17).normal(size=(5,9))
    expected = ref.hessian(modes=modes)
    gradient = ref.energy_gradient.copy()
    calls = []
    original = RHFHessian._first_order_mo_response_data
    def tracked(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(RHFHessian, '_first_order_mo_response_data', tracked)
    actual = ref.hessian(modes=modes, columns=[2,0])
    assert len(calls) == 2
    np.testing.assert_allclose(actual, expected[:,:,[2,0]], atol=2e-10, rtol=0)
    np.testing.assert_allclose(ref.energy_gradient, gradient, atol=2e-10, rtol=0)
    assert ref.derivative_info['columns'] == [2,0]
    with pytest.raises(ValueError, match='unique valid'):
        ref.hessian(modes=modes, columns=[0,0])


def test_cd_first_response_packed_solve_matches_full():
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.factor_curvature import FactorCurvature
    from pyqed.qchem.basis import _pair_factors_to_full
    from pyqed.gw.response import _factor_response
    from pyqed.gw.response_storage import ResponseStore, reference_scope, storage, read
    mol = Molecule(atom='O 0 0 0; H 0 1.4 1.1; H .2 -1.6 1.',
                   unit='bohr', basis='cc-pvdz').build(eri='cd', options={
                       'low_rank_tol': 1e-10, 'eri_screen_tol': 0.})
    fc = FactorCurvature(mol, prepare_first=False)
    mode = np.random.default_rng(43).normal(size=(1, 9))
    columns = fc.integral_response._cd_columns(mode)[0]
    root = fc.root_derivative(columns[:, fc.pivots])
    full = _pair_factors_to_full(columns, mol.nao).reshape(len(root), -1)
    expected = fc.solve(full-root@fc.factors.reshape(len(root), -1))
    store, cache = ResponseStore(), {}
    try:
        with storage(store), reference_scope({'factor_curvature': fc}, store):
            actual = _factor_response(mol, mode, order=1, cache=cache)
            np.testing.assert_allclose(actual.d[0], expected.reshape(fc.factors.shape), atol=2e-10, rtol=0)
            np.testing.assert_allclose(read(cache['factor_first'][0][0]), root, atol=1e-13, rtol=0)
    finally:
        store.directory.cleanup()


@pytest.mark.parametrize('budget, calls', [(0., 3), (1000., 2), ('two_outputs', 2)])
@pytest.mark.parametrize('workers', [1, 4])
def test_cd_first_response_batch_reuses_integrals(budget, calls, workers, monkeypatch):
    from types import SimpleNamespace
    from contextlib import closing
    from pyqed.qchem import Molecule
    from pyqed.qchem.eri_response import CoulombIntegrals
    from pyqed.gw.response import _factor_response
    from pyqed.gw.response_storage import (
        ResponseStore, reference_scope, storage, first_columns, _batched_columns)
    mol = Molecule(atom='O 0 0 0; H 0 1.4 1.1; H .2 -1.6 1.',
                   unit='bohr', basis='sto-3g').build(eri='cd', workers=workers, options={
                       'low_rank_tol': 1e-10, 'eri_screen_tol': 0.})
    if budget == 'two_outputs':
        data = mol._cd_factor_data
        size = len(data['pivots'])*data['nao']*(data['nao']+1)//2*8
        budget = (2*size+1)*4/1e6
    modes = np.random.default_rng(44).normal(size=(3, 9))
    expected = _factor_response(mol, modes, order=1).d
    original = CoulombIntegrals._cd_columns
    seen = []
    def counted(self, directions, *args, **kwargs):
        seen.append(len(directions))
        return original(self, directions, *args, **kwargs)
    monkeypatch.setattr(CoulombIntegrals, '_cd_columns', counted)
    store, reference = ResponseStore(), {}
    try:
        batches = _batched_columns(SimpleNamespace(mol=mol), modes,
                                   [(i, i) for i in range(3)], 'first', budget)
        with closing(batches):
            for (i, _), columns in batches:
                with storage(store), reference_scope(reference, store), first_columns(columns):
                    actual = _factor_response(mol, modes[i:i+1], order=1, cache={})
                    np.testing.assert_allclose(actual.d[0], expected[i], atol=2e-12, rtol=0)
        assert len(seen) == calls
    finally:
        store.directory.cleanup()


@pytest.mark.parametrize('cls', [BSE, TDA])
@pytest.mark.parametrize('screening', ['hf', 'qp'])
@pytest.mark.parametrize('eri', ['dense', 'cd', 'ri'])
@pytest.mark.parametrize('response_method', ['projected', 'physical'])
def test_projected_excitation_response_matches_dense(cls, screening, eri, monkeypatch, response_method):
    ref = _point(np.array([[0.,0.,0.],[0.,0.,1.5],[1.8,0.,.3]]),
                 cls,screening,eri,charge=1)
    modes = np.random.default_rng(73).normal(size=(2,9))
    expected = ref.hessian(modes=modes,excitation_response='dense')
    expected_gradient = ref.energy_gradient.copy()
    from pyqed.gw import response_storage
    if response_method == 'physical':
        from pyqed.gw import adjoint
        monkeypatch.setattr(adjoint, '_FACTOR_BLOCK', 4)
    original = response_storage.reference_value
    def no_dense_eigenvectors(key, compute):
        assert key != ('adjoint','excitation_basis')
        if response_method == 'physical' and key[0] == 'adjoint' and isinstance(key[1], tuple) and key[1][0] == 'root':
            assert key[1][1] < 2  # Only the two dynamic GW roots remain.
        return original(key,compute)
    monkeypatch.setattr(response_storage,'reference_value',no_dense_eigenvectors)
    actual = ref.hessian(modes=modes,excitation_response=response_method)
    np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=0)
    np.testing.assert_allclose(ref.energy_gradient,expected_gradient,atol=2e-9,rtol=0)
    assert ref.derivative_info['excitation_response_diagnostics']
    assert max(x['residual'] for x in ref.derivative_info['excitation_response_diagnostics']) < 1e-10
    if response_method == 'physical':
        np.testing.assert_allclose(ref.gradient(modes=modes,excitation_response='physical'),
                                   expected_gradient,atol=2e-9,rtol=0)


def test_projected_response_keeps_complement_and_rejects_failed_solve(monkeypatch):
    from pyqed.gw import excitation_response as response
    diagonal = np.array([.2,.20001,.7,1.1])
    state = np.eye(4)[0]
    rhs = np.array([.1,.02,.03,-.04])
    actual, shift, info = response.projected_response(
        lambda v: diagonal*v,diagonal,diagonal[0],state,state,rhs)
    expected = np.r_[0.,-rhs[1:]/(diagonal[1:]-diagonal[0])]
    np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-12)
    assert shift == rhs[0]
    assert info['physical_residual'] < 1e-10
    def failed(*args, **kwargs):
        return np.zeros(4), 1
    monkeypatch.setattr(response,'gmres',failed)
    with pytest.raises(RuntimeError,match='Projected BSE response failed'):
        response.projected_response(lambda v:diagonal*v,diagonal,diagonal[0],state,state,rhs)


@pytest.mark.parametrize('response_method', ['projected', 'physical'])
def test_projected_response_multiple_states_selected_columns(response_method):
    ref = _point(np.array([[0.,0.,0.],[0.,0.,1.5],[1.8,0.,.3]]), BSE, eri='cd', charge=1)
    modes = np.random.default_rng(74).normal(size=(2,9))
    expected = ref.hessian(state_ids=[0,1],modes=modes,excitation_response='dense')
    actual = ref.hessian(state_ids=[1,0],modes=modes,columns=[1],excitation_response=response_method)
    np.testing.assert_allclose(actual,expected[[1,0],:,1:2],atol=2e-9,rtol=0)
    assert [x['state_id'] for x in ref.derivative_info['excitation_response_diagnostics']] == [1,0]
    assert all(x['column'] == 1 for x in ref.derivative_info['excitation_response_diagnostics'])


@pytest.mark.parametrize('response_method', ['dense', 'projected', 'physical'])
def test_hessian_refines_stationary_states_without_mutating_reference(response_method):
    ref = _point(np.array([[0.,0.,0.],[0.,0.,1.5],[1.8,0.,.3]]), BSE, eri='cd', charge=1)
    modes = np.random.default_rng(75).normal(size=(2,9))
    expected = ref.hessian(modes=modes, excitation_response=response_method)
    angle = 1e-7
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    ref.x[:] = ref.x@rotation
    ref.y[:] = ref.y@rotation
    x, y = ref.x.copy(), ref.y.copy()
    actual = ref.hessian(modes=modes, excitation_response=response_method)
    np.testing.assert_allclose(actual,expected,atol=2e-10,rtol=0)
    np.testing.assert_array_equal(ref.x,x)
    np.testing.assert_array_equal(ref.y,y)


@pytest.mark.parametrize('columns', [[4,0,2], list(range(7))])
@pytest.mark.parametrize('rank,nao', [(4,3),(36,9)])
def test_cd_curvature_column_weights_match_unpacked_reference(columns, rank, nao):
    from types import SimpleNamespace
    from pyqed.gw.adjoint import _cd_curvature_weights
    from pyqed.gw.response_storage import ResponseStore, read
    rng = np.random.default_rng(83)
    u, metric = rng.normal(size=(rank,nao,nao)), rng.normal(size=(rank,rank))
    fc = SimpleNamespace(factors=np.zeros_like(u), pivots=np.linspace(0,nao*(nao+1)//2-1,rank,dtype=int),
        pullback=lambda bar:(bar[0]*u,bar[0]*metric))
    first = []
    for _ in range(7):
        l = np.tril(rng.normal(size=(rank,rank)))
        b = rng.normal(size=(rank,nao,nao))
        first.append((l,b+b.transpose(0,2,1)))
    caches = [dict(factor_first=((l,),(b.reshape(rank,-1),))) for l,b in first]
    bars = [(None,None,(np.array([scale]),)) for scale in (1.,2.)]
    store = ResponseStore()
    try:
        weights, packed_first = _cd_curvature_weights(fc,bars,caches,store,columns)
        for scale, (raw, rows, selected_only) in zip((1.,2.),weights):
            i,j = np.tril_indices(nao)
            expected_raw = u[:,i,j]+u[:,j,i]
            expected_raw[:,i == j] *= .5
            expected_raw[:,fc.pivots] += metric
            np.testing.assert_allclose(read(raw),scale*expected_raw,atol=2e-12,rtol=0)
            assert len(rows) == (len(columns) if selected_only else len(first))
            for x,(lx,bx) in enumerate(first):
                for y in columns:
                    ly,by = first[y]
                    ry, connection = read(rows[y])
                    px = read(packed_first[x])[1]
                    if selected_only:
                        actual = np.sum(lx*connection)+np.sum(ry*px)
                    else:
                        rx,mx = read(rows[x])
                        actual = (np.sum(rx*read(packed_first[y])[1])
                                  +np.sum(ry*px)+np.sum(mx*ly))
                    full = u.reshape(rank,-1)
                    expected = scale*(-np.sum((lx.T@full)*by.reshape(rank,-1))
                        -np.sum((ly.T@full)*bx.reshape(rank,-1))
                        -2*np.sum((metric@lx)*ly))
                    np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=0)
    finally:
        store.directory.cleanup()


@pytest.mark.parametrize('response', ['dense','physical'])
def test_gradient_workspace_reused_by_subset_hessian(response, monkeypatch):
    from pyqed.qchem import basis_derivatives
    from pyqed.qchem.orbital_response import OrbitalResponse
    ref = _point(np.array([[0.,0.,0.],[0.,0.,1.5],[1.8,0.,.3]]),BSE,'hf','cd',charge=1)
    with ref.derivative_workspace():
        modes=np.random.default_rng(92).normal(size=(3,9))
        options=dict(state_ids=[0],gw_response='spectral',excitation_response=response)
        gradient=ref.gradient(modes=modes,**options).copy()
        workspace=ref._adjoint_workspace
        original=basis_derivatives.directional_one_electron_derivatives
        def no_first(*args,**kwargs):
            assert kwargs.get('order',2) != 1, 'Repeated first-order integral work'
            return original(*args,**kwargs)
        monkeypatch.setattr(basis_derivatives,'directional_one_electron_derivatives',no_first)
        def no_rebuild(*args,**kwargs):
            raise AssertionError('Rebuilt cached orbital solver')
        from pyqed.qchem.eri_response import CoulombIntegrals
        columns=CoulombIntegrals._cd_columns
        def no_first_columns(*args,**kwargs):
            assert kwargs.get('order',1) != 1, 'Repeated first CD integral columns'
            return columns(*args,**kwargs)
        from pyqed.qchem.hf.hessian import RHFHessian
        solve=RHFHessian._solve_orbital_system
        def shared_solve(self,*args,**kwargs):
            assert kwargs.get('orbital_solver') is workspace['reference']['adjoint_orbital_solver']
            return solve(self,*args,**kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(OrbitalResponse,'rhf',no_rebuild)
            patch.setattr(RHFHessian,'_solve_orbital_system',shared_solve)
            patch.setattr(CoulombIntegrals,'_cd_columns',no_first_columns)
            warm=ref.hessian(modes=modes[[2,0]],**options).copy()
        assert ref._adjoint_workspace is workspace
        np.testing.assert_allclose(ref.energy_gradient,gradient[:,[2,0]],atol=2e-10,rtol=0)
        from pathlib import Path
        folder=Path(workspace['store'].directory.name)
        assert folder.exists()
        ref.clear_derivative_cache()
        assert not folder.exists()
        monkeypatch.setattr(basis_derivatives,'directional_one_electron_derivatives',original)
        cold=ref.hessian(modes=modes[[2,0]],**options)
        np.testing.assert_allclose(warm,cold,atol=2e-10,rtol=0)


def test_adjoint_workspace_invalidates_changed_reference():
    from pathlib import Path
    ref=_point(np.array([[0.,0.,0.],[0.,0.,1.5],[1.8,0.,.3]]),BSE,'hf','cd',charge=1)
    with ref.derivative_workspace():
        modes=np.eye(9)[:1]
        ref.gradient(state_ids=[0],modes=modes)
        previous=ref._adjoint_workspace
        path=Path(previous['store'].directory.name)
        # In-place input changes must invalidate even if object identity survives.
        ref.e[0]+=1e-10
        ref.gradient(state_ids=[0],modes=modes)
        assert ref._adjoint_workspace is not previous
        assert not path.exists()


def test_copied_bse_does_not_own_original_workspace():
    from copy import copy, deepcopy
    from pathlib import Path
    ref=_point(np.array([[0.,0.,0.],[0.,0.,1.5],[1.8,0.,.3]]),BSE,'hf','cd',charge=1)
    with ref.derivative_workspace():
        modes=np.eye(9)[:1]
        expected=ref.gradient(state_ids=[0],modes=modes)
        original=ref._adjoint_workspace
        folder=Path(original['store'].directory.name)
        assert deepcopy(original) is original
        clone=copy(ref)
        clone.clear_derivative_cache()
        assert folder.exists()
        clone=copy(ref)
        np.testing.assert_allclose(clone.gradient(state_ids=[0],modes=modes),expected,atol=1e-12)
        assert clone._adjoint_workspace is not original
        clone.clear_derivative_cache()
        assert folder.exists()
        ref.clear_derivative_cache()
        assert not folder.exists()


@pytest.mark.parametrize('cls', [BSE, TDA])
def test_combined_derivatives_and_cleanup(cls, monkeypatch):
    from pathlib import Path
    ref = _point(np.array([[0.,0.,0.],[0.,0.,1.5],[1.8,0.,.3]]), cls, 'hf', 'cd', charge=1)
    modes = np.random.default_rng(104).normal(size=(2,9))
    options = dict(state_ids=[0], modes=modes, excitation_response='physical')
    expected_gradient = ref.gradient(**options).copy()
    expected_hessian = ref.hessian(columns=[1], **options).copy()
    def forbidden(*args, **kwargs):
        raise AssertionError('Combined derivatives must not run a separate gradient')
    with monkeypatch.context() as patch:
        patch.setattr(ref, 'gradient', forbidden)
        gradient, hessian = ref.derivatives(order=2, columns=[1], **options)
    np.testing.assert_allclose(gradient, expected_gradient, atol=2e-10, rtol=0)
    np.testing.assert_allclose(hessian, expected_hessian, atol=2e-10, rtol=0)
    assert gradient is ref.energy_gradient and hessian is ref.energy_hessian
    assert ref._adjoint_workspace is None and ref._first_response_cache is None
    np.testing.assert_allclose(ref.derivatives(order=1, **options), expected_gradient, atol=2e-10)
    assert ref._adjoint_workspace is None


def test_combined_derivatives_cleanup_on_failure(monkeypatch):
    from pathlib import Path
    ref = _point(np.array([[0.,0.,0.],[0.,0.,1.5],[1.8,0.,.3]]), BSE, 'hf', 'cd', charge=1)
    def fail(*args, **kwargs):
        raise RuntimeError('response failed')
    monkeypatch.setattr(ref, 'hessian', fail)
    with pytest.raises(RuntimeError, match='response failed'):
        ref.derivatives(order=2)
    assert ref._adjoint_workspace is None and ref._first_response_cache is None
    with pytest.raises(ValueError, match='order'):
        ref.derivatives(order=3)
