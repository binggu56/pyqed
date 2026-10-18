import numpy as np
import pytest

from test_su2_letta_closure import model
from pyqed.letta.parameter_operator import TiedParameterOperator, iter_parameter_blocks
from pyqed.mps.nonabelian import environment as env


@pytest.mark.parametrize('scale', [1., 1e-24, 1e24])
def test_metric_whitening_is_invariant_to_uniform_tensor_scale(scale):
    from pyqed.letta.su2_qchem import _ReducedMetricComponents
    block = np.array([[2., .3j], [-.3j, 1.]])
    metric = _ReducedMetricComponents(2, [(np.arange(2), scale*block)])
    transform, *_ = metric.whitener(rtol=1e-11, sparse=True)
    assert transform.shape == (2, 2)
    np.testing.assert_allclose(transform.conj().T@metric.dense()@transform, np.eye(2), atol=1e-12)
    vector = np.array([1.+2j, 3.-4j])*np.sqrt(scale)
    actual = metric.whitened_norm(vector, rtol=1e-11)
    base = _ReducedMetricComponents(2, [(np.arange(2), block)])
    np.testing.assert_allclose(actual, base.whitened_norm(vector/np.sqrt(scale), rtol=1e-11), atol=1e-12)


def test_sparse_local_coordinates_preserve_discarded_incumbent():
    from pyqed.letta.parameter_operator import preserve_metric_reference
    from pyqed.letta.su2_qchem import _ReducedMetricComponents
    metric = _ReducedMetricComponents(3, [(np.arange(3), np.diag([1., 1e-14, 0.]))])
    transform, *_ = metric.whitener(rtol=1e-11, sparse=True)
    current = np.array([1., 1e7j, 0.])
    augmented, lost = preserve_metric_reference(metric, transform, current)
    assert transform.shape == (3, 1) and augmented.shape == (3, 2)
    assert lost == pytest.approx(.5)
    np.testing.assert_allclose(augmented@(augmented.conj().T@metric.matvec(current)), current)
    np.testing.assert_allclose(augmented.conj().T@metric.matvec(augmented.toarray()), np.eye(2))
    same, lost = preserve_metric_reference(metric, augmented, current)
    assert same.shape == augmented.shape and lost < 1e-25


def test_metric_reference_does_not_normalize_unresolved_cancellation():
    from pyqed.letta.parameter_operator import preserve_metric_reference
    from pyqed.letta.su2_qchem import _ReducedMetricComponents
    metric = _ReducedMetricComponents(2, [(np.arange(2), np.array([[1., 1.], [1., 1.+1e-15]]))])
    transform, *_ = metric.whitener(rtol=1e-11, sparse=True)
    augmented, missing = preserve_metric_reference(metric, transform, np.array([1., 0.]))
    assert augmented.shape == transform.shape == (2, 1)
    assert missing < 1e-14


def test_local_solver_uses_actual_norm_after_imperfect_whitening(monkeypatch):
    from scipy.sparse import csr_matrix
    from pyqed.letta.parameter_operator import solve_parameter_operator
    from pyqed.letta.su2_qchem import _ReducedMetricComponents
    metric = _ReducedMetricComponents(2, [(np.arange(2), np.eye(2))])
    transform = csr_matrix(np.diag([1., 1.+1e-7]))
    monkeypatch.setattr(metric, 'whitener', lambda **kwargs: (transform, None, None, 0))
    h = np.array([[-2., .3], [.3, -1.]])
    with TiedParameterOperator(2, [(np.arange(2), np.arange(2), h)]) as operator:
        candidate, _, report = solve_parameter_operator(operator, metric, np.array([1., .5]),
                                                        tolerance=1e-12)
    np.testing.assert_allclose(report['operator_energy'], np.linalg.eigvalsh(h)[0], atol=1e-12)
    assert report['operator_residual'] < 1e-12
    assert np.linalg.norm(h@candidate-report['operator_energy']*candidate) < 1e-12


def test_local_solver_uses_factored_norm_for_energy_and_residual():
    from scipy.linalg import eigh
    from pyqed.letta.parameter_operator import solve_parameter_operator
    from pyqed.letta.su2_qchem import _ReducedMetricComponents
    n = np.array([[1., .2], [.2, 2.]])
    h = np.array([[-2., .3], [.3, -1.]])
    metric = _ReducedMetricComponents(2, [(np.arange(2), n+1e-8*np.eye(2))])
    current = np.array([1., .5])
    with TiedParameterOperator(2, [(np.arange(2), np.arange(2), h)]) as operator:
        candidate, transform, report = solve_parameter_operator(
            operator, metric, current, norm_matvec=lambda v: n@v, tolerance=1e-12)
    assert report['factored_norm']
    np.testing.assert_allclose(report['operator_energy'], eigh(h, n)[0][0], atol=1e-12)
    np.testing.assert_allclose(report['energy_before'], current@h@current/(current@n@current), atol=1e-14)
    assert np.linalg.norm(transform.conj().T@(h@candidate-report['operator_energy']*(n@candidate))) < 1e-12


@pytest.mark.parametrize('spill', [False, True])
def test_transformed_diagonal_keeps_complex_off_diagonal_couplings(tmp_path, spill):
    from scipy.sparse import csr_matrix
    rng = np.random.default_rng(841)
    rows, columns = np.array([0, 2, 0]), np.array([1, 2, 1])
    values = rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3))
    transform = rng.normal(size=(3, 4))+1j*rng.normal(size=(3, 4))
    transform[0, 2:] = 0
    transform[2, :2] = 0
    dense = np.zeros((3, 3), dtype=complex)
    np.add.at(dense, (rows[:, None], columns[None, :]), values)
    with TiedParameterOperator(3, [(rows, columns, values)],
                               spill_directory=tmp_path if spill else None) as op:
        np.testing.assert_allclose(op.transformed_diagonal(csr_matrix(transform)),
                                   np.diag(transform.conj().T@dense@transform), rtol=1e-13, atol=1e-13)
        with pytest.raises(ValueError, match='sparse coordinates'):
            op.transformed_diagonal(transform)


def test_frozen_operator_snapshot_provenance_and_truncation(tmp_path):
    import json
    from examples.qchem.benchmark_su2_parameter_operator import snapshot_blocks
    (tmp_path/'operator-provenance.json').write_text(json.dumps(
        dict(sha256='checkpoint', site=2, parameters=3, blocks=2)))
    np.savez(tmp_path/'operator-index.npz', dimension=3, rows=[0, 2, 1], columns=[1, 0, 2],
             row_offsets=[0, 2, 3], col_offsets=[0, 1, 3], payload_offsets=[0, 32])
    values = np.array([1+2j, 3-4j, 5+6j, 7-8j])
    values.tofile(tmp_path/'operator-values.bin')
    kwargs = dict(sha256='checkpoint', site=2, dimension=3)
    with TiedParameterOperator(3, snapshot_blocks(tmp_path, **kwargs), spill_directory=tmp_path) as op:
        dense = np.array([[0, values[0], 0], [values[2], 0, values[3]], [0, values[1], 0]])
        vector = np.array([2+1j, 4-3j, 6+5j])
        np.testing.assert_array_equal(op.matvec(vector), dense@vector)
    for key, bad in [('sha256', 'different'), ('site', 1), ('dimension', 4)]:
        with pytest.raises(ValueError, match='match'):
            list(snapshot_blocks(tmp_path, **(kwargs | {key: bad})))
    with (tmp_path/'operator-values.bin').open('r+b') as stream:
        stream.truncate(63)
    with pytest.raises(OSError, match='Incomplete'):
        list(snapshot_blocks(tmp_path, **kwargs))


@pytest.mark.parametrize('spill', [False, True])
@pytest.mark.parametrize('block_bytes', [64, 2**20])
def test_parameter_operator_matches_complex_reduced_action(spill, block_bytes, tmp_path):
    state = model(4, [(0, 1), (0, 3), (1, 2), (2, 3)], 'general')
    try:
        rng = np.random.default_rng(728)
        for i in range(state.nsites):
            c = state._pack_site(i)
            state._set_site_vector(i, c+.1j*rng.normal(size=c.size))
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo)
        site = 1
        plan = state._wigner_eckart_route_plan(site)
        left, right = chain.left_envs[site].data, chain.right_envs[site].data
        sizes = []
        def blocks():
            for rows, columns, values in iter_parameter_blocks(
                    plan, state.mpo[site], left, right, block_bytes=block_bytes, route_tile=2):
                sizes.append(values.nbytes)
                yield rows, columns, values
        with TiedParameterOperator(plan.nparameters, blocks(),
                                   spill_directory=tmp_path if spill else None) as operator:
            assert sizes and max(sizes) <= block_bytes
            for c in [state._pack_site(site), rng.normal(size=plan.nparameters)+1j*rng.normal(size=plan.nparameters)]:
                expected = state._one_site_parameter_adjoint(
                    site, sites[site], plan.tensor(c), state.mpo[site], left, right)
                np.testing.assert_allclose(operator.matvec(c), expected, rtol=1e-11, atol=1e-12)
            if spill:
                assert operator.disk_bytes == sum(sizes)
            with pytest.raises(ValueError, match='shape'):
                operator.matvec(np.zeros(1))
        with pytest.raises(RuntimeError, match='closed'):
            operator.matvec(c)
        assert not list(tmp_path.iterdir())
    finally:
        state.close()


def test_parameter_operator_storage_guards_and_ownership(tmp_path):
    indices = np.arange(2)
    matrix = np.eye(2, dtype=complex)
    for kwargs, message in [({'storage_bytes': 1}, 'resident'),
                            ({'spill_directory': tmp_path, 'disk_bytes': 1}, 'disk')]:
        with pytest.raises(MemoryError, match=message):
            TiedParameterOperator(2, [(indices, indices, matrix)], **kwargs)
        assert not list(tmp_path.iterdir())
    with TiedParameterOperator(2, [(indices, indices, matrix)]) as operator:
        indices[:] = 0
        matrix[:] = 0
        np.testing.assert_array_equal(operator.matvec([1, 2]), [1, 2])
    with TiedParameterOperator(2, [(np.arange(2), np.arange(2), np.eye(2))],
                               spill_directory=tmp_path) as operator:
        operator.file.truncate(operator.disk_bytes-1)
        with pytest.raises(OSError, match='Incomplete'):
            operator.matvec([1, 2])


def test_parameter_operator_matches_independent_adjacent_pair_projection():
    state = model(4, [(0, 1), (0, 3), (1, 2), (2, 3)], 'general')
    try:
        rng = np.random.default_rng(117)
        for i in range(state.nsites):
            c = state._pack_site(i)
            state._set_site_vector(i, c+.1j*rng.normal(size=c.size))
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo)
        site = 1
        plan = state._wigner_eckart_route_plan(site)
        blocks = iter_parameter_blocks(plan, state.mpo[site],
                                       chain.left_envs[site].data, chain.right_envs[site].data)
        with TiedParameterOperator(plan.nparameters, blocks) as operator:
            _, support, reference, *_ = state._frontier_wigner_eckart_matrix_free_problem(
                site, site, support_rtol=0.)
            vectors = []
            for _ in range(2):
                vector = np.zeros(plan.nparameters, dtype=complex)
                vector[support] = rng.normal(size=len(support))+1j*rng.normal(size=len(support))
                vectors.append(vector)
                np.testing.assert_allclose(operator.matvec(vector)[support], reference(vector[support]),
                                           rtol=1e-11, atol=1e-12)
            a, b = vectors
            np.testing.assert_allclose(np.vdot(a, operator.matvec(b)), np.vdot(operator.matvec(a), b),
                                       rtol=1e-11, atol=1e-12)
            from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo
            from pyqed.letta.parameter_operator import solve_parameter_operator
            identity = _identity_mpo_factors_for_sites_and_mpo(sites, state.mpo)
            norm_chain = env.BlockSparseEnvironmentChain.build(sites, identity)
            metric = state._one_site_norm_components(site, identity[site],
                norm_chain.left_envs[site].data, norm_chain.right_envs[site].data)
            candidate, transform, report = solve_parameter_operator(
                operator, metric, state._pack_site(site), maxiter=160, max_space=32)
            energy = report['operator_energy']
            vector = transform.conj().T@metric.matvec(candidate)
            def action(vector):
                return transform.conj().T@operator.matvec(transform@vector)
            restricted = transform[support]
            matrix = restricted.conj().T@np.column_stack([reference(v) for v in restricted.toarray().T])
            expected = np.linalg.eigvalsh((matrix+matrix.conj().T)/2)[0]
            np.testing.assert_allclose(energy, expected, rtol=0, atol=1e-9)
            assert np.linalg.norm(action(vector)-energy*vector) < 1e-9
    finally:
        state.close()


@pytest.mark.parametrize('spill', [False, True])
def test_parameter_operator_diagonal_handles_overlap_and_repeated_indices(spill, tmp_path):
    blocks = [(np.array([2, 0]), np.array([0, 2]), np.array([[1j, 2], [3, 4j]])),
              (np.array([1, 1]), np.array([1, 2]), np.array([[5, 6], [7, 8]])),
              (np.array([2]), np.array([2]), np.array([[9]]))]
    with TiedParameterOperator(3, blocks, spill_directory=tmp_path if spill else None) as operator:
        retained = operator.resident_bytes
        np.testing.assert_array_equal(operator.diagonal(), [3, 12, 11])
        eye = np.eye(3)
        dense = np.column_stack([operator.matvec(v) for v in eye])
        np.testing.assert_array_equal(operator.diagonal(), np.diag(dense))
        assert operator.resident_bytes == retained
    with pytest.raises(RuntimeError, match='closed'):
        operator.diagonal()


def test_sparse_component_whitener_preserves_complex_support_and_cache():
    from pyqed.letta.su2_qchem import _ReducedMetricComponents
    from scipy.sparse import issparse
    metric = _ReducedMetricComponents(8, [
        (np.array([0, 3]), np.array([[2, 1j], [-1j, 3]])),
        (np.array([1, 6]), np.diag([4., 1e-15])),
        (np.array([2, 4, 5, 7]), np.diag([0., 5., 6., 0.]))])
    cache = {}
    revision = lambda indices, block: hash((indices.tobytes(), block.tobytes()))
    dense, values, backend, _ = metric.whitener(rtol=1e-11, component_cache=cache, revision=revision)
    sparse, actual_values, actual_backend, hits = metric.whitener(
        rtol=1e-11, component_cache=cache, revision=revision, sparse=True)
    assert issparse(sparse) and hits == len(metric.components)
    np.testing.assert_array_equal(sparse.toarray(), dense)
    np.testing.assert_array_equal(actual_values, values)
    assert actual_backend == backend
    np.testing.assert_allclose((sparse.conj().T@metric.dense()@sparse).real, np.eye(sparse.shape[1]), atol=1e-12)
    for actual, expected in zip(metric.orthonormal_components(sparse), metric.orthonormal_components(dense)):
        np.testing.assert_array_equal(actual, expected)


def test_sparse_component_whitener_does_not_allocate_full_height_panels(monkeypatch):
    from pyqed.letta.su2_qchem import _ReducedMetricComponents
    metric = _ReducedMetricComponents(100000, [(np.array([7, 90000]), np.diag([2., 3.]))])
    original = np.zeros
    def bounded(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 2 and shape[0] == metric.dimension:
            raise AssertionError('Allocated a full-height whitening panel')
        return original(shape, *args, **kwargs)
    monkeypatch.setattr(np, 'zeros', bounded)
    transform, *_ = metric.whitener(rtol=1e-11, sparse=True)
    assert transform.shape == (100000, 2) and transform.nnz == 2


@pytest.mark.parametrize('sparse', [False, True])
def test_whitener_keeps_small_off_diagonal_terms_in_weak_metric_directions(sparse):
    from pyqed.letta.su2_qchem import _ReducedMetricComponents
    metric = _ReducedMetricComponents(3, [
        (np.array([0]), np.ones((1, 1))),
        (np.array([1, 2]), np.array([[1e-10, 9e-12], [9e-12, 1e-10]]))])
    transform, values, backend, _ = metric.whitener(rtol=1e-11, sparse=sparse)
    assert backend == 'block_eigh'
    np.testing.assert_allclose(transform.conj().T@metric.dense()@transform, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.sort(values), [9.1e-11, 1.09e-10, 1.], rtol=1e-12, atol=0.)


def test_compiled_parameter_routes_match_numpy_and_validate_indices():
    from pyqed.mps.nonabelian import _su2_kernel
    rng = np.random.default_rng(54)
    left, right = rng.normal(size=(2, 4, 3)), rng.normal(size=(3, 4, 5))
    weights, metric = rng.normal(size=(2, 3)), rng.normal(size=3)
    bra = np.array([[0, 1, 2], [1, 0, 3], [0, 3, 1]], dtype=np.int64)
    ket = np.array([[1, 2, 0], [0, 1, 4], [1, 0, 2]], dtype=np.int64)
    actual = rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2))
    expected = actual.copy()
    for oi, ol, ore in bra:
        for ii, il, ir in ket:
            expected[oi, ii] += np.sum(left[:, ol, il, None]*right[None, :, ore, ir]*weights*metric)
    _su2_kernel.accumulate_parameter_routes(actual, left, right, weights, metric, bra, ket)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    bad = bra.copy()
    bad[0, 0] = 2
    with pytest.raises(ValueError, match='out of bounds'):
        _su2_kernel.accumulate_parameter_routes(actual, left, right, weights, metric, bad, ket)
    with pytest.raises(ValueError, match='shapes'):
        _su2_kernel.accumulate_parameter_routes(actual, left, right, weights, metric[:1], bra, ket)


def test_parameter_route_compensation_survives_channel_cancellation():
    from pyqed.mps.nonabelian import _su2_kernel
    output = np.zeros((1, 1), dtype=complex)
    correction = np.zeros_like(output)
    routes = np.zeros((1, 3), dtype=np.int64)
    ones = np.ones((1, 1, 1))
    for value in [1e16, 1., 1., -1e16]:
        _su2_kernel.accumulate_parameter_routes(
            output, ones, ones, np.array([[value]]), np.ones(1),
            routes, routes, correction)
    assert output[0, 0] == 2.


@pytest.mark.parametrize('compensated', [False, True])
def test_real_parameter_builder_dispatch_and_tiny_imaginary_data(monkeypatch, compensated):
    from pyqed.mps.nonabelian import _su2_kernel
    state = model(4, [(0, 3)], 'general')
    try:
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo)
        site = 1
        plan = state._wigner_eckart_route_plan(site)
        left, right = chain.left_envs[site].data, chain.right_envs[site].data
        kwargs = dict(block_bytes=64, route_tile=2)
        expected = list(iter_parameter_blocks(plan, state.mpo[site], left, right, compiled=False, **kwargs))
        original = _su2_kernel.accumulate_parameter_routes
        calls = []
        def observed(*args):
            calls.append(1)
            return original(*args)
        monkeypatch.setattr(_su2_kernel, 'accumulate_parameter_routes', observed)
        actual = list(iter_parameter_blocks(plan, state.mpo[site], left, right,
                                            compensated=compensated, **kwargs))
        assert calls and len(actual) == len(expected)
        count = len(calls)
        calls.clear()
        list(iter_parameter_blocks(plan, state.mpo[site], left, right,
                                   block_bytes=64, route_tile=1))
        assert len(calls) == count
        for (ar, ac, a), (br, bc, b) in zip(actual, expected):
            np.testing.assert_array_equal(ar, br)
            np.testing.assert_array_equal(ac, bc)
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        varied = {key: {ch: np.asarray(value)*(1+1e-20j) for ch, value in channels.items()}
                  for key, channels in left.items()}
        if compensated:
            with pytest.raises(NotImplementedError, match='real scalar-physical'):
                list(iter_parameter_blocks(plan, state.mpo[site], varied, right,
                                           compensated=True, **kwargs))
        complex_blocks = list(iter_parameter_blocks(plan, state.mpo[site], varied, right, **kwargs))
        assert len(complex_blocks) == len(expected)
        for (_, _, a), (_, _, b) in zip(complex_blocks, expected):
            np.testing.assert_allclose(a.imag/1e-20, b.real, rtol=1e-11, atol=1e-10)
    finally:
        state.close()


def test_relaxation_aligns_phase_and_backtracks_in_physical_metric():
    from scipy.sparse.linalg import aslinearoperator
    from pyqed.letta.parameter_operator import relax_parameter_candidate
    metric = aslinearoperator(np.diag([2., 3.]))
    operator = aslinearoperator(np.diag([0., 3.]))
    current = np.array([np.sqrt(.99/2), np.sqrt(.01/3)], dtype=complex)
    root = np.array([1/np.sqrt(2), 0.], dtype=complex)
    outputs = []
    for phase in [1., -1., np.exp(.7j)]:
        trial, weight = relax_parameter_candidate(operator, metric, current, root*phase,
            relaxation=1.99, energy_before=.01, root_energy=0.)
        assert 1 < weight < 1.99
        np.testing.assert_allclose(np.vdot(trial, metric@trial), 1., atol=1e-14)
        assert np.vdot(trial, operator@trial).real <= .0095+1e-12
        outputs.append(trial)
    np.testing.assert_allclose(outputs, np.broadcast_to(outputs[0], (3, 2)), atol=1e-14)
    unchanged, weight = relax_parameter_candidate(operator, metric, current, root,
        relaxation=1., energy_before=.01, root_energy=0.)
    assert unchanged is root and weight == 1.


def test_reduced_adjoint_preserves_small_channel_contributions(monkeypatch):
    from types import SimpleNamespace
    from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
    q = SpinChargeSector(0, SU2Irrep(0))
    tensor = SimpleNamespace(data={(q, q, q): np.ones((1, 1, 1))})
    core = SimpleNamespace(right_channel_irreps={i: SU2Irrep(0) for i in range(4)})
    terms = {(0, i): np.full((1, 1, 1, 1), value)
             for i, value in enumerate([1e16, 1., 1., -1e16])}
    monkeypatch.setattr(env, '_left_reduced_rank_coupled_block', lambda *args: terms)
    left = {(q, q): {0: np.ones((1, 1, 1))}}
    right = {(q, q): {i: np.ones((1, 1, 1)) for i in range(4)}}
    actual = env.one_site_reduced_adjoint(tensor, tensor, core, left, right)
    np.testing.assert_array_equal(actual[q, q, q], np.full((1, 1, 1), 2.))


@pytest.mark.parametrize('compiled', [False, True])
def test_parameter_assembly_reuses_consecutive_channels(monkeypatch, compiled):
    from types import SimpleNamespace
    from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
    q = SpinChargeSector(0, SU2Irrep(0))
    plan = SimpleNamespace(_group_by_key={
        (q, q, q): (np.array([0]), np.array([0]), (1, 1, 1))})
    core = SimpleNamespace(right_channel_irreps={i: SU2Irrep(0) for i in range(3)})
    pairs = [(0, 0), (0, 1), (1, 1), (1, 2), (0, 2)]
    terms = {pair: np.ones((1, 1, 1, 1)) for pair in pairs}
    monkeypatch.setattr(env, '_left_reduced_rank_coupled_block', lambda *args: terms)

    class Channels(dict):
        def __init__(self, count):
            super().__init__((i, np.full((1, 1, 1), i+1.)) for i in range(count))
            self.reads = []

        def __getitem__(self, key):
            self.reads.append(key)
            return super().__getitem__(key)

    left, right = Channels(2), Channels(3)
    blocks = list(iter_parameter_blocks(plan, core, {(q, q): left},
                                       {(q, q): right}, compiled=compiled))
    assert left.reads == [0, 1, 0]
    assert right.reads == [0, 1, 2]
    np.testing.assert_array_equal(blocks[0][2], [[sum((l+1)*(r+1) for l,r in pairs)]])
