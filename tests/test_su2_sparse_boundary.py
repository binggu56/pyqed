import numpy as np
import pytest
import weakref

from test_su2_letta_closure import model
from pyqed.mps.nonabelian import environment as env
from pyqed.mps.nonabelian.sparse_boundary import SparseBoundary


def test_sparse_boundary_does_not_retain_itself_through_metadata():
    from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
    q = SpinChargeSector(0, SU2Irrep(0))
    boundary = SparseBoundary({(q, q, 0): np.ones((1, 2, 2))}, side='right', bond=1)
    boundary.array(0)
    assert boundary.table.stats['block_pool']['stored_bytes'] == boundary.nbytes
    reference = weakref.ref(boundary)
    del boundary
    assert reference() is None


def test_boundary_decode_writes_directly_to_final_storage(monkeypatch):
    from scipy.sparse import csr_matrix
    from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
    q = SpinChargeSector(0, SU2Irrep(0))
    parts = [np.eye(4, dtype=np.float32), 2*np.eye(4, dtype=np.float64)]
    boundary = SparseBoundary({(q, q, 0): parts}, side='right', bond=1,
                              decode_bytes=0)
    original = csr_matrix.toarray
    outputs = []

    def direct(self, order=None, out=None):
        assert out is not None
        outputs.append(out)
        return original(self, order=order, out=out)

    monkeypatch.setattr(csr_matrix, 'toarray', direct)
    actual = boundary.array(0)
    np.testing.assert_array_equal(actual, np.stack(parts))
    assert actual.dtype == np.float64
    assert all(np.shares_memory(view, actual) for view in outputs)
    assert boundary.stats['decoded_bytes'] == 0


def sparse(block, side, bond):
    return SparseBoundary({(qb, qk, ch): np.asarray(a).real
                           for (qb, qk), channels in block.data.items()
                           for ch, a in channels.items()}, side=side, bond=bond)


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('release_recoupling', [False, True])
def test_sparse_recursion_matches_independent_reduced_boundaries(side, release_recoupling):
    state = model(4, [(0, 3)], 'general')
    try:
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo, boundaries_only=True)
        expected = chain.left_envs[0] if side == 'left' else chain.right_envs[-1]
        start = 0 if side == 'left' else state.nsites-1
        actual = sparse(expected, side, start)
        indices = range(state.nsites-1) if side == 'left' else range(state.nsites-1, 0, -1)
        for i in indices:
            expected = expected.advance(state.mpo[i], sites[i], sites[i])
            actual = actual.advance(state.mpo[i], sites[i], moving_environment=state._su2_moving_environment,
                                    release_recoupling=release_recoupling)
            if release_recoupling:
                assert not state.mpo[i]._environment_reduced_block_cache
            reference = {(qb, qk, ch): a for (qb, qk), channels in expected.data.items()
                         for ch, a in channels.items()}
            for key in set(reference) | set(actual.blocks):
                value = np.stack([p.toarray() for p in actual.blocks[key]]) if key in actual.blocks else 0
                np.testing.assert_allclose(value, reference.get(key, 0), rtol=1e-11, atol=1e-10)
            with pytest.raises(RuntimeError, match='no dense arena'):
                _ = actual.table.block_pool.data
    finally:
        state.close()


def test_sparse_boundary_budgets_and_reactivation():
    state = model(4, [(0, 3)], 'general')
    try:
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo, boundaries_only=True)
        parent = sparse(chain.right_envs[-1], 'right', state.nsites-1)
        owner = state._su2_moving_environment
        packed = chain.right_envs[-1].ensure_packed(side='right', bond=state.nsites-1)
        labels, revision = env._packed_boundary_labels(packed)
        owner.install_boundary('right', state.nsites-1, np.ascontiguousarray(packed.block_pool.data.real),
                               packed.block_pool.offsets, labels, revision, 123)
        with pytest.raises(MemoryError, match='arena budget'):
            parent.advance(state.mpo[-1], sites[-1], moving_environment=owner, arena_bytes=1)
        with pytest.raises(MemoryError, match='storage budget'):
            parent.advance(state.mpo[-1], sites[-1], moving_environment=owner, storage_bytes=1)
        saved = {k: a.copy() for k, a in sites[-1].data.items()}
        for a in sites[-1].data.values():
            a[...] = 0
        zero = parent.advance(state.mpo[-1], sites[-1], moving_environment=owner)
        assert zero.nbytes == 0
        for k, a in sites[-1].data.items():
            a[...] = saved[k]
        restored = parent.advance(state.mpo[-1], sites[-1], moving_environment=owner)
        assert restored.nbytes > 0
        restored.decode_bytes = 1
        for i in range(len(restored.keys)):
            restored.array(i)
        assert restored._decoded_bytes == 0
        assert not restored._decoded
        assert owner.boundary_installed('right', state.nsites-1, revision, 123)
        np.testing.assert_array_equal(owner.boundary_values('right', state.nsites-1),
                                      packed.block_pool.data.real)
        assert not owner.release_boundary('right', state.nsites+1)
        assert not owner.release_boundary('right', state.nsites+2)
    finally:
        state.close()


def test_local_action_cache_eviction():
    cache = env._ArrayLRU(32)
    for i in range(10):
        cache.get(i, lambda: np.ones(4))
        assert cache.nbytes == 32
        assert len(cache.values) == 1
    large = cache.get('large', lambda: np.ones(5))
    assert large.size == 5
    assert 'large' not in cache.values
    assert cache.nbytes == 32


@pytest.mark.parametrize('cache_bytes', [0, 64, 8 * 2**20])
def test_sparse_one_site_adjoint(cache_bytes):
    state = model(4, [(0, 3)], 'general')
    try:
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo)
        left = [sparse(chain.left_envs[0], 'left', 0)]
        right = [None] * state.nsites
        right[-1] = sparse(chain.right_envs[-1], 'right', state.nsites-1)
        owner = state._su2_moving_environment
        for i in range(state.nsites-1):
            left.append(left[-1].advance(state.mpo[i], sites[i], moving_environment=owner))
        for i in range(state.nsites-1, 0, -1):
            right[i-1] = right[i].advance(state.mpo[i], sites[i], moving_environment=owner)
        for i, site in enumerate(sites):
            left[i].decode_bytes = right[i].decode_bytes = 64
            expected = env.one_site_reduced_adjoint(
                site, site, chain.mpo_factors[i], chain.left_envs[i].data,
                chain.right_envs[i].data, cache_bytes=0)
            actual = env.one_site_reduced_adjoint(
                site, site, chain.mpo_factors[i], left[i], right[i], cache_bytes=cache_bytes)
            assert actual.keys() == expected.keys()
            for key in expected:
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-11, atol=1e-10)
            assert left[i]._decoded_bytes <= 64
            assert right[i]._decoded_bytes <= 64
    finally:
        state.close()


@pytest.mark.parametrize('stride', [1, 2, 3])
def test_sparse_checkpoint_iterator(stride):
    from pyqed.mps.nonabelian.sparse_boundary import iter_right_boundaries
    state = model(4, [(0, 3)], 'general')
    try:
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo)
        endpoint = SparseBoundary.from_reduced(chain.right_envs[-1], side='right', bond=3)
        iterator = iter_right_boundaries(sites, chain.mpo_factors, endpoint,
            moving_environment=state._su2_moving_environment, stride=stride)
        seen = []
        for i, block in iterator:
            seen.append(i)
            reference = chain.right_envs[i].data
            for (qb, qk), channels in reference.items():
                for ch, expected in channels.items():
                    actual = block.get((qb, qk), {}).get(ch, 0)
                    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-10)
        assert seen == list(range(4))
    finally:
        state.close()


def test_solver_sparse_parameter_actions_without_dense_stack(monkeypatch):
    state = model(4, [(0, 3)], 'general')
    try:
        expected = state.reduced_parameter_actions()
        build = env.BlockSparseEnvironmentChain.build
        def endpoints_only(*args, **kwargs):
            assert kwargs.get('boundaries_only')
            return build(*args, **kwargs)
        monkeypatch.setattr(env.BlockSparseEnvironmentChain, 'build', endpoints_only)
        actual = state.reduced_parameter_actions(sparse_hamiltonian=True)
        for arrays, references in zip(actual, expected):
            for a, b in zip(arrays, references):
                np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-10)
    finally:
        state.close()


def test_cached_local_actions_do_not_redecode_right_blocks(monkeypatch):
    state = model(4, [(0, 3)], 'general')
    try:
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo)
        i = 1
        left = SparseBoundary.from_reduced(chain.left_envs[i], side='left', bond=i, decode_bytes=0)
        right = SparseBoundary.from_reduced(chain.right_envs[i], side='right', bond=i, decode_bytes=0)
        reads = []
        decode = right.array
        def counted(index):
            reads.append(index)
            return decode(index)
        monkeypatch.setattr(right, 'array', counted)
        expected = env.one_site_reduced_adjoint(sites[i], sites[i], chain.mpo_factors[i],
            chain.left_envs[i].data, chain.right_envs[i].data)
        actual = env.one_site_reduced_adjoint(sites[i], sites[i], chain.mpo_factors[i], left, right)
        assert reads
        assert len(reads) == len(set(reads))
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        assert right._decoded_bytes == 0
    finally:
        state.close()


def test_output_batching_preserves_values_and_arena_bounds(monkeypatch):
    state = model(4, [(0, 3)], 'general')
    try:
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo)
        parent = SparseBoundary.from_reduced(chain.right_envs[2], side='right', bond=2)
        parent.decode_bytes = 0
        decode = parent.array
        decoded = []

        def decode_one(index):
            assert all(reference() is None for reference in decoded)
            array = decode(index)
            decoded.append(weakref.ref(array))
            return array

        monkeypatch.setattr(parent, 'array', decode_one)
        owner = state._su2_moving_environment
        *_, routes, specs = env._plan_rank_coupled_boundary(
            state.mpo[2], sites[2], parent.table, sites[2], side='right', require_real=True)
        groups = {key: set() for key in specs}
        for p, a, b, w, key in routes:
            groups[key].add(p)
        budget = max(max(int(np.prod(specs[key]))*8,
                        sum(int(parent.offsets[i+1]-parent.offsets[i])*8 for i in ids))
                     for key, ids in groups.items())
        small = parent.advance(state.mpo[2], sites[2], moving_environment=owner, arena_bytes=budget)
        small_stats = dict(parent.last_advance)
        large = parent.advance(state.mpo[2], sites[2], moving_environment=owner)
        assert small_stats['completed_batches'] > parent.last_advance['completed_batches']
        assert small_stats['maximum_gather_bytes'] <= budget
        assert small_stats['maximum_output_bytes'] <= budget
        assert small.blocks.keys() == large.blocks.keys()
        for key in small.blocks:
            for a, b in zip(small.blocks[key], large.blocks[key]):
                np.testing.assert_array_equal(a.toarray(), b.toarray())
    finally:
        state.close()


def test_disk_boundaries_match_memory_and_release_files(tmp_path):
    state = model(4, [(0, 3)], 'general')
    try:
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo, boundaries_only=True)
        disk = memory = SparseBoundary.from_reduced(chain.right_envs[-1], side='right', bond=3)
        handles = []
        for i in range(3, 0, -1):
            memory = memory.advance(state.mpo[i], sites[i], moving_environment=state._su2_moving_environment)
            disk = disk.advance(state.mpo[i], sites[i], moving_environment=state._su2_moving_environment,
                                spill_directory=tmp_path)
            handles.append(weakref.ref(disk.blocks.file))
            assert disk.stats['resident_sparse_bytes'] == 0
            assert disk.stats['disk_bytes'] > 0
            for j, key in enumerate(disk.keys):
                expected = np.stack([a.toarray() for a in memory.blocks[key]])
                np.testing.assert_array_equal(disk.array(j), expected)
        del disk
        assert all(handle() is None for handle in handles)
        assert not list(tmp_path.iterdir())
    finally:
        state.close()


def test_disk_block_views_survive_later_reads_and_detect_truncation(tmp_path):
    from scipy.sparse import bsr_matrix
    from pyqed.mps.nonabelian.sparse_boundary import _DiskBlocks
    blocks = _DiskBlocks(tmp_path, 2**20)
    try:
        first = (bsr_matrix(np.eye(4), blocksize=(2, 2)),
                 bsr_matrix(np.zeros((4, 4)), blocksize=(1, 1)))
        blocks['a'] = first
        saved = blocks['a']
        blocks['b'] = (bsr_matrix(np.ones((4, 4)), blocksize=(2, 2)),)
        np.testing.assert_array_equal(blocks['b'][0].toarray(), np.ones((4, 4)))
        for actual, expected in zip(saved, first):
            np.testing.assert_array_equal(actual.toarray(), expected.toarray())
        blocks.file.truncate(blocks.disk_bytes-1)
        with pytest.raises(OSError, match='Incomplete sparse boundary'):
            blocks['b']
    finally:
        blocks.file.close()


def test_disk_parameter_actions_and_budget_cleanup(tmp_path):
    state = model(4, [(0, 3)], 'general')
    try:
        expected = state.reduced_parameter_actions()
        actual = state.reduced_parameter_actions(sparse_hamiltonian=True, spill_directory=tmp_path)
        for arrays, refs in zip(actual, expected):
            for a, b in zip(arrays, refs):
                np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-10)
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo, boundaries_only=True)
        parent = SparseBoundary.from_reduced(chain.right_envs[-1], side='right', bond=3)
        with pytest.raises(MemoryError, match='disk storage'):
            parent.advance(state.mpo[-1], sites[-1], moving_environment=state._su2_moving_environment,
                           spill_directory=tmp_path, disk_bytes=1)
        assert not list(tmp_path.iterdir())
    finally:
        state.close()


@pytest.mark.parametrize('side,site_index', [('left', 1), ('right', 2)])
def test_split_parent_gathering_preserves_route_accumulation(side, site_index):
    state = model(4, [(0, 3)], 'general')
    try:
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo)
        block = (chain.left_envs if side == 'left' else chain.right_envs)[site_index]
        parent = SparseBoundary.from_reduced(block, side=side, bond=site_index)
        core, site = state.mpo[site_index], sites[site_index]
        *_, routes, specs = env._plan_rank_coupled_boundary(
            core, site, parent.table, site, side=side, require_real=True)
        budget = max(max(int(np.prod(shape))*8 for shape in specs.values()),
                     max(int(parent.offsets[p+1]-parent.offsets[p])*8 for p, *_ in routes))
        split = parent.advance(core, site, moving_environment=state._su2_moving_environment,
                               arena_bytes=budget)
        stats = dict(parent.last_advance)
        assert stats['completed_gathers'] > stats['completed_batches']
        assert max(stats['maximum_gather_bytes'], stats['maximum_output_bytes']) <= budget
        whole = parent.advance(core, site, moving_environment=state._su2_moving_environment)
        assert split.keys == whole.keys
        for i in range(len(split.keys)):
            np.testing.assert_array_equal(split.array(i), whole.array(i))
    finally:
        state.close()


@pytest.mark.parametrize('stride,updates', [(1, 3), (2, 4), (3, 3)])
def test_disk_action_checkpoint_spacing(stride, updates, tmp_path, monkeypatch):
    state = model(4, [(0, 3)], 'general')
    try:
        expected = state.reduced_parameter_actions()
        advance = SparseBoundary.advance
        right_updates = []
        def observed(parent, *args, **kwargs):
            child = advance(parent, *args, **kwargs)
            if parent.side == 'right':
                right_updates.append(child.bond)
            return child
        monkeypatch.setattr(SparseBoundary, 'advance', observed)
        actual = state.reduced_parameter_actions(sparse_hamiltonian=True,
            spill_directory=tmp_path, checkpoint_stride=stride)
        assert len(right_updates) == updates
        for arrays, refs in zip(actual, expected):
            for a, b in zip(arrays, refs):
                np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-10)
    finally:
        state.close()


def test_disk_checkpoints_discard_construction_decode_cache(tmp_path, monkeypatch):
    from pyqed.mps.nonabelian.sparse_boundary import iter_right_boundaries
    state = model(4, [(0, 3)], 'general')
    retained = []
    try:
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo, boundaries_only=True)
        endpoint = SparseBoundary.from_reduced(chain.right_envs[-1], side='right', bond=3)
        advance = SparseBoundary.advance
        def observed(parent, *args, **kwargs):
            child = advance(parent, *args, **kwargs)
            retained.append(child)
            return child
        monkeypatch.setattr(SparseBoundary, 'advance', observed)
        iterator = iter_right_boundaries(sites, state.mpo, endpoint,
            moving_environment=state._su2_moving_environment, stride=1, spill_directory=tmp_path)
        try:
            assert next(iterator)[0] == 0
            assert len(retained) == 3
            assert all(block.stats['decoded_bytes'] == 0 for block in retained)
        finally:
            iterator.close()
    finally:
        retained.clear()
        state.close()


def test_local_action_preserves_arbitrarily_small_imaginary_ket():
    state = model(4, [(0, 3)], 'general')
    try:
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo)
        i = 1
        left = sparse(chain.left_envs[i], 'left', i)
        right = sparse(chain.right_envs[i], 'right', i)
        expected = env.one_site_reduced_adjoint(sites[i], sites[i], state.mpo[i], left, right)
        ket = state._wigner_eckart_route_plan(i).tensor(state._pack_site(i)*(1+1e-20j))
        actual = env.one_site_reduced_adjoint(sites[i], ket, state.mpo[i], left, right)
        for key in expected:
            np.testing.assert_allclose(actual[key].real, expected[key].real, rtol=1e-11, atol=1e-10)
            np.testing.assert_allclose(actual[key].imag/1e-20, expected[key].real, rtol=1e-11, atol=1e-10)
    finally:
        state.close()


def test_zero_ket_blocks_skip_work_without_pruning_bra_outputs(monkeypatch):
    state = model(4, [(0, 3)], 'general')
    try:
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo)
        i = 1
        route = state._wigner_eckart_route_plan(i)
        ket = route.tensor(state._pack_site(i))
        zero_bra = route.tensor(np.zeros_like(state._pack_site(i)))
        left, right = chain.left_envs[i].data, chain.right_envs[i].data
        expected = env.one_site_reduced_adjoint(sites[i], ket, state.mpo[i], left, right)
        actual = env.one_site_reduced_adjoint(zero_bra, ket, state.mpo[i], left, right)
        assert any(np.any(a) for a in expected.values())
        assert actual.keys() == expected.keys()
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        saved = {key:a.copy() for key,a in ket.data.items()}
        for a in ket.data.values():
            a[...] = 0
        with monkeypatch.context() as patch:
            def forbidden(*args, **kwargs):
                raise AssertionError('Zero ket requested a recoupling block')
            patch.setattr(env, '_left_reduced_rank_coupled_block', forbidden)
            zero = env.one_site_reduced_adjoint(sites[i], ket, state.mpo[i], left, right)
        assert zero.keys() == expected.keys()
        assert all(not np.any(a) for a in zero.values())
        for key,a in ket.data.items():
            a[...] = saved[key]
        restored = env.one_site_reduced_adjoint(sites[i], ket, state.mpo[i], left, right)
        for key in expected:
            np.testing.assert_array_equal(restored[key], expected[key])
    finally:
        state.close()


@pytest.mark.parametrize('disk', [False, True])
def test_route_support_does_not_decode_sparse_blocks(tmp_path, monkeypatch, disk):
    from pyqed.mps.nonabelian.sparse_boundary import _DiskBlocks
    state = model(4, [(0, 3)], 'general')
    try:
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo, boundaries_only=True)
        parent = sparse(chain.right_envs[-1], 'right', state.nsites-1)
        if disk:
            blocks = _DiskBlocks(tmp_path, 2**20)
            for key, parts in parent.blocks.items():
                blocks[key] = parts
            parent = SparseBoundary(blocks, side='right', bond=state.nsites-1)
        expected = env._plan_rank_coupled_boundary(
            state.mpo[-1], sites[-1], parent.table, sites[-1], side='right', require_real=True)
        def forbidden(*args):
            raise AssertionError('Decoded a boundary merely to inspect support')
        monkeypatch.setattr(parent, 'array', forbidden)
        actual = env._plan_rank_coupled_boundary(
            state.mpo[-1], sites[-1], parent.table, sites[-1], side='right', require_real=True)
        assert actual[-2:] == expected[-2:]
        assert parent.stats['decoded_bytes'] == 0
    finally:
        state.close()


@pytest.mark.parametrize('disk', [False, True])
def test_sparse_support_keeps_tiny_values_and_excludes_explicit_zeros(tmp_path, disk):
    from scipy.sparse import bsr_matrix
    from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
    from pyqed.mps.nonabelian.sparse_boundary import _DiskBlocks
    q = SpinChargeSector(0, SU2Irrep(0))
    blocks = _DiskBlocks(tmp_path, 2**20) if disk else {}
    for channel, value in enumerate([0., 1e-200]):
        part = bsr_matrix((np.full((1, 1, 1), value), np.array([0]), np.array([0, 1])), shape=(1, 1))
        blocks[q, q, channel] = (part,)
    boundary = SparseBoundary(blocks, side='right', bond=1)
    for index, (_, _, channel) in enumerate(boundary.keys):
        assert boundary.has_nonzero(index) == bool(channel)
    assert boundary.stats['decoded_bytes'] == 0


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_disk_component_descriptors_preserve_shapes_and_dtypes(dtype, tmp_path):
    from scipy.sparse import bsr_matrix
    from pyqed.mps.nonabelian.sparse_boundary import _DiskBlocks
    blocks = _DiskBlocks(tmp_path, 2**20)
    try:
        parts = [bsr_matrix(np.arange(16, dtype=dtype).reshape(4, 4), blocksize=(2, 2)),
                 bsr_matrix((4, 4), dtype=dtype, blocksize=(1, 1))]
        parts[0].indices = parts[0].indices.astype(np.int64)
        parts[0].indptr = parts[0].indptr.astype(np.int64)
        blocks['mixed'] = parts
        for actual, expected in zip(blocks['mixed'], parts):
            assert actual.data.dtype == expected.data.dtype
            assert actual.blocksize == expected.blocksize
            np.testing.assert_array_equal(actual.toarray(), expected.toarray())
    finally:
        blocks.file.close()


@pytest.mark.parametrize('side', ['left', 'right'])
def test_grouped_routes_preserve_every_route_and_order(side):
    state = model(4, [(0, 3)], 'general')
    try:
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo, boundaries_only=True)
        i = 0 if side == 'left' else state.nsites-1
        block = chain.left_envs[0] if side == 'left' else chain.right_envs[-1]
        parent = SparseBoundary.from_reduced(block, side=side, bond=i)
        for _ in range(2):
            args = (state.mpo[i], sites[i], parent.table, sites[i])
            ordinary = env._plan_rank_coupled_boundary(*args, side=side, require_real=True)
            compact = env._plan_rank_coupled_boundary(*args, side=side, require_real=True, grouped_routes=True)
            assert ordinary[-1] == compact[-1]
            expected = {}
            for p, a, b, w, key in ordinary[-2]:
                expected.setdefault(key, []).extend((p, a, b, w))
            assert list(expected) == list(compact[-2])
            for key, values in compact[-2].items():
                assert values.itemsize == 8
                assert list(values) == expected[key]
            parent = parent.advance(state.mpo[i], sites[i], moving_environment=state._su2_moving_environment)
            i += 1 if side == 'left' else -1
    finally:
        state.close()
