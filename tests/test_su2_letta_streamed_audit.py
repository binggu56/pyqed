import numpy as np
import pytest

from test_su2_letta_closure import model
from pyqed.letta import SU2LETTA
from pyqed.letta.su2_qchem import _ReducedMetricComponents
from pyqed.mps.nonabelian.environment import BlockSparseEnvironmentChain
from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo


@pytest.mark.parametrize('complex_state', [False, True])
@pytest.mark.parametrize('tie', ['physical', 'fusion'])
def test_direct_conditional_metric_matches_independent_norm_actions(complex_state, tie):
    state = model(4, [(0,3)], 'general')
    if tie == 'fusion':
        base = state._base_sites
        state.close()
        state = SU2LETTA(None, base_sites=base, nelec=4, spin=0, D=2,
                         graph=[(0,3)], tie=tie, virtual_boundary='periodic')
    rng = np.random.default_rng(821)
    try:
        for i in range(state.nsites):
            c = state._pack_site(i) + .03*rng.normal(size=state._pack_site(i).size)
            if complex_state:
                c = c + .03j*rng.normal(size=c.size)
            c[::3] = 0
            state._set_site_vector(i, c)
        sites = state.materialize()
        identity = _identity_mpo_factors_for_sites_and_mpo(sites, state.mpo)
        chain = BlockSparseEnvironmentChain.build(sites, identity)
        for site, tensor in enumerate(sites):
            route = state._wigner_eckart_route_plan(site)
            core = chain.mpo_factors[site]
            left, right = chain.left_envs[site].data, chain.right_envs[site].data
            metric = state._one_site_norm_components(site, core, left, right)
            direction = rng.normal(size=route.nparameters) + 1j*rng.normal(size=route.nparameters)
            reference = state._one_site_parameter_adjoint(
                site, tensor, route.tensor(direction), core, left, right)
            np.testing.assert_allclose(metric.matvec(direction), reference, rtol=1e-11, atol=1e-10)
    finally:
        state.close()


def test_component_whitened_norm_matches_explicit_whitener():
    rng = np.random.default_rng(20)
    a = rng.normal(size=(3,2)) + 1j*rng.normal(size=(3,2))
    metric = _ReducedMetricComponents(6, [([0,2,4], a@a.conj().T),
                                           ([1,3,5], np.diag([2.,1e-14,3.]))])
    vector = rng.normal(size=6) + 1j*rng.normal(size=6)
    whitener, *_ = metric.whitener(rtol=1e-11)
    np.testing.assert_allclose(metric.whitened_norm(vector, rtol=1e-11),
                               np.linalg.norm(whitener.conj().T@vector), rtol=1e-14)


def test_streamed_actions_consume_boundaries_and_match_full_stacks(monkeypatch):
    state = model(4, [(0,3)], 'general')
    try:
        sites = state.materialize()
        identity = _identity_mpo_factors_for_sites_and_mpo(sites, state.mpo)
        references = [state._parameter_chain_actions(
            sites, BlockSparseEnvironmentChain.build(sites, factors))
            for factors in (state.mpo, identity)]
        chains = []
        build = BlockSparseEnvironmentChain.build
        def observed(*args, **kwargs):
            assert kwargs['sweep_direction'] == 'lr'
            chain = build(*args, **kwargs)
            assert all(block is None for block in chain.left_envs[1:])
            chains.append(chain)
            return chain
        monkeypatch.setattr(BlockSparseEnvironmentChain, 'build', observed)
        adjoint = state._one_site_parameter_adjoint
        def checked(site, *args):
            assert all(block is None for block in chains[-1].right_envs[:site])
            return adjoint(site, *args)
        monkeypatch.setattr(state, '_one_site_parameter_adjoint', checked)
        for actual, expected in zip(state.reduced_parameter_actions(), references):
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        assert len(chains) == 2 and all(not chain.right_envs for chain in chains)
    finally:
        state.close()


@pytest.mark.parametrize('complex_state', [False, True])
def test_streamed_stationarity_matches_pair_metric_reference(complex_state, monkeypatch):
    state = model(4, [(0,3)], 'general')
    rng = np.random.default_rng(628)
    try:
        if complex_state:
            for i in range(state.nsites):
                c = state._pack_site(i)
                state._set_site_vector(i, c+.02j*rng.normal(size=c.size))
        sites = state.materialize()
        identity = _identity_mpo_factors_for_sites_and_mpo(sites, state.mpo)
        h_chain = BlockSparseEnvironmentChain.build(sites, state.mpo)
        n_chain = BlockSparseEnvironmentChain.build(sites, identity)
        h = state._parameter_chain_actions(sites, h_chain)
        n = state._parameter_chain_actions(sites, n_chain)
        anchor = state._pack_site(0)
        norm = float(np.real(np.vdot(anchor, n[0])))
        energy = float(np.real(np.vdot(anchor, h[0])) / norm)
        expected = []
        for site in range(state.nsites):
            _, support, _, _, _, _, info = state._frontier_wigner_eckart_matrix_free_problem(
                site, min(site,state.nsites-2), sites=sites,
                environment_sweeps=(h_chain,n_chain))
            reference = info['_projected_metric']
            direct = state._one_site_norm_components(
                site, n_chain.mpo_factors[site], n_chain.left_envs[site].data,
                n_chain.right_envs[site].data).restrict(support)
            np.testing.assert_allclose(direct.dense(), reference.dense(), rtol=1e-11, atol=1e-10)
            whitener, *_ = reference.whitener(rtol=1e-11)
            expected.append(np.linalg.norm(whitener.conj().T@(h[site]-energy*n[site])[support])/np.sqrt(norm))
        def forbidden(*args, **kwargs):
            raise AssertionError('closed audit constructed a pair Hamiltonian')
        monkeypatch.setattr(state, '_frontier_wigner_eckart_matrix_free_problem', forbidden)
        actual = state.local_tangent_residual()
        np.testing.assert_allclose(state.tangent_residuals, expected, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(actual, max(expected), rtol=1e-10, atol=1e-11)
    finally:
        state.close()


@pytest.mark.parametrize('stride', [1, 2, 3, 6])
@pytest.mark.parametrize('complex_state', [False, True])
def test_checkpoint_right_boundaries_match_full_stack(stride, complex_state):
    state = model(4, [(0,3)], 'general')
    try:
        if complex_state:
            rng = np.random.default_rng(146)
            for i in range(state.nsites):
                c = state._pack_site(i)
                state._set_site_vector(i, c + .03j*rng.normal(size=c.size))
        sites = state.materialize()
        reference = BlockSparseEnvironmentChain.build(sites, state.mpo)
        chain = BlockSparseEnvironmentChain.build(sites, state.mpo, boundaries_only=True)
        assert all(block is None for block in chain.right_envs[:-1])
        seen = []
        for i, block in chain.iter_right_boundaries(stride):
            seen.append(i)
            assert all(b is None for b in chain.right_envs[:i])
            assert sum(b is not None for b in chain.right_envs) <= (
                int(np.ceil(len(sites)/stride)) + stride - 1)
            assert block.data.keys() == reference.right_envs[i].data.keys()
            for key, value in block.data.items():
                expected = reference.right_envs[i].data[key]
                assert value.data.keys() == expected.data.keys()
                for channel, array in value.items():
                    np.testing.assert_array_equal(array, expected[channel])
        assert seen == list(range(len(sites)))
        assert not chain.right_envs
        chain = BlockSparseEnvironmentChain.build(sites, state.mpo, boundaries_only=True)
        iterator = chain.iter_right_boundaries(stride)
        next(iterator)
        iterator.close()
        assert not chain.right_envs
    finally:
        state.close()



def test_checkpoint_restore_does_not_contract_disposable_state(tmp_path, monkeypatch):
    state = model(4, [(0,3)], 'general')
    try:
        path = state.save_checkpoint(tmp_path/'state.chk')
        def forbidden(*args, **kwargs):
            raise AssertionError('checkpoint restore evaluated a disposable state')
        with monkeypatch.context() as patch:
            patch.setattr(type(state), 'expectation', forbidden)
            restored = type(state).load_checkpoint(path)
        try:
            assert restored.energy == state.energy
            for i in range(state.nsites):
                np.testing.assert_array_equal(restored._pack_site(i), state._pack_site(i))
            np.testing.assert_allclose(restored.expectation(), state.energy, rtol=1e-12, atol=1e-12)
        finally:
            restored.close()
    finally:
        state.close()


def test_checkpoint_without_energy_evaluates_restored_tensors_once(tmp_path, monkeypatch):
    import pickle
    state = model(4, [(0,3)], 'general')
    try:
        state._set_site_vector(1, state._pack_site(1)*1.09)
        path = state.save_checkpoint(tmp_path/'state.chk')
        with path.open('rb') as handle:
            payload = pickle.load(handle)
        payload.pop('energy')
        with path.open('wb') as handle:
            pickle.dump(payload, handle)
        expectation = type(state).expectation
        calls = []
        def checked(restored, *args, **kwargs):
            for i in range(state.nsites):
                np.testing.assert_array_equal(restored._pack_site(i), state._pack_site(i))
            calls.append(True)
            return expectation(restored, *args, **kwargs)
        monkeypatch.setattr(type(state), 'expectation', checked)
        restored = type(state).load_checkpoint(path)
        try:
            assert len(calls) == 1
            np.testing.assert_allclose(restored.energy, state.energy, rtol=1e-12, atol=1e-12)
        finally:
            restored.close()
    finally:
        state.close()


def test_channel_resolved_update_omits_unused_transition_plan(monkeypatch):
    from pyqed.letta import su2_qchem
    state = model(4, [(0,3)], 'general')
    try:
        def forbidden(*args, **kwargs):
            raise AssertionError('channel-resolved update built an unused transition plan')
        monkeypatch.setattr(su2_qchem.AdjacentPairTransitionPlan, 'build', forbidden)
        before = state.expectation()
        state.optimize_site(1, local_eigensolver='dense', max_local_parameters=512)
        assert state.expectation() <= before + 1e-10
    finally:
        state.close()


@pytest.mark.parametrize('complex_state', [False, True])
def test_projected_default_moving_environments_match_rebuilding(complex_state):
    states = [model(4, [(0,3)], 'general') for _ in range(2)]
    try:
        if complex_state:
            rng = np.random.default_rng(832)
            for i in range(states[0].nsites):
                c = states[0]._pack_site(i)
                c = c + .02j*rng.normal(size=c.size)
                for state in states:
                    state._set_site_vector(i, c)
        for state, options in zip(states, ({}, {'reuse_environments':False})):
            state.run(nsweeps=1, algorithm='projected', local_eigensolver='dense',
                      tol=0, **options)
        assert states[0].history[-1]['environment_reuse']
        assert not states[1].history[-1]['environment_reuse']
        np.testing.assert_allclose(states[0].expectation(), states[1].expectation(),
                                   rtol=0, atol=1e-10)
        np.testing.assert_allclose(states[0].local_tangent_residual(),
                                   states[1].local_tangent_residual(), rtol=1e-8, atol=1e-10)
    finally:
        for state in states:
            state.close()


@pytest.mark.parametrize('side', ['left', 'right'])
def test_boundary_zero_site_support_is_recomputed_after_reactivation(side, monkeypatch):
    from pyqed.mps.nonabelian import environment as env
    state = model(4, [(0,3)], 'general')
    try:
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo, boundaries_only=True)
        i = 0 if side == 'left' else state.nsites-1
        parent = chain.left_envs[0] if side == 'left' else chain.right_envs[-1]
        child = i+1 if side == 'left' else i-1
        owner = state._su2_moving_environment
        packed = parent.ensure_packed(side=side, bond=i)
        labels, revision = env._packed_boundary_labels(packed)
        owner.install_boundary(side, i, np.ascontiguousarray(packed.block_pool.data.real),
                               packed.block_pool.offsets, labels, revision, 1)
        array = next(iter(sites[i].data.values()))
        saved = array.copy()
        assert np.any(saved)
        convert = env._real64_contiguous_or_none
        def checked(value):
            if value is array:
                assert np.any(value), 'Zero site block reached numerical packing'
            return convert(value)
        sizes = []
        for step in range(2):
            array[...] = 0 if step == 0 else saved
            monkeypatch.setattr(env, '_real64_contiguous_or_none', convert)
            expected = parent.advance(state.mpo[i], sites[i], sites[i])
            monkeypatch.setattr(env, '_real64_contiguous_or_none', checked)
            data, packed, _, _ = env._contract_rank_coupled_boundary_cpp(
                state.mpo[i], sites[i], parent, sites[i], moving_environment=owner,
                side=side, parent_bond=i, child_bond=child, numeric_revision=step+2)
            sizes.append(int(packed.block_pool.offsets[-1]))
            for key in set(expected.data) | set(data):
                a, b = expected.data.get(key, {}), data.get(key, {})
                for channel in set(dict(a.items())) | set(dict(b.items())):
                    np.testing.assert_allclose(a.get(channel, 0), b.get(channel, 0),
                                               rtol=1e-11, atol=1e-10)
        assert sizes[1] > sizes[0]
    finally:
        state._su2_moving_environment.clear_boundaries()
        state.close()


@pytest.mark.parametrize('side', ['left', 'right'])
def test_reduced_core_cache_shares_virtual_spins_but_keeps_physical_charge(side):
    from pyqed.mps.nonabelian import environment as env
    from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
    state = model(4, [(0,3)], 'general')
    try:
        core = state.mpo[1]
        contract = (env._left_reduced_rank_coupled_block if side == 'left'
                    else env._right_reduced_rank_coupled_block)
        virtual = SpinChargeSector(2, SU2Irrep(0))
        shifted = SpinChargeSector(4, SU2Irrep(0))
        empty = SpinChargeSector(0, SU2Irrep(0))
        double = SpinChargeSector(2, SU2Irrep(0))
        core._environment_reduced_block_cache.clear()
        reference = contract(core, shifted, shifted, empty, empty, shifted, shifted)
        core._environment_reduced_block_cache.clear()
        first = contract(core, virtual, virtual, empty, empty, virtual, virtual)
        second = contract(core, shifted, shifted, empty, empty, shifted, shifted)
        assert first and reference
        assert second is first
        assert first.keys() == reference.keys()
        for key in first:
            np.testing.assert_array_equal(first[key], reference[key])
        occupied = contract(core, virtual, virtual, double, double, shifted, shifted)
        assert occupied is not first
        assert len(core._environment_reduced_block_cache) == 2
    finally:
        state.close()


@pytest.mark.parametrize('side', ['left', 'right'])
def test_boundary_skips_virtual_routes_for_absent_physical_support(side, monkeypatch):
    from pyqed.mps.nonabelian import environment as env
    state = model(4, [(0,3)], 'general')
    try:
        site = state.materialize_site(1)
        left, physical, right = next(iter(site.data))
        core = state.mpo[1]
        unsupported = set()
        for term in core.reduced_terms:
            rank = env._reduced_operator_rank(term.reduced_operator)
            blocks = [term.reduced_operator.component_block(m, physical, physical)
                      for m in env.ordered_two_m_values(rank)]
            if not any(b is not None and np.any(b) for b in blocks):
                unsupported.add(id(term.visible_virtual_block))
        assert unsupported
        routes = env.iter_virtual_routes
        def checked(block):
            assert id(block) not in unsupported
            return routes(block)
        monkeypatch.setattr(env, 'iter_virtual_routes', checked)
        core._environment_reduced_block_cache.clear()
        contract = (env._left_reduced_rank_coupled_block if side == 'left'
                    else env._right_reduced_rank_coupled_block)
        contract(core, left, left, physical, physical, right, right)
    finally:
        state.close()


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('complex_storage', [False, True])
def test_boundary_real_storage_converts_each_site_block_once_per_role(side, complex_storage, monkeypatch):
    from collections import Counter
    from pyqed.mps.nonabelian import environment as env
    state = model(4, [(0,3)], 'general')
    try:
        sites = state.materialize()
        if complex_storage:
            for site in sites:
                for key, array in tuple(site.data.items()):
                    site.data[key] = array.astype(np.complex128)
                for key in tuple(site.metadata):
                    if key.startswith('_rank_coupled_site_entries_by_'):
                        site.metadata.pop(key)
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo, boundaries_only=True)
        block = chain.left_envs[0] if side == 'left' else chain.right_envs[-1]
        owner = state._su2_moving_environment
        convert = env._real64_contiguous_or_none
        bond = 0 if side == 'left' else state.nsites-1
        packed = block.ensure_packed(side=side, bond=bond)
        labels, revision = env._packed_boundary_labels(packed)
        owner.install_boundary(side, bond, np.ascontiguousarray(packed.block_pool.data.real),
                               packed.block_pool.offsets, labels, revision, 1)
        calls = Counter()
        def counted(array):
            calls[id(array)] += 1
            return convert(array)
        monkeypatch.setattr(env, '_real64_contiguous_or_none', counted)
        indices = range(state.nsites-1) if side == 'left' else range(state.nsites-1,0,-1)
        for i in indices:
            expected = block.advance(state.mpo[i], sites[i], sites[i])
            calls.clear()
            child = i+1 if side == 'left' else i-1
            data, packed, topology, revision = env._contract_rank_coupled_boundary_cpp(
                state.mpo[i], sites[i], block, sites[i], moving_environment=owner,
                side=side, parent_bond=i, child_bond=child, numeric_revision=i+1)
            used = [calls[id(a)] for a in sites[i].data.values()]
            assert any(used)
            assert max(used) <= 2  # One conversion per bra/ket role, not per route.
            for key in set(expected.data) | set(data):
                a, b = expected.data.get(key, {}), data.get(key, {})
                for channel in set(dict(a.items())) | set(dict(b.items())):
                    np.testing.assert_allclose(a.get(channel, 0), b.get(channel, 0),
                                               rtol=1e-11, atol=1e-10)
            block = (env.LeftBlock if side == 'left' else env.RightBlock)(
                data, rank_coupled=True, packed_table=packed, cpp_owned_boundary=True,
                cpp_topology_revision=topology, cpp_numeric_revision=revision)
    finally:
        state._su2_moving_environment.clear_boundaries()
        state.close()


@pytest.mark.parametrize('side', ['left', 'right'])
def test_direct_boundary_prunes_forbidden_spin_outputs_before_allocation(side, monkeypatch):
    from pyqed.mps.nonabelian import environment as env
    state = model(4, [(0,3)], 'general')
    try:
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites,state.mpo,boundaries_only=True)
        block = chain.left_envs[0] if side == 'left' else chain.right_envs[-1]
        owner = state._su2_moving_environment
        build = env._empty_packed_boundary_from_specs
        checked = []
        def guarded(specs, **kwargs):
            for (bra,ket,_channel),shape in specs.items():
                j,k,rank = bra.irrep.two_j,ket.irrep.two_j,shape[0]-1
                assert abs(j-k) <= rank <= j+k
                assert (j+k+rank) % 2 == 0
            checked.append(True)
            return build(specs, **kwargs)
        monkeypatch.setattr(env,'_empty_packed_boundary_from_specs',guarded)
        indices = range(state.nsites-1) if side == 'left' else range(state.nsites-1,0,-1)
        for i in indices:
            expected = block.advance(state.mpo[i],sites[i],sites[i])
            child = i+1 if side == 'left' else i-1
            data,packed,topology,revision = env._contract_normal_complementary_boundary_cpp(
                state.mpo[i],sites[i],block,sites[i],moving_environment=owner,
                side=side,parent_bond=i,child_bond=child,numeric_revision=i+1)
            for key in set(expected.data)|set(data):
                a,b = expected.data.get(key,{}),data.get(key,{})
                for channel in set(dict(a.items()))|set(dict(b.items())):
                    np.testing.assert_allclose(a.get(channel,0),b.get(channel,0),
                                               rtol=1e-11,atol=1e-10)
            block = (env.LeftBlock if side == 'left' else env.RightBlock)(
                data,rank_coupled=True,packed_table=packed,cpp_owned_boundary=True,
                cpp_topology_revision=topology,cpp_numeric_revision=revision)
        assert len(checked) == state.nsites-1
    finally:
        state._su2_moving_environment.clear_boundaries()
        state.close()


@pytest.mark.parametrize('disk', [False, True])
def test_sparse_stationarity_matches_dense_audit(disk, tmp_path):
    state = model(4, [(0, 3)], 'general')
    try:
        expected = state.local_tangent_residual()
        residuals = state.tangent_residuals
        coefficients = [state._pack_site(i).copy() for i in range(state.nsites)]
        actual = state.local_tangent_residual(
            sparse_hamiltonian=True, spill_directory=tmp_path if disk else None)
        assert all(not core._environment_reduced_block_cache for core in state.mpo)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(state.tangent_residuals, residuals, rtol=1e-10, atol=1e-11)
        for i, vector in enumerate(coefficients):
            np.testing.assert_array_equal(state._pack_site(i), vector)
        assert not list(tmp_path.iterdir())
    finally:
        state.close()


def test_sparse_audit_releases_recoupling_on_failure(monkeypatch, tmp_path):
    state = model(4, [(0, 3)], 'general')
    action = state._one_site_parameter_adjoint
    def interrupted(*args, **kwargs):
        action(*args, **kwargs)
        raise RuntimeError('interrupted audit')
    monkeypatch.setattr(state, '_one_site_parameter_adjoint', interrupted)
    try:
        with pytest.raises(RuntimeError, match='interrupted audit'):
            state.local_tangent_residual(sparse_hamiltonian=True, spill_directory=tmp_path)
        assert all(not core._environment_reduced_block_cache for core in state.mpo)
        assert not list(tmp_path.iterdir())
    finally:
        state.close()


def test_tied_adjoint_consumes_block_iterator_without_collecting():
    import weakref
    state = model(4, [(0, 3)], 'general')
    try:
        plan = state._wigner_eckart_route_plan(1)
        tensor = plan.tensor(np.arange(plan.nparameters, dtype=float))
        expected = plan.adjoint(tensor.data)
        refs = []
        def blocks():
            for key, template in tensor.data.items():
                # The consumer may still own its immediately preceding block,
                # but must not retain the full expanded output.
                assert all(ref() is None for ref in refs[:-1])
                block = template.copy()
                refs.append(weakref.ref(block))
                yield key, block
        np.testing.assert_array_equal(plan.adjoint(blocks()), expected)
        assert all(ref() is None for ref in refs)
    finally:
        state.close()


def test_stationarity_releases_each_metric_before_building_next(monkeypatch):
    import weakref
    state = model(4, [(0, 3)], 'general')
    previous = []
    build = state._one_site_norm_components
    def observed(*args, **kwargs):
        assert all(ref() is None for ref in previous)
        metric = build(*args, **kwargs)
        previous.append(weakref.ref(metric))
        return metric
    monkeypatch.setattr(state, '_one_site_norm_components', observed)
    try:
        residual = state.local_tangent_residual()
        assert np.isfinite(residual)
        assert len(previous) == state.nsites
        assert all(ref() is None for ref in previous)
    finally:
        state.close()
