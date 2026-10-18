import numpy as np
import pytest

from test_su2_letta_closure import model
from pyqed.letta.mpo_parameter_operator import MPOParameterOperator
from pyqed.letta.parameter_operator import TiedParameterOperator, iter_parameter_blocks, solve_parameter_operator
from pyqed.mps.nonabelian import environment as env
from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo


@pytest.mark.parametrize('cache_bytes', [1, 32*2**20])
@pytest.mark.parametrize('complex_vector', [False, True])
def test_prepared_mpo_actions_match_independent_and_assembled(tmp_path, cache_bytes, complex_vector):
    state = model(4, [(0, 3)], 'general')
    rng = np.random.default_rng(92)
    try:
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo)
        for i in range(state.nsites):
            plan = state._wigner_eckart_route_plan(i)
            left, right = chain.left_envs[i].data, chain.right_envs[i].data
            with MPOParameterOperator(plan, state.mpo[i], left, right,
                    storage_bytes=cache_bytes) as direct, TiedParameterOperator(
                    plan.nparameters, iter_parameter_blocks(plan, state.mpo[i], left, right)) as assembled:
                for _ in range(2):
                    v = rng.normal(size=plan.nparameters)
                    if complex_vector:
                        v = v+1j*rng.normal(size=v.size)
                    reference = state._one_site_parameter_adjoint(i, sites[i], plan.tensor(v), state.mpo[i], left, right)
                    np.testing.assert_allclose(direct.matvec(v), reference, atol=1e-11, rtol=1e-11)
                    np.testing.assert_allclose(direct.matvec(v), assembled.matvec(v), atol=1e-11, rtol=1e-11)
                    assert direct.resident_bytes <= cache_bytes
                    assert direct.disk_bytes == 0
                    assert not state.mpo[i]._environment_reduced_block_cache
                if cache_bytes > 1:
                    assert direct.cache_hits > 0
            with pytest.raises(RuntimeError, match='closed'):
                direct.matvec(v)
    finally:
        state.close()


def test_prepared_mpo_davidson_matches_assembled(tmp_path):
    state = model(4, [(0, 3)], 'general')
    try:
        sites = state.materialize(); i = 1
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo)
        identity = _identity_mpo_factors_for_sites_and_mpo(sites, state.mpo)
        norms = env.BlockSparseEnvironmentChain.build(sites, identity)
        metric = state._one_site_norm_components(i, identity[i], norms.left_envs[i].data, norms.right_envs[i].data)
        plan = state._wigner_eckart_route_plan(i)
        left, right = chain.left_envs[i].data, chain.right_envs[i].data
        with MPOParameterOperator(plan, state.mpo[i], left, right) as direct, TiedParameterOperator(
                plan.nparameters, iter_parameter_blocks(plan, state.mpo[i], left, right)) as assembled:
            _, _, a = solve_parameter_operator(direct, metric, state._pack_site(i), preconditioner='constant')
            _, _, b = solve_parameter_operator(assembled, metric, state._pack_site(i), preconditioner='constant')
            np.testing.assert_allclose(a['operator_energy'], b['operator_energy'], atol=1e-10)
            assert a['operator_residual'] <= 1e-10
    finally:
        state.close()


def test_prepared_mpo_sweep_never_assembles_parameter_matrix(tmp_path, monkeypatch):
    from pyqed.letta import parameter_sweep
    state = model(4, [(0, 3)], 'general')
    def forbidden(*args, **kwargs):
        raise AssertionError('Assembled the parameter-space Hamiltonian')
    monkeypatch.setattr(parameter_sweep, 'iter_parameter_blocks', forbidden)
    try:
        before = state.expectation()
        rows = state.sweep_parameter_blocks(spill_directory=tmp_path,
            operator_mode='mpo', factored_norm=True, tolerance=1e-9)
        assert all(r['operator_mode'] == 'mpo' for r in rows)
        assert all(r['accepted'] and r['independently_verified'] for r in rows)
        assert state.energy < before
        np.testing.assert_allclose(state.energy, state.expectation(), atol=1e-10)
    finally:
        state.close()


def test_prepared_mpo_constructor_does_not_expand_factors(monkeypatch):
    state = model(4, [(0, 3)], 'general')
    try:
        plan = state._wigner_eckart_route_plan(1)
        def forbidden(*args, **kwargs):
            raise AssertionError('Expanded factors during construction')
        monkeypatch.setattr(env, '_left_reduced_rank_coupled_block', forbidden)
        with MPOParameterOperator(plan, state.mpo[1], {}, {}, storage_bytes=1) as operator:
            assert operator.resident_bytes == operator.disk_bytes == 0
            assert operator.factor_builds == 0
    finally:
        state.close()


def test_prepared_mpo_active_factor_budget():
    state = model(4, [(0, 3)], 'general')
    try:
        sites = state.materialize()
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo)
        plan = state._wigner_eckart_route_plan(1)
        with MPOParameterOperator(plan, state.mpo[1], chain.left_envs[1].data,
                chain.right_envs[1].data, factor_bytes=1) as operator:
            with pytest.raises(MemoryError, match='factor_bytes'):
                operator.matvec(state._pack_site(1))
            assert not state.mpo[1]._environment_reduced_block_cache
    finally:
        state.close()
