import numpy as np
import pytest

from test_su2_letta_closure import model


@pytest.mark.parametrize('fail_audit', [False, True])
@pytest.mark.parametrize('verify', [False, True])
@pytest.mark.parametrize('accurate', [False, True])
def test_cycle_convergence_requires_fresh_final_state_audit(tmp_path, monkeypatch, fail_audit, verify, accurate):
    state = model(4, [(0, 3)], 'general')
    calls = []
    audit = state.local_tangent_residual
    def checked(**kwargs):
        calls.append(1)
        assert kwargs['sparse_hamiltonian'] and kwargs['support_rtol'] == 0.
        actual = audit(**kwargs)
        return 1. if fail_audit else actual
    monkeypatch.setattr(state, 'local_tangent_residual', checked)
    try:
        result = state.run_parameter_blocks(spill_directory=tmp_path, nsweeps=3,
            tol=1e-8, residual_tol=1e-8, checkpoint=tmp_path/'cycle.chk', verify=verify,
            compensated=accurate, factored_norm=accurate)
        assert result is state and calls
        assert all(row['factored_norm'] == accurate and row['compensated'] == accurate
                   for cycle in state.history for row in cycle['updates'])
        assert state.converged == state.success == (not fail_audit)
        np.testing.assert_allclose(state.energy, state.expectation(), atol=1e-9)
        assert state.history[-1]['converged'] == (not fail_audit)
        if fail_audit:
            assert state.history[-1]['max_tangent_residual'] == 1.
        else:
            assert state.history[-1]['max_tangent_residual'] <= 1e-8
        assert (tmp_path/'cycle.chk').exists()
    finally:
        state.close()


def test_direction_change_reuses_updated_boundaries(tmp_path, monkeypatch):
    from pyqed.mps.nonabelian.sparse_boundary import SparseBoundary
    state = model(4, [(0, 3)], 'general')
    cache, calls = {}, []
    advance = SparseBoundary.advance
    def counted(*args, **kwargs):
        calls.append(1)
        return advance(*args, **kwargs)
    monkeypatch.setattr(SparseBoundary, 'advance', counted)
    try:
        first = state.sweep_parameter_blocks(spill_directory=tmp_path, _checkpoint_cache=cache)
        assert len(calls) == 2*(state.nsites-1)
        second = state.sweep_parameter_blocks(spill_directory=tmp_path, direction='rl', _checkpoint_cache=cache)
        assert len(calls) == 3*(state.nsites-1)
        assert not any(r['boundary_checkpoints_reused'] for r in first)
        assert all(r['boundary_checkpoints_reused'] for r in second)
        np.testing.assert_allclose(state.energy, state.expectation(), atol=1e-9)
        vector = state._pack_site(0)
        state._set_site_vector(0, vector*1.001)
        third = state.sweep_parameter_blocks(spill_directory=tmp_path, _checkpoint_cache=cache)
        assert not any(r['boundary_checkpoints_reused'] for r in third)
        assert len(calls) == 5*(state.nsites-1)
    finally:
        cache.clear()
        state.close()


@pytest.mark.parametrize('stride', [1, 2])
@pytest.mark.parametrize('direction', ['lr', 'rl'])
def test_parameter_sweep_updates_closed_state_with_fresh_energy(stride, direction, tmp_path, monkeypatch):
    state = model(4, [(0, 3), (1, 2)], 'general')
    try:
        before = state.expectation()
        original = [state._pack_site(i).copy() for i in range(state.nsites)]
        def forbidden(*args, **kwargs):
            raise AssertionError('Used adjacent-pair local solver')
        monkeypatch.setattr(state, '_frontier_wigner_eckart_matrix_free_problem', forbidden)
        updates = state.sweep_parameter_blocks(spill_directory=tmp_path,
                                               checkpoint_stride=stride, direction=direction, tolerance=1e-9)
        order = range(state.nsites) if direction == 'lr' else range(state.nsites-1, -1, -1)
        assert [r['site'] for r in updates] == list(order)
        assert any(r['accepted'] for r in updates)
        assert all(r['independently_verified'] for r in updates)
        for r in updates:
            if r['accepted']:
                assert r['residual'] <= 1e-9 and r['energy_after'] <= r['energy_before']+1e-10
            else:
                np.testing.assert_array_equal(state._pack_site(r['site']), original[r['site']])
        assert any(not np.array_equal(state._pack_site(i), old) for i, old in enumerate(original))
        fresh = state.expectation()
        assert fresh < before-1e-6
        np.testing.assert_allclose(state.energy, fresh, rtol=0, atol=1e-9)
        for left, right in zip(updates, updates[1:]):
            np.testing.assert_allclose(left['energy_after'], right['energy_before'], atol=1e-9)
        assert not state.converged and not state.success
    finally:
        state.close()


def test_parameter_sweep_rolls_back_partial_pass(tmp_path):
    state = model(4, [(0, 3)], 'general')
    try:
        original = [state._pack_site(i).copy() for i in range(state.nsites)]
        energy = state.energy
        state.converged = state.success = True
        state.message = 'Original status'
        def interrupt(report):
            if report['site'] == 1:
                assert not state.converged and not state.success
                raise RuntimeError('Injected interrupted pass')
        with pytest.raises(RuntimeError, match='Injected'):
            state.sweep_parameter_blocks(spill_directory=tmp_path, callback=interrupt, tolerance=1e-9)
        for i, vector in enumerate(original):
            np.testing.assert_array_equal(state._pack_site(i), vector)
        assert state.energy == energy
        assert state.converged and state.success and state.message == 'Original status'
    finally:
        state.close()


@pytest.mark.parametrize('failed_residual', [1., np.inf, np.nan])
def test_parameter_sweep_rejects_unconverged_candidates(tmp_path, monkeypatch, failed_residual):
    from pyqed.letta import parameter_sweep
    original_solve = parameter_sweep.solve_parameter_operator
    def unconverged(*args, **kwargs):
        candidate, transform, report = original_solve(*args, **kwargs)
        report['operator_residual'] = failed_residual
        return candidate, transform, report
    monkeypatch.setattr(parameter_sweep, 'solve_parameter_operator', unconverged)
    state = model(4, [(0, 3)], 'general')
    def forbidden(*args, **kwargs):
        raise AssertionError('Independent action evaluated for an ineligible candidate')
    monkeypatch.setattr(state, '_one_site_parameter_adjoint', forbidden)
    try:
        original = [state._pack_site(i).copy() for i in range(state.nsites)]
        updates = state.sweep_parameter_blocks(spill_directory=tmp_path, accept_verified_descent=True)
        assert not any(r['accepted'] for r in updates)
        assert not any(r['independently_verified'] for r in updates)
        assert all(r['verification_skipped'] == 'operator_residual' for r in updates)
        for i, vector in enumerate(original):
            np.testing.assert_array_equal(state._pack_site(i), vector)
    finally:
        state.close()


@pytest.mark.parametrize('factored_norm', [False, True])
def test_relaxed_sweep_checks_roots_and_updated_energy(tmp_path, factored_norm):
    state = model(4, [(0, 3), (1, 2)], 'general')
    try:
        before = state.expectation()
        updates = state.sweep_parameter_blocks(spill_directory=tmp_path,
            relaxation=1.8, tolerance=1e-9, factored_norm=factored_norm)
        relaxed = [r for r in updates if r['relaxation_used'] > 1]
        assert relaxed
        for row in updates:
            if row['accepted']:
                assert row['root_residual'] <= 1e-9
                assert row['root_operator_residual'] <= 1e-9
                assert np.isfinite(row['residual'])
                assert row['energy_after'] <= row['energy_before']+1e-10
        assert state.energy < before
        np.testing.assert_allclose(state.energy, state.expectation(), atol=1e-9)
        assert not state.converged
    finally:
        state.close()


def test_parameter_sweep_releases_recoupled_core_caches(tmp_path):
    state = model(4, [(0, 3)], 'general')
    try:
        state.sweep_parameter_blocks(spill_directory=tmp_path)
        assert all(not core._environment_reduced_block_cache for core in state.mpo)
        np.testing.assert_allclose(state.energy, state.expectation(), atol=1e-9)
        def interrupted(row):
            raise RuntimeError('Stop pass')
        with pytest.raises(RuntimeError, match='Stop pass'):
            state.sweep_parameter_blocks(spill_directory=tmp_path, callback=interrupted)
        assert all(not core._environment_reduced_block_cache for core in state.mpo)
    finally:
        state.close()


def test_verified_descent_is_distinct_from_local_stationarity(tmp_path, monkeypatch):
    from pyqed.letta import parameter_sweep
    state = model(4, [(0, 3)], 'general')
    solve = parameter_sweep.solve_parameter_operator
    adjoint = state._one_site_parameter_adjoint
    captured = {}
    def observed_solve(*args, **kwargs):
        candidate, transform, report = solve(*args, **kwargs)
        captured['candidate'] = candidate
        return candidate, transform, report
    def perturbed_adjoint(site, bra, ket, core, left, right):
        value = adjoint(site, bra, ket, core, left, right)
        if core is state.mpo[site]:
            c = captured['candidate']
            perturbation = np.ones_like(c)
            perturbation -= c*np.vdot(c, perturbation)/np.vdot(c, c)
            value = value+1e-5*perturbation
        return value
    monkeypatch.setattr(parameter_sweep, 'solve_parameter_operator', observed_solve)
    monkeypatch.setattr(state, '_one_site_parameter_adjoint', perturbed_adjoint)
    try:
        strict = state.sweep_parameter_blocks(spill_directory=tmp_path)
        assert any(not r['root_converged'] for r in strict)
        assert all(not r['accepted'] for r in strict if not r['root_converged'])
        rows = state.sweep_parameter_blocks(spill_directory=tmp_path, accept_verified_descent=True)
        inexact = [r for r in rows if r['accepted_by_descent']]
        assert inexact
        assert all(r['accepted'] and not r['root_converged'] for r in inexact)
        assert all(r['energy_after'] < r['energy_before']-1e-10 for r in inexact)
        monkeypatch.setattr(state, '_one_site_parameter_adjoint', adjoint)
        np.testing.assert_allclose(state.energy, state.expectation(), atol=1e-9)
        assert not state.converged
        with pytest.raises(ValueError, match='independent'):
            state.sweep_parameter_blocks(spill_directory=tmp_path, verify=False, accept_verified_descent=True)
    finally:
        state.close()


def test_final_audit_reuses_only_current_right_boundaries(tmp_path, monkeypatch):
    from pyqed.mps.nonabelian.sparse_boundary import SparseBoundary
    state = model(4, [(0, 3)], 'general')
    cache, calls = {}, []
    advance = SparseBoundary.advance
    def counted(*args, **kwargs):
        calls.append(1)
        return advance(*args, **kwargs)
    monkeypatch.setattr(SparseBoundary, 'advance', counted)
    def audit(cache=None):
        calls.clear()
        value = state.local_tangent_residual(sparse_hamiltonian=True,
            spill_directory=tmp_path, checkpoint_stride=1, _checkpoint_cache=cache)
        return value, np.array(state.tangent_residuals), len(calls)
    try:
        state.sweep_parameter_blocks(spill_directory=tmp_path, _checkpoint_cache=cache)
        # A forward pass retains left boundaries, which cannot supply this audit.
        assert audit(cache)[2] == 2*(state.nsites-1)
        state.sweep_parameter_blocks(spill_directory=tmp_path, direction='rl', _checkpoint_cache=cache)
        reused = audit(cache)
        fresh = audit()
        assert reused[2] == state.nsites-1
        assert fresh[2] == 2*(state.nsites-1)
        np.testing.assert_allclose(reused[1], fresh[1], atol=1e-12, rtol=1e-10)
        assert len(cache['blocks']) == state.nsites
        # An intervening tensor update invalidates the cache, even if tiny.
        state._set_site_vector(0, state._pack_site(0)*1.001)
        assert audit(cache)[2] == 2*(state.nsites-1)
    finally:
        cache.clear()
        state.close()


def test_requested_cycle_audit_does_not_bypass_energy_stability(tmp_path, monkeypatch):
    state = model(4, [(0, 3)], 'general')
    calls = []
    def audit(**kwargs):
        assert kwargs['_checkpoint_cache']['signature'][-1] == 'lr'
        calls.append(1)
        return 0.
    monkeypatch.setattr(state, 'local_tangent_residual', audit)
    try:
        state.run_parameter_blocks(spill_directory=tmp_path, nsweeps=1,
            tol=1e-30, audit_each_cycle=True)
        assert calls == [1]
        assert abs(state.history[-1]['delta_energy']) > 1e-30
        assert state.history[-1]['max_tangent_residual'] == 0.
        assert not state.converged and not state.success
    finally:
        state.close()
