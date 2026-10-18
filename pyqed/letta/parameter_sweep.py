"""Moving reduced environments for disk-backed tied-parameter updates.

Uses the reduced contractions of Weichselbaum, PRB 86, 245124 (2012),
doi:10.1103/PhysRevB.86.245124, adapted to LETTA ties and scalar closed seams.
This experimental coordinate-descent pass preserves the incumbent in
each retained metric space. It is not a canonical PBC solver or a certificate
of full-space/global convergence. Hamiltonian boundaries currently require
real data; norm environments retain the existing reduced representation.
"""
import time

import numpy as np

from .parameter_operator import TiedParameterOperator, iter_parameter_blocks, solve_parameter_operator, _positive_integer
from .parameter_operator import relax_parameter_candidate
from pyqed.mps.nonabelian import environment as env
from pyqed.mps.nonabelian.sparse_boundary import SparseBoundary, iter_sweep_boundaries
from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo


def run_parameter_blocks(state, *, spill_directory, nsweeps=2, tol=1e-9,
                         residual_tol=1e-8, consecutive_cycles=1, checkpoint=None,
                         reset_history=True, audit_max_parameters=4096, audit_each_cycle=False, **options):
    """Run complete LR/RL cycles with updated-boundary reuse and final audits.

    Convergence means energy stability and fresh retained-metric one-site
    stationarity, not a global optimum. The final audit uses the independent
    reduced adjoint path. See the module reference for method fidelity.
    Stride 1 retains O(n) disk-backed checkpoints across direction changes;
    larger strides recompute segments independently on each pass. Final audits
    reuse current right Hamiltonian boundaries from the reverse pass, while
    rebuilding left boundaries, local adjoint actions, and norm metrics.
    ``audit_each_cycle`` measures stationarity even before energy stabilizes;
    it does not relax the energy or local-update convergence requirements.
    """
    nsweeps = _positive_integer(nsweeps, 'nsweeps')
    consecutive_cycles = _positive_integer(consecutive_cycles, 'consecutive_cycles')
    audit_max_parameters = _positive_integer(audit_max_parameters, 'audit_max_parameters')
    if not all(np.isfinite(x) and x > 0 for x in (tol, residual_tol)):
        raise ValueError('Convergence tolerances must be finite and positive')
    if 'direction' in options or '_checkpoint_cache' in options:
        raise ValueError('The cycle driver controls traversal and boundary reuse')
    cache = {} if options.get('checkpoint_stride', 1) == 1 else None
    if reset_history:
        state.history = []
    state.converged = state.success = False
    streak = 0
    try:
        for _ in range(nsweeps):
            started = time.perf_counter()
            updates = []
            for direction in ('lr', 'rl'):
                updates.extend(sweep_parameter_blocks(state, spill_directory=spill_directory,
                    direction=direction, _checkpoint_cache=cache, **options))
            delta = state.energy-(updates[0]['energy_before']+state.ecore)
            residual = None
            audit_seconds = 0.
            stable = abs(delta) <= tol and all(row['accepted'] for row in updates)
            if audit_each_cycle or stable:
                sparse = state.virtual_boundary == 'periodic'
                audit_started = time.perf_counter()
                residual = state.local_tangent_residual(
                    metric_rtol=options.get('metric_rtol', 1e-11), support_rtol=0.,
                    max_parameters=audit_max_parameters, sparse_hamiltonian=sparse,
                    spill_directory=spill_directory if sparse else None,
                    checkpoint_stride=options.get('checkpoint_stride', 1) if sparse else None,
                    _checkpoint_cache=cache)
                audit_seconds = time.perf_counter()-audit_started
            streak = streak+1 if stable and residual is not None and residual <= residual_tol else 0
            state.converged = state.success = streak >= consecutive_cycles
            state.history.append(dict(sweep=len(state.history)+1, algorithm='parameter_blocks',
                energy=state.energy, delta_energy=delta, updates=updates,
                max_tangent_residual=residual, audit_elapsed_s=audit_seconds,
                converged=state.converged,
                elapsed_s=time.perf_counter()-started))
            state.message = ('Converged retained-metric stationarity after complete LR/RL cycles'
                             if state.converged else 'Completed parameter-block cycle; convergence not established')
            if checkpoint is not None:
                state.save_checkpoint(checkpoint)
            if state.converged:
                break
    finally:
        if cache is not None:
            cache.clear()
    return state


def sweep_parameter_blocks(state, *, spill_directory, tolerance=1e-10,
                           metric_rtol=1e-11, maxiter=300, max_space=96,
                           block_bytes=2**20, storage_bytes=32*2**20,
                           checkpoint_stride=1, direction='lr', verify=True, callback=None,
                           relaxation=1., accept_verified_descent=False,
                           compensated=False, factored_norm=False, operator_mode='assembled',
                           _checkpoint_cache=None):
    """Update sites in either direction and return per-site diagnostics.

    Opposite-side checkpoints are consumed while the updated side advances once.
    ``verify=True`` independently checks eligible candidates with reduced adjoints.
    Candidates failing the stored-operator residual are rejected immediately;
    their residual diagnostics remain unverified stored-operator values.
    Numerical budgets apply per operator/boundary, not to total process memory.
    An exception rolls back this pass; failed local residual/variational checks
    leave that site unchanged. The caller owns sweep history and checkpoints.
    ``1 < relaxation < 2`` enables energy-guarded overrelaxation inspired by
    Oseledets et al., doi:10.1002/nla.2459; see ``relax_parameter_candidate``
    for the changes and missing convergence guarantees. Independent validation
    checks both the local root and any extrapolated update. An extrapolated
    update is not itself a stationary local root.
    ``accept_verified_descent=True`` can accept a stored-operator root whose
    independent residual misses tolerance if its independently checked energy
    drops by more than the numerical acceptance floor. Such an update is
    reported as nonstationary; final-state convergence criteria are unchanged.
    Experimental ``compensated`` assembly and ``factored_norm`` actions improve
    local contraction accuracy at added cost; neither changes acceptance or
    final convergence tolerances. Compensation requires real compiled terms.
    ``operator_mode='mpo'`` prepares factored MPO/right-environment contractions
    instead of local matrix blocks and uses constant-preconditioned Davidson.
    It supports scalar physical multiplicities and retains independent checks;
    see ``MPOParameterOperator`` for storage and precision limitations.
    """
    if operator_mode not in {'assembled', 'mpo'}:
        raise ValueError('operator_mode must be assembled or mpo')
    if direction not in {'lr', 'rl'}:
        raise ValueError('direction must be lr or rl')
    if not np.isfinite(relaxation) or not 1 <= relaxation < 2:
        raise ValueError('relaxation must lie in [1, 2)')
    if accept_verified_descent and not verify:
        raise ValueError('Verified descent requires independent per-update checks')
    if _checkpoint_cache is not None and checkpoint_stride != 1:
        raise ValueError('Cross-pass checkpoint reuse requires stride 1')
    originals = [state._pack_site(i).copy() for i in range(state.nsites)]
    old_energy = state.energy
    old_status = state.converged, state.success, state.message
    sites = state.materialize()
    chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo, boundaries_only=True)
    identity = _identity_mpo_factors_for_sites_and_mpo(sites, state.mpo)
    norms = env.BlockSparseEnvironmentChain.build(sites, identity, boundaries_only=True)
    forward = direction == 'lr'
    moving_side, frozen_side = ('left', 'right') if forward else ('right', 'left')
    moving_bond, frozen_bond = (0, state.nsites-1) if forward else (state.nsites-1, 0)
    moving = SparseBoundary.from_reduced(getattr(chain, moving_side+'_envs')[moving_bond],
                                        side=moving_side, bond=moving_bond)
    endpoint = SparseBoundary.from_reduced(getattr(chain, frozen_side+'_envs')[frozen_bond],
                                          side=frozen_side, bond=frozen_bond)
    boundary_peaks = dict(route_count=0, route_pool_bytes=0, maximum_gather_bytes=0,
                          maximum_output_bytes=0, output_block_count=0,
                          reduced_cache_array_bytes=0)
    def advance_h(parent, i):
        core = chain.mpo_factors[i]
        try:
            child = parent.advance(core, sites[i],
                moving_environment=state._su2_moving_environment, spill_directory=spill_directory,
                release_recoupling=True)
        finally:
            core._environment_reduced_block_cache.clear()
        for key in boundary_peaks:
            boundary_peaks[key] = max(boundary_peaks[key], parent.last_advance.get(key, 0))
        parent.release_decoded()
        return child
    def advance_n(parent, i):
        return parent.advance(identity[i], sites[i], sites[i])
    suffixes = iter_sweep_boundaries(state.nsites, endpoint, advance_h,
                                     stride=checkpoint_stride, direction=direction)
    norm_moving = getattr(norms, moving_side+'_envs')[moving_bond]
    norm_suffixes = iter_sweep_boundaries(state.nsites,
        getattr(norms, frozen_side+'_envs')[frozen_bond], advance_n,
        stride=checkpoint_stride, direction=direction)
    retained, collected = {}, {}
    reused = False
    if _checkpoint_cache is not None:
        signature = (id(state), state._state_revision, tuple(map(id, state.mpo)), direction)
        if _checkpoint_cache.get('signature') == signature:
            retained = _checkpoint_cache.pop('blocks')
            order = range(state.nsites) if forward else range(state.nsites-1, -1, -1)
            suffixes = ((i, retained[i][0]) for i in order)
            norm_suffixes = ((i, retained.pop(i)[1]) for i in order)
            reused = True
        _checkpoint_cache.clear()
        collected[moving_bond] = moving, norm_moving
    updates = []
    previous = None
    try:
        for (site, frozen), (norm_site, norm_frozen) in zip(suffixes, norm_suffixes, strict=True):
            if site != norm_site:
                raise RuntimeError('Hamiltonian and norm sweep sites differ')
            left, right = (moving, frozen) if forward else (frozen, moving)
            norm_left, norm_right = (norm_moving, norm_frozen) if forward else (norm_frozen, norm_moving)
            started = time.perf_counter()
            core = chain.mpo_factors[site]
            plan = state._wigner_eckart_route_plan(site)
            metric = state._one_site_norm_components(site, identity[site], norm_left.data, norm_right.data)
            def apply_norm(vector):
                return state._one_site_parameter_adjoint(
                    site, sites[site], plan.tensor(vector), identity[site],
                    norm_left.data, norm_right.data)
            norm_matvec = apply_norm if factored_norm else None
            trial_metric = metric
            if factored_norm:
                from scipy.sparse.linalg import LinearOperator
                trial_metric = LinearOperator((plan.nparameters, plan.nparameters),
                                              matvec=apply_norm, dtype=complex)
            if operator_mode == 'mpo':
                from .mpo_parameter_operator import MPOParameterOperator
                local_operator = MPOParameterOperator(plan, core, left, right,
                    storage_bytes=storage_bytes)
            else:
                local_operator = TiedParameterOperator(plan.nparameters,
                    iter_parameter_blocks(plan, core, left, right, block_bytes=block_bytes,
                                          compensated=compensated),
                    storage_bytes=storage_bytes, spill_directory=spill_directory)
            with local_operator as operator:
                build_seconds = time.perf_counter()-started
                candidate, transform, report = solve_parameter_operator(
                    operator, metric, state._pack_site(site), metric_rtol=metric_rtol,
                    tolerance=tolerance, maxiter=maxiter, max_space=max_space,
                    norm_matvec=norm_matvec,
                    preconditioner='constant' if operator_mode == 'mpo' else 'diagonal')
                report.update(operator_resident_bytes=operator.resident_bytes,
                              operator_disk_bytes=operator.disk_bytes, build_seconds=build_seconds,
                              compensated=bool(compensated or operator_mode == 'mpo'),
                              operator_mode=operator_mode)
                trial, weight = None, 1.
                if relaxation > 1 and report['operator_residual'] <= tolerance:
                    trial, weight = relax_parameter_candidate(operator, trial_metric,
                        state._pack_site(site), candidate, relaxation=relaxation,
                        energy_before=report['energy_before'], root_energy=report['operator_energy'])
                    if weight > 1:
                        tn, th = trial_metric.matvec(trial), operator.matvec(trial)
                        trial_norm = float(np.vdot(trial, tn).real)
                        trial_operator_energy = float(np.vdot(trial, th).real/trial_norm)
                        trial_operator_residual = float(np.linalg.norm(
                            transform.conj().T@(th-trial_operator_energy*tn))/np.sqrt(trial_norm))
            before, energy = report['energy_before'], report['operator_energy']
            residual = report['operator_residual']
            if previous is not None and abs(before-previous) > 1e-8:
                raise ArithmeticError('Moving local energy is inconsistent with the preceding update')
            independently_verified = bool(verify and np.isfinite(residual) and residual <= tolerance)
            if independently_verified:
                tensor = plan.tensor(candidate)
                h = state._one_site_parameter_adjoint(site, sites[site], tensor, core, left, right)
                n = state._one_site_parameter_adjoint(site, sites[site], tensor,
                                                     identity[site], norm_left.data, norm_right.data)
                norm = float(np.vdot(candidate, n).real)
                if not np.isfinite(norm) or norm <= 0:
                    raise ArithmeticError('Invalid independently checked candidate norm')
                energy = float(np.vdot(candidate, h).real/norm)
                residual = float(np.linalg.norm(transform.conj().T@(h-energy*n))/np.sqrt(norm))
                del tensor, h, n
            root_converged = bool(residual <= tolerance and report['operator_residual'] <= tolerance)
            descent_floor = max(1e-10, 64*np.finfo(float).eps*max(1., abs(before), abs(energy)))
            descent = bool(accept_verified_descent and np.isfinite(residual)
                           and report['operator_residual'] <= tolerance
                           and before-energy > descent_floor)
            accepted = bool(np.isfinite(energy) and energy <= before+1e-10
                            and (root_converged or descent))
            report.update(root_energy=energy, root_residual=residual,
                          root_converged=root_converged,
                          accepted_by_descent=accepted and not root_converged,
                          root_operator_residual=report['operator_residual'],
                          relaxation_requested=relaxation, relaxation_used=1.)
            if accepted and weight > 1:
                trial_energy, trial_residual = trial_operator_energy, trial_operator_residual
                if verify:
                    tensor = plan.tensor(trial)
                    h = state._one_site_parameter_adjoint(site, sites[site], tensor, core, left, right)
                    n = state._one_site_parameter_adjoint(site, sites[site], tensor,
                        identity[site], norm_left.data, norm_right.data)
                    trial_norm = float(np.vdot(trial, n).real)
                    if np.isfinite(trial_norm) and trial_norm > 0:
                        trial_energy = float(np.vdot(trial, h).real/trial_norm)
                        trial_residual = float(np.linalg.norm(
                            transform.conj().T@(h-trial_energy*n))/np.sqrt(trial_norm))
                    else:
                        trial_energy = np.inf
                    del tensor, h, n
                if (np.isfinite(trial_energy) and np.isfinite(trial_residual)
                        and trial_energy <= before+1e-10):
                    candidate, energy, residual = trial, trial_energy, trial_residual
                    report.update(relaxation_used=weight, operator_energy=trial_operator_energy,
                                  operator_residual=trial_operator_residual, norm=trial_norm)
            if accepted:
                state._set_site_vector(site, candidate)
                sites[site] = state.materialize_site(site)
                state.converged = state.success = False
                state.message = f'Parameter-block {direction} pass in progress; convergence not assessed'
            previous = energy if accepted else before
            report.update(site=site, direction=direction, energy_after=previous, residual=residual,
                          accepted=accepted, independently_verified=independently_verified,
                          verification_skipped=('operator_residual' if verify and not independently_verified
                                                else None),
                          boundary_peaks=dict(boundary_peaks),
                          boundary_checkpoints_reused=reused,
                          elapsed=time.perf_counter()-started)
            updates.append(report)
            state.energy = previous+state.ecore
            if callback is not None:
                callback(report)
            del left, right, norm_left, norm_right, frozen, norm_frozen, candidate, transform, metric
            if site != (state.nsites-1 if forward else 0):
                moving = advance_h(moving, site)
                norm_moving = advance_n(norm_moving, site)
                if _checkpoint_cache is not None:
                    collected[moving.bond] = moving, norm_moving
    except BaseException:
        for site, vector in enumerate(originals):
            state._set_site_vector(site, vector)
        state.energy = old_energy
        state.converged, state.success, state.message = old_status
        collected.clear()
        raise
    finally:
        suffixes.close()
        norm_suffixes.close()
        norms.right_envs.clear()
        retained.clear()
        for core in chain.mpo_factors:
            core._environment_reduced_block_cache.clear()
    if _checkpoint_cache is not None:
        _checkpoint_cache.update(
            signature=(id(state), state._state_revision, tuple(map(id, state.mpo)), 'rl' if forward else 'lr'),
            blocks=collected)
    state.converged = state.success = False
    state.message = f'Completed one {direction} parameter-block pass; global convergence not assessed'
    return updates
