"""Audit a disk-backed one-site operator in tied SU(2) parameters.

Builds only the required left/right boundaries. This is a frozen local-action
benchmark, not an optimization or molecular energy convergence certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
from pyqed.letta import SU2LETTA
from pyqed.letta.parameter_operator import TiedParameterOperator, iter_parameter_blocks, solve_parameter_operator
from pyqed.mps.nonabelian import environment as env
from pyqed.mps.nonabelian.sparse_boundary import SparseBoundary


def snapshot_blocks(directory, *, sha256, site, dimension):
    """Stream a frozen operator only for the exact checkpoint and site."""
    provenance = json.loads((directory/'operator-provenance.json').read_text())
    if any(provenance.get(key) != value for key, value in
           [('sha256', sha256), ('site', site), ('parameters', dimension)]):
        raise ValueError('Operator snapshot does not match checkpoint and site')
    with np.load(directory/'operator-index.npz', allow_pickle=False) as data:
        if int(data['dimension']) != dimension:
            raise ValueError('Operator snapshot dimension mismatch')
        rows, columns = data['rows'], data['columns']
        ro, co, offsets = data['row_offsets'], data['col_offsets'], data['payload_offsets']
        count = provenance['blocks']
        if (ro.shape != (count+1,) or co.shape != (count+1,) or offsets.shape != (count,)
                or ro[0] != 0 or co[0] != 0 or ro[-1] != len(rows) or co[-1] != len(columns)
                or np.any(np.diff(ro) < 0) or np.any(np.diff(co) < 0)):
            raise ValueError('Invalid operator snapshot index')
        sizes = np.diff(ro)*np.diff(co)*16
        expected_offsets = np.r_[0, np.cumsum(sizes)[:-1]] if count else np.array([], dtype=int)
        if not np.array_equal(offsets, expected_offsets):
            raise ValueError('Invalid operator snapshot payload offsets')
        with (directory/'operator-values.bin').open('rb') as stream:
            stream.seek(0, 2)
            if stream.tell() != int(sizes.sum()):
                raise OSError('Incomplete operator snapshot payload')
            stream.seek(0)
            for i, size in enumerate(sizes):
                raw = stream.read(int(size))
                if len(raw) != size:
                    raise OSError('Incomplete operator snapshot payload')
                yield (rows[ro[i]:ro[i+1]], columns[co[i]:co[i+1]],
                       np.frombuffer(raw, dtype=complex).reshape(ro[i+1]-ro[i], co[i+1]-co[i]))


def solve_local(state, sites, operator, h_left, h_right, args):
    """Trial local solve with sparse metric coordinates; do not update state."""
    from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo
    started = time.perf_counter()
    identity = _identity_mpo_factors_for_sites_and_mpo(sites, state.mpo)
    chain = env.BlockSparseEnvironmentChain.build(sites, identity, boundaries_only=True)
    left, right = chain.left_envs[0], chain.right_envs[-1]
    for i in range(args.site):
        left = left.advance(identity[i], sites[i], sites[i])
    for i in range(state.nsites-1, args.site, -1):
        right = right.advance(identity[i], sites[i], sites[i])
    metric = state._one_site_norm_components(args.site, identity[args.site], left.data, right.data)
    plan = state._wigner_eckart_route_plan(args.site)
    def factored_norm(vector):
        return state._one_site_parameter_adjoint(args.site, sites[args.site], plan.tensor(vector),
                                                 identity[args.site], left.data, right.data)
    norm_matvec = factored_norm if args.factored_norm else None
    candidate, transform, report = solve_parameter_operator(
        operator, metric, state._pack_site(args.site), metric_rtol=args.metric_rtol,
        tolerance=args.tolerance, maxiter=args.maxiter, max_space=args.max_space,
        preconditioner=args.preconditioner, norm_matvec=norm_matvec)
    tensor = state._wigner_eckart_route_plan(args.site).tensor(candidate)
    phase_started = time.perf_counter()
    fresh_h = state._one_site_parameter_adjoint(args.site, sites[args.site], tensor,
                                               state.mpo[args.site], h_left, h_right)
    fresh_n = state._one_site_parameter_adjoint(args.site, sites[args.site], tensor,
                                               identity[args.site], left.data, right.data)
    fresh_norm = float(np.vdot(candidate, fresh_n).real)
    if not np.isfinite(fresh_norm) or fresh_norm <= 0:
        raise ArithmeticError('Invalid independently evaluated candidate norm')
    fresh_energy = float(np.vdot(candidate, fresh_h).real/fresh_norm)
    fresh_residual = float(np.linalg.norm(transform.conj().T@(fresh_h-fresh_energy*fresh_n))/np.sqrt(fresh_norm))
    verification_seconds = time.perf_counter()-phase_started
    stored_h = operator.matvec(candidate)
    stored_n = (factored_norm if args.factored_norm else metric.matvec)(candidate)
    projected_h_error = transform.conj().T@(fresh_h-stored_h)
    projected_n_error = transform.conj().T@(fresh_n-stored_n)
    report['action_discrepancy'] = dict(
        h_raw=float(np.linalg.norm(fresh_h-stored_h)),
        n_raw=float(np.linalg.norm(fresh_n-stored_n)),
        h_projected=float(np.linalg.norm(projected_h_error)),
        energy_n_projected=float(abs(fresh_energy)*np.linalg.norm(projected_n_error)),
        combined_projected=float(np.linalg.norm(projected_h_error-fresh_energy*projected_n_error)),
        transform_max_abs=float(np.max(np.abs(transform.data))))
    np.savez_compressed(args.out/'action_discrepancy.npz',
                        candidate=candidate, stored_h=stored_h, fresh_h=fresh_h,
                        stored_n=stored_n, fresh_n=fresh_n,
                        projected_h_error=projected_h_error,
                        projected_n_error=projected_n_error)
    accepted = bool(np.isfinite(fresh_energy) and fresh_residual <= args.tolerance
                    and report['operator_residual'] <= args.tolerance
                    and fresh_energy <= report['energy_before']+1e-10)
    np.save(args.out/'candidate.npy', candidate, allow_pickle=False)
    report.update(energy_after=fresh_energy, residual=fresh_residual, accepted=accepted,
                  verification_seconds=verification_seconds,
                  seconds=time.perf_counter()-started, state_updated=False)
    return report


def run(args):
    if args.compensated and args.operator_snapshot:
        raise ValueError('Compensated assembly cannot reuse an operator snapshot')
    if args.solve_only and not args.solve:
        raise ValueError('--solve-only requires --solve')
    if args.solve and (not 0 < args.metric_rtol < 1 or not np.isfinite(args.tolerance)
                       or args.tolerance <= 0 or args.maxiter < 1 or args.max_space < 1):
        raise ValueError('Invalid local-solver tolerance or iteration limit')
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    state = SU2LETTA.load_checkpoint(args.checkpoint)
    operator = None
    result = dict(site=args.site, checkpoint=str(args.checkpoint), failure=None,
                  compensated=bool(args.compensated))
    with args.checkpoint.open('rb') as stream:
        result['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    def progress(**row):
        row['elapsed'] = time.perf_counter()-started
        row['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
        with (args.out/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    (args.out/'progress.jsonl').write_text('')
    try:
        if not 0 <= args.site < state.nsites:
            raise ValueError('Site is outside the chain')
        progress(event='checkpoint_loaded')
        sites = state.materialize()
        progress(event='sites_materialized')
        chain = env.BlockSparseEnvironmentChain.build(sites, state.mpo, boundaries_only=True)
        left = SparseBoundary.from_reduced(chain.left_envs[0], side='left', bond=0)
        right = SparseBoundary.from_reduced(chain.right_envs[-1], side='right', bond=state.nsites-1)
        for side, indices in [('left', range(args.site)),
                              ('right', range(state.nsites-1, args.site, -1))]:
            boundary = left if side == 'left' else right
            for i in indices:
                boundary = boundary.advance(chain.mpo_factors[i], sites[i],
                    moving_environment=state._su2_moving_environment, spill_directory=args.spill_directory,
                    release_recoupling=True)
                progress(event='boundary', side=side, bond=boundary.bond, **boundary.stats)
            if side == 'left':
                left = boundary
            else:
                right = boundary
        del boundary
        result['boundary_seconds'] = time.perf_counter()-started
        plan = state._wigner_eckart_route_plan(args.site)
        core = chain.mpo_factors[args.site]
        begin = time.perf_counter()
        count = 0
        def blocks():
            nonlocal count
            source = (snapshot_blocks(args.operator_snapshot, sha256=result['sha256'],
                                      site=args.site, dimension=plan.nparameters)
                      if args.operator_snapshot else
                      iter_parameter_blocks(plan, core, left, right, block_bytes=args.block_bytes,
                                            compensated=args.compensated))
            for block in source:
                count += 1
                if count % 100 == 0:
                    progress(event='operator_blocks', count=count)
                yield block
        if args.profile_assembly:
            import cProfile
            import pstats
            profile = cProfile.Profile()
            profile.enable()
        try:
            operator = TiedParameterOperator(plan.nparameters, blocks(),
                spill_directory=args.spill_directory, storage_bytes=args.storage_bytes)
        finally:
            if args.profile_assembly:
                profile.disable()
                profile.dump_stats(str(args.out/'assembly.prof'))
                with (args.out/'assembly_profile.txt').open('w') as stream:
                    pstats.Stats(profile, stream=stream).sort_stats('cumulative').print_stats(50)
        result.update(parameters=plan.nparameters, blocks=count,
                      operator_snapshot=str(args.operator_snapshot) if args.operator_snapshot else None,
                      build_seconds=time.perf_counter()-begin,
                      resident_bytes=operator.resident_bytes, disk_bytes=operator.disk_bytes)
        progress(event='operator_built', **{k:result[k] for k in ('blocks','build_seconds','resident_bytes','disk_bytes')})
        # Preserve the costly frozen operator for further local-solver diagnostics.
        row_offsets = np.r_[0, np.cumsum([len(r) for r, _, _ in operator.blocks])]
        col_offsets = np.r_[0, np.cumsum([len(c) for _, c, _ in operator.blocks])]
        np.savez(args.out/'operator-index.npz', dimension=operator.dimension,
                 row_offsets=row_offsets, col_offsets=col_offsets,
                 rows=np.concatenate([r for r, _, _ in operator.blocks]) if operator.blocks else np.array([],dtype=np.int64),
                 columns=np.concatenate([c for _, c, _ in operator.blocks]) if operator.blocks else np.array([],dtype=np.int64),
                 payload_offsets=np.array([p for _, _, p in operator.blocks],dtype=np.int64))
        operator.file.seek(0)
        with (args.out/'operator-values.bin').open('wb') as stream:
            shutil.copyfileobj(operator.file, stream, length=2**20)
        (args.out/'operator-provenance.json').write_text(json.dumps(
            {key:result[key] for key in ('checkpoint','sha256','site','parameters','blocks')},indent=2))
        rng = np.random.default_rng(72)
        vectors = [state._pack_site(args.site),
                   rng.normal(size=plan.nparameters)+1j*rng.normal(size=plan.nparameters)]
        if args.solve_only:
            vectors = []
        result['action_probes_skipped'] = args.solve_only
        rows = []
        for index, vector in enumerate(vectors):
            begin = time.perf_counter()
            expected = state._one_site_parameter_adjoint(args.site, sites[args.site],
                plan.tensor(vector), core, left, right)
            reference_seconds = time.perf_counter()-begin
            begin = time.perf_counter()
            actual = operator.matvec(vector)
            seconds = time.perf_counter()-begin
            np.save(args.out/f'action_{index}.npy', actual, allow_pickle=False)
            passed = bool(np.allclose(actual, expected, rtol=1e-11, atol=1e-12))
            rows.append(dict(vector=index, direct_seconds=reference_seconds,
                             projected_seconds=seconds, max_error=float(np.max(np.abs(actual-expected))),
                             passed=passed))
            progress(event='comparison', **rows[-1])
        result['comparisons'] = rows
        if not all(row['passed'] for row in rows):
            raise AssertionError('Projected action differs from independent reduced adjoint')
        if args.solve:
            result['local_solve'] = solve_local(state, sites, operator, left, right, args)
            progress(event='local_solve', **result['local_solve'])
            if not result['local_solve']['accepted']:
                raise AssertionError('Local candidate failed residual or variational acceptance')
    except Exception as exc:
        result['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        if operator is not None:
            operator.close()
        state.close()
    result['elapsed'] = time.perf_counter()-started
    result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
    (args.out/'result.json').write_text(json.dumps(result, indent=2))
    plot(result, args.out)
    print(json.dumps(result, indent=2), flush=True)
    if result['failure']:
        raise SystemExit(result['failure'])


def plot(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    rows = result.get('comparisons', [])
    if rows:
        counts = np.arange(1, 81)
        for row in rows:
            label = 'Direct incumbent action' if row['vector'] == 0 else 'Direct complex action'
            ax.plot(counts, counts*row['direct_seconds'], label=label)
        ax.plot(counts, result['build_seconds']+counts*np.mean([r['projected_seconds'] for r in rows]), label='Build and apply blocks')
        ax.legend()
    ax.set(xlabel='Local actions', ylabel='Estimated time (s)',
           title=f"Frozen SU(2) site {result['site']}"+(' — failed validation' if result['failure'] else ''))
    if rows:
        fig.savefig(out/'parameter_operator.png', dpi=180)
        fig.savefig(out/'parameter_operator.pdf')
    plt.close(fig)
    if 'local_solve' in result:
        solve = result['local_solve']
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
        axes[0].bar(['Incumbent', 'Candidate'], [0., solve['energy_after']-solve['energy_before']])
        axes[0].set(ylabel='Energy minus incumbent (Eh)')
        axes[1].bar(['Independently checked\nretained-space residual'], [max(solve['residual'], 1e-16)])
        axes[1].axhline(solve['tolerance'], color='black', linestyle='--', label='Requested tolerance')
        axes[1].set(yscale='log', ylabel='Hartree')
        axes[1].legend()
        fig.suptitle('Local trial accepted' if solve['accepted'] else 'Local trial rejected')
        fig.savefig(out/'local_solve.png', dpi=180)
        fig.savefig(out/'local_solve.pdf')
        plt.close(fig)
        if 'action_discrepancy' in solve:
            discrepancy = solve['action_discrepancy']
            fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
            keys = ['h_projected', 'energy_n_projected', 'combined_projected']
            ax.bar(['Hamiltonian', 'Energy × norm', 'Combined'],
                   [max(discrepancy[key], 1e-18) for key in keys])
            ax.axhline(solve['tolerance'], color='black', linestyle='--',
                       label='Requested residual tolerance')
            ax.set(yscale='log', ylabel='Projected action discrepancy (Eh)',
                   title='Independent versus stored local actions')
            ax.legend()
            fig.savefig(out/'action_discrepancy.png', dpi=180)
            fig.savefig(out/'action_discrepancy.pdf')
            plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--site', required=True, type=int)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--spill-directory', required=True, type=Path)
    parser.add_argument('--block-bytes', type=int, default=2**20)
    parser.add_argument('--storage-bytes', type=int, default=32*2**20)
    parser.add_argument('--operator-snapshot', type=Path,
                        help='Reuse a frozen operator from this benchmark for the exact checkpoint and site')
    parser.add_argument('--solve', action='store_true', help='Also trial a local Davidson solve without changing the checkpoint')
    parser.add_argument('--profile-assembly', action='store_true',
                        help='Profile operator construction only; profiled timings include instrumentation overhead')
    parser.add_argument('--compensated', action='store_true',
                        help='Experimentally compensate real compiled route accumulation')
    parser.add_argument('--factored-norm', action='store_true',
                        help='Use the reduced factored norm action inside Davidson')
    parser.add_argument('--solve-only', action='store_true', help='Skip repeated action probes; still independently verify the trial candidate')
    parser.add_argument('--metric-rtol', type=float, default=1e-11)
    parser.add_argument('--tolerance', type=float, default=1e-10)
    parser.add_argument('--maxiter', type=int, default=300)
    parser.add_argument('--max-space', type=int, default=96)
    parser.add_argument('--preconditioner', choices=['constant', 'diagonal'], default='constant',
                        help='Exact diagonal in sparse metric coordinates, or constant Davidson preconditioning')
    run(parser.parse_args())
