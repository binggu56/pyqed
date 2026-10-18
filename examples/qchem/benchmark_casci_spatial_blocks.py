"""Compare matrix-free and assembled D3h CASCI with symmetry disabled.

H6 consists of two concentric equilateral triangles, in bohr. The default
6-31G active spaces contain complete orbital multiplets; alternate --basis
and --active-orbitals choices must also retain complete multiplets. Both modes use the same
symmetry-adapted RHF orbitals and PyQED packed AO integrals. Off-mode E timings
include the ground state needed to reach the first E pair (three roots);
targeted runs request one complete multiplet. Warm timings include integral
transformation, operator setup, eigensolve, residual checks and reporting.
Cold timings clear spatial/string transformation and contraction-plan caches on every
repeat, after compilation warmup; they exclude imports, JIT compilation and RHF.
Only immutable transformation plans are cached; no previous CI root is reused.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/matplotlib-codex')
import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import time
import numpy as np
from pyqed import num_workers
from pyqed.qchem import Molecule, RHF
from pyqed.qchem.mcscf.direct_ci import CASCI
from pyqed.qchem.ci import spatial_blocks
from pyqed.qchem.spatial_symmetry import _cached_string_transform

ATOM = ';'.join(f'H {r*np.cos(t):.16g} {r*np.sin(t):.16g} 0'
               for r in (1.4, 3.1) for t in np.arange(3)*2*np.pi/3)


@contextmanager
def assembled_reference(enabled):
    original = spatial_blocks.matrix_free_row_action
    try:
        if enabled:
            spatial_blocks.matrix_free_row_action = lambda *args, **kwargs: None
        yield
    finally:
        spatial_blocks.matrix_free_row_action = original


def plot(rows, directory, basis="6-31g"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = list(dict.fromkeys(row['case'] for row in rows))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    x = np.arange(len(labels))
    for mode, shift, color in [('off', -.25, '#777777'), ('assembled', 0, '#D55E00'), ('matrix_free', .25, '#0072B2')]:
        if not any(row['mode'] == mode for row in rows):
            continue
        selected = [next(row for row in rows if row['case']==name and row['mode']==mode) for name in labels]
        medians = np.array([row['median_seconds'] for row in selected])
        intervals = np.array([np.quantile(row['times'], [.25, .75]) for row in selected]).T
        axes[0].bar(x+shift, medians, width=.24, color=color, label={'off':'Symmetry off', 'assembled':'Assembled', 'matrix_free':'Matrix-free'}[mode],
                    yerr=[medians-intervals[0], intervals[1]-medians], capsize=2)
    axes[0].set(yscale='log', ylabel='Total CASCI time (s)', title='Warm median and interquartile range')
    axes[0].legend(frameon=False, loc='upper left', fontsize=8)
    cold = [next(row for row in rows if row['case']==name and row['mode']=='matrix_free') for name in labels]
    cold_medians = np.array([row['cold_median_seconds'] for row in cold])
    cold_intervals = np.array([np.quantile([s['seconds'] for s in row['cold_samples']],
                                         [.25, .75]) for row in cold]).T
    axes[2].bar(x-.25, cold_medians, width=.24,
                yerr=[cold_medians-cold_intervals[0], cold_intervals[1]-cold_medians],
                capsize=2, color='#009E73', label='Cold plans')
    axes[2].bar(x, [row['median_seconds'] for row in cold], width=.24,
                color='#0072B2', label='Warm plans')
    off_times = [next(row['median_seconds'] for row in rows
                      if row['case'] == name and row['mode'] == 'off') for name in labels]
    axes[2].bar(x+.25, off_times, width=.24, color='#777777', label='Symmetry off')
    axes[2].set(yscale='log', ylabel='Total CASCI time (s)', title='Cold and warm symmetry versus off')
    axes[2].legend(frameon=False, loc='upper left', fontsize=8)
    errors = [next(row['energy_error'] for row in rows if row['case']==name and row['mode']=='matrix_free') for name in labels]
    axes[1].scatter(x, np.maximum(errors, 1e-15), color='#0072B2')
    axes[1].set(yscale='log', ylabel='Maximum energy difference (Ha)', title='Matrix-free versus symmetry off')
    axes[1].axhline(1e-8, color='#777777', ls='--', lw=1, label='Validation tolerance')
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=35, ha='right')
        ax.spines[['top','right']].set_visible(False)
    partner_note = 'both E partners retained' if any(row['target'] == "E'" for row in rows) else 'A1′ target'
    fig.suptitle(f'D3h H₆ / {basis} · six active electrons · {partner_note}')
    fig.savefig(directory/'casci-spatial-blocks.png', dpi=300)
    fig.savefig(directory/'casci-spatial-blocks.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--active-orbitals', type=int, nargs='+', default=[6, 9, 11])
    parser.add_argument('--basis', default='6-31g')
    parser.add_argument('--wfnsym', nargs='+', choices=["A1'", "E'"], default=["A1'", "E'"])
    parser.add_argument('--modes', nargs='+', choices=['off', 'assembled', 'matrix_free'],
                        default=['off', 'assembled', 'matrix_free'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output-dir', type=Path, default=Path('/private/tmp/casci-spatial-blocks'))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if not {'off', 'matrix_free'}.issubset(args.modes):
        parser.error('--modes must include off and matrix_free')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    refs = {}
    for enabled in (True, False):
        mol = Molecule(atom=ATOM, unit='bohr', basis=args.basis, symmetry=enabled).build(eri='s8')
        mf = RHF(mol); mf.chkfile = None
        mf.run(verbose=0, tol=1e-11)
        refs[enabled] = mf
    coeff = refs[True].mo_coeff.copy()

    def execute(ncas, target, mode):
        gc.collect()
        mc = CASCI(refs[mode != 'off'], ncas=ncas, nelecas=6, multiplicity=1, verbose=0)
        mc.tol = 1e-11
        mc.direct_ci_residual_tol = 1e-9
        start = time.perf_counter()
        with assembled_reference(mode == 'assembled'):
            mc.run(mo_coeff=coeff, wfnsym=target if mode != 'off' else None,
                   nstates=1 if mode != 'off' or target=="A1'" else 3)
        seconds = time.perf_counter()-start
        assert mc.converged
        selected = slice(None) if mode != 'off' or target=="A1'" else slice(1,3)
        energies = np.asarray(mc.e_tot)[selected]
        roots = np.asarray(mc.ci)[selected]
        gram_error = float(np.max(abs(roots.conj() @ roots.T - np.eye(len(roots)))))
        assert gram_error < 1e-8
        if target=="E'":
            assert len(energies)==2
            np.testing.assert_allclose(energies[0], energies[1], atol=1e-8, rtol=0)
        diagnostics = mc.direct_ci_diagnostics
        if mode == 'matrix_free':
            assert diagnostics['matrix_free'] and diagnostics['block_nnz'] == 0
        covariance_error = 0.0
        if mode != 'off':
            assert mc.state_irreps == (target,)*len(energies)
            if target=="E'":
                action = mc.spatial_symmetry
                for i in range(2):
                    for j in range(2):
                        covariance_error = max(covariance_error, float(np.linalg.norm(
                            action.apply(roots[j], i, j)-roots[i])))
                assert covariance_error < 1e-8
        assert np.max(diagnostics.get('unprojected_residual_norms', [0])) < 1e-8
        return dict(seconds=seconds, energies=energies.tolist(), gram_error=gram_error,
            covariance_error=covariance_error, backend=mc.solver_backend,
            diagnostics={key: diagnostics[key] for key in ('symmetry_blocked',
                'transformation_cache_hit', 'determinant_dimension', 'row_dimension',
                'block_nnz', 'matrix_free', 'representative_rows', 'reconstruction_condition',
                'contraction_plan_cache_hit', 'transformation_seconds', 'assembly_seconds', 'davidson_seconds')
                if key in diagnostics},
            residual_max=float(np.max(diagnostics.get('unprojected_residual_norms', [0]))))

    rows = []
    for ncas in args.active_orbitals:
        for target in args.wfnsym:
            case = f'{ncas} orbitals {target}'
            # Compile first, then measure cold transformation setup separately.
            execute(ncas, target, 'matrix_free')
            cold_samples = []
            for repeat in range(args.repeats):
                spatial_blocks._PLAN_CACHE.clear(); spatial_blocks._ROW_PLAN_CACHE.clear()
                _cached_string_transform.cache_clear()
                cold = execute(ncas, target, 'matrix_free')
                assert not cold['diagnostics']['transformation_cache_hit']
                assert not cold['diagnostics']['contraction_plan_cache_hit']
                cold_samples.append(cold)
                print(case, 'cold', f"{cold['seconds']:.4f}s", flush=True)
            for mode in args.modes:
                execute(ncas, target, mode)
            runs = {mode: [] for mode in args.modes}
            for repeat in range(args.repeats):
                order = args.modes if repeat % 2 == 0 else list(reversed(args.modes))
                for mode in order:
                    result = execute(ncas, target, mode)
                    runs[mode].append(result)
                    print(case, mode, f"{result['seconds']:.4f}s", result['energies'], flush=True)
            reference = np.array(runs['off'][-1]['energies'])
            for sample in cold_samples:
                np.testing.assert_allclose(sample['energies'], reference, atol=1e-8, rtol=0)
            for mode, samples in runs.items():
                error = max(float(np.max(abs(np.array(sample['energies'])-reference))) for sample in samples)
                assert error < 1e-8, (case, mode, error)
                rows.append(dict(case=case, ncas=ncas, target=target, mode=mode,
                    times=[sample['seconds'] for sample in samples],
                    median_seconds=float(np.median([sample['seconds'] for sample in samples])),
                    energy_error=error, cold_samples=cold_samples if mode=='matrix_free' else [],
                    cold_median_seconds=float(np.median([s['seconds'] for s in cold_samples])) if mode=='matrix_free' else None, samples=samples))
            payload = dict(atom=ATOM, basis=args.basis, nelecas=6, workers=num_workers(),
                blas_threads=1, repeats=args.repeats, residual_tolerance=1e-9,
                timing_scope=__doc__, rows=rows)
            (args.output_dir/'benchmark.json').write_text(json.dumps(payload, indent=2))
    plot(rows, args.output_dir, args.basis)
    print('Saved', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
