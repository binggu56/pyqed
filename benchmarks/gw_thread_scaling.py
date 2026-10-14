"""Molecular G0W0 thread scaling, separating OpenMP controls from BLAS.

Run with PYTHONPATH=. and initial BLAS/OpenMP environment limits of one.
Reuses the charge-screening benchmark's CD/RHF preparation and geometry.
No OpenMP speedup is claimed unless a loaded OpenMP runtime is observed.
"""
import argparse
import contextlib
import hashlib
import json
import os
import platform
import time
import cProfile
from unittest.mock import patch
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from benchmarks.benchmark_bse_molecules import CASES
from benchmarks.benchmark_gw_tda_memory import geometry
from pyqed.qchem import Molecule
from pyqed.gw.gw import GW
import importlib
g = importlib.import_module('pyqed.gw.gw')


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for mode in dict.fromkeys(row['mode'] for row in report['rows']):
        rows = [r for r in report['rows'] if r['mode'] == mode]
        baseline = rows[0]['median_seconds']
        x = [r['threads'] for r in rows]
        label = 'Orbital OpenMP (BLAS=1)' if mode == 'openmp' else 'BLAS threads (orbital threads=1)'
        axes[0].plot(x, [r['median_seconds'] for r in rows], 'o-', label=label)
        axes[1].plot(x, [baseline/r['median_seconds'] for r in rows], 'o-', label=label)
        failed = [r for r in rows if not r['energy_agreement']]
        for ax, values in zip(axes, ([r['median_seconds'] for r in failed],
                                   [baseline/r['median_seconds'] for r in failed])):
            if failed:
                ax.scatter([r['threads'] for r in failed], values, marker='x',
                           s=110, color='red', zorder=5, label='QP agreement failed')
    for ax in axes:
        ax.set_xlabel('Requested threads')
        ax.set_xticks(report['threads'])
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Median GW wall time (s)')
    axes[1].set_ylabel('Speedup relative to one thread')
    axes[1].axhline(1, color='.6', linestyle='--')
    fig.suptitle(f"{report['molecule'].capitalize()}/{report['basis']} G₀W₀; CD integrals; setup excluded")
    fig.savefig(output/f"{report['molecule']}_gw_thread_scaling.png", dpi=180)
    plt.close(fig)


def plot_comparison(reports, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    error = 0.
    for report in reports:
        rows = [r for r in report['rows'] if r['mode'] == 'openmp']
        x = [r['threads'] for r in rows]
        label = f"{report['molecule'].capitalize()}/{report['basis']} ({report['nao']} orbitals)"
        axes[0].plot(x, [r['median_seconds'] for r in rows], 'o-', label=label)
        axes[1].plot(x, [r['speedup'] for r in rows], 'o-', label=label)
        error = max(error, max(r['max_qp_difference'] for r in rows))
    for ax in axes:
        ax.set(xlabel='Orbital OpenMP threads (BLAS=1)', xticks=reports[0]['threads'])
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set(ylabel='Median GW wall time (s)', yscale='log')
    axes[1].set(ylabel='Speedup versus compiled one-thread solve')
    fig.suptitle(f'Complete-screening G₀W₀; maximum QP difference {error:.2g} Ha')
    fig.savefig(output/'molecular_gw_openmp_comparison.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--basis', default='6-31g')
    parser.add_argument('--molecule', choices=['pyrazine', 'naphthalene'], default='pyrazine')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--modes', nargs='+', choices=['openmp', 'blas'], default=['openmp', 'blas'])
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--compare', type=Path, nargs='+', help='Plot existing benchmark reports together')
    args = parser.parse_args()
    if args.repeats < 1 or args.threads[0] != 1 or min(args.threads) < 1:
        parser.error('Positive repeats and thread counts starting with one are required')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.compare:
        plot_comparison([json.loads(p.read_text()) for p in args.compare], args.output)
        return
    path = args.output/'report.json'
    if args.plot_only:
        plot(json.loads(path.read_text()), args.output)
        return
    atom = CASES['pyrazine'] if args.molecule == 'pyrazine' else geometry(2)
    report = dict(molecule=args.molecule, basis=args.basis, modes=args.modes,
                  threads=args.threads, repeats=args.repeats,
                  machine=platform.platform(), cpu_count=os.cpu_count(), rows=[],
                  geometry_angstrom=atom, cd_tolerance=1e-10,
                  source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                 for p in [__file__, 'pyqed/gw/gw.py', 'pyqed/gw/screening.py', 'pyqed/gw/dyson.cpp']})
    with threadpool_limits(limits=1), (args.output/'preparation.log').open('w') as log, contextlib.redirect_stdout(log):
        start = time.perf_counter()
        mol = Molecule(atom=atom, basis=args.basis, unit='angstrom')
        mol.build(eri='cd', options={'low_rank_tol': 1e-10, 'eri_screen_tol': 0., 'workers': 1})
        mf = mol.RHF().run(tol=1e-11, max_cycle=100, verbose=0)
        assert mf.converged
        report.update(preparation_seconds=time.perf_counter()-start, nao=mol.nao, rhf_energy=float(mf.e_tot))
    report['libraries'] = threadpool_info()
    report['openmp_available'] = g._dyson is not None and g._dyson.openmp_available()
    if not report['openmp_available']:
        raise RuntimeError('This benchmark requires the GW extension built with OpenMP')
    with threadpool_limits(limits=1), patch.object(g, '_dyson', None), \
            (args.output/'python_reference.log').open('w') as log, contextlib.redirect_stdout(log):
        profiler = cProfile.Profile()
        start = time.perf_counter()
        profiler.enable()
        ref = GW(mf, screening='TDH', eta=1e-3).run()
        profiler.disable()
        report['python_reference_seconds_profiled'] = time.perf_counter()-start
        profiler.dump_stats(str(args.output/'python_reference.prof'))
        reference = ref.e_qp.copy()
        report['reference_qp_energies'] = reference.tolist()
    print(f"Python reference: {report['python_reference_seconds_profiled']:.3f} s (profiled)", flush=True)
    for mode in args.modes:
        rows = {n: dict(mode=mode, threads=n, seconds=[], max_qp_difference=0., max_qp_residual=0.) for n in args.threads}
        for repeat in range(args.repeats+1):
            order = args.threads if repeat % 2 == 0 else args.threads[::-1]
            for n in order:
                with threadpool_limits(limits=1), threadpool_limits(limits=n, user_api=mode):
                    pools = threadpool_info()
                    with (args.output/f'{mode}_{n}_{repeat}.log').open('w') as log, contextlib.redirect_stdout(log):
                        start = time.perf_counter()
                        gw = GW(mf, screening='TDH', eta=1e-3,
                                threads=n if mode == 'openmp' else 1).run()
                        elapsed = time.perf_counter()-start
                assert np.all(gw.qp_weights > 0) and np.max(gw.qp_residuals) <= 1e-9
                row = rows[n]
                row.update(libraries=pools, threading_info=gw.threading_info, qp_energies=gw.e_qp.tolist(),
                           max_qp_difference=max(row['max_qp_difference'], float(np.max(np.abs(gw.e_qp-reference)))),
                           max_qp_residual=max(row['max_qp_residual'], float(np.max(gw.qp_residuals))))
                row['energy_agreement'] = row['max_qp_difference'] <= 1e-9
                if repeat:
                    row['seconds'].append(elapsed)
                print(f"{mode} threads={n} pass={repeat}: {elapsed:.3f} s; QP difference={row['max_qp_difference']:.3g} Ha", flush=True)
            report['rows'] = [r for r in report['rows'] if r['mode'] != mode] + list(rows.values())
            path.write_text(json.dumps(report, indent=2))
        for row in rows.values():
            row['median_seconds'] = float(np.median(row['seconds']))
            row['speedup'] = float(np.median(rows[1]['seconds']))/row['median_seconds']
    path.write_text(json.dumps(report, indent=2))
    plot(report, args.output)
    if not all(row['energy_agreement'] for row in report['rows']):
        raise RuntimeError('QP energy agreement failed; see report and figure')


if __name__ == '__main__':
    main()
