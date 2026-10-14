"""Molecular full-BSE/TDA OpenMP scaling with complete RI/CD screening.

Run with PYTHONPATH=. and initial BLAS/OpenMP environment limits of one.
Compares the NumPy action, compiled serial action and orbital-factor OpenMP.
Setup and screening caches are excluded; no transition matrix is materialized.
Static dielectric screening exercises the same BSE path used after contour GW,
but QP preparation uses spectral GW and is not a contour-GW timing benchmark.
"""
import argparse
import contextlib
import cProfile
import hashlib
import importlib
import json
import platform
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.linalg import cho_factor
from threadpoolctl import threadpool_info, threadpool_limits
from benchmarks.benchmark_bse_molecules import CASES
from benchmarks.benchmark_gw_tda_memory import geometry
from pyqed.qchem import Molecule
from pyqed.gw.gw import GW
from pyqed.gw.contour import ContourDeformation
from pyqed.gw.screening import StaticScreening
from pyqed.gw.bse import BSE, TDA

b = importlib.import_module('pyqed.gw.bse')


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for label, marker in [('TDA', 'o'), ('Full BSE', 's')]:
        rows = [r for r in report['rows'] if r['solver'] == label and r['method'] != 'NumPy']
        baseline = rows[0]['median_solve_seconds']
        x = [r['threads'] for r in rows]
        axes[0].plot(x, [r['median_action_seconds'] for r in rows], marker+'-', label=label)
        axes[1].plot(x, [r['median_solve_seconds'] for r in rows], marker+'-', label=label)
        axes[2].plot(x, [baseline/r['median_solve_seconds'] for r in rows], marker+'-', label=label)
    for ax in axes:
        ax.set(xlabel='BSE OpenMP threads (BLAS=1)', xticks=report['threads'])
        ax.grid(alpha=.25)
        ax.legend()
    axes[0].set_ylabel('Median eight-vector action (s)')
    axes[1].set_ylabel('Median five-root solve (s)')
    axes[2].set_ylabel('Solve speedup versus compiled serial')
    fig.suptitle(f"{report['molecule'].capitalize()}/{report['basis']}; complete static screening")
    fig.savefig(output/f"{report['molecule']}_bse_openmp_scaling.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--molecule', choices=['pyrazine', 'naphthalene'], default='pyrazine')
    p.add_argument('--basis', default='6-31g')
    p.add_argument('--screening', choices=['static', 'spectral'], default='static')
    p.add_argument('--threads', nargs='+', type=int, default=[1, 2, 4, 8])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--plot-only', action='store_true')
    args = p.parse_args()
    if args.repeats < 1 or min(args.threads) < 1 or args.threads[0] != 1:
        p.error('Positive repeats and thread counts starting at one are required')
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'report.json'
    if args.plot_only:
        plot(json.loads(path.read_text()), args.output)
        return
    if b._bse_contractions is None or not b._bse_contractions.openmp_available():
        raise RuntimeError('Build BSE contractions with OpenMP before benchmarking')
    atom = CASES['pyrazine'] if args.molecule == 'pyrazine' else geometry(2)
    report = dict(molecule=args.molecule, basis=args.basis, screening=args.screening,
                  geometry_angstrom=atom, cd_tolerance=1e-10, threads=args.threads,
                  repeats=args.repeats, machine=platform.platform(), rows=[],
                  source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                      for name in [__file__, 'pyqed/gw/bse.py', 'pyqed/gw/bse_contractions.cpp']})
    with threadpool_limits(limits=1), (args.output/'preparation.log').open('w', buffering=1) as log, contextlib.redirect_stdout(log):
        started = time.perf_counter()
        mol = Molecule(atom=atom, basis=args.basis, unit='angstrom').build(eri='cd',
            options={'low_rank_tol': 1e-10, 'eri_screen_tol': 0., 'workers': 1})
        mf = mol.RHF().run(tol=1e-11, max_cycle=100, verbose=0)
        assert mf.converged
        gw = GW(mf, screening='TDH', eta=1e-3, threads=max(args.threads)).run()
        if args.screening == 'static':
            contour = ContourDeformation(gw, gw.e_mf[::2], [gw.nocc//2], prepare=False)
            contour.static_screening = StaticScreening(contour.factors,
                cho_factor(contour.dielectric(0., imaginary=True), overwrite_a=True), gw.max_memory)
            gw._contour = contour
        report.update(preparation_seconds=time.perf_counter()-started, nao=mol.nao,
                      qp_energies=gw.e_qp.tolist(), rhf_energy=float(mf.e_tot))
        template = BSE(gw)
        template._ensure_screening()
        screening = b._bse_screening(template)
        screening.static_occupied(template.nocc)
        screening.static_ov(template.nocc)
    report['libraries'] = threadpool_info()
    print('Preparation complete:', report['nao'], 'orbitals,', args.screening, 'screening', flush=True)
    dim = template.nocc*(template.nso-template.nocc)
    trial = np.random.default_rng(931).normal(size=(dim, 8))
    references = {}
    configurations = [('NumPy', 1)]+[('OpenMP', n) for n in args.threads]
    kernel = b._bse_contractions
    with threadpool_limits(limits=1):
        for label, cls, action in [('TDA', TDA, b._bse_tda_matmat), ('Full BSE', BSE, b._bse_full_matmat)]:
            vectors = trial if label == 'TDA' else np.vstack((trial, trial[::-1]))
            rows = {key: dict(solver=label, method=key[0], threads=key[1], action_seconds=[],
                             solve_seconds=[], max_action_difference=0., max_energy_difference=0.) for key in configurations}
            for repeat in range(args.repeats+1):
                for method, n in configurations if repeat % 2 == 0 else configurations[::-1]:
                    row = rows[method, n]
                    with patch.object(b, '_bse_contractions', None if method == 'NumPy' else kernel):
                        model = cls(gw, threads=n)
                        started = time.perf_counter()
                        value = action(model, vectors, 8)
                        elapsed_action = time.perf_counter()-started
                        if (label, 'action') not in references:
                            references[label, 'action'] = value.copy()
                            profiler = cProfile.Profile()
                            profiler.runcall(action, model, vectors, 8)
                            profiler.dump_stats(str(args.output/f'{label.replace(" ", "_")}_numpy.prof'))
                        difference = float(np.max(np.abs(value-references[label, 'action'])))
                        np.testing.assert_allclose(value, references[label, 'action'], atol=1e-10, rtol=1e-11)
                        started = time.perf_counter()
                        model.run(nroots=5, batch_columns=8, tol=1e-9, max_cycle=200)
                        elapsed_solve = time.perf_counter()-started
                    if (label, 'energy') not in references:
                        references[label, 'energy'] = model.e.copy()
                    error = float(np.max(np.abs(model.e-references[label, 'energy'])))
                    np.testing.assert_allclose(model.e, references[label, 'energy'], atol=1e-8, rtol=0)
                    if repeat:
                        row['action_seconds'].append(elapsed_action)
                        row['solve_seconds'].append(elapsed_solve)
                    row.update(max_action_difference=max(row['max_action_difference'], difference),
                               max_energy_difference=max(row['max_energy_difference'], error),
                               energies=model.e.tolist(), residuals=np.asarray(model.info['residual_norms']).tolist(),
                               threading_info=getattr(model, 'threading_info', None))
                    print(f'{label} {method} threads={n} pass={repeat}: action {elapsed_action:.4f} s, solve {elapsed_solve:.3f} s', flush=True)
                report['rows'] = [r for r in report['rows'] if r['solver'] != label]+list(rows.values())
                path.write_text(json.dumps(report, indent=2))
            for row in rows.values():
                row['median_action_seconds'] = float(np.median(row['action_seconds']))
                row['median_solve_seconds'] = float(np.median(row['solve_seconds']))
    path.write_text(json.dumps(report, indent=2))
    plot(report, args.output)


if __name__ == '__main__':
    main()
