"""Measure BSE batch width independently of auxiliary OpenMP thread count.

Run with PYTHONPATH=. and initial BLAS/OpenMP limits of one. Uses complete
pyrazine/CD static screening and the same prepared QP reference for every
solve. No extra excitation-space or screening truncation is introduced.
"""
import argparse
import contextlib
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor
from threadpoolctl import threadpool_limits
from benchmarks.benchmark_bse_molecules import CASES
from pyqed.qchem import Molecule
from pyqed.gw.gw import GW
from pyqed.gw.bse import BSE, TDA
from pyqed.gw.contour import ContourDeformation
from pyqed.gw.screening import StaticScreening


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, solver in zip(axes, ['TDA', 'Full BSE']):
        for threads in report['threads']:
            rows = [r for r in report['rows'] if r['solver'] == solver and r['threads'] == threads]
            explicit = [r for r in rows if r['batch'] != 'auto']
            line, = ax.plot([r['batch'] for r in explicit], [r['median_seconds'] for r in explicit],
                           'o-', label=f'{threads} thread' if threads == 1 else f'{threads} threads')
            auto = next(r for r in rows if r['batch'] == 'auto')
            ax.scatter(auto['selected_batch'], auto['median_seconds'], marker='*', s=170, color=line.get_color())
        ax.set(title=solver, xlabel='Trial-vector batch width', ylabel='Median five-root solve (s)',
               xscale='log', yscale='log')
        ax.set_xticks(report['batches'])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle(f"Pyrazine/{report['basis']}: auxiliary tiles parallelize independently of vector batches; stars = auto")
    fig.savefig(output/'pyrazine_bse_batch_scaling.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--basis', default='6-31g')
    p.add_argument('--threads', type=int, nargs='+', default=[1, 8])
    p.add_argument('--batches', type=int, nargs='+', default=[1, 2, 4, 8, 16])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--plot-only', action='store_true')
    args = p.parse_args()
    if args.repeats < 1 or min(args.threads+args.batches) < 1:
        p.error('Positive repeat, thread and batch counts are required')
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'report.json'
    if args.plot_only:
        plot(json.loads(path.read_text()), args.output)
        return
    report = dict(basis=args.basis, molecule='pyrazine', threads=args.threads, batches=args.batches,
                  repeats=args.repeats, cd_tolerance=1e-10, geometry_angstrom=CASES['pyrazine'], rows=[],
                  source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                      for name in [__file__, 'pyqed/gw/bse.py', 'pyqed/gw/bse_contractions.cpp', 'pyqed/linalg/row_product.hpp']})
    with threadpool_limits(limits=1), (args.output/'preparation.log').open('w', buffering=1) as log, contextlib.redirect_stdout(log):
        mol = Molecule(atom=CASES['pyrazine'], basis=args.basis, unit='angstrom').build(eri='cd',
            options={'low_rank_tol': 1e-10, 'eri_screen_tol': 0., 'workers': 1})
        mf = mol.RHF().run(tol=1e-11, max_cycle=100, verbose=0)
        assert mf.converged
        gw = GW(mf, eta=1e-3, threads=max(args.threads)).run()
        contour = ContourDeformation(gw, gw.e_mf[::2], [int(gw.nocc//2)], prepare=False)
        contour.static_screening = StaticScreening(contour.factors,
            cho_factor(contour.dielectric(0., imaginary=True), overwrite_a=True), gw.max_memory)
        gw._contour = contour
        screening = contour.static_screening
        screening.static_occupied(int(gw.nocc//2))
        screening.static_ov(int(gw.nocc//2))
        report.update(nao=mol.nao, qp_energies=gw.e_qp.tolist())
    print('Preparation complete', flush=True)
    configurations = [(n, width) for n in args.threads for width in args.batches+['auto']]
    with threadpool_limits(limits=1):
        for label, cls in [('TDA', TDA), ('Full BSE', BSE)]:
            reference = None
            rows = {key: dict(solver=label, threads=key[0], batch=key[1], seconds=[],
                              max_energy_difference=0., max_residual=0.) for key in configurations}
            for repeat in range(args.repeats+1):
                for n, batch in configurations if repeat % 2 == 0 else configurations[::-1]:
                    with (args.output/'solves.log').open('a') as log, contextlib.redirect_stdout(log):
                        model = cls(gw, threads=n)
                        start = time.perf_counter()
                        model.run(nroots=5, batch_columns=batch, tol=1e-9, max_cycle=200)
                        elapsed = time.perf_counter()-start
                    if reference is None:
                        reference = model.e.copy()
                    error = float(np.max(abs(model.e-reference)))
                    np.testing.assert_allclose(model.e, reference, atol=1e-8, rtol=0)
                    row = rows[n, batch]
                    if repeat:
                        row['seconds'].append(elapsed)
                    row.update(selected_batch=model.batch_columns, energies=model.e.tolist(),
                        max_energy_difference=max(row['max_energy_difference'], error),
                        max_residual=max(row['max_residual'], float(np.max(model.info['residual_norms']))),
                        threading_info=model.threading_info)
                    print(f'{label} threads={n} batch={batch} pass={repeat}: {elapsed:.3f} s', flush=True)
                report['rows'] = [r for r in report['rows'] if r['solver'] != label]+list(rows.values())
                path.write_text(json.dumps(report, indent=2))
            for row in rows.values():
                row['median_seconds'] = float(np.median(row['seconds']))
    path.write_text(json.dumps(report, indent=2))
    plot(report, args.output)


if __name__ == '__main__':
    main()
