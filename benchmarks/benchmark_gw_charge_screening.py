"""Compare exact charge-reduced GW with the original full-spin equations.

Use PYTHONPATH=. and one BLAS/OpenMP thread. CD integrals and RHF are shared
between methods. Full-spin screening is a benchmark reference only. No
screening poles are truncated in the charge calculation. Outputs belong
outside the repository; --new-only supports larger production checks.
"""
import argparse
import contextlib
import importlib
import hashlib
import os
import json
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from pyqed.qchem import Molecule
from pyqed.gw.bse import BSE
from benchmarks.benchmark_bse_molecules import CASES

g = importlib.import_module('pyqed.gw.gw')


def full_spin(gw, using_tda=False, using_casida=True, method='TDH'):
    assert method == 'TDH' and using_casida and not using_tda
    gw._M = None
    gw._charge_screening = None
    return g._casida_eigh(*g.rpa_AB_matrices(gw, method=method))


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout='constrained')
    labels = list(report['methods'])
    colors = ['#0072B2', '#009E73'][-len(labels):]
    for ax, key, title in zip(axes, ['median_gw_seconds', 'bse_preparation_seconds', 'bse_seconds'],
                              ['G₀W₀', 'BSE screening preparation', 'BSE: five roots']):
        values = [report['methods'][name][key] for name in labels]
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, labels=[f'{v:.3g} s' for v in values], padding=4, fontsize=9)
        ax.set_ylim(0, max(values)*1.25)
        ax.set_ylabel('Time (s)')
        ax.set_title(title)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f"Pyrazine / {report['basis']}, CD; one thread, exact charge reduction")
    for ext in ['png', 'pdf']:
        fig.savefig(output/f'gw_charge_screening.{ext}', dpi=300)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--basis', default='6-31g')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--new-only', action='store_true')
    p.add_argument('--plot-only', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'report.json'
    if args.plot_only:
        plot(json.loads(path.read_text()), args.output)
        return
    report = dict(basis=args.basis, cd_tolerance=1e-10, repeats=args.repeats, methods={},
        geometry_angstrom=CASES['pyrazine'], numpy=np.__version__,
        threads={key: os.environ.get(key) for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']},
        source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                       for name in ['pyqed/gw/gw.py', 'pyqed/gw/bse.py', __file__]})
    with (args.output/'preparation.log').open('w', buffering=1) as log, contextlib.redirect_stdout(log):
        started = time.perf_counter()
        mol = Molecule(atom=CASES['pyrazine'], basis=args.basis, unit='angstrom')
        mol.build(eri='cd', options={'low_rank_tol': 1e-10, 'eri_screen_tol': 0., 'workers': 1})
        mf = mol.RHF().run(tol=1e-11, max_cycle=100, verbose=0)
        assert mf.converged
        report['rhf_and_integral_seconds'] = time.perf_counter()-started
    charge_rpa = g.rpa
    methods = ['charge'] if args.new_only else ['full spin', 'charge']
    objects = {}
    for name in methods:
        report['methods'][name] = dict(gw_seconds=[], qp_energies=[], qp_weights=[], qp_residuals=[])
    for repeat in range(args.repeats):
        for name in methods[::1 if repeat % 2 == 0 else -1]:
            with (args.output/f'{name.replace(" ", "_")}_{repeat}.log').open('w', buffering=1) as log, contextlib.redirect_stdout(log):
                with patch.object(g, 'rpa', charge_rpa if name == 'charge' else full_spin):
                    start = time.perf_counter()
                    gw = g.GW(mf, screening='TDH', eta=1e-3).run()
                    elapsed = time.perf_counter()-start
            assert np.all(gw.qp_weights > 0) and np.max(gw.qp_residuals) <= 1e-9
            objects[name] = gw
            report['methods'][name]['gw_seconds'].append(elapsed)
            report['methods'][name]['qp_energies'].append(gw.e_qp.tolist())
            report['methods'][name]['qp_weights'].append(gw.qp_weights.tolist())
            report['methods'][name]['qp_residuals'].append(gw.qp_residuals.tolist())
            print(name, repeat+1, elapsed, flush=True)
            path.write_text(json.dumps(report, indent=2))
    for name, gw in objects.items():
        row = report['methods'][name]
        row['median_gw_seconds'] = float(np.median(row['gw_seconds']))
        started = time.perf_counter()
        model = BSE(gw)
        reused = model._M is not None
        model._ensure_screening()
        row['bse_preparation_seconds'] = time.perf_counter()-started
        started = time.perf_counter()
        model.run(nroots=5, low_rank=True, batch_columns=8, tol=1e-9, max_cycle=200)
        row.update(bse_seconds=time.perf_counter()-started, screening_reused=reused,
                   bse_roots=model.e.tolist(), residuals=model.info['residual_norms'].tolist(),
                   screening_modes=len(model.e_rpa))
        print(name, 'BSE', row['bse_seconds'], 'reuse', reused, flush=True)
    if not args.new_only:
        old, new = [report['methods'][name] for name in methods]
        qp_error = float(np.max(np.abs(np.array(old['qp_energies'])-new['qp_energies'])))
        bse_error = float(np.max(np.abs(np.array(old['bse_roots'])-new['bse_roots'])))
        report.update(max_qp_difference=qp_error, max_bse_difference=bse_error,
                      gw_speedup=old['median_gw_seconds']/new['median_gw_seconds'])
        path.write_text(json.dumps(report, indent=2))
        plot(report, args.output)
        assert qp_error < 1e-8 and bse_error < 1e-8
    path.write_text(json.dumps(report, indent=2))
    plot(report, args.output)


if __name__ == '__main__':
    main()
