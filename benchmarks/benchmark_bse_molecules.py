"""Real-molecule G0W0/full-BSE solver qualification against dense references.

Exact-integral references use PySCF RHF; --integrals cd uses PyQED RHF
and AO Cholesky factors throughout GW/BSE. Both use exact-frequency
G0W0/TDH screening. --basis selects the orbital basis (default cc-pVDZ). Dense A/B matrices and full spectra are reference calculations only.
Solver timings exclude SCF, GW, screening, compilation, and dense validation;
the public solvers' own post-normalization checks remain timed. No persistent
wavefunction database is produced. Use a temporary output directory and the
repository's single-thread BLAS/OpenMP environment.
"""
import argparse
import contextlib
import hashlib
import json
import signal
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import eig, eigvalsh
from pyscf import gto, scf, ao2mo, __version__ as pyscf_version
from pyqed.gw.gw import GW
from pyqed.gw.bse import BSE, bse_AB_matrices, solve_bse, _bse_full_matmat

CASES = {
    'LiH': 'Li 0 0 0; H 0 0 1.60',
    'HF': 'H 0 0 0; F 0 0 .917',
    'H2O': 'O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
    'NH3': 'N 0 0 .116; H 0 .939 -.271; H .8132 -.4695 -.271; H -.8132 -.4695 -.271',
    'N2': 'N 0 0 0; N 0 0 1.098',
    'N2_stretched': 'N 0 0 0; N 0 0 1.80',
    # Idealized planar D2h geometry shared with benchmark_pyrazine_rhf.py.
    'pyrazine': ('N 0 1.340 0; C 1.160 .670 0; C 1.160 -.670 0; '
                 'N 0 -1.340 0; C -1.160 -.670 0; C -1.160 .670 0; '
                 'H 2.095 1.210 0; H 2.095 -1.210 0; '
                 'H -2.095 -1.210 0; H -2.095 1.210 0'),
}
METHODS = ['arpack', 'davidson', 'davidson_batched']


def dense_reference_blocks(model):
    """Independent dense A/B equations, for benchmark validation only."""
    no = model.nocc
    nv = model.nso-no
    o, v = slice(None, no), slice(no, None)
    f = model._pair_factors
    if f is not None:
        direct = np.einsum('Pia,Pjb->iajb', f[:, o, v], f[:, o, v], optimize=True)
        a = 2*direct-np.einsum('Pab,Pij->iajb', f[:, v, v], f[:, o, o], optimize=True)
        b = 2*direct-np.einsum('Paj,Pib->iajb', f[:, v, o], f[:, o, v], optimize=True)
    else:
        eri = model.eri
        a = 2*eri[v,o,v,o].transpose(1,0,3,2)-eri[v,v,o,o].transpose(2,0,3,1)
        b = 2*eri[v,o,o,v].transpose(1,0,2,3)-eri[v,o,o,v].transpose(2,0,1,3)
    m = model._M
    a += 4*np.einsum('ijL,abL,L->iajb', m[o,o], m[v,v], 1/model.e_rpa, optimize=True)
    b += 4*np.einsum('ibL,ajL,L->iajb', m[o,v], m[v,o], 1/model.e_rpa, optimize=True)
    a, b = a.reshape(no*nv, no*nv), b.reshape(no*nv, no*nv)
    a[np.diag_indices(no*nv)] += (model.e_qp[v][None, :]-model.e_qp[o][:, None]).ravel()
    return a, b


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases = report['cases']
    labels = [c['name'].replace('_', '\n') for c in cases]
    colors = ['#0072B2', '#D55E00', '#009E73']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout='constrained')
    for j, (method, color) in enumerate(zip(METHODS, colors)):
        x = np.arange(len(cases))+(j-1)*.23
        times, errors, counts = [], [], []
        for case in cases:
            row = case.get('solvers', {}).get(method, {})
            good = row.get('passed', False)
            times.append(row.get('median_seconds', np.nan) if good else np.nan)
            errors.append(max([s['max_residual'] for s in row.get('samples', []) if s['passed']], default=np.nan) if good else np.nan)
            counts.append(row.get('median_operator_columns', np.nan) if good else np.nan)
        axes[0].bar(x, times, .23, color=color, label=method.replace('_', ' '))
        axes[1].scatter(x, errors, color=color, marker=['o','s','^'][j], s=32)
        axes[2].bar(x, counts, .23, color=color)
        for i, seconds in enumerate(times):
            if not np.isfinite(seconds):
                axes[0].text(x[i], .02, '×', transform=axes[0].get_xaxis_transform(),
                             ha='center', color=color, fontsize=15)
    for ax in axes:
        ax.set_xlim(-.5, len(cases)-.5)
        ax.set_xticks(np.arange(len(cases)), labels, fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    if not any(r.get('passed', False) for c in cases for r in c.get('solvers', {}).values()):
        axes[0].set_ylim(1e-4, 1.)
    axes[0].set_yscale('log'); axes[0].set_ylabel('Median solve time (s)')
    valid_times = [r['median_seconds'] for c in cases for r in c.get('solvers', {}).values() if r.get('passed')]
    if valid_times:
        axes[0].set_ylim(10**np.floor(np.log10(min(valid_times))),
                         10**np.ceil(np.log10(max(valid_times))))
    axes[1].set_yscale('log'); axes[1].set_ylabel('Maximum normalized residual (Ha)')
    axes[1].axhline(report['tolerance'], color='black', ls='--', lw=1)
    axes[2].set_ylabel('Median operator columns')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle(f"G₀W₀/BSE, {report['basis']}, {report.get('integrals', 'exact')}: five roots, single thread; × = failed validation")
    for ext in ['png', 'pdf']:
        fig.savefig(output/f'bse_molecule_comparison.{ext}', dpi=300)
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    for i, case in enumerate(cases):
        for root, energy in enumerate(case.get('reference_roots', [])):
            ax.plot(i+(root-2)*.055, energy*27.211386245988, ['o', 's', '^', 'D', 'v'][root], color=plt.get_cmap('tab10')(root), ms=5,
                    label=f'Root {root+1}' if i == 0 else None)
    ax.set_xticks(np.arange(len(cases)), labels)
    ax.set_ylabel('Dense-reference excitation energy (eV)')
    ax.set_title(f"Lowest full-BSE excitations, G₀W₀/{report['basis']}")
    ax.spines[['top', 'right']].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center', ncol=5, frameon=False)
    for ext in ['png', 'pdf']:
        fig.savefig(output/f'bse_molecule_excitation_energies.{ext}', dpi=300)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    p.add_argument('--basis', default='cc-pvdz')
    p.add_argument('--integrals', choices=['exact', 'cd'], default='exact')
    p.add_argument('--cd-tol', type=float, default=1e-10)
    p.add_argument('--solver-timeout', type=float, default=120.)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--plot-only', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'report.json'
    if args.plot_only:
        plot(json.loads(path.read_text()), args.output)
        return
    report = dict(basis=args.basis, integrals=args.integrals, cd_tolerance=args.cd_tol,
        mean_field='PyQED RHF' if args.integrals == 'cd' else 'PySCF RHF', method='PyQED G0W0/TDH full BSE',
        solver_timeout=args.solver_timeout, eta=1e-3, tolerance=1e-9, roots=5, max_cycle=200, repeats=args.repeats,
        scipy=scipy.__version__, numpy=np.__version__, pyscf=pyscf_version,
        source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                      ['pyqed/gw/bse.py', 'pyqed/gw/gw.py', 'pyqed/linalg/nonsymmetric.py',
                       'pyqed/linalg/nonsymmetric_davidson.hpp', __file__]}, cases=[])
    for name in args.cases:
        case = dict(name=name, geometry_angstrom=CASES[name], solvers={})
        report['cases'].append(case)
        print(name, 'preparing RHF/G0W0/screening', flush=True)
        try:
            start = time.perf_counter()
            with (args.output/f'{name}_preparation.log').open('w', buffering=1) as log, contextlib.redirect_stdout(log):
                if args.integrals == 'cd':
                    from pyqed.qchem import Molecule
                    mol = Molecule(atom=CASES[name], basis=args.basis, unit='angstrom')
                    mol.build(eri='cd', options={'low_rank_tol': args.cd_tol,
                                                'eri_screen_tol': 0.0, 'workers': 1})
                    case['integral_seconds'] = time.perf_counter()-start
                    stage = time.perf_counter()
                    mf = mol.RHF().run(tol=1e-11, max_cycle=100, verbose=0)
                else:
                    mol = gto.M(atom=CASES[name], basis=args.basis, unit='Angstrom', verbose=0)
                    case['integral_seconds'] = time.perf_counter()-start
                    stage = time.perf_counter()
                    mf = scf.RHF(mol)
                    mf.chkfile = None
                    mf.conv_tol = 1e-11
                    mf.kernel()
                case['rhf_seconds'] = time.perf_counter()-stage
                print('RHF complete', case['rhf_seconds'], flush=True)
                if not mf.converged:
                    raise RuntimeError('RHF did not converge')
                stage = time.perf_counter()
                gw = GW(mf, ao2mofn=None if args.integrals == 'cd' else ao2mo.general,
                        screening='TDH', eta=1e-3).run()
                case['gw_seconds'] = time.perf_counter()-stage
                print('GW complete', case['gw_seconds'], flush=True)
                case['factor_rank'] = None if gw._pair_factors is None else len(gw._pair_factors)
                if args.integrals == 'cd' and (gw.eri is not None or gw._pair_factors is None):
                    raise RuntimeError('CD benchmark did not retain factorized integrals')
                if not np.all(np.isfinite(gw.e_qp)):
                    raise RuntimeError('Nonfinite quasiparticle energies')
                model = BSE(gw)
                model._ensure_screening()
                a, b = dense_reference_blocks(model)
            h = np.block([[a, b], [-b, -a]])
            w, v = eig(h)
            stable = np.max(abs(w.imag)) < 1e-8 and min(eigvalsh(a-b)[0], eigvalsh(a+b)[0]) > 0
            positive = np.sort(w.real[(w.real > 0) & (abs(w.imag) < 1e-8)])
            reference = positive[:5]
            if len(reference) != 5:
                raise RuntimeError('Dense reference lacks five positive real roots')
            trials = np.random.default_rng(682).normal(size=(len(h), 7))
            action_error = float(np.max(abs(_bse_full_matmat(model, trials, 3)-h@trials)))
            if action_error > 1e-10:
                raise RuntimeError(f'Dense A/B and production action disagree: {action_error}')
            case.update(qp_fallbacks=(args.output/f'{name}_preparation.log').read_text().count('Newton-Raphson unconverged'),
                dimension=len(h), scf_energy=float(mf.e_tot), qp_energies=gw.e_qp.tolist(),
                reference_roots=reference.tolist(), stable=bool(stable),
                min_a_minus_b=float(eigvalsh(a-b)[0]), min_a_plus_b=float(eigvalsh(a+b)[0]),
                max_spectrum_imaginary=float(max(abs(w.imag))), action_error=action_error,
                preparation_seconds=time.perf_counter()-start)
            path.write_text(json.dumps(report, indent=2))
            print(name, 'reference ready', case['dimension'], 'dimensions', flush=True)
            def timeout(signum, frame):
                raise TimeoutError(f'Solver exceeded {args.solver_timeout:g} seconds')
            signal.signal(signal.SIGALRM, timeout)
            def run(method):
                signal.setitimer(signal.ITIMER_REAL, args.solver_timeout)
                started = time.perf_counter()
                try:
                    roots, vectors, info = solve_bse(model, nroots=5, tol=1e-9, max_cycle=200,
                        eigensolver='arpack' if method == 'arpack' else 'davidson',
                        batch_columns=8 if method == 'davidson_batched' else None, return_info=True)
                    seconds = time.perf_counter()-started
                    error = float(max(abs(roots-reference)))
                    residual = float(max(np.linalg.norm(h@vectors-vectors*roots, axis=0)))
                    passed = bool(info['converged'] and stable and error < 1e-8 and residual <= 1e-9 and info['metric_error'] <= 1e-8)
                    return dict(passed=passed, seconds=seconds, roots=roots.tolist(), max_error=error,
                        max_residual=residual, metric_error=info['metric_error'], operator_columns=info['operator_columns'],
                        iterations=info.get('davidson', {}).get('iterations'),
                        message='' if passed else 'Dense spectrum/stability/residual validation failed')
                except Exception as exc:
                    return dict(passed=False, seconds=time.perf_counter()-started,
                        message=f'{type(exc).__name__}: {exc}')
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
            for method in METHODS:
                warmup = run(method)
                case['solvers'][method] = dict(warmup=warmup, samples=[])
                print(name, method, 'warmup', warmup['passed'], warmup.get('message'), flush=True)
            for repeat in range(args.repeats):
                for method in np.random.default_rng(880+repeat).permutation(METHODS):
                    if not case['solvers'][method]['warmup']['passed']:
                        continue
                    sample = run(method)
                    case['solvers'][method]['samples'].append(sample)
                    print(name, method, 'repeat', repeat+1, sample['passed'], sample['seconds'], flush=True)
                    path.write_text(json.dumps(report, indent=2))
            for method, row in case['solvers'].items():
                row['passed'] = all(s['passed'] for s in row['samples']) and row['warmup']['passed']
                if row['passed']:
                    row['median_seconds'] = float(np.median([s['seconds'] for s in row['samples']]))
                    row['median_operator_columns'] = float(np.median([s['operator_columns'] for s in row['samples']]))
                print(name, method, 'PASS' if row['passed'] else 'FAIL', row.get('median_seconds'), flush=True)
        except Exception as exc:
            case['preparation_error'] = f'{type(exc).__name__}: {exc}'
            print(name, case['preparation_error'], flush=True)
        path.write_text(json.dumps(report, indent=2))
    plot(report, args.output)


if __name__ == '__main__':
    main()
