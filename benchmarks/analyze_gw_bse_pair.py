"""Archive a matched eight-thread molecular GW/BSE pair and plot diagnostics."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np


def analyze(results, output):
    inp = json.loads((results/'input.json').read_text())
    for name, digest in inp['input_hashes'].items():
        assert hashlib.sha256((results/name).read_bytes()).hexdigest() == digest
    rows = [json.loads((results/f'{solver}-8-1.json').read_text()) for solver in ['PyQED', 'PySCF']]
    control = json.loads((results/'PySCF-8-common.json').read_text())
    assert all(r['success'] and r['threads']==8 for r in [*rows, control])
    np.testing.assert_array_equal(rows[0]['qp'], control['qp'])
    own = np.asarray(rows[0]['excitation_eV'])-rows[1]['excitation_eV']
    shared = np.asarray(rows[0]['excitation_eV'])-control['excitation_eV']
    qp = (np.asarray(rows[0]['qp'])-rows[1]['qp'])*27.211386245988
    native_fit = np.load(results/'PyQED-8-1-fit.npz')
    reference_fit = np.load(results/'PySCF-8-1-fit.npz')
    sigma = native_fit['sigma']
    other = reference_fit['sigma'].copy()
    # Occupied native orbitals sample the lower half-plane; PySCF samples
    # the upper half-plane at the same Fermi-shifted frequencies.
    np.testing.assert_allclose(abs(native_fit['omega'].imag),
        np.broadcast_to(reference_fit['omega'].imag, sigma.shape), atol=1e-14)
    np.testing.assert_allclose(reference_fit['omega'].real, native_fit['ef'], atol=1e-14)
    lower = native_fit['omega'][:,1].imag < 0
    other[lower] = other[lower].conj()
    report = dict(input=inp, pyqed=rows[0], pyscf=rows[1], common_qp_control=control,
        excitation_difference_eV=own.tolist(), common_qp_difference_eV=shared.tolist(),
        max_qp_difference_eV=float(max(abs(qp))),
        frontier_qp_difference_eV=qp[inp['nocc']-2:inp['nocc']+2].tolist(),
        max_qp_difference_orbital=int(np.argmax(abs(qp))),
        sigma_max_difference_Ha=float(np.max(abs(sigma-other))),
        total_ratio=rows[0]['total_seconds']/rows[1]['total_seconds'])
    repeated = {}
    for solver in ['PyQED', 'PySCF']:
        entries = [json.loads(p.read_text()) for p in sorted(results.glob(f'{solver}-8-common*.json'))]
        assert all(r['success'] for r in entries)
        for r in entries:
            np.testing.assert_array_equal(rows[0]['qp'], r['qp'])
        times = [r['stages']['BSE'] for r in entries]
        repeated[solver] = dict(seconds=times, median_seconds=float(np.median(times)),
                               maximum_residual=max(max(r['physical_residuals']) for r in entries))
    report['repeated_common_qp_bse'] = repeated
    for p in results.glob('*-8-common*.json'):
        shutil.copy2(p, output/p.name)
    (output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    for name in ['input.json', 'input_validation.json', 'PyQED-8-1.json', 'PySCF-8-1.json', 'PySCF-8-common.json']:
        shutil.copy2(results/name, output/name)
    return report


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [report['pyqed'], report['pyscf']]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    axes = axes.ravel()
    bottom = np.zeros(2)
    for key, label, color in [('setup_transform', 'MO transform', '#8196a9'), ('GW', 'GW', '#0072B2'), ('BSE', 'BSE', '#E69F00')]:
        values = np.array([r['stages'][key] for r in rows])
        axes[0].bar(range(2), values, bottom=bottom, color=color, label=label)
        bottom += values
    for i, row in enumerate(rows):
        axes[0].text(i, bottom[i]+max(bottom)*.03, f"{row['total_seconds']:.2f} s", ha='center')
    axes[0].set(xticks=[0,1], xticklabels=['PyQED', 'PySCF 2.14'], ylabel='Wall time (s)', ylim=(0,max(bottom)*1.22), title='Matched GW + full BSE · 8 threads')
    axes[0].legend(fontsize=8)
    roots = np.arange(1,len(rows[0]['excitation_eV'])+1)
    for key, label, marker in [('excitation_difference_eV', 'Independent GW energies', 'o'), ('common_qp_difference_eV', 'Identical GW energies', 's')]:
        axes[1].plot(roots, np.asarray(report[key])*1e6, marker+'-', label=label)
    axes[1].set(xlabel='Singlet root', ylabel='PyQED − PySCF (µeV)', xticks=roots, title='Excitation energy agreement')
    axes[1].legend(fontsize=8)
    for row, label in zip(rows, ['PyQED', 'PySCF']):
        axes[2].semilogy(roots, row['physical_residuals'], 'o-', label=label)
    axes[2].axhline(1e-6, color='gray', ls='--', label='Requested tolerance')
    axes[2].set(xlabel='Singlet root', ylabel='Fresh physical XY residual (Ha)', xticks=roots, title='Independent convergence audit')
    axes[2].legend(fontsize=8)
    for index, solver in enumerate(['PyQED', 'PySCF']):
        values = report['repeated_common_qp_bse'][solver]['seconds']
        median = np.median(values)
        axes[3].bar(index, median, color=['#0072B2', '#009E73'][index], width=.6)
        axes[3].scatter([index]*len(values), values, color='black', s=20)
        axes[3].text(index, max(values)+.2, f'{median:.2f} s', ha='center')
    upper = max(max(r['seconds']) for r in report['repeated_common_qp_bse'].values())
    axes[3].set(xticks=[0,1], xticklabels=['PyQED', 'PySCF 2.14'], ylabel='BSE wall time (s)',
                ylim=(0,upper*1.2), title='Shared GW energies · BSE-only repeats')
    inp = report['input']
    fig.suptitle(f"{inp['molecule'].capitalize()} / {inp['basis']} — {inp['nao']} orbitals, BSE dimension {inp['full_bse_dimension']:,}")
    fig.savefig(output/f"{inp['molecule']}_gw_bse_8_threads.png", dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plot-only', action='store_true')
    args = p.parse_args();args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.output/'comparison.json').read_text()) if args.plot_only else analyze(args.results, args.output)
    plot(report, args.output)
    print(json.dumps({k:v for k,v in report.items() if k not in ['input','pyqed','pyscf','common_qp_control']}, indent=2))
