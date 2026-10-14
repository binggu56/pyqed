"""Audit the eight-thread projection optimization and regenerate its figure.

Run with --baseline/--results pointing to raw matched benchmark directories.
Use --plot-only to regenerate the figure from archived comparison.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def analyze(args):
    baseline, results = args.baseline, args.results
    names = ['PyQED-8-1', 'PySCF-8-1']
    before = json.loads((baseline / f'{names[0]}.json').read_text())
    rows = [json.loads((results / f'{name}.json').read_text()) for name in names]
    assert all(r['success'] and r['threads'] == 8 for r in [before, *rows])
    old_input = json.loads((baseline / 'input.json').read_text())
    new_input = json.loads((results / 'input.json').read_text())
    assert old_input['input_hashes'] == new_input['input_hashes']
    for name, digest in new_input['input_hashes'].items():
        assert hashlib.sha256((results / name).read_bytes()).hexdigest() == digest
    old_fit = np.load(baseline / f'{names[0]}-fit.npz')
    new_fit = np.load(results / f'{names[0]}-fit.npz')
    sigma_delta = float(np.max(abs(old_fit['sigma'] - new_fit['sigma'])))
    excitation_delta = np.asarray(rows[0]['excitation_eV']) - before['excitation_eV']
    qp_delta = float(np.max(abs(np.asarray(rows[0]['qp']) - before['qp'])))
    assert sigma_delta < 1e-13 and qp_delta < 1e-9
    assert np.max(abs(excitation_delta)) < args.energy_tolerance
    cross_delta = np.asarray(rows[0]['excitation_eV']) - rows[1]['excitation_eV']
    report = dict(stage=args.stage, energy_tolerance_eV=args.energy_tolerance,
                  before=before, optimized=rows[0], pyscf=rows[1],
                  sigma_max_delta_Ha=sigma_delta, qp_max_delta_Ha=qp_delta,
                  excitation_delta_eV=excitation_delta.tolist(),
                  cross_code_excitation_delta_eV=cross_delta.tolist(),
                  identical_input_hashes=new_input['input_hashes'])
    paired = results / 'paired_bse.json'
    if args.stage == 'BSE' and paired.exists():
        report['paired_bse'] = json.loads(paired.read_text())
        for row in report['paired_bse']:
            if row['stage'] == 'solve':
                assert max(row['residuals']) < 1e-6
                np.testing.assert_allclose(np.asarray(row['energies'])*27.211386245988,
                                           before['excitation_eV'], atol=args.energy_tolerance, rtol=0)
    (args.output / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    for name in names:
        shutil.copy2(results / f'{name}.json', args.output / f'{name}.json')
    shutil.copy2(results / 'input.json', args.output / 'input.json')
    return report


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = [report[k] for k in ['before', 'optimized', 'pyscf']]
    paired = report.get('paired_bse')
    panels = 3 if paired else 2
    fig, axes = plt.subplots(1, panels, figsize=(5*panels, 4.5), constrained_layout=True)
    bottom = np.zeros(3)
    for key, label, color in [('setup_transform', 'MO setup', '#8196a9'),
                              ('GW', 'GW', '#3274a1'), ('BSE', 'BSE', '#e0a045')]:
        values = np.array([r['stages'][key] for r in rows])
        axes[0].bar(range(3), values, bottom=bottom, label=label, color=color, width=.6)
        bottom += values
    for i, r in enumerate(rows):
        axes[0].text(i, bottom[i]+1, f"{r['total_seconds']:.2f} s", ha='center')
    axes[0].set(xticks=range(3), xticklabels=['PyQED\nbefore', 'PyQED\noptimized', 'PySCF\n2.14'],
                ylabel='Wall time (s)', ylim=(0, max(bottom)*1.18), title='End-to-end solve at 8 threads')
    axes[0].legend(frameon=False, loc='upper right')
    if paired:
        times = [[r['seconds'] for r in paired if r['stage']=='solve' and r['label']==label]
                 for label in ['before', 'after']]
        means = [np.mean(v) for v in times]
        axes[1].bar([0, 1], means, color=['#8196a9', '#3274a1'], width=.6)
        for i, values in enumerate(times):
            axes[1].scatter([i]*len(values), values, color='black', s=20)
            axes[1].text(i, max(values)+.15, f'{means[i]:.2f} s', ha='center')
        axes[1].set(xticks=[0, 1], xticklabels=['Before', 'Optimized'], ylabel='BSE solve time (s)',
                    title='Paired BSE-only runs · shared setup', ylim=(0, max(map(max, times))*1.2))
    scale = 1e6 if max(abs(np.asarray(report['excitation_delta_eV']))) > 1e-8 else 1e12
    delta = np.asarray(report['excitation_delta_eV']) * scale
    axes[-1].bar(np.arange(1, 6), delta, color='#3274a1', width=.6)
    axes[-1].axhline(0, color='0.5', linewidth=.8)
    axes[-1].set(xlabel='Singlet excitation', ylabel='After − before (µeV)' if scale==1e6 else 'After − before (10⁻¹² eV)',
                xticks=np.arange(1, 6), title='Excitation energy verification')
    axes[-1].text(.05, .93, f"Max |ΔE| = {max(abs(delta))/scale:.2e} eV\nImaginary self-energy Δ = {report['sigma_max_delta_Ha']:.1e} Ha",
                 transform=axes[-1].transAxes, va='top')
    bound = max(.08, max(abs(delta))*1.8)
    axes[-1].set_ylim(-bound, bound)
    fig.suptitle('Anthracene / cc-pVDZ · matched RI-HF orbitals and HF-gap RPA screening')
    filename = ('anthracene_bse_optimization_8_threads.png' if report.get('stage') == 'BSE'
                else 'anthracene_gw_projection_8_threads.png')
    fig.savefig(output / filename, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--stage', choices=['GW', 'BSE'], default='GW')
    parser.add_argument('--energy-tolerance', type=float, default=1e-8,
                        help='Before/after excitation audit threshold in eV')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        report = json.loads((args.output / 'comparison.json').read_text())
    else:
        if args.baseline is None or args.results is None:
            parser.error('--baseline and --results are required for the numerical audit')
        report = analyze(args)
    plot(report, args.output)
    print(json.dumps({k:v for k,v in report.items() if k not in ['before','optimized','pyscf']}, indent=2))
