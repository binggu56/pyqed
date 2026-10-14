"""Archive and plot the matched eight-thread anthracene BSE solver trial."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = ['PyQED-8-response-final', 'PySCF-8-reference-final', 'PyQED-8-lanczos-final']
    labels = ['PyQED Davidson', 'PySCF 2.14', 'PyQED Lanczos']
    rows = [json.loads((args.results/(name+'.json')).read_text()) for name in names]
    summary = {}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout='constrained')
    colors = ['#0072B2', '#009E73', '#D55E00']
    reference = np.asarray(rows[1]['excitation_eV'])
    for index, (label, row, color) in enumerate(zip(labels, rows, colors)):
        converged = row.get('success', False)
        method = row['solver_info'].get('lanczos', row['solver_info'].get('response', row['solver_info']))
        history = method.get('history', [])
        energy = np.asarray(row['excitation_eV'] if converged else history[-1]['energies'])* (1 if converged else 27.211386245988)
        residual = np.asarray(row['physical_residuals'] if converged else history[-1]['residual_norms'])
        seconds = row['stages']['BSE']
        summary[label] = dict(converged=converged, seconds=seconds,
            iterations=method.get('iterations'), subspace=method.get('subspace', history[-1]['subspace']),
            max_residual=float(max(residual)), excitation_eV=energy.tolist(),
            max_energy_difference_eV=float(max(abs(energy-reference))),
            timers=row['timers'], error=row.get('error'))
        axes[0].bar(index, seconds, color=color, hatch='' if converged else '//')
        axes[0].text(index, seconds*1.12, f'{seconds:.2f}s'+('' if converged else '\nunconverged'), ha='center', fontsize=9)
        if history and 'residual_norms' in history[0]:
            axes[1].semilogy([r['iteration'] for r in history], [max(r['residual_norms']) for r in history], color=color, label=label)
        else:
            axes[1].semilogy(method['iterations'], max(residual), 'o', color=color, label=label+' (final audit)')
        axes[2].plot(range(1, 6), energy-reference, 'o-', color=color, label=label+('' if converged else ' (unconverged Ritz values)'))
    axes[0].set(xticks=range(3), xticklabels=['PyQED\nDavidson', 'PySCF\n2.14', 'PyQED\nLanczos'], ylabel='BSE wall time (s, log scale)', title='Screening + solve + internal checks', yscale='log')
    axes[0].set_ylim(top=max(r['seconds'] for r in summary.values())*2.2)
    axes[1].axhline(1e-6, ls='--', color='gray', label='Requested tolerance')
    axes[1].set(xlabel='Block iteration', ylabel='Maximum XY residual (Ha)', title='Convergence')
    axes[1].legend(fontsize=8)
    axes[2].set(xlabel='Singlet root', ylabel='Energy difference from PySCF (eV)', title='Identical saved GW energies', xticks=range(1,6), yscale='symlog')
    axes[2].legend(fontsize=7)
    fig.suptitle('Anthracene / cc-pVDZ — full BSE, 8 threads, shared HF orbitals and W₀')
    fig.savefig(args.output/'anthracene_bse_lanczos_8_threads.png', dpi=170)
    plt.close(fig)
    (args.output/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
    for name in names:
        shutil.copyfile(args.results/(name+'.json'), args.output/(name+'.json'))
    shutil.copyfile(args.results/'input.json', args.output/'input.json')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
