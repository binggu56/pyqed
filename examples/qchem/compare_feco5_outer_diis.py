#!/usr/bin/env python3
"""Plot and summarize controlled Fe(CO)5 outer-DIIS calculation outputs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def compare(baseline, revised, output, *, restart=False, experimental=None,
            revised_label=None):
    output.mkdir(parents=True, exist_ok=True)
    folders = [baseline, revised] + ([experimental] if experimental is not None else [])
    records = []
    for folder in folders:
        path = folder / 'result.json'
        if not path.exists():
            path = folder / 'progress.json'
        records.append(json.loads(path.read_text()))
    keys = ('basis', 'atom_angstrom', 'ncas', 'nelecas', 'cd_tol', 'macro_tol',
            'orbital_gradient_tol', 'ci_tol', 'optimizer', 'optimizer_max_steps', 'diis')
    for key in keys:
        if any(record[key] != records[0][key] for record in records[1:]):
            raise ValueError(f'Calculation settings differ: {key}')
    offsets = [json.loads((folder / 'config.json').read_text()).get('continued_after_macro', 0)
               for folder in folders]
    if len(set(offsets)) != 1 or bool(offsets[0]) != restart:
        raise ValueError('Both calculations must have the same initial macro offset')
    offset = offsets[0]
    for folder, record in zip(folders[1:], records[1:]):
        if restart:
            np.testing.assert_array_equal(np.load(baseline / 'orbitals_initial.npy'),
                                          np.load(folder / 'orbitals_initial.npy'))
        with np.load(baseline / 'rhf_orbitals.npz') as a, np.load(folder / 'rhf_orbitals.npz') as b:
            np.testing.assert_array_equal(a['mo_coeff'], b['mo_coeff'])
            np.testing.assert_array_equal(a['mo_energy'], b['mo_energy'])
        np.testing.assert_allclose(records[0]['energy_history_hartree'][offset],
                                   record['energy_history_hartree'][offset], atol=1e-10, rtol=0)
    if revised_label is None:
        revised_label = {'step': 'Original displacement DIIS',
                         'transported_step': 'Aligned displacement DIIS',
                         'gradient': 'Post-CI gradient DIIS'}.get(
                             records[1].get('diis_residual'), 'Revised outer DIIS')
    labels = ['Previous displacement DIIS', revised_label]
    if experimental is not None:
        labels.append('Post-CI gradient DIIS')
    summary = {'settings': {key: records[0][key] for key in keys}, 'same_initial_orbitals_verified': True, 'initial_macro_offset': offset,
               'timing_note': 'Runs may overlap; compare macrosteps, not wall-time speedup.', 'variants': {}}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    initial = records[0]['energy_history_hartree'][offset]
    for label, record in zip(labels, records):
        energies = np.asarray(record['energy_history_hartree'][offset:])
        rows = [row for row in record['macro_diagnostics'] if row.get('accepted') and 'gn' in row and row['macro'] > offset]
        macros = [row['macro'] - offset for row in rows]
        axes[0].plot(np.arange(len(energies)), energies - initial, 'o-', ms=3, label=label)
        axes[1].semilogy(macros, [row['gn'] for row in rows], 'o-', ms=3, label=label)
        axes[2].semilogy(np.arange(1, len(energies)), np.maximum(abs(np.diff(energies)), 1e-16),
                         'o-', ms=3, label=label)
        summary['variants'][label] = dict(
            accepted_macros=len(rows), energy_hartree=float(energies[-1]),
            final_gradient_norm=rows[-1]['gn'],
            macro_converged=record.get('macro_converged', False),
            solver_converged=record.get('solver_converged'),
            rejected_trials=sum(row.get('rej', 0) for row in rows),
            casci_trials=len(rows) + sum(row.get('rej', 0) for row in rows),
            diis_rejected_macros=sum(row.get('diis_rejected', False) for row in rows),
            extrapolated_macros=sum(row.get('diis_used', False) for row in rows)
                                if any('diis_used' in row for row in rows) else None,
        )
    axes[0].set(xlabel='Accepted CO macrostep', ylabel='Energy change from initial CASCI ($E_h$)')
    axes[1].set(xlabel='Accepted CO macrostep', ylabel='Physical orbital gradient norm ($E_h$)')
    axes[2].set(xlabel='Accepted CO macrostep', ylabel='Absolute energy change per step ($E_h$)')
    axes[1].axhline(records[0]['orbital_gradient_tol'], color='gray', ls='--', label='Gradient tolerance')
    axes[2].axhline(records[0]['macro_tol'], color='gray', ls='--', label='Energy tolerance')
    for axis in axes:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    start_label = 'same saved CO start' if restart else 'same RHF frontier start'
    converged_count = sum(bool(record.get('macro_converged', False)) for record in records)
    fig.suptitle(f'Fe(CO)₅ · def2-SVP · COCAS(8,8) · CD 10⁻⁸ · {start_label}\n'
                 f'CO runs converged: {converged_count}/{len(records)}')
    name = 'feco5_cocas88_outer_diis_restart_comparison' if restart else 'feco5_cocas88_outer_diis_comparison'
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'{name}.{suffix}', dpi=170)
    plt.close(fig)
    (output / ('restart_comparison.json' if restart else 'comparison.json')).write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--revised', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--experimental', type=Path, help='Include the optional gradient-residual experiment')
    parser.add_argument('--revised-label', help='Explicit label for an archived driver variant')
    parser.add_argument('--restart', action='store_true', help='Compare continuations from the same saved CO orbitals')
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.revised, args.output, restart=args.restart, experimental=args.experimental, revised_label=args.revised_label), indent=2))
