"""Validate and plot an interleaved gw_openmp_phases.py comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    data = json.loads((args.directory/'results.json').read_text())
    rows = data['rows']
    reference = next(r for r in rows if r['variant'] == 'baseline' and r['threads'] == 1)
    errors = {}
    for key in ('qp', 'weights', 'sigma_real', 'sigma_imag'):
        errors[key] = max(float(np.max(np.abs(np.asarray(r[key])-reference[key]))) for r in rows)
        assert errors[key] < 1e-9, (key, errors[key])
    errors['residuals'] = max(float(np.max(np.abs(r['residuals']))) for r in rows)
    assert errors['residuals'] < 1e-9
    errors['kernel_max_absolute'] = 0.
    for r in data['kernels']:
        if r['variant'] != 'optimized':
            continue
        name = f"{r['kernel']}-{r['rank']}.npy"
        before = np.load(args.directory/f'baseline-{name}')
        after = np.load(args.directory/f'optimized-{name}')
        np.testing.assert_allclose(after, before, atol=1e-11, rtol=1e-11)
        errors['kernel_max_absolute'] = max(errors['kernel_max_absolute'], float(np.max(np.abs(after-before))))
        assert r['max_thread_difference'] < 1e-11
    variants = ('baseline', 'optimized')
    medians = {v: {str(t): float(np.median([r['seconds'] for r in rows
                                         if r['variant'] == v and r['threads'] == t]))
                   for t in (1, 4)} for v in variants}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout='constrained')
    colors = ('#8996a8', '#147d92')
    x = np.arange(2)
    for j, (v, color) in enumerate(zip(variants, colors)):
        axes[0].bar(x+(j-.5)*.35, list(medians[v].values()), .35, label=v, color=color)
    axes[0].set(xticks=x, xticklabels=['1 thread', '4 threads'], ylabel='Seconds (median of 3)',
                title='Water/cc-pVDZ: complete GW roots')
    axes[0].legend(frameon=False)
    for j, (v, color) in enumerate(zip(variants, colors)):
        times = [np.median(next(r['seconds'] for r in data['kernels'] if r['variant'] == v
                               and r['rank'] == 512 and r['kernel'] == 'dielectric' and r['threads'] == t))
                 for t in (1, 4)]
        axes[1].bar(x+(j-.5)*.35, times, .35, color=color)
    axes[1].set(xticks=x, xticklabels=['1 thread', '4 threads'], ylabel='Seconds (median of 3)',
                title='Complex dielectric: 512 × 8192 factors')
    labels = ['QP energy', 'QP weight', 'Re Σ / derivative', 'Im Σ / derivative']
    values = [errors[k] for k in ('qp', 'weights', 'sigma_real', 'sigma_imag')]
    axes[2].barh(labels, np.maximum(values, 1e-25), color=colors[1])
    axes[2].set(xscale='log', xlabel='Maximum absolute difference', title='All runs vs baseline, 1 thread')
    axes[2].axvline(1e-9, color='#a33b36', linestyle='--', label='Validation threshold')
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle('Fixed screening, quadrature and solver tolerances; BLAS threads = 1', fontsize=12)
    fig.savefig(args.directory/'gw_openmp_comparison.png', dpi=170)
    summary = dict(median_seconds=medians, max_absolute_errors=errors,
                   baseline_qp_hartree=reference['qp'])
    (args.directory/'validation.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
