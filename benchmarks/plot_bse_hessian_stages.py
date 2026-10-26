"""Plot stage timing and sampled process memory from an adjoint Hessian run."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output
    run = json.loads((out/'adjoint.json').read_text())
    h = np.load(out/'adjoint_hessian.npy')[0]
    selected = run['columns']
    stages = run['derivative_info']['timings']
    measurements = run['stage_measurements']
    translation = np.tile(np.eye(3), (h.shape[0]//3, 1))
    events = run.get('stage_events', [])
    column_times = {name: [b['seconds']-a['seconds'] for a,b in zip(events, events[1:])
                          if a['stage'] == name] for name in ('reverse_response', 'integral_curvature')}
    diagnostics = dict(
        column_wall_seconds=column_times,
        shape=list(h.shape), finite=bool(np.isfinite(h).all()),
        max_selected_asymmetry=float(np.max(abs(h[selected]-h[selected].T))),
        max_translation_residual=float(np.max(abs(h.T@translation))),
        bottleneck=max(stages, key=stages.get),
        stage_wall_fractions={k:v/run['seconds'] for k,v in stages.items()},
        memory_note='Sampled process RSS includes shared reference data; stage peaks are not additive.')
    (out/'stage_diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(stages)
    labels = ['First\nresponse', 'Adjoint\nresponse', 'Integral\ncurvature']
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    wall = [stages[k] for k in names]
    cpu = [measurements[k]['cpu_seconds'] for k in names]
    for offset, values, label, color in ((-.18, wall, 'Wall', '#0072B2'),
                                         (.18, cpu, 'CPU', '#D55E00')):
        bars = axes[0].bar(x+offset, values, width=.36, label=label, color=color)
        axes[0].bar_label(bars, fmt='%.1f', padding=3, fontsize=8)
    axes[0].set(xticks=x, xticklabels=labels, ylabel='Time (s)', title='a  Stage time')
    axes[0].legend(frameon=False)
    axes[0].margins(y=.25)
    bars = axes[1].bar(x, [measurements[k]['peak_rss_bytes']/2**30 for k in names], color='#009E73')
    axes[1].bar_label(bars, fmt='%.2f', padding=3)
    axes[1].set(xticks=x, xticklabels=labels, ylabel='Peak process RSS (GiB)', title='b  Stage memory')
    axes[1].margins(y=.2)
    axes[2].axis('off')
    axes[2].set_title('c  Run and consistency checks')
    lines = [f"Total wall: {run['seconds']:.1f} s", f"Peak RSS: {run['peak_rss_bytes']/2**30:.3f} GiB",
             f"Workers: {run['workers']}; BLAS threads: {run['blas_threads']}",
             f"Selected asymmetry: {diagnostics['max_selected_asymmetry']:.2e}",
             f"Translation residual: {diagnostics['max_translation_residual']:.2e}",
             'Residual units: Hartree/bohr²', 'State 0: first excited total-energy surface',
             'Reference setup excluded from timing']
    for i, line in enumerate(lines):
        axes[2].text(0, .94-.115*i, line, transform=axes[2].transAxes, fontsize=9)
    for ax in axes[:2]:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f"Pyrazine/{run['basis']}: {h.shape[0]} × {h.shape[1]} Z-vector Hessian")
    for extension in ('png', 'pdf'):
        fig.savefig(out/f'pyrazine_hessian_stages.{extension}', dpi=300)
    print(json.dumps(diagnostics, indent=2))


if __name__ == '__main__':
    main()
