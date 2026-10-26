"""Compare saved early/late projected integral runs and draw diagnostics."""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
p.add_argument('--control', default='late')
p.add_argument('--candidate', default='early')
a = p.parse_args()
root = a.directory
labels = [a.control, a.candidate]
reports = [json.loads((root / f'{tag}.json').read_text()) for tag in labels]
with np.load(root / f'{a.control}.npz') as old, np.load(root / f'{a.candidate}.npz') as new:
    np.testing.assert_array_equal(old['pairs'], new['pairs'])
    np.testing.assert_array_equal(old['pivots'], new['pivots'])
    error = np.abs(old['curvature'] - new['curvature'])
    np.testing.assert_allclose(old['curvature'], new['curvature'], atol=2e-11, rtol=2e-11)
first_error = 0.
if all((root / f'{tag}-first.npy').exists() for tag in labels):
    old, new = [np.load(root / f'{tag}-first.npy', mmap_mode='r') for tag in labels]
    assert old.shape == new.shape
    for mode in range(len(old)):
        for start in range(0, old.shape[1], 32):
            x, y = old[mode,start:start+32], new[mode,start:start+32]
            np.testing.assert_allclose(x,y,atol=2e-11,rtol=2e-11)
            first_error = max(first_error,float(np.max(np.abs(x-y))))
else:
    assert reports[0]['first_hash'] == reports[1]['first_hash']
summary = {'curvature_speedup': reports[0]['stage_seconds']['curvature'] / reports[1]['stage_seconds']['curvature'],
           'first_speedup': reports[0]['stage_seconds']['first'] / reports[1]['stage_seconds']['first'],
           'max_absolute_error': float(error.max()), 'first_max_absolute_error': first_error,
           'first_columns_bitwise_equal': reports[0]['first_hash'] == reports[1]['first_hash']}
(root / 'comparison.json').write_text(json.dumps(summary, indent=2))
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout='constrained')
colors = ['#7B8CA3', '#167D9A']
for i, stage in enumerate(('first', 'curvature')):
    x = np.arange(2) + (i - .5) * .32
    axes[0].bar(x, [r['stage_seconds'][stage] for r in reports], .32, label=stage, color=colors[i])
axes[0].set(xticks=[0,1], xticklabels=labels, ylabel='Wall time (s)')
axes[0].set_ylim(0,1.35*max(max(r['stage_seconds'].values()) for r in reports))
axes[0].legend(frameon=False)
for i, field in enumerate(('process_peak_rss', 'curvature')):
    values = [r[field] if i == 0 else r['stage_peak_rss'][field] for r in reports]
    axes[1].bar(np.arange(2) + (i-.5)*.32, np.array(values)/2**20,
                .32, color=colors[i], label='Overall' if i == 0 else 'Curvature')
axes[1].set(xticks=[0,1], xticklabels=labels, ylabel='Peak RSS (MiB)')
axes[1].set_ylim(0, 1.35 * max(r['process_peak_rss']/2**20 for r in reports))
axes[1].legend(frameon=False)
axes[2].plot(np.arange(1,len(error)+1), error, '.', color=colors[1])
axes[2].set(xlabel='Selected mode pair', ylabel='Absolute contraction difference')
fig.suptitle('Pyrazine/cc-pVDZ: 22 dimensionless modes, 3 columns\nSynthetic integral sensitivity; not a complete GW–BSE Hessian', fontsize=11)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
for ext in ('png','pdf'):
    fig.savefig(root / f'projected_curvature.{ext}', dpi=300)
print(json.dumps(summary, indent=2))

weight_paths = [root / f'weights-{tag}.json' for tag in labels]
if all(path.exists() for path in weight_paths):
    weight_reports = [json.loads(path.read_text()) for path in weight_paths]
    differences = {}
    for name in ('raw','row','connection'):
        arrays = [np.load(path.with_name(path.stem+'-'+name+'.npy'),mmap_mode='r') for path in weight_paths]
        assert arrays[0].shape == arrays[1].shape
        difference = 0.
        for start in range(0,len(arrays[0]),32):
            x,y = [array[start:start+32] for array in arrays]
            np.testing.assert_allclose(x,y,atol=2e-10,rtol=2e-11)
            difference = max(difference,float(np.max(np.abs(x-y))))
        differences[name] = difference
    assert weight_reports[0]['hashes'][3:] == weight_reports[1]['hashes'][3:]
    summary['weight_max_absolute_errors'] = differences
    summary['weight_preparation_speedup'] = weight_reports[0]['wall_seconds']/weight_reports[1]['wall_seconds']
    (root / 'comparison.json').write_text(json.dumps(summary,indent=2))
    fig,axes = plt.subplots(1,2,figsize=(7,3.5),layout='constrained')
    axes[0].bar(labels,[r['wall_seconds'] for r in weight_reports],color=colors)
    axes[0].set_ylabel('Weight preparation (s)')
    axes[1].bar(labels,[r['peak_rss']/2**20 for r in weight_reports],color=colors)
    axes[1].set_ylabel('Stage peak RSS (MiB)')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Synthetic curvature-weight preparation\n1848 factors, 104 AOs; three directions, one column',fontsize=11)
    for ext in ('png','pdf'):
        fig.savefig(root / f'curvature_weights.{ext}',dpi=300)
    print(json.dumps(summary,indent=2))
