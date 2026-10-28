"""Compare completed same-checkpoint molecular CASSCF memory reports."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def compare(before,after,output,before_label='Previous workspace',after_label='Blocked workspace',
            figure_name='pentacene_blocked_memory_comparison.png'):
    rows=[json.loads(path.read_text()) for path in (before,after)]
    assert rows[0]['config'] == rows[1]['config']
    for row in rows:
        assert row['converged'] and row['failure'] is None
        assert row['workspace_calls']['applied'] > 0
        np.testing.assert_allclose(row['spin_square'],0,atol=1e-8,rtol=0)
    delta=np.asarray(rows[1]['energies'])-rows[0]['energies']
    np.testing.assert_allclose(delta,0,atol=1e-7,rtol=0)
    output.mkdir(parents=True,exist_ok=True)
    report=dict(before_report=str(before),after_report=str(after),
        before_rss_gib=rows[0]['peak_process_rss_gib'],after_rss_gib=rows[1]['peak_process_rss_gib'],
        before_seconds=rows[0]['casscf_seconds'],after_seconds=rows[1]['casscf_seconds'],
        max_energy_change_hartree=float(np.max(abs(delta))))
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(10,6.8),layout='constrained')
    labels=(before_label,after_label)
    for label,row in zip(labels,rows):
        t,rss=np.asarray(row['rss_samples']).T
        axes[0,0].plot(t/60,rss,label=label)
    axes[0,0].set(xlabel='SA-CASSCF elapsed (min)',ylabel='Process RSS (GiB)')
    axes[0,0].legend()
    axes[0,1].bar([v.split()[0] for v in labels],[r['peak_process_rss_gib'] for r in rows])
    axes[0,1].set(ylabel='Peak process RSS (GiB)')
    axes[1,0].bar([v.split()[0] for v in labels],[r['casscf_seconds']/60 for r in rows])
    axes[1,0].set(ylabel='SA-CASSCF time (min)')
    axes[1,1].plot(range(6),delta,'o-')
    axes[1,1].set(xlabel='Singlet root',ylabel='After − before energy (hartree)')
    fig.suptitle('Pentacene dimer: RI SA6 singlet CAS(4e,4o), same RHF checkpoint')
    fig.savefig(output/figure_name,dpi=160)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--before-label',default='Previous workspace')
    p.add_argument('--after-label',default='Blocked workspace')
    p.add_argument('--figure-name',default='pentacene_blocked_memory_comparison.png')
    args=p.parse_args()
    compare(args.before,args.after,args.output,args.before_label,args.after_label,args.figure_name)
