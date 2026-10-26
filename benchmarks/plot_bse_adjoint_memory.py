"""Compare completed pyrazine adjoint benchmark directories, including accuracy."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('before',type=Path)
    p.add_argument('after',type=Path)
    p.add_argument('--labels', nargs=2, default=['Previous','Optimized'])
    p.add_argument('--memory-metric', choices=['peak_rss_bytes', 'process_peak_rss_bytes'],
                   default='peak_rss_bytes')
    args=p.parse_args()
    records=[json.loads((d/'adjoint.json').read_text()) for d in (args.before,args.after)]
    refs=[np.load(d/'adjoint_reference.npz') for d in (args.before,args.after)]
    for key in refs[0].files:
        np.testing.assert_allclose(refs[0][key],refs[1][key],atol=1e-10,rtol=0)
    quantity=records[0].get('quantity','gradient')
    assert records[1].get('quantity','gradient') == quantity
    np.testing.assert_array_equal(records[0]['directions'],records[1]['directions'])
    old,new=[np.load(d/('adjoint_'+quantity+'.npy')) for d in (args.before,args.after)]
    forward_path=args.before/('forward_'+quantity+'.npy')
    forward=np.load(forward_path) if forward_path.exists() else None
    result=dict(before=records[0],after=records[1],
        speedup=records[0]['seconds']/records[1]['seconds'],
        memory_metric=args.memory_metric,
        peak_reduction_fraction=1-records[1][args.memory_metric]/records[0][args.memory_metric],
        max_change=float(np.max(abs(new-old))),
        max_forward_difference=float(np.max(abs(new-forward))) if forward is not None else None,
        rms_forward_difference=float(np.sqrt(np.mean((new-forward)**2))) if forward is not None else None)
    (args.after/'memory_comparison.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(10,3.4),layout='constrained')
    for ax,key,scale,label in ((axs[0],'seconds',1.,'Wall time (s)'),
                               (axs[1],args.memory_metric,1024**3,
                                ('OS peak RSS (GiB)' if args.memory_metric == 'process_peak_rss_bytes'
                                 else 'Sampled peak RSS (GiB)'))):
        bars=ax.bar(args.labels,[r[key]/scale for r in records],
                    color=['#0072B2','#D55E00'])
        ax.bar_label(bars,fmt='%.2f',padding=3)
        ax.set_ylabel(label);ax.margins(y=.2)
    axs[2].plot(np.arange(new.size),(new-old).ravel(),'o',ms=4,color='#0072B2')
    unit='Hartree/bohr²' if quantity == 'hessian' else 'Hartree/bohr'
    axs[2].set(xlabel='Hessian block entry' if quantity == 'hessian' else 'Cartesian coordinate',
               ylabel=args.labels[1]+' − '+args.labels[0]+' ('+unit+')')
    axs[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    shape = f' ({new.shape[-2]} × {new.shape[-1]})' if quantity == 'hessian' else ''
    fig.suptitle('Pyrazine/'+records[0].get('basis', 'cc-pVDZ')+' Z-vector '+quantity+shape)
    if result['max_change'] == 0:
        axs[2].text(.5, .85, 'Bitwise identical', transform=axs[2].transAxes, ha='center')
        axs[2].set_yticks([0])
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    for ext in ('png','pdf'):fig.savefig(args.after/('bse_adjoint_memory.'+ext),dpi=300)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
