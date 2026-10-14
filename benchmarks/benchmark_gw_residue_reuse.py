"""Benchmark exact residue reuse and independent GW orbital solves.

Use PYTHONPATH=. and single-threaded BLAS/OMP environment defaults. The public
GW threads=1/4 budget controls parallel work without nested worker pools.
"""
import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from pyqed.qchem import Molecule
from pyqed.gw.gw import GW
from pyqed.gw import contour


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True,help='Saved contour.py before optimization')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.repeats<1: parser.error('repeats must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('pyqed.gw._residue_reference',args.baseline)
    baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    kernels=dict(baseline=baseline.kernel,optimized=contour.kernel)
    with threadpool_limits(limits=1):
        with (args.output/'preparation.log').open('w') as log,contextlib.redirect_stdout(log):
            mol=Molecule(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                         basis='cc-pvdz',unit='angstrom').build(eri='cd',options={'workers':1})
            mf=mol.RHF().run(tol=1e-11,verbose=0)
            assert mf.converged
            template=GW(mf,freq_int='cd',eta=1e-6,threads=1)
        rows=[]
        configs=[(v,t) for t in (1,4) for v in kernels]
        for selected in ([4,5],[3,4,5,6]):
            reference=None
            # Warm up each implementation before collecting timings.
            for v,t in configs:
                gw=GW(mf,freq_int='cd',eta=1e-6,threads=t,mo_pair_factors=template._pair_factors)
                with patch.object(contour,'kernel',kernels[v]):
                    gw.run(orbs=selected,nw=32,residue_solver='direct',screening_tol=1e-11)
            for repeat in range(args.repeats):
                for v,t in (configs if repeat%2==0 else configs[::-1]):
                    gw=GW(mf,freq_int='cd',eta=1e-6,threads=t,mo_pair_factors=template._pair_factors)
                    with patch.object(contour,'kernel',kernels[v]):
                        start=perf_counter()
                        gw.run(orbs=selected,nw=32,residue_solver='direct',screening_tol=1e-11)
                        elapsed=perf_counter()-start
                    info=gw._contour.screening_info.copy()
                    sigma=np.array([[gw._contour.evaluate(p,float(mf.mo_energy[p])+shift)
                                     for shift in (-.037,0.,.037)] for p in selected])
                    values=dict(qp=gw.e_qp[selected],weights=gw.qp_weights[selected],
                                sigma_real=sigma.real,sigma_imag=sigma.imag)
                    if reference is None: reference={k:a.copy() for k,a in values.items()}
                    errors={k:float(np.max(np.abs(a-reference[k]))) for k,a in values.items()}
                    for k,a in values.items():
                        np.testing.assert_allclose(a,reference[k],atol=1e-9,rtol=0,err_msg=k)
                    assert np.max(gw.qp_residuals[selected])<1e-9
                    assert info['failed_solves']==0
                    row=dict(variant=v,threads=t,orbs=selected,repeat=repeat,seconds=elapsed,
                             errors=errors,screening=info,threading=gw.threading_info,
                             residuals=gw.qp_residuals[selected].tolist(),
                             **{k:a.tolist() for k,a in values.items()})
                    rows.append(row)
                    print(json.dumps({k:row[k] for k in ('variant','threads','orbs','seconds','errors','screening')}),flush=True)
            # Retain partial results if a later window fails.
            data=dict(case='water/cc-pVDZ',nw=32,eta=1e-6,screening_tol=1e-11,
                      libraries=threadpool_info(),rows=rows)
            (args.output/'results.json').write_text(json.dumps(data,indent=2))
    plot(data,args.output)


def plot(data,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    summary={}
    for ax,selected in zip(axes,([4,5],[3,4,5,6])):
        key=str(len(selected))+' orbitals';summary[key]={}
        for j,v in enumerate(('baseline','optimized')):
            times=[np.median([r['seconds'] for r in data['rows'] if r['variant']==v
                            and r['threads']==t and r['orbs']==selected]) for t in (1,4)]
            summary[key][v]=dict(zip(('1','4'),map(float,times)))
            ax.bar(np.arange(2)+(j-.5)*.35,times,.35,label=v,color=('#8896a9','#127c92')[j])
        ax.set(xticks=[0,1],xticklabels=['1 thread','4 threads'],ylabel='Median seconds',
               title=f'Complete GW: orbitals {selected}')
    axes[0].legend(frameon=False)
    fig.suptitle('Water/cc-pVDZ; full screening; identical solver controls; BLAS=1')
    fig.savefig(output/'gw_residue_reuse.png',dpi=170)
    plt.close(fig)
    summary['max_absolute_errors']={k:max(r['errors'][k] for r in data['rows'])
                                    for k in ('qp','weights','sigma_real','sigma_imag')}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
