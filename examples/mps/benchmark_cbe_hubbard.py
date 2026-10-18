"""Compare reduced one-site CBE and two-site DMRG on open Hubbard chains.

Run with PYTHONPATH=. and BLAS/OpenMP environment limits set to one. The
compiled owner uses pyqed.num_workers(). Compare shrewd selection with the
explicit DD-residual reference using --selection shrewd exact. One sweep
is one directional pass.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from examples.mps.hubbard_2d_dmrg import (build_random_reduced_spatial_mps,
    build_2d_hubbard_mpo, normalized_energy, run_sweeps)


def calculate(n,algorithm,*,D=32,passes=12,expansion=8,selection="shrewd",preselection=None):
    started=time.perf_counter()
    sites=build_random_reduced_spatial_mps(n,seed=7,bond_multiplicity=1,cap_multiplicity=True)
    mpo=build_2d_hubbard_mpo(sites,n,1,hopping_t=1,onsite_u=4)
    owner=mpo[0].normal_complementary_owner
    setup=time.perf_counter()-started; tick=time.perf_counter()
    result=run_sweeps(sites,mpo_factors=mpo,algorithm=algorithm,cbe_expansion=expansion,
        cbe_selection=selection,cbe_preselection=preselection,
        nsweeps=passes,max_bond=D,max_bond_mode='reduced',cutoff=1e-12,
        su2_moving_environment=owner,canonical_local_norm=False,
        conv_tol=1e-9,converge_on_full_sweeps=True,
        local_solver_kwargs={'tol':1e-11,'tol_residual':1e-8,'itermax':300,'max_space':48},
        verbose=1)
    elapsed=time.perf_counter()-tick
    energy,norm=normalized_energy(result['sites'],mpo)
    history=[]
    for row in result['history']:
        objectives=[u['local_objective'] for u in row['updates']]
        history.append(dict(energy=float(row['energy']),
            residual=max(float(o.get('residual',0)) for o in objectives),
            local_converged=all(o.get('davidson_converged',False) for o in objectives),
            added=sum(o.get('cbe_added_multiplets',0) for o in objectives),
            one_site_action=all(o.get('cbe_one_site_action',False) for o in objectives),
            compiled_update=all(o.get('cbe_compiled_update',False) for o in objectives)))
    return dict(n=n,algorithm=algorithm,selection=selection,D=D,expansion=expansion,setup_s=setup,sweep_s=elapsed,
                energy=energy,norm=norm,history=history,passes=len(history))


def plot(rows,output):
    import os
    os.environ.setdefault("MPLCONFIGDIR",str(output/"mpl-cache"))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    lengths=sorted({r['n'] for r in rows})
    reference={n:min(r['energy'] for r in rows if r['n']==n) for n in lengths}
    colors={n:f'C{i}' for i,n in enumerate(lengths)}
    labels=[]
    for i,r in enumerate(rows):
        method=('CBE '+r.get('selection','exact')) if r['algorithm']=='one_site_cbe' else 'Two-site'
        label=f"{r['n']} sites, {method}"
        gap=[max(abs(h['energy']-reference[r['n']]),1e-13) for h in r['history']]
        axes[0].semilogy(range(1,r['passes']+1),gap,
                        '.-' if r.get('selection')=='shrewd' and method!='Two-site' else '.--',color=colors[r['n']],label=label)
        axes[1].bar(i,r['sweep_s'],color=colors[r['n']],
                    alpha=1 if r.get('selection')=='shrewd' and method!='Two-site' else .45)
        axes[1].text(i,r['sweep_s'],f"{r['sweep_s']:.1f}s",ha='center',va='bottom')
        labels.append(f"{r['n']} sites\n{method}")
    axes[0].set(xlabel='Directional pass',ylabel='|E − lowest final energy| / t')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    axes[1].set(ylabel='Sweep time (s)',xticks=range(len(rows)),xticklabels=labels)
    axes[1].tick_params(axis='x', labelsize=8)
    axes[1].set_ylim(0,max(r['sweep_s'] for r in rows)*1.12)
    fig.suptitle(f"Open half-filled Hubbard chains, U/t=4, D={rows[0]['D']} reduced multiplets")
    fig.savefig(output/'cbe-hubbard-comparison.png',dpi=170)



if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sites',type=int,nargs='+',default=[8,12])
    p.add_argument('--algorithm',nargs='+',default=['two_site','one_site_cbe'])
    p.add_argument('--D',type=int,default=32);p.add_argument('--passes',type=int,default=12)
    p.add_argument('--expansion',type=int,default=8);p.add_argument('--output',type=Path,default=Path('/private/tmp/cbe-hubbard'))
    p.add_argument('--selection',nargs='+',choices=['shrewd','exact'],default=['shrewd'])
    p.add_argument('--preselection',type=int)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=[]
    for n in a.sites:
        for algorithm in a.algorithm:
            for selection in (a.selection if algorithm=='one_site_cbe' else ['exact']):
                rows.append(calculate(n,algorithm,D=a.D,passes=a.passes,expansion=a.expansion,
                                      selection=selection,preselection=a.preselection))
                (a.output/'results.json').write_text(json.dumps(rows,indent=2))
                print(rows[-1],flush=True)
    plot(rows,a.output)
