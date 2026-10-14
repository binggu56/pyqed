"""Matched molecular RI-G0W0/full-singlet-BSE comparison with PySCF 2.14.

Shared RI-HF orbitals and AO factors; independent AO-to-MO, GW and BSE work.
GW uses HF-gap direct RPA and Padé continuation; BSE uses the same HF-gap static W0 in both implementations.
PySCF screening helper receives HF energies; its BSE diagonal retains GW QPs.
Preparation and independent final residual checks are outside solve timings.
Use PYTHONPATH=. and single-thread environment defaults; --threads supplies
one active BLAS team (GW/PySCF BSE) or OpenMP team (PyQED BSE contractions).
"""
import argparse
import contextlib
import cProfile
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import time
from unittest.mock import patch

import numpy as np
import scipy.linalg as la
import pyscf
from pyscf import df, gto, lib, scf
from pyscf.gw import gw_ac, bse as pbse
from threadpoolctl import threadpool_limits, threadpool_info
from pyqed.qchem.basis import PackedRIFactors
from pyqed.gw.gw import GW
from pyqed.gw.bse import BSE, _bse_full_matmat
from pyqed.gw.contour import ContourDeformation
from pyqed.gw.screening import StaticScreening
from benchmark_gw_tda_memory import geometry


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))+'\n')


def prepare(args):
    lib.num_threads(4)
    with threadpool_limits(limits=4, user_api='blas'):
        start=time.perf_counter()
        atom=geometry(args.rings)
        mol=gto.M(atom=atom, basis=args.basis, unit='angstrom', verbose=0, max_memory=8000)
        mf=scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
        mf.chkfile=None; mf.conv_tol=1e-11; mf.conv_tol_grad=1e-7
        mf.kernel()
        assert mf.converged
        factors=np.concatenate(list(mf.with_df.loop()))
        np.save(args.output/'ao_factors.npy',factors)
        np.savez(args.output/'orbitals.npz',energy=mf.mo_energy,coeff=mf.mo_coeff,occ=mf.mo_occ,e_tot=mf.e_tot)
        np.testing.assert_allclose(mf.mo_coeff.T@mf.get_ovlp()@mf.mo_coeff,np.eye(mol.nao),atol=1e-10)
        report=dict(molecule={2:'naphthalene',3:'anthracene',4:'tetracene',5:'pentacene'}.get(args.rings, f'{args.rings}-acene'),atom=atom,basis=args.basis,
                    auxbasis='cc-pvdz-ri',nao=mol.nao,nocc=mol.nelectron//2,naux=len(factors),
                    full_bse_dimension=2*(mol.nelectron//2)*(mol.nao-mol.nelectron//2),
                    scf_energy=mf.e_tot,preparation_seconds=time.perf_counter()-start,
                    pyscf=pyscf.__version__,platform=platform.platform(),
                    input_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.output/'ao_factors.npy',args.output/'orbitals.npz']})
        dump(args.output/'input.json',report)
        print(json.dumps(report,indent=2),flush=True)


def restore(output):
    inp=json.loads((output/'input.json').read_text()); saved=np.load(output/'orbitals.npz')
    mol=gto.M(atom=inp['atom'],basis=inp['basis'],unit='angstrom',verbose=0,max_memory=8000)
    mf=scf.RHF(mol).density_fit(auxbasis=inp['auxbasis']); mf.chkfile=None
    mf.mo_coeff=saved['coeff'];mf.mo_energy=saved['energy'];mf.mo_occ=saved['occ'];mf.e_tot=float(saved['e_tot']);mf.converged=True
    factors=np.load(output/'ao_factors.npy')
    mf.with_df._cderi=factors
    mol.eri_factors=PackedRIFactors(factors,mol.nao)
    return mf,inp


def validate_inputs(args):
    """Compare full transformed factors and independently assembled screening."""
    mf,inp=restore(args.output)
    lib.num_threads(args.threads)
    with threadpool_limits(limits=args.threads,user_api='blas'):
        q=GW(mf,freq_int='ac',vhf_df=True,threads=args.threads)
        p=gw_ac.GWAC(mf);p.initialize_df();pf=p.ao2mo()
        qf=q._pair_factors
        factor_error=max(float(np.max(abs(qf[i:i+8]-pf[i:i+8]))) for i in range(0,len(pf),8))
        response=ContourDeformation(q,mf.mo_energy,range(len(mf.mo_energy)),nw=100,prepare=False)
        no=inp['nocc'];dielectric_errors=[]
        for frequency in (0.,.5,5.):
            eps_p=np.eye(len(pf))-gw_ac.get_rho_response(frequency,mf.mo_energy,pf[:,:no,no:])
            eps_q=response.dielectric(frequency,imaginary=True)
            dielectric_errors.append(float(np.max(abs(eps_q-eps_p))))
        eps=response.dielectric(0.,imaginary=True)
        chol=la.cho_factor(eps,lower=True,check_finite=False)
        ii,ia=pbse._get_lpq_bar_by_block([no],mf.mo_energy[None,:],[pf[:,:no,:no]],[pf[:,:no,no:]])
        errors=[]
        for block,other in [(qf[:,:no,:no],ii[0]),(qf[:,:no,no:],ia[0])]:
            expected=la.cho_solve(chol,block.reshape(len(qf),-1)).reshape(block.shape).transpose(1,0,2)
            errors.append(float(np.max(abs(expected-other))))
        report=dict(mo_factor_max_error=factor_error,dielectric_max_errors=dielectric_errors,
                    static_screened_block_max_errors=errors,frequencies_Ha=[0.,.5,5.])
        assert max([factor_error,*dielectric_errors,*errors])<1e-10,report
        dump(args.output/'input_validation.json',report)
        print(json.dumps(report),flush=True)


def run(args):
    mf,inp=restore(args.output)
    lib.num_threads(args.threads)
    assert lib.num_threads()==args.threads
    label=f'{args.solver}-{args.threads}-{args.repeat}'
    common_qp=np.load(args.common_qp)['qp'] if args.common_qp else None
    row=dict(solver=args.solver,threads=args.threads,repeat=args.repeat,stages={},nroots=args.roots,
             gw_tolerance=1e-7,bse_residual_tolerance=1e-6,nw=100,pade_npts=18,
             gw_screening='HF-gap direct RPA',bse_screening='HF-gap static direct RPA (shared W0)',
             timers={},profile=bool(args.profile), max_space=args.max_space,
             ac_stability_checked=(not args.unchecked_ac if args.solver=='PyQED' else False) if not args.common_qp else None,
             pyscf_pade_evaluator=('corrected' if args.correct_pyscf_pade else 'stock') if args.solver=='PySCF' else None,
             max_cycle=args.max_cycle,bse_eigensolver=args.bse_eigensolver if args.solver=='PyQED' else 'response',
             common_qp_file=str(args.common_qp) if args.common_qp else None,
             common_qp_sha256=hashlib.sha256(args.common_qp.read_bytes()).hexdigest() if args.common_qp else None)
    row['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),Path(gw_ac.__file__),Path(pbse.__file__),
         Path(importlib.import_module('pyqed.gw.ac').__file__),
         Path(importlib.import_module('pyqed.gw.bse').__file__),
         Path('pyqed/gw/bse_contractions.cpp'),Path('pyqed/gw/bse_lanczos.py'),
         Path(importlib.import_module('pyqed.gw._bse_contractions').__file__)]}
    current_stage=['setup_transform']
    captured_sigma=[]
    def wrap(fn,name):
        def timed(*a,**kw):
            t=time.perf_counter()
            try:
                result=fn(*a,**kw)
                if name=='imaginary self-energy':captured_sigma.append(result)
                return result
            finally:
                record=row['timers'].setdefault(current_stage[0]+' '+name,dict(seconds=0.,calls=0))
                record['seconds']+=time.perf_counter()-t;record['calls']+=1
        return timed
    def stage(name,fn):
        current_stage[0]=name
        t=time.perf_counter()
        try:
            value=fn()
        except Exception as error:
            row.update(success=False, error=str(error))
            row['stages'][name]=time.perf_counter()-t
            dump(args.output/f'{label}.json',row)
            raise
        row['stages'][name]=time.perf_counter()-t
        print(label,name,row['stages'][name],flush=True);dump(args.output/f'{label}.json',row)
        return value
    profiler=cProfile.Profile() if args.profile else None
    with threadpool_limits(limits=args.threads,user_api='blas'),contextlib.ExitStack() as stack:
        if args.correct_pyscf_pade and args.solver=='PySCF':
            from pyscf.gw.utils import ac_grid
            from analyze_matched_gw_bse import corrected_pade
            stack.enter_context(patch.object(ac_grid,'pade_thiele_ndarray',corrected_pade))
        if args.solver=='PyQED':
            for name in ['dielectric','screened_elements']:
                stack.enter_context(patch.object(ContourDeformation,name,wrap(getattr(ContourDeformation,name),name)))
            module=importlib.import_module('pyqed.gw.bse')
            stack.enter_context(patch.object(module,'_factorized_bse_action',wrap(module._factorized_bse_action,'actions')))
        else:
            history=[]
            update=pbse.update_mp_mm
            def tracked_update(Mp,Mm,tri_vec,apb_prod,amb_prod,ntri,nprod):
                history.append(dict(iteration=len(history)+1,subspace=ntri,new_columns=ntri-nprod))
                return update(Mp,Mm,tri_vec,apb_prod,amb_prod,ntri,nprod)
            stack.enter_context(patch.object(pbse,'update_mp_mm',tracked_update))
            screening=pbse._get_lpq_bar_by_block
            def matched_screening(nocc,mo_energy,Lii,Lia):
                return screening(nocc,np.asarray([mf.mo_energy]),Lii,Lia)
            stack.enter_context(patch.object(pbse,'_get_lpq_bar_by_block',matched_screening))
            for module,name,label_timer in [(gw_ac,'get_sigma','imaginary self-energy'),
                (gw_ac,'get_rho_response','dielectric'),(gw_ac,'get_id_minus_pi_inv','inverse'),
                (pbse,'_get_lpq_bar_by_block','screening'),(pbse,'_bse_contraction','actions')]:
                stack.enter_context(patch.object(module,name,wrap(getattr(module,name),label_timer)))
        if profiler:profiler.enable()
        begin=time.perf_counter()
        if args.solver=='PyQED':
            gw=stage('setup_transform',lambda:GW(mf,freq_int='ac',vhf_df=True,threads=args.threads))
            def progress(obj,index):
                if (index+1)%20==0:
                    print(label,'GW quadrature',index+1,'/100',flush=True)
            if common_qp is None:
                try:
                    stage('GW',lambda:gw.run(nw=100,ac='pade',pade_npts=18,tol=1e-7,maxiter=100,callback=progress,stability_tol=None if args.unchecked_ac else 1e-4))
                finally:
                    row['gw_accuracy']=gw.ac_info.get('stability')
                    dump(args.output/f'{label}.json',row)
            else:
                gw.e_qp=common_qp.copy();gw.qp_residuals=np.zeros_like(common_qp)
                gw.ac_info={'frequencies':[],'fit_and_root_seconds':0.}
            row['gw_frequency_totals']={key:sum(item[key] for item in gw.ac_info['frequencies'])
                                       for key in ['dielectric_seconds','inverse_seconds','projection_seconds']}
            row['gw_fit_seconds']=gw.ac_info['fit_and_root_seconds']
            row['gw_projection_setup_seconds']=gw.ac_info.get('projection_setup_seconds',0.)
            row['gw_projection_packed_bytes']=gw.ac_info.get('projection_packed_bytes',0)
            qp=gw.e_qp.copy(); factors=gw._pair_factors
            def solve_bse():
                response=ContourDeformation(gw,mf.mo_energy,range(len(qp)),nw=8,prepare=False)
                chol=la.cho_factor(response.dielectric(0.,imaginary=True),lower=True,check_finite=False)
                b=BSE(gw,threads=args.threads, screening_energies='hf')
                b._static_screening=StaticScreening(factors,chol,b.max_memory)
                try:
                    b.run(nroots=args.roots,tol=1e-6,max_cycle=args.max_cycle,
                          max_space=args.max_space,eigensolver=args.bse_eigensolver,return_vectors=True)
                finally:
                    row['solver_info']=b.info
                    row['threading_info']=b.threading_info
                return b
            b=stage('BSE',solve_bse)
            energies=b.e.copy(); xy=np.vstack((b.x,b.y));row['solver_info']=b.info
            row['threading_info']=b.threading_info
            row['qp_residual_max']=float(np.max(gw.qp_residuals))
        else:
            def setup():
                g=gw_ac.GWAC(mf);g.initialize_df();g.Lpq=g.ao2mo();g.vhf_df=True
                g.nw=100;g.qpe_tol=1e-7;g.qpe_max_iter=100;g.ac='pade';g.ac_pade_npts=18
                return g
            gw=stage('setup_transform',setup)
            if common_qp is None:stage('GW',gw.kernel)
            else:gw.mo_energy=common_qp.copy()
            qp=gw.mo_energy.copy();factors=gw.Lpq
            def solve_bse():
                b=pbse.BSE(gw);b.nroot=args.roots;b.init_ntri=args.roots;b.max_expand=args.roots
                b.max_vec=args.max_space;b.max_iter=args.max_cycle;b.residue_thresh=1e-12
                b.kernel('s');return b
            b=stage('BSE',solve_bse)
            row['solver_info']=dict(converged=True,history=history,iterations=len(history),
                operator_columns=sum(item['new_columns'] for item in history))
            energies=b.exci.copy();xy=np.vstack((b.X_vec[0].reshape(len(energies),-1).T,b.Y_vec[0].reshape(len(energies),-1).T))
            shifts=np.diag(gw.vk-gw.vxc) if common_qp is None else np.zeros_like(common_qp)
            row['qp_residual_max']=0. if common_qp is not None else float(max(abs(qp[p]-mf.mo_energy[p]-gw.acobj[p].ac_eval(qp[p]).real-shifts[p]) for p in range(len(qp))))
        row['total_seconds']=time.perf_counter()-begin
        if profiler:profiler.disable();profiler.dump_stats(str(args.output/f'{label}.prof'))
        row['threadpools']=threadpool_info()
        row['pyscf_version']=pyscf.__version__
        if common_qp is not None:
            row['common_qp_file']=str(args.common_qp)
        elif args.solver=='PyQED':
            np.savez(args.output/f'{label}-fit.npz',sigma=gw.ac_sigma,omega=gw.ac_omega,
                     coeff=gw.ac_coeff,nodes=gw.ac_nodes,static=gw.ac_static,ef=gw.ac_info['fermi_level'])
        else:
            np.savez(args.output/f'{label}-fit.npz',coeff=gw.acobj.coeff,
                     omega=gw.acobj.omega,indices=gw.acobj.idx,static=shifts,sigma=captured_sigma[0][0])
        indices=np.unique([0,inp['nocc']-1,inp['nocc'],len(qp)-1])
        row['factor_sample']=factors[::max(1,len(factors)//8)][:,indices][:,:,indices]
        row['excitation_eV']=energies*27.211386245988;row['qp']=qp
        row['success']=False
        np.savez(args.output/f'{label}.npz',qp=qp,energies=energies,xy=xy)
        dump(args.output/f'{label}.json',row)
        # Independent residual gate in the same physical XY normalization.
        current_stage[0]='audit'
        check=GW(mf,freq_int='ac',vhf_df=True,mo_pair_factors=factors)
        check.e_qp=qp
        response=ContourDeformation(check,mf.mo_energy,range(len(qp)),nw=8,prepare=False)
        chol=la.cho_factor(response.dielectric(0.,imaginary=True),lower=True,check_finite=False)
        audit=BSE(check,threads=args.threads, screening_energies='hf');audit._static_screening=StaticScreening(factors,chol,audit.max_memory)
        dim=xy.shape[0]//2;norm=np.sum(xy[:dim]**2-xy[dim:]**2,axis=0)
        assert np.all(norm>0)
        xy=xy/np.sqrt(norm)
        residual=np.linalg.norm(_bse_full_matmat(audit,xy)-xy*energies,axis=0)
        row['physical_residuals']=residual;row['success']=bool(np.max(residual)<1e-6 and row['qp_residual_max']<1e-7)
        dump(args.output/f'{label}.json',row)
        assert row['success'],row
        print(label,'PASS',row['total_seconds'],row['excitation_eV'],residual,flush=True)


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pstats
    inp=json.loads((output/'input.json').read_text())
    rows=[json.loads(p.read_text()) for p in output.glob('Py*.json')]
    rows=[r for r in rows if r.get('success') and r['repeat'].isdigit() and not r['profile'] and r['bse_screening'].startswith('HF-gap')]
    summary={}
    reference=next(r for r in rows if r['solver']=='PySCF' and r['threads']==1)
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    colors={'PyQED':'#0072B2','PySCF':'#D55E00'}
    for solver in colors:
        summary[solver]={}
        for n in (1,2,4,8):
            selected=[r for r in rows if r['solver']==solver and r['threads']==n]
            assert selected,(solver,n)
            summary[solver][n]=dict(total_seconds=float(np.median([r['total_seconds'] for r in selected])),
                stages={s:float(np.median([r['stages'][s] for r in selected])) for s in ['setup_transform','GW','BSE']},
                excitation_eV=np.mean([r['excitation_eV'] for r in selected],axis=0),repeats=len(selected),
                max_physical_residual=max(max(r['physical_residuals']) for r in selected))
        times=[summary[solver][n]['total_seconds'] for n in (1,2,4,8)]
        axes[0,0].plot([1,2,4,8],times,'o-',label=solver,color=colors[solver])
        axes[0,1].plot([1,2,4,8],times[0]/np.asarray(times),'o-',label=solver,color=colors[solver])
        for n in (1,2,4,8):
            delta=(summary[solver][n]['excitation_eV']-reference['excitation_eV'])*1000
            axes[1,1].plot(range(1,len(delta)+1),delta,'o-',label=f'{solver}, {n}')
    labels=[f'{s}\n{n} threads' for s in colors for n in (1,4)]
    bottom=np.zeros(4)
    for stage in ['setup_transform','GW','BSE']:
        heights=[summary[s][n]['stages'][stage] for s in colors for n in (1,4)]
        axes[1,0].bar(labels,heights,bottom=bottom,label=stage);bottom+=heights
    axes[0,0].set(xlabel='Thread budget',ylabel='Median GW+BSE time (s)',xticks=[1,2,4,8])
    axes[0,1].set(xlabel='Thread budget',ylabel='Speedup versus one thread',xticks=[1,2,4,8])
    axes[1,0].set(ylabel='Median stage time (s)')
    axes[1,1].set(xlabel='Singlet root',ylabel='Difference from PySCF / 1 thread (meV)')
    for ax in axes.flat:ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle(f"{inp['molecule'].capitalize()} / {inp['basis']}: RI-G0W0 + full BSE with identical HF-based W0\nShared orbitals and AO factors; fresh transformations, GW and BSE per run")
    fig.savefig(output/'matched_gw_bse_comparison.png',dpi=180);plt.close(fig)
    excitation=np.array([r['excitation_eV'] for r in rows]);qp=np.array([r['qp'] for r in rows])
    comparison=dict(summary=summary,excitation_spread_meV=np.ptp(excitation,axis=0)*1000,
                    max_qp_spread_eV=float(np.max(np.ptp(qp,axis=0))*27.211386245988),
                    max_physical_residual=max(max(r['physical_residuals']) for r in rows),runs=len(rows))
    dump(output/'comparison.json',comparison)
    for profile in output.glob('*.prof'):
        with profile.with_suffix('.txt').open('w') as stream:
            pstats.Stats(str(profile),stream=stream).strip_dirs().sort_stats('cumulative').print_stats(35)
    print(json.dumps(comparison,indent=2,default=lambda v:v.tolist()),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--rings',type=int,default=3);p.add_argument('--basis',default='cc-pvdz')
    p.add_argument('--solver',choices=['PyQED','PySCF'],default='PyQED');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--repeat',default='pilot');p.add_argument('--roots',type=int,default=5);p.add_argument('--profile',action='store_true')
    p.add_argument('--validate-inputs',action='store_true')
    p.add_argument('--common-qp',type=Path,help='BSE-only control; skip GW and use saved QPs')
    p.add_argument('--plot-only',action='store_true')
    p.add_argument('--unchecked-ac',action='store_true',help='Explicitly reproduce historical unchecked Padé fits')
    p.add_argument('--correct-pyscf-pade',action='store_true',help='Label and apply the PySCF 2.14 continued-fraction evaluator correction')
    p.add_argument('--bse-eigensolver',choices=['response','davidson','lanczos'],default='response')
    p.add_argument('--max-space',type=int,default=160)
    p.add_argument('--max-cycle',type=int,default=200)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.plot_only:plot(args.output)
    elif args.prepare:prepare(args)
    elif args.validate_inputs:validate_inputs(args)
    else:run(args)

if __name__=='__main__':main()
