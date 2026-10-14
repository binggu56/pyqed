"""Fast frontier RI-GW/full-XY-BSE pilot with full HF-gap screening.

Reuse the matched benchmark's orbitals and factor infrastructure. Padé targets
that fail the existing 16/18/20-point stability check are excluded; BSE uses
only the contiguous passing frontier window. This is a restricted-space
calculation, not a converged full excitation spectrum. PySCF BSE is adapted
to the same window and full-space static screening, without changing its
response solver. GW and BSE formulations/references follow
``gw_bse_matched_scaling.py`` and the public PyQED solver documentation.
"""
import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import scipy.linalg as la
from pyscf import lib
from pyscf.gw import gw_ac, bse as pbse
from threadpoolctl import threadpool_limits
from gw_bse_matched_scaling import restore, dump
from pyqed.gw.gw import GW
from pyqed.gw.bse import BSE, _bse_full_matmat
from pyqed.gw.contour import ContourDeformation
from pyqed.gw.screening import StaticScreening


def plot(report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for row in report['windows']:
        ax[0].plot(range(1,6),row['pyqed_eV'],'o-',label=str(row['window']))
        ax[1].plot(range(1,6),(np.asarray(row['pyqed_eV'])-row['pyscf_independent_eV'])*1000,'o-',label=str(row['window']))
    windows=[tuple(w['window']) for w in report['windows']]
    labels=[str(w) for w in windows];x=np.arange(len(windows));width=.35
    ax[2].bar(x-width/2,[report['timings'][f'PyQED BSE {o}/{v}'] for o,v in windows],width,label='PyQED')
    ax[2].bar(x+width/2,[report['timings'][f'PySCF BSE common {o}/{v}'] for o,v in windows],width,label='PySCF, same QPs')
    ax[0].set(xlabel='Singlet state',ylabel='Excitation (eV)',title='Restricted BSE windows: [occupied, virtual]')
    ax[1].set(xlabel='Singlet state',ylabel='PyQED − PySCF (meV)',title='Independent GW energies')
    ax[2].set(xticks=x,xticklabels=labels,ylabel='BSE time (s)',title='8 threads; single solves\nShared screening setup excluded')
    for axis in ax:axis.legend(fontsize=8);axis.grid(axis='y',alpha=.2)
    fig.suptitle('Pentacene / cc-pVDZ: fast frontier GW + full-XY BSE\nFull-space screening; restricted excitation space, not full-spectrum convergence')
    fig.savefig(out/'pentacene_fast_window_gw_bse.png',dpi=160)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--occupied',type=int,default=24)
    p.add_argument('--virtual',type=int,default=48)
    p.add_argument('--plot-only',action='store_true')
    p.add_argument('--resume-bse',action='store_true')
    p.add_argument('--bse-thread-check',action='store_true')
    p.add_argument('--matched-gw-timing',action='store_true',help='Retiming with exactly the passing PySCF target set')
    a=p.parse_args(); out=a.output
    if a.plot_only:
        plot(json.loads((out/'fast_window_report.json').read_text()),out);return
    mf,inp=restore(out); no=inp['nocc']; lib.num_threads(a.threads)
    if a.bse_thread_check:
        old=json.loads((out/'fast_window_report.json').read_text())
        window=tuple(old['passing_frontier_window'])
        with threadpool_limits(limits=a.threads,user_api='blas'):
            gw=GW(mf,freq_int='ac',vhf_df=True,threads=a.threads)
            gw.e_qp=np.load(out/'PyQED-window-gw.npz')['qp']
            response=ContourDeformation(gw,mf.mo_energy,[no],prepare=False)
            response.static_screening=StaticScreening(gw._pair_factors,
                la.cho_factor(response.dielectric(0.,imaginary=True),lower=True,check_finite=False),gw.max_memory)
            gw._ac_screening=response
        rows=[]
        with threadpool_limits(limits=1,user_api='blas'):
            for n in (1,2,4,8):
                times=[]
                for repeat in range(3):
                    b=BSE(gw,orbital_window=window,threads=n, screening_energies='hf')
                    start=time.perf_counter()
                    b.run(nroots=5,tol=1e-6,max_cycle=200,max_space=160,eigensolver='response',return_vectors=True)
                    times.append(time.perf_counter()-start)
                    np.testing.assert_allclose(b.e*27.211386245988,old['windows'][-1]['pyqed_eV'],rtol=0,atol=1e-7)
                    assert b.info['converged']
                rows.append(dict(threads=n,seconds=times,median_seconds=float(np.median(times)),threading_info=b.threading_info))
        dump(out/'bse_thread_check.json',dict(window=window,rows=rows,best_threads=min(rows,key=lambda x:x['median_seconds'])['threads']))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
        ax.plot([x['threads'] for x in rows],[x['median_seconds'] for x in rows],'o-')
        ax.set(xticks=[1,2,4,8],xlabel='BSE OpenMP threads (BLAS = 1)',ylabel='Median solve time (s)',
               title='Pentacene: 11 occupied / 22 virtual BSE\nThree solves per thread count; shared screening excluded')
        fig.savefig(out/'pentacene_fast_bse_threads.png',dpi=160)
        return
    if a.matched_gw_timing:
        old=json.loads((out/'fast_window_report.json').read_text())
        wo,wv=old['passing_frontier_window']; ids=np.arange(no-wo,no+wv)
        with threadpool_limits(limits=a.threads,user_api='blas'):
            start=time.perf_counter(); gw=GW(mf,freq_int='ac',vhf_df=True,threads=a.threads)
            setup=time.perf_counter()-start; start=time.perf_counter()
            gw.run(orbs=ids,nw=100,ac='pade',pade_npts=18,tol=1e-7,maxiter=100,stability_tol=1e-4)
            seconds=time.perf_counter()-start
        previous=np.load(out/'PyQED-window-gw.npz')['qp']
        np.testing.assert_allclose(gw.e_qp[ids],previous[ids],rtol=0,atol=1e-9)
        dump(out/'matched_gw_timing.json',dict(threads=a.threads,targets=ids,setup_seconds=setup,
             gw_seconds=seconds,total_seconds=setup+seconds,stability=gw.ac_info['stability'],
             max_change_eV=float(np.max(abs(gw.e_qp[ids]-previous[ids]))*27.211386245988)))
        return
    report=dict(input=inp,threads=a.threads,scope=__doc__,timings={},windows=[],success=False)
    if a.resume_bse:
        report=json.loads((out/'fast_window_report.json').read_text());report['windows']=[]
    def save(): dump(out/'fast_window_report.json',report)
    def timed(name,fn):
        t=time.perf_counter()
        try:return fn()
        finally:
            report['timings'][name]=time.perf_counter()-t
            save(); print(name,report['timings'][name],flush=True)
    with threadpool_limits(limits=a.threads,user_api='blas'):
        gw=GW(mf,freq_int='ac',vhf_df=True,threads=a.threads) if a.resume_bse else timed('PyQED setup',lambda:GW(mf,freq_int='ac',vhf_df=True,threads=a.threads))
        targets=np.arange(no-a.occupied,no+a.virtual)
        def run_gw():
            try:
                gw.run(orbs=targets,nw=100,ac='pade',pade_npts=18,tol=1e-7,maxiter=100,
                       stability_tol=1e-4,callback=lambda g,i:print('GW quadrature',i+1,flush=True) if (i+1)%20==0 else None)
            except RuntimeError:
                if not gw.ac_info.get('stability',{}).get('unstable_orbitals'):raise
                print('Restricting BSE to passing contiguous frontier targets',flush=True)
        if not a.resume_bse:
            timed('PyQED GW attempted window',run_gw)
            report['gw_info']=gw.ac_info
            np.savez(out/'PyQED-window-gw.npz',qp=gw.e_qp,weights=gw.qp_weights,sigma=gw.ac_sigma,omega=gw.ac_omega)
            left=right=no
            while left>targets[0] and np.isfinite(gw.e_qp[left-1]):left-=1
            while right<=targets[-1] and np.isfinite(gw.e_qp[right]):right+=1
            if min(no-left,right-no)<2:raise RuntimeError('Insufficient stable frontier orbitals')
            report['passing_frontier_window']=[no-left,right-no]
            report['qp_max_residual_Ha']=float(np.max(np.abs(gw.qp_residuals[left:right])))
            report['qp_min_weight']=float(np.min(gw.qp_weights[left:right]))
            report['gw_frequency_totals']={k:sum(x[k] for x in gw.ac_info['frequencies']) for k in ['dielectric_seconds','inverse_seconds','projection_seconds']}
        else:
            saved=np.load(out/'PyQED-window-gw.npz');gw.e_qp=saved['qp'];gw.qp_weights=saved['weights']
            wo,wv=report['passing_frontier_window'];left,right=no-wo,no+wv
            response=ContourDeformation(gw,mf.mo_energy,range(left,right),nw=100,prepare=False)
            response.static_screening=StaticScreening(gw._pair_factors,la.cho_factor(response.dielectric(0.,imaginary=True),lower=True,check_finite=False),gw.max_memory)
            gw._ac_screening=response
        # The production AC path retains full-space static screening for BSE.
        response=gw._ac_screening
        chol=response.static_screening.cholesky
        factors=gw._pair_factors
        windows=sorted(set((max(2,int((no-left)*f)),max(2,int((right-no)*f))) for f in (.5,.75,1.)))
        pg=timed('PySCF setup',lambda:gw_ac.GWAC(mf))
        def reference_gw():
            pg.initialize_df(); pg.Lpq=pg.ao2mo();pg.vhf_df=True
            pg.orbs=list(range(left,right));pg.nw=100;pg.ac='pade';pg.ac_pade_npts=18
            pg.qpe_tol=1e-7;pg.qpe_max_iter=100;pg.kernel()
        if not a.resume_bse:
            timed('PySCF transform and GW passing window',reference_gw)
            reference_qp=pg.mo_energy.copy()
        else:
            pg.initialize_df();pg.Lpq=pg.ao2mo()
            reference_qp=np.full_like(mf.mo_energy,np.nan)
            reference_qp[left:right]=gw.e_qp[left:right]-np.asarray(report['reference_qp_difference_eV'])/27.211386245988
            pg.mo_energy=reference_qp.copy()
        np.savez(out/'PySCF-window-gw.npz',qp=reference_qp)
        report['reference_qp_difference_eV']=((gw.e_qp[left:right]-reference_qp[left:right])*27.211386245988).tolist()
        report['comparison_scope']='PySCF stock Padé evaluator; independent GW on passing targets only. Different GW target counts make GW timings non-equivalent.'
        eps_p=np.eye(len(factors))-gw_ac.get_rho_response(0.,mf.mo_energy,pg.Lpq[:,:no,no:])
        pchol=la.cho_factor(eps_p,lower=True,check_finite=False)
        report['factor_error']=float(np.max(np.abs(pg.Lpq-factors)))
        for wo,wv in windows:
            ids=np.arange(no-wo,no+wv); row=dict(window=[wo,wv],dimension=2*wo*wv)
            b=BSE(gw,orbital_window=(wo,wv),threads=a.threads, screening_energies='hf')
            timed(f'PyQED BSE {wo}/{wv}',lambda:b.run(nroots=5,tol=1e-6,max_cycle=200,max_space=160,eigensolver='response',return_vectors=True))
            row.update(pyqed_eV=b.e*27.211386245988,solver_info=b.info,threading_info=b.threading_info)
            xy=np.vstack((b.x,b.y)); residual=np.linalg.norm(_bse_full_matmat(b,xy)-xy*b.e,axis=0)
            row['pyqed_residuals']=residual
            assert b.info['converged'] and max(residual)<1e-6
            # Project the excitation space, retaining independently computed full W0.
            def full_screen(nocc,mo_energy,Lii,Lia):
                def solve(block):
                    return np.ascontiguousarray(la.cho_solve(pchol,block.reshape(len(factors),-1)).reshape(block.shape).transpose(1,0,2))
                return [solve(x) for x in Lii],[solve(x) for x in Lia]
            for common in (False,True):
                ref=pbse.BSE(pg);ref.nocc=np.array([wo]);ref.nmo=wo+wv
                qp=gw.e_qp if common else reference_qp
                ref.mo_energy=qp[ids][None,:];ref.mo_coeff=mf.mo_coeff[:,ids][None,:,:]
                ref.Lpq=np.ascontiguousarray(pg.Lpq[:,ids[:,None],ids][None,:,:,:])
                ref.nroot=5;ref.init_ntri=5;ref.max_expand=5;ref.max_vec=160;ref.max_iter=200;ref.residue_thresh=1e-12
                label='common' if common else 'independent'
                with patch.object(pbse,'_get_lpq_bar_by_block',full_screen):
                    timed(f'PySCF BSE {label} {wo}/{wv}',lambda:ref.kernel('s'))
                row[f'pyscf_{label}_eV']=ref.exci*27.211386245988
                rxy=np.vstack((ref.X_vec[0].reshape(5,-1).T,ref.Y_vec[0].reshape(5,-1).T))
                saved=b.e_qp.copy();b.e_qp=qp[ids]
                rres=np.linalg.norm(_bse_full_matmat(b,rxy)-rxy*ref.exci,axis=0);b.e_qp=saved
                row[f'pyscf_{label}_residuals']=rres
                assert max(rres)<1e-6
            report['windows'].append(row);save()
        report['success']=True
        if len(windows)>1:
            report['last_window_change_eV']=np.asarray(report['windows'][-1]['pyqed_eV'])-report['windows'][-2]['pyqed_eV']
        save()
    plot(report,out)
    print('PASS',report['passing_frontier_window'],flush=True)

if __name__=='__main__':main()
