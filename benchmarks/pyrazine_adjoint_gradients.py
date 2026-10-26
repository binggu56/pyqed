"""Isolated-process BSE gradient and analytic Hessian benchmarks.

Each method reconstructs the same idealized geometry and tight CD/RHF/exact-GW
reference. Timings exclude reference preparation, include the requested directions,
and use state_ids=[0] (first excitation). RSS is sampled every 20 ms; lifetime
OS peak is also recorded separately. Run with PYTHONPATH=. and --output DIR.
Use --quantity hessian --directions 3 for selected pyrazine displacements,
or --quantity hessian --system water for a complete small-system Hessian.
The three pyrazine directions are Cartesian N1-x, N1-y, N1-z in bohr;
all diagonal and mixed entries are evaluated. No finite differences.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter, process_time
from threading import Event, Thread
import numpy as np
import psutil


def worker(args):
    if args.fixed_pivots is None:
        return _worker(args)
    from pyqed.qchem import basis as gaussian
    build_factors = gaussian._matrix_free_spherical_cd_factors
    pivots = np.load(args.fixed_pivots)
    def fixed_factors(*values, **options):
        return build_factors(*values, **options, fixed_pivots=pivots)
    gaussian._matrix_free_spherical_cd_factors = fixed_factors
    try:
        return _worker(args)
    finally:
        gaussian._matrix_free_spherical_cd_factors = build_factors


def _worker(args):
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    from pyqed import num_workers
    from pyqed.qchem import Molecule
    from pyqed.gw import GW,BSE
    from examples.qchem.gw.pyrazine_gw_bse import ATOM
    import torch
    torch.set_num_threads(1)
    out=args.output
    start=perf_counter()
    basis=args.basis or ('cc-pvdz' if args.system == 'pyrazine' else 'sto-3g')
    atom=ATOM if args.system == 'pyrazine' else [('O',(0,0,0)),('H',(0,.7409,.5821)),('H',(.1058,-.8467,.5292))]
    mol=Molecule(atom=atom,unit='angstrom',basis=basis)
    if args.displacement:
        coords=mol.atom_coords()
        coords.flat[args.displacement_coordinate] += args.displacement
        mol=Molecule(atom=list(zip(mol.atom_symbols(),coords)),unit='bohr',basis=basis)
    mol.build(eri='cd',workers=num_workers(),
        options={'eri_screen_tol':0.,'low_rank_tol':1e-10})
    mf=mol.RHF().run(tol=getattr(args,'scf_tol',1e-11),
                     conv_tol_grad=getattr(args,'scf_gradient_tol',None))
    if not mf.converged:
        raise RuntimeError('RHF did not converge')
    gw=GW(mf,freq_int='exact',eta=0.,threads=num_workers()).run()
    if getattr(args,'qp_tol',None) is not None:
        gw.e_qp=gw._qp_spectrum.solve(gw,range(mol.nao),tol=args.qp_tol)
    ref=BSE(gw,screening_energies='hf',threads=num_workers()).run(nroots=2,low_rank=not getattr(args,'dense_bse',False),tol=getattr(args,'bse_tol',1e-9))
    np.savez(out/(args.method+'_reference.npz'),coords=mol.atom_coords(),rhf=mf.e_tot,
        mo_energy=mf.mo_energy,qp=gw.e_qp,excitation=ref.e)
    np.save(out/(args.method+'_cd_pivots.npy'),mol._cd_factor_data['pivots'])
    spectrum=gw._qp_spectrum
    probe=int(np.argmin(np.min(abs(gw.e_qp[:,None]-spectrum.brackets),axis=1)))
    den=gw.e_qp[probe]-spectrum.centers
    residues=(2*spectrum.couplings[probe]**2).ravel()
    nearest=int(np.argmin(abs(den)))
    with np.errstate(divide='ignore',invalid='ignore'):
        slope=1+np.sum(np.divide(residues,den**2,out=np.zeros_like(residues),where=residues>0))
        sigma=np.sum(np.divide(residues,den,out=np.zeros_like(residues),where=residues>0))
    np.savez(out/(args.method+'_qp_selection.npz'),roots=gw.e_qp,
        weights=gw.qp_weights,residuals=gw.qp_residuals,brackets=spectrum.brackets,
        probe_index=probe,probe_full_weight=1/slope,
        probe_full_residual=gw.e_qp[probe]-spectrum.reference[probe]-spectrum.static[probe]-sigma,
        probe_min_pole_distance=abs(den[nearest]),probe_nearest_residue=residues[nearest])
    del den,residues
    report=dict(dense_bse=getattr(args,'dense_bse',False),bse_tol=getattr(args,'bse_tol',1e-9),qp_tol=getattr(args,'qp_tol',None),scf_tol=getattr(args,'scf_tol',1e-11),
        scf_gradient_tol=getattr(args,'scf_gradient_tol',None),scf_info=mf.scf_info,process_reused=getattr(args,'reuse_process',False),fixed_pivots=None if args.fixed_pivots is None else str(args.fixed_pivots), displacement_coordinate=args.displacement_coordinate, displacement_bohr=args.displacement, method=args.method,reference_seconds=perf_counter()-start,nao=int(mol.nao),
        nocc=int(ref.nocc),rank=len(mol.eri_factors),workers=num_workers(),blas_threads=1,
        state_ids=[0],coordinates=3*mol.natom,basis=basis,system=args.system,quantity=args.quantity,eri='cd',screening='hf',max_memory_MB=args.memory)
    if args.reference_only:
        (out/'reference_only.json').write_text(json.dumps(report,indent=2))
        print(report,flush=True)
        return
    import hashlib
    from pyqed.qchem import basis as gaussian
    extension = Path(gaussian._integrals_cpp.__file__)
    with extension.open('rb') as stream:
        report['integral_extension_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    report['integral_extension_path'] = str(extension)
    modes=np.eye(3*mol.natom)[:args.directions] if args.directions else None
    report['directions'] = (np.eye(3*mol.natom) if modes is None else modes).tolist()
    done=Event(); samples=[psutil.Process().memory_info().rss]
    stage = ['first_response' if args.quantity == 'hessian' and args.method == 'adjoint' else 'derivatives']
    stage_samples = {stage[0]: [samples[0]]}
    stage_cpu = {}
    stage_clock = [process_time()]
    stage_events = []
    class StageMonitor(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if not message.startswith('Adjoint Hessian '):
                return
            if 'first response:' in message:
                target = 'reverse_response'
            elif 'reverse response:' in message:
                target = 'integral_curvature'
            elif 'integral contraction:' in message:
                target = 'reverse_response'
            else:
                return
            now = process_time()
            stage_cpu[stage[0]] = stage_cpu.get(stage[0], 0.) + now-stage_clock[0]
            rss = psutil.Process().memory_info().rss
            stage_samples.setdefault(stage[0], []).append(rss)
            stage[0], stage_clock[0] = target, now
            stage_samples.setdefault(target, []).append(rss)
            stage_events.append(dict(seconds=perf_counter()-start, stage=target, rss_bytes=rss))
    stage_monitor = StageMonitor()
    logging.getLogger('pyqed.gw.adjoint').addHandler(stage_monitor)
    def sample():
        count=0
        while not done.wait(.02):
            samples.append(psutil.Process().memory_info().rss)
            stage_samples.setdefault(stage[0], []).append(samples[-1])
            count+=1
            if count % 500 == 0:
                (out/(args.method+'_progress.json')).write_text(json.dumps(dict(elapsed=perf_counter()-start,peak_rss_bytes=max(samples),current_rss_bytes=samples[-1])))
    monitor=Thread(target=sample,daemon=True);monitor.start()
    start=perf_counter()
    cpu_start=process_time()
    try:
        if args.quantity == 'gradient':
            g=ref.gradient(state_ids=[0],method=args.method,max_memory=args.memory,modes=modes,
                           **({'gw_response':args.gw_response} if args.method=='adjoint' else {}))
        else:
            h=ref.hessian(state_ids=[0],method='adjoint' if args.method=='adjoint' else 'mixed',
                          max_memory=args.memory,modes=modes,
                          **({'excitation_response':args.excitation_response, 'response_tol':args.response_tol} if args.method=='adjoint' else {}),
                          **({'gw_response':args.gw_response} if args.method=='adjoint' else {}),
                          **({'columns':args.columns} if getattr(args,'columns',None) is not None else {}))
            g=ref.energy_gradient
            np.save(out/(args.method+'_hessian.npy'),h)
            report['columns']=ref.derivative_info.get('columns',list(range(h.shape[2])))
            block=h[:,report['columns'],:]
            report['max_asymmetry']=float(np.max(abs(block-block.transpose(0,2,1))))
            report['derivative_info']=ref.derivative_info
        np.save(out/(args.method+'_gradient.npy'),g)
        report['success']=True
    except Exception as error:
        report.update(success=False,error=repr(error))
        raise
    finally:
        report.update(seconds=perf_counter()-start,cpu_seconds=process_time()-cpu_start,baseline_rss_bytes=samples[0],peak_rss_bytes=max(samples),
            incremental_peak_bytes=max(samples)-samples[0])
        done.set();monitor.join()
        logging.getLogger('pyqed.gw.adjoint').removeHandler(stage_monitor)
        stage_cpu[stage[0]] = stage_cpu.get(stage[0], 0.) + process_time()-stage_clock[0]
        report['stage_measurements'] = {name: dict(peak_rss_bytes=max(values),
            entry_rss_bytes=values[0], peak_above_first_entry_bytes=max(values)-values[0],
            cpu_seconds=stage_cpu.get(name, 0.)) for name, values in stage_samples.items()}
        report['stage_events'] = stage_events
        import resource
        report['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        (out/(args.method+'.json')).write_text(json.dumps(report,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()))
        print(report,flush=True)


def summarize(out):
    records=[json.loads((out/(m+'.json')).read_text()) for m in ('forward','adjoint')]
    reference=[np.load(out/(m+'_reference.npz')) for m in ('forward','adjoint')]
    differences={k:float(np.max(abs(reference[0][k]-reference[1][k]))) for k in reference[0].files}
    for k in reference[0].files:
        np.testing.assert_allclose(reference[0][k],reference[1][k],atol=1e-10,rtol=0)
    quantity=records[0].get('quantity','gradient')
    values=[np.load(out/(m+'_'+quantity+'.npy')) for m in ('forward','adjoint')]
    delta=values[1]-values[0]
    report=dict(runs=records,reference_differences=differences,max_error=float(np.max(abs(delta))),
        rms_error=float(np.sqrt(np.mean(delta**2))),speedup=records[0]['seconds']/records[1]['seconds'],quantity=quantity)
    gradients=[np.load(out/(m+'_gradient.npy')) for m in ('forward','adjoint')]
    report['gradient_max_error']=float(np.max(abs(gradients[1]-gradients[0])))
    report['scaled_max_error']=report['max_error']/max(float(np.max(abs(values[0]))),1e-300)
    (out/'comparison.json').write_text(json.dumps(report,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
    for ax,key,label,scale in ((axs[0],'seconds',quantity.title()+' wall time (s)',1),
                              (axs[1],'peak_rss_bytes','Sampled peak RSS (GiB)',1024**3)):
        ax.bar(['forward','adjoint'],[r[key]/scale for r in records],color=['#0072B2','#D55E00'])
        ax.set_ylabel(label)
    axs[2].plot(np.arange(delta.size),delta.ravel(),'o',ms=3,color='#0072B2')
    axs[2].set(xlabel='Flattened '+quantity+' component',
        ylabel='Adjoint − forward (Hartree/bohr'+('²)' if quantity=='hessian' else ')'))
    for ax in axs: ax.spines[['top','right']].set_visible(False)
    for ext in ('png','pdf'):fig.savefig(out/(records[0].get('system','pyrazine')+'_adjoint_'+quantity+'.'+ext),dpi=300)
    print(json.dumps(report,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',choices=['forward','adjoint'])
    p.add_argument('--quantity',choices=['gradient','hessian'],default='gradient')
    p.add_argument('--system',choices=['pyrazine','water'],default='pyrazine')
    p.add_argument('--dense-bse',action='store_true',help='Small-system reference diagonalization.')
    p.add_argument('--bse-tol',type=float,default=1e-9)
    p.add_argument('--qp-tol',type=float,help='Refine selected exact GW roots to this tolerance.')
    p.add_argument('--scf-tol',type=float,default=1e-11)
    p.add_argument('--scf-gradient-tol',type=float,help='Explicit SCF orbital-gradient convergence tolerance.')
    p.add_argument('--basis',help='Override the system default basis.')
    p.add_argument('--response-tol', type=float, default=1e-11)
    p.add_argument('--excitation-response', choices=['dense','projected','physical'], default='dense')
    p.add_argument('--gw-response', choices=['dense','spectral'], default='dense')
    p.add_argument('--columns',type=int,nargs='+',help='Selected adjoint Hessian columns; retain all rows.')
    p.add_argument('--directions',type=int,default=0,help='First N Cartesian directions; zero means all.')
    p.add_argument('--memory',type=float,default=12000)
    p.add_argument('--displacement-coordinate',type=int,default=0)
    p.add_argument('--displacement',type=float,default=0.,help='Signed displacement in bohr, for gradient validation.')
    p.add_argument('--summarize',action='store_true')
    p.add_argument('--fixed-pivots',type=Path,help='Recorded CD pivots for a fixed-auxiliary-space derivative check.')
    p.add_argument('--reference-only',action='store_true',help='Save reference and QP-selection diagnostics without derivatives.')
    a=p.parse_args()
    if a.columns is not None and (a.method != 'adjoint' or a.quantity != 'hessian'):
        p.error('--columns requires --method adjoint --quantity hessian')
    if a.reference_only and a.method is None:
        p.error('--reference-only requires --method')
    a.output.mkdir(parents=True,exist_ok=True)
    if a.summarize:summarize(a.output)
    elif a.method:worker(a)
    else:
        for method in ('adjoint','forward'):
            with (a.output/(method+'.log')).open('w') as stream:
                subprocess.run([sys.executable,"-u",__file__,'--output',str(a.output),'--method',method,
                    '--memory',str(a.memory),'--quantity',a.quantity,'--system',a.system,'--directions',str(a.directions),'--displacement-coordinate',str(a.displacement_coordinate),
                    '--displacement='+str(a.displacement)]
                    +(['--dense-bse'] if a.dense_bse else [])+['--bse-tol',str(a.bse_tol)]
                    +([] if a.qp_tol is None else ['--qp-tol',str(a.qp_tol)])
                    +['--scf-tol',str(a.scf_tol)]
                    +([] if a.scf_gradient_tol is None else ['--scf-gradient-tol',str(a.scf_gradient_tol)])
                    +([] if a.basis is None else ['--basis',a.basis])
                    +([] if a.fixed_pivots is None else ['--fixed-pivots',str(a.fixed_pivots)])
                    +(['--excitation-response',a.excitation_response,'--response-tol',str(a.response_tol),'--gw-response',a.gw_response] if method=='adjoint' else []),stdout=stream,stderr=subprocess.STDOUT,check=True)
        summarize(a.output)
if __name__=='__main__':main()
