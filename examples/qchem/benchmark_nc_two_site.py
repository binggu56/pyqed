"""Isolated-process Li2 NC CBE/two-site time-to-accuracy and peak RSS comparison.

Uses identical packed-s8 RHF data generation, complete Dooh multiplets, seed,
per-sector multiplicity cap, local residual tolerance and central thread budget.
Each fresh process measures all imports, molecular preparation, NC construction
and converged sweeps, then snapshots peak RSS before verification/plotting.
The independent previously validated MPO energies are explicit references;
this is a matched finite-D accuracy comparison, not an exact CAS claim.
"""
from time import perf_counter
_STARTED=perf_counter()
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys


def worker(args):
    import numpy as np
    import hashlib
    from pyqed import num_workers
    from pyqed.qchem import Molecule,RHF
    from pyqed.qchem.symmetry import mo_symmetry_representations
    from pyqed.qchem.mcscf.direct_ci import transform_active_space_spatial_integrals
    from pyqed.mps.nonabelian.multiplet_dmrg import MultipletDMRG
    from examples.qchem.benchmark_nc_stability import guards
    print('Preparing packed-s8 molecular data',flush=True)
    mol=Molecule(atom='Li 0 0 -2; Li 0 0 2',unit='bohr',basis='6-311g',symmetry='Dooh').build(eri='s8')
    mf=RHF(mol,init_guess='atom');mf.chkfile=None;mf.run(verbose=0,tol=1e-12)
    assert mf.converged
    ncas=mf.mo_coeff.shape[1]-1
    assert ncas==25
    reps=mo_symmetry_representations(mol,mf.mo_coeff[:,1:],generators=True)
    h,v,ecore=transform_active_space_spatial_integrals(mf,mf.mo_coeff,ncas,1)
    print('Building direct NC solver',flush=True)
    setup_start=perf_counter()
    with guards():
        solver=MultipletDMRG.from_integrals(h,v,nelec=4,spin=0,D=2,ecore=ecore,
            n_threads=num_workers(),point_group='Dooh',orbital_representations=reps,wfnsym=args.target,seed=11)
        print('Starting '+args.algorithm+' sweeps',flush=True)
        solver.D=args.D
        setup=perf_counter()-setup_start
        samples=[];finish=solver._finish_cycle
        def record(h,n):
            finish(h,n)
            samples.append(dict(seconds=perf_counter()-_STARTED,energy=solver.energy,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        solver._finish_cycle=record
        solver.run(algorithm=args.algorithm,nsweeps=30,tol=1e-10,local_tol=1e-11,
                   cbe_expansion=4,cbe_preselection=16,verbose=1)
    end=perf_counter();peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    diagnostic=solver.symmetry_diagnostics()
    rows=json.loads(args.reference.read_text())
    reference=next(r['energy'] for r in rows if r['molecule']=='Li2' and r['target']==args.target and r['mode']=='mpo')
    error=abs(solver.energy-reference)
    result=dict(algorithm=args.algorithm,target=args.target,D=args.D,seed_D=2,seed=11,
        n_threads=num_workers(),blas_threads=1,outer_workers=1,total_seconds=end-_STARTED,
        solver_seconds=end-setup_start,setup_seconds=setup,peak_rss_bytes=peak,
        energy=solver.energy,reference_energy=reference,absolute_error=error,
        integral_sha256=hashlib.sha256(np.ascontiguousarray(h).tobytes()+np.ascontiguousarray(v).tobytes()+np.asarray(ecore).tobytes()).hexdigest(),
        accuracy_hartree=args.accuracy,converged=bool(solver.success),cycles=len(solver.history),
        partners=len(solver.states),covariance_error=float(diagnostic['covariance_error']),
        orthogonality_error=float(diagnostic['orthogonality_error']),
        partner_energy_error=float(np.max(abs(np.asarray(diagnostic['partner_energies_hartree'])-solver.e_tot))),
        max_local_residual=max(s['residual_norm'] for c in solver.history for s in c['sites']),
        hamiltonian_mpo=hasattr(solver,'mpo'),history=samples)
    result['passed']=(result['converged'] and error<args.accuracy and
        max(result['covariance_error'],result['orthogonality_error'],result['partner_energy_error'])<1e-8
        and result['partners']==(1 if args.target=='A1g' else 2) and not result['hamiltonian_mpo'])
    import pickle
    with args.output.with_suffix('.states.pkl').open('wb') as stream:
        pickle.dump(solver.states,stream,protocol=pickle.HIGHEST_PROTOCOL)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    solver.close()
    if not result['passed']: raise RuntimeError('Benchmark accuracy/partner checks failed; see result JSON')



def validate_partners(args):
    import pickle
    import numpy as np
    from pyqed import num_workers
    from pyqed.qchem import Molecule,RHF
    from pyqed.qchem.symmetry import mo_symmetry_representations
    from pyqed.qchem.mcscf.direct_ci import transform_active_space_spatial_integrals
    from pyqed.mps.nonabelian.multiplet_dmrg import MultipletDMRG
    mol=Molecule(atom='Li 0 0 -2; Li 0 0 2',unit='bohr',basis='6-311g',symmetry='Dooh').build(eri='s8')
    mf=RHF(mol,init_guess='atom');mf.chkfile=None;mf.run(verbose=0,tol=1e-12)
    assert mf.converged
    reps=mo_symmetry_representations(mol,mf.mo_coeff[:,1:],generators=True)
    h,v,ecore=transform_active_space_spatial_integrals(mf,mf.mo_coeff,25,1)
    print('Building independent reduced MPO for partner verification (outside timing)',flush=True)
    reference=MultipletDMRG.from_integrals(h,v,nelec=4,spin=0,D=16,ecore=ecore,
        n_threads=num_workers(),point_group='Dooh',orbital_representations=reps,wfnsym='A1g',
        contractions='reduced',seed=11)
    rows=json.loads((args.output/'results.json').read_text())
    checks=[]
    for row in rows:
        stem=f"{row['target']}-{row['algorithm']}-{row['repeat']}"
        with (args.output/(stem+'.states.pkl')).open('rb') as stream:
            states=pickle.load(stream)
        energies=np.asarray([state.expectation(reference.mpo)+ecore for state in states])
        error=float(np.max(abs(energies-row['energy'])))
        checks.append(dict(run=stem,partner_energies=energies.real.tolist(),
                           max_error=error,passed=error<1e-8))
        print(checks[-1],flush=True)
        assert error<1e-8, checks[-1]
    (args.output/'partner_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    reference.close()


def plot(rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    targets=['A1g','E1u'];colors=['#276FBF','#D95F02']
    for j,algorithm in enumerate(['one_site_cbe','two_site']):
        selected=[[r for r in rows if r['target']==t and r['algorithm']==algorithm] for t in targets]
        for ax,key,scale in [(axs[0],'total_seconds',1),(axs[1],'peak_rss_bytes',1024**2)]:
            means=[np.median([r[key]/scale for r in group]) for group in selected]
            ax.bar(np.arange(2)+(j-.5)*.35,means,.35,color=colors[j],label=['NC CBE','NC two-site'][j])
            for i,group in enumerate(selected):
                ax.scatter([i+(j-.5)*.35]*len(group),[r[key]/scale for r in group],color='black',s=12,zorder=3)
        for t,group in zip(targets,selected):
            r=group[0]
            axs[2].semilogy([x['seconds'] for x in r['history']],
                [max(abs(x['energy']-r['reference_energy']),1e-14) for x in r['history']],
                '-o' if t=='A1g' else '--s',color=colors[j],markersize=3,label=f'{algorithm}: {t}')
    for ax in axs[:2]: ax.set_xticks(range(2),['A1g (1 partner)','E1u (2 partners)'])
    axs[0].set_ylabel('Total wall time (s)');axs[0].legend()
    axs[1].set_ylabel('Peak process RSS (MiB)')
    axs[2].set(xlabel='Elapsed total time (s)',ylabel='Error vs independent MPO (hartree)')
    axs[2].axhline(rows[0]['accuracy_hartree'],color='gray',ls=':');axs[2].legend(fontsize=7)
    fig.suptitle('Li2 CAS(4e,25o), Dooh complete partners — fresh processes, equal accuracy')
    fig.savefig(output/'li2_nc_cbe_vs_two_site.png',dpi=160);plt.close(fig)


def main(args):
    # Shared timestamp-based runtime caches can be rebuilt by unrelated checkouts.
    os.environ.setdefault('PYQED_MPS_CPP_BUILD',str(args.output.resolve()/'runtime_cache'))
    import hashlib
    from pyqed.mps.nonabelian import _su2_kernel
    args.output.mkdir(parents=True,exist_ok=True)
    runtime=Path(os.environ['PYQED_MPS_CPP_BUILD'])
    provenance=dict(runtime_cache=str(runtime),reference=str(args.reference),reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        extension_sha256=hashlib.sha256(Path(_su2_kernel.__file__).read_bytes()).hexdigest(),
        timing='Fresh process imports + packed-s8 RHF + integral transform + NC setup + sweeps; diagnostics excluded',
        memory='OS process high-water RSS measured before diagnostics; fresh process per sample',
        reference_scope='Independently validated fixed-D=16 one-site MPO; not exact CAS',accuracy=args.accuracy,
        repeats=args.repeats,python=sys.version)
    root=Path(__file__).resolve().parents[2]
    paths=[Path(__file__)]+sorted(p for p in (root/'pyqed').rglob('*')
        if p.suffix in ('.py','.hpp','.cpp','.pxi','.pyx','.so') and '__pycache__' not in p.parts)
    provenance['sources']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    archive=args.output/'sources';archive.mkdir(exist_ok=True)
    for p in paths:
        (archive/provenance['sources'][str(p.relative_to(root))]).write_bytes(p.read_bytes())
    (args.output/'reference_results.json').write_bytes(args.reference.read_bytes())
    provenance['runtime_files']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime.glob('*') if p.suffix in ('.so','.stamp')}
    def verify_sources():
        for name,digest in provenance['runtime_files'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Runtime cache changed during benchmark: '+name)
        for name,digest in provenance['sources'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Source changed during benchmark: '+name)
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    env=dict(os.environ,PYTHONPATH='.',PYTHONHASHSEED='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLCONFIGDIR='/private/tmp/pyqed-mpl-cache')
    rows=[]
    for repeat in range(args.repeats):
        for target in ['A1g','E1u']:
            algorithms=['one_site_cbe','two_site'] if repeat%2==0 else ['two_site','one_site_cbe']
            for algorithm in algorithms:
                stem=f'{target}-{algorithm}-{repeat+1}';path=args.output/(stem+'.json')
                command=[sys.executable,__file__,'--worker','--algorithm',algorithm,'--target',target,
                    '--output',str(path),'--reference',str(args.reference),'--D',str(args.D),'--accuracy',str(args.accuracy)]
                verify_sources()
                print('Running '+stem,flush=True)
                with (args.output/(stem+'.log')).open('w') as log:
                    subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                row=json.loads(path.read_text());row['repeat']=repeat+1
                if rows:
                    assert row['integral_sha256']==rows[0]['integral_sha256'], 'Integral data changed between runs'
                    assert row['n_threads']==rows[0]['n_threads'], 'Thread budget changed between runs'
                rows.append(row)
                (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
                print({k:v for k,v in row.items() if k!='history'},flush=True)
    verify_sources()
    with (args.output/'partner_validation.log').open('w') as log:
        subprocess.run([sys.executable,__file__,'--validate','--output',str(args.output)],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    from statistics import median
    summary=[]
    for target in ['A1g','E1u']:
        record=dict(target=target)
        for algorithm in ['one_site_cbe','two_site']:
            runs=[r for r in rows if r['target']==target and r['algorithm']==algorithm]
            record[algorithm]={key:median(r[key] for r in runs) for key in
                ['total_seconds','solver_seconds','setup_seconds','peak_rss_bytes','cycles']}
            record[algorithm]['total_seconds_range']=[min(r['total_seconds'] for r in runs),max(r['total_seconds'] for r in runs)]
            record[algorithm]['max_energy_error']=max(r['absolute_error'] for r in runs)
        record['cbe_speedup']=record['two_site']['total_seconds']/record['one_site_cbe']['total_seconds']
        record['two_site_to_cbe_peak_memory_ratio']=record['two_site']['peak_rss_bytes']/record['one_site_cbe']['peak_rss_bytes']
        summary.append(record)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot(rows,args.output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--validate',action='store_true')
    p.add_argument('--algorithm',default='two_site');p.add_argument('--target',default='A1g')
    p.add_argument('--D',type=int,default=16);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--accuracy',type=float,default=1e-8)
    p.add_argument('--reference',type=Path,default=Path('/private/tmp/nc-cbe-production-complete/results.json'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    worker(args) if args.worker else validate_partners(args) if args.validate else main(args)
