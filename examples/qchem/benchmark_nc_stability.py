"""Larger-space and sustained-sweep validation of direct compiled NC CBE.

Uses packed s8 AO integrals, complete orbital shells, serial outer work,
central compiled thread budget, and one BLAS thread. Molecular cases use
four active electrons in all remaining 6-311G orbitals. References are
explicit reduced-MPO one-site sweeps at the same final per-sector D.
NC starts at a smaller multiplicity and grows with compiled shrewd CBE.
The minimum sweep count deliberately prevents early convergence.
Outputs and reproducible figures are kept outside the repository.
"""
import argparse
import ctypes
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import platform
import resource
import psutil
from time import perf_counter, time_ns
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
from pyqed import num_workers
from pyqed.qchem import Molecule,RHF
from pyqed.mps.nonabelian.multiplet_dmrg import MultipletDMRG
from pyqed.mps.nonabelian.multiplet_operator import MultipletMPO
from examples.qchem._allocation_probe import allocated_bytes

CASES={'Be':('Be 0 0 0','SO3','L0',0),
       'LiH':('Li 0 0 -1.5; H 0 0 1.5','Coov','A1',0),
       'Li2':('Li 0 0 -2; Li 0 0 2','Dooh','A1g',1)}


class _HeapStatistics(ctypes.Structure):
    _fields_=[('blocks_in_use',ctypes.c_uint),('size_in_use',ctypes.c_size_t),
              ('max_size_in_use',ctypes.c_size_t),('size_allocated',ctypes.c_size_t)]


_heap_statistics=None
if platform.system()=='Darwin':
    _heap_statistics=ctypes.CDLL(None).malloc_zone_statistics
    _heap_statistics.argtypes=[ctypes.c_void_p,ctypes.POINTER(_HeapStatistics)]
    _heap_statistics.restype=None


def live_heap_bytes():
    """Allocated malloc ranges, including C++/NumPy, excluding allocator caches."""
    return allocated_bytes()


def allocator_bytes():
    """Allocator statistics, which can include storage beyond in-use blocks."""
    if _heap_statistics is None: return None
    stats=_HeapStatistics();_heap_statistics(None,ctypes.byref(stats))
    return stats.size_in_use


def guards():
    from pyqed.mps.nonabelian import cbe,cbe_selection,multiplet_dmrg
    from pyqed.mps.nonabelian.multiplet_operator import MultipletMPO
    from pyqed.mps.nonabelian.fusion import TensorFusionMap
    from pyqed.mps.nonabelian.nc_multiplet import NCMultipletDMRG
    stack=ExitStack()
    for owner,name in [(MultipletMPO,'__init__'),(TensorFusionMap,'expand'),
                       (cbe,'expand_multiplet_bond'),(cbe,'multiplet_pair_action'),
                       (cbe,'truncate_multiplet_bond'),(cbe_selection,'shrewd_subspace'),
                       (cbe_selection,'multiplet_halves'),(multiplet_dmrg,'_contract'),
                       (np.linalg,'svd'),(np.linalg,'qr'),(np.linalg,'eigh'),
                       (NCMultipletDMRG,'_hamiltonian_action')]:
        stack.enter_context(patch.object(owner,name,side_effect=AssertionError('Forbidden fallback: '+name)))
    return stack


def run(args):
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[2]
    from pyqed.mps.nonabelian import _su2_kernel
    paths=list((root/'pyqed/mps/nonabelian').glob('nc_*'))+[
        root/'pyqed/mps/nonabelian/cbe.py',root/'pyqed/mps/nonabelian/cbe_selection.py',
        root/'pyqed/mps/nonabelian/_su2_kernel.pyx',root/'pyqed/mps/nonabelian/multiplet_construction.py',
        root/'pyqed/qchem/hf/rhf.py',Path(__file__).resolve(),
        Path(__file__).with_name('_allocation_probe.py'),Path(__file__).with_name('_allocation_probe.c')]
    archive=output/'sources';archive.mkdir(exist_ok=True)
    for path in paths:
        if path.is_file():
            data=path.read_bytes();(archive/hashlib.sha256(data).hexdigest()).write_bytes(data)
    provenance=dict(basis=args.basis,ao_integrals='s8',active_electrons=4,hf_initial_guess='atom',hf_tolerance=1e-12,
        seed=11,seed_D=args.seed_D,D=args.D,nsweeps=args.sweeps,min_sweeps=args.sweeps,
        tolerance=1e-9,local_tolerance=1e-11,n_threads=num_workers(),outer_workers=1,
        heap_accounting='malloc_in_use_ranges' if _heap_statistics else None,
        python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        cache_policy='Shared process, MPO then NC; timings are not cold-start timings.',
        sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()},
        extension_sha256=hashlib.sha256(Path(_su2_kernel.__file__).read_bytes()).hexdigest())
    previous_provenance={}
    if args.resume and (output/'results.json').exists() and not (output/'provenance.json').exists():
        raise ValueError('Resume requires the original provenance.')
    if args.resume and (output/'provenance.json').exists():
        previous_provenance=json.loads((output/'provenance.json').read_text())
        if previous_provenance['extension_sha256']!=provenance['extension_sha256']:
            raise ValueError('Resume requires the original compiled solver.')
        for name,digest in previous_provenance['sources'].items():
            if '/nc_' in name and provenance['sources'].get(name)!=digest:
                raise ValueError('Resume requires the original direct NC implementation.')
        if previous_provenance['basis']!=args.basis:
            raise ValueError('Resume requires the original molecular basis.')
        prior=output/f'provenance-{time_ns()}.json'
        prior.write_bytes((output/'provenance.json').read_bytes())
        provenance['prior_run']=prior.name
        provenance['cache_policy']='Resumed process; completed verified jobs retained. Remaining jobs run MPO then NC with shared CG caches.'
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    rows=[]
    if args.resume and (output/'results.json').exists():
        previous=json.loads((output/'results.json').read_text())
        completed={(r['molecule'],r['target']) for r in previous if r['mode']=='nc'
            and previous_provenance.get('heap_accounting')==provenance['heap_accounting']
            and r['converged'] and r.get('mpo_error',1)<args.energy_tolerance
            and r.get('mpo_partner_expectation_error',1)<1e-8
            and r.get('tail_live_heap_growth_bytes',float('inf')) is not None
            and r.get('tail_live_heap_growth_bytes',float('inf'))<16*1024**2
            and r.get('tail_in_use_slope_bytes_per_cycle',float('inf')) is not None
            and r.get('tail_in_use_slope_bytes_per_cycle',float('inf'))<64*1024
            and r.get('tail_in_use_range_bytes',float('inf'))<4*1024**2
            and max(r.get('covariance_error',1),r.get('orthogonality_error',1),r.get('partner_energy_error',1))<1e-8
            and r.get('tail_energy_spread',1)<args.energy_tolerance
            and r.get('rss_envelope_growth_bytes',float('inf'))<100*1024**2
            and not r.get('hamiltonian_mpo',True)
            and r.get('max_sector_multiplicity',args.D+1)<=args.D
            and len(r['history'])==args.sweeps and r['D']==args.D and r['seed_D']==args.seed_D}
        rows=[r for r in previous if (r['molecule'],r['target']) in completed]
    else: completed=set()
    jobs=[(name,None) for name in args.molecules]
    if args.partners:
        jobs += [(name,{'Be':'L1','LiH':'E1','Li2':'E1u'}[name]) for name in args.molecules]
    for name,target_override in jobs:
        atom,group,target,core=CASES[name]
        target=target_override or target
        if (name,target) in completed:
            print(f'{name}: {target}, reusing verified completed run',flush=True)
            continue
        print(f'{name}: RHF / {args.basis} / packed s8',flush=True)
        mol=Molecule(atom=atom,unit='bohr',basis=args.basis,symmetry=group).build(eri='s8')
        mf=RHF(mol,init_guess="atom");mf.chkfile=None;mf.run(verbose=0,tol=1e-12)
        assert mf.converged, "Molecular RHF reference did not converge"
        ncas=mf.mo_coeff.shape[1]-core
        print(f'{name}: CAS(4e,{ncas}o) {group} {target}',flush=True)
        from pyqed.qchem.symmetry import mo_symmetry_representations
        from pyqed.qchem.mcscf.direct_ci import transform_active_space_spatial_integrals
        reps=mo_symmetry_representations(mol,mf.mo_coeff[:,core:],generators=True)
        h,v,ecore=transform_active_space_spatial_integrals(mf,mf.mo_coeff,ncas,core)
        settings=dict(nelec=4,spin=0,D=args.D,ecore=ecore,n_threads=num_workers(),
                      point_group=group,orbital_representations=reps,wfnsym=target,seed=11)
        energies={}
        for mode in ('mpo','nc'):
            print(f'{name}: {mode} start',flush=True);start=perf_counter()
            with ExitStack() as construction:
                if mode=='nc':
                    construction.enter_context(patch.object(MultipletMPO,'__init__',
                        side_effect=AssertionError('MPO construction entered NC setup')))
                solver=MultipletDMRG.from_integrals(h,v,**dict(settings,D=args.seed_D if mode=='nc' else args.D),
                                                   contractions='reduced' if mode=='mpo' else None)
            setup=perf_counter()-start;solver.D=args.D
            samples=[]
            finish=solver._finish_cycle
            def record(h,n):
                finish(h,n)
                if mode=='nc':
                    samples.append(dict(resident_bytes=psutil.Process().memory_info().rss,
                        live_heap_bytes=live_heap_bytes(),allocator_bytes=allocator_bytes(),
                        boundary_count=len(solver.nc._installed),
                        cached_routes=solver.nc._route_cache_size,
                        boundary_bytes=sum(v.nbytes for blocks in solver.nc._installed.values() for v in blocks.values())))
            with patch.object(solver,'_finish_cycle',record), guards() if mode=='nc' else ExitStack():
                options=dict(nsweeps=args.sweeps,tol=1e-9,local_tol=1e-11,verbose=1)
                if mode=='nc': options.update(algorithm='one_site_cbe',cbe_expansion=args.expansion,
                                              cbe_preselection=args.preselection,min_sweeps=args.sweeps)
                solver.run(**options)
            seconds=perf_counter()-start
            diagnostic=solver.symmetry_diagnostics()
            history=[dict(cycle=c['cycle'],energy=c['energy'],
                          max_residual=max((s.get('residual_norm',0) for s in c['sites']),default=0),
                          added=sum(s.get('cbe_added_multiplets',0) for s in c['sites'])) for c in solver.history]
            if mode=='nc':
                for cycle,sample in zip(history,samples): cycle.update(sample)
            tail=np.array([c['energy'] for c in history[-min(10,len(history)):]])
            row=dict(molecule=name,group=group,target=target,nelec=4,ncas=ncas,D=args.D,seed_D=args.seed_D,
                     mode=mode,seconds=seconds,setup_seconds=setup,energy=solver.energy,
                     converged=bool(solver.success),partners=len(solver.states),
                     covariance_error=float(diagnostic['covariance_error']),
                     orthogonality_error=float(diagnostic['orthogonality_error']),
                     partner_energy_error=float(np.max(abs(np.asarray(diagnostic['partner_energies_hartree'])-solver.e_tot))),
                     tail_energy_spread=float(np.ptp(tail)),history=history,
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     max_sector_multiplicity=max(max(leg.dims.values()) for leg in solver.bond_legs),
                     hamiltonian_mpo=hasattr(solver,'mpo'))
            if mode=='nc':
                row.update(cbe_added_multiplets=solver.diagnostics.get('cbe_added_multiplets',0),
                           owner_stats=solver.nc.owner.stats,
                           tail_rss_growth_bytes=max(0,samples[-1]['resident_bytes']-samples[-min(10,len(samples))]['resident_bytes']))
            energies[mode]=solver.energy
            if mode=='mpo': reference_mpo=solver.mpo
            else:
                actual=np.array([state.expectation(reference_mpo)+ecore for state in solver.states])
                row['mpo_error']=abs(energies['nc']-energies['mpo'])
                row['mpo_partner_expectations']=actual.real.tolist()
                row['mpo_partner_expectation_error']=float(np.max(abs(actual-solver.e_tot)))
                row['rss_envelope_growth_bytes']=max(0,max(c['resident_bytes'] for c in history[-10:])
                    -max(c['resident_bytes'] for c in history[:10]))
                heap=[c['live_heap_bytes'] for c in history if c['live_heap_bytes'] is not None]
                row['tail_live_heap_growth_bytes']=max(0,heap[-1]-min(heap[-10:])) if heap else None
                tail_heap=np.asarray(heap[-min(20,max(2,len(heap)//2)):])
                row['tail_in_use_slope_bytes_per_cycle']=float(np.polyfit(np.arange(len(tail_heap)),tail_heap,1)[0]) if len(tail_heap)>1 else None
                row['tail_in_use_range_bytes']=float(np.ptp(tail_heap)) if len(tail_heap) else None
            rows.append(row)
            (output/'results.json').write_text(json.dumps(rows,indent=2,default=str)+'\n')
            plot(rows,output)
            print(json.dumps({k:v for k,v in row.items() if k not in ('history','owner_stats')}),flush=True)
            assert row['converged'] and row['covariance_error']<1e-8 and row['orthogonality_error']<1e-8,row
            assert row['partner_energy_error']<1e-8,row
            if mode=='nc':
                assert row['mpo_error']<args.energy_tolerance,row
                assert len(history)==args.sweeps and row['tail_energy_spread']<args.energy_tolerance,row
                assert row['mpo_partner_expectation_error']<1e-8,row
                # RSS can oscillate when macOS compresses/reactivates allocator pages.
                # Independently test actual in-use ranges, not allocator statistics.
                assert row['rss_envelope_growth_bytes']<100*1024**2,row
                if row['tail_live_heap_growth_bytes'] is not None:
                    assert row['tail_in_use_range_bytes']<4*1024**2,row
                    assert row['tail_in_use_slope_bytes_per_cycle'] is None or row['tail_in_use_slope_bytes_per_cycle']<64*1024,row
                assert len({c['boundary_count'] for c in history[-10:]})==1,row
                assert all(c['cached_routes']<=solver.nc._route_cache_limit for c in history),row
                assert not row['hamiltonian_mpo'] and row['max_sector_multiplicity']<=args.D,row
            solver.close()
    return rows


def plot(rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    axes=axes.ravel()
    for row in rows:
        label=f"{row['molecule']} {row['target']} ({row['nelec']}e,{row['ncas']}o) {row['mode']}"
        e=np.array([c['energy'] for c in row['history']])
        axes[0].semilogy(np.arange(1,len(e)+1),np.maximum(abs(e-e[-1]),1e-15),label=label)
        if row['mode']=='nc':
            rss=np.array([c['resident_bytes'] for c in row['history']],dtype=float)
            axes[3].plot([c['cycle'] for c in row['history']],(rss-rss[0])/1024**2,
                         label=f"{row['molecule']} {row['target']}")
            axes[1].plot([c['cycle'] for c in row['history']],[c['added'] for c in row['history']],label=f"{row['molecule']} {row['target']}")
            axes[2].semilogy([c['cycle'] for c in row['history']],
                np.maximum([c['max_residual'] for c in row['history']],1e-15),label=f"{row['molecule']} {row['target']}")
    for ax in axes:
        ax.set_xlabel('Full sweep cycle');ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('|E − final E| (hartree)');axes[1].set_ylabel('Added multiplets per cycle')
    axes[2].set_ylabel('Largest local Davidson residual')
    axes[3].set_ylabel('RSS change since first cycle (MiB)')
    for i,ax in enumerate(axes):
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=6 if i==0 else 8)
    fig.suptitle('Direct compiled NC CBE · larger active spaces and sustained sweeps')
    fig.savefig(output/'nc_cbe_stability.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('/private/tmp/nc-cbe-stability'))
    p.add_argument('--molecules',nargs='+',choices=list(CASES),default=list(CASES))
    p.add_argument('--resume',action='store_true')
    p.add_argument('--partners',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--basis',default='6-311g');p.add_argument('--D',type=int,default=16)
    p.add_argument('--seed-D',type=int,default=2);p.add_argument('--expansion',type=int,default=4)
    p.add_argument('--preselection',type=int,default=16);p.add_argument('--sweeps',type=int,default=50)
    p.add_argument('--energy-tolerance',type=float,default=1e-7)
    with threadpool_limits(limits=1): run(p.parse_args())
