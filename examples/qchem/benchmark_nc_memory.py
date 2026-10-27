"""Audit Li2's largest NC CBE cases with actual in-use allocation ranges.

The numerical extension and computational sources must match the completed
all-partner MPO benchmark. This adds independent memory measurements without
rebuilding its reference MPOs. No Hamiltonian MPO enters these NC runs.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
from pyqed import num_workers
from pyqed.qchem import Molecule,RHF
from pyqed.qchem.symmetry import mo_symmetry_representations
from pyqed.qchem.mcscf.direct_ci import transform_active_space_spatial_integrals
from pyqed.mps.nonabelian import _su2_kernel
from pyqed.mps.nonabelian.multiplet_dmrg import MultipletDMRG
from pyqed.mps.nonabelian.multiplet_operator import MultipletMPO
from examples.qchem.benchmark_nc_stability import live_heap_bytes,allocator_bytes,guards


def run(folder,sweeps):
    root=Path(__file__).resolve().parents[2]
    provenance=json.loads((folder/'provenance.json').read_text())
    extension=hashlib.sha256(Path(_su2_kernel.__file__).read_bytes()).hexdigest()
    assert extension==provenance['extension_sha256']
    for name,digest in provenance['sources'].items():
        if name.startswith('pyqed/'):
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    references=json.loads((folder/'results.json').read_text())
    expected={r['target']:r for r in references if r['molecule']=='Li2' and r['mode']=='mpo'}
    assert set(expected)=={'A1g','E1u'} and all(r['converged'] for r in expected.values())
    assert live_heap_bytes() is not None,'This in-use range audit requires macOS.'
    paths=[Path(__file__),Path(__file__).with_name('_allocation_probe.py'),
           Path(__file__).with_name('_allocation_probe.c'),Path(__file__).with_name('benchmark_nc_stability.py')]
    result=dict(extension_sha256=extension,
        reference_sha256=hashlib.sha256((folder/'results.json').read_bytes()).hexdigest(),
        sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        heap_accounting='malloc_in_use_ranges',nsweeps=sweeps,cases=[])
    for path in paths:
        data=path.read_bytes();(folder/'sources'/hashlib.sha256(data).hexdigest()).write_bytes(data)
    mol=Molecule(atom='Li 0 0 -2; Li 0 0 2',unit='bohr',basis=provenance['basis'],symmetry='Dooh').build(eri='s8')
    mf=RHF(mol,init_guess='atom');mf.chkfile=None;mf.run(verbose=0,tol=1e-12)
    assert mf.converged
    h,v,ecore=transform_active_space_spatial_integrals(mf,mf.mo_coeff,25,1)
    reps=mo_symmetry_representations(mol,mf.mo_coeff[:,1:],generators=True)
    for target in ('A1g','E1u'):
        start=perf_counter()
        with patch.object(MultipletMPO,'__init__',side_effect=AssertionError('MPO entered memory audit')):
            solver=MultipletDMRG.from_integrals(h,v,nelec=4,spin=0,D=2,ecore=ecore,
                n_threads=num_workers(),point_group='Dooh',orbital_representations=reps,wfnsym=target,seed=11)
        solver.D=16;samples=np.zeros((sweeps,6));counter=[0];finish=solver._finish_cycle
        def record(h,n):
            finish(h,n);i=counter[0];counter[0]+=1
            samples[i]=[i+1,live_heap_bytes(),allocator_bytes(),
                sum(v.nbytes for blocks in solver.nc._installed.values() for v in blocks.values()),
                len(solver.nc._installed),solver.nc._route_cache_size]
            print(target,i+1,solver.energy,int(samples[i,1]),flush=True)
        with patch.object(solver,'_finish_cycle',record),guards():
            solver.run(algorithm='one_site_cbe',cbe_expansion=4,cbe_preselection=16,
                nsweeps=sweeps,min_sweeps=sweeps,local_tol=1e-11,tol=1e-9)
        diagnostic=solver.symmetry_diagnostics()
        tail=samples[-20:,1]
        slope=float(np.polyfit(np.arange(len(tail)),tail,1)[0])
        case=dict(target=target,energy=solver.energy,partners=len(solver.states),
            seconds=perf_counter()-start,mpo_error=abs(solver.energy-expected[target]['energy']),
            covariance_error=float(diagnostic['covariance_error']),
            orthogonality_error=float(diagnostic['orthogonality_error']),
            tail_in_use_slope_bytes_per_cycle=slope,tail_in_use_range_bytes=float(np.ptp(tail)),
            samples=samples.tolist(),sample_columns=['cycle','in_use_bytes','allocator_bytes',
                'boundary_bytes','boundary_count','cached_routes'])
        result['cases'].append(case)
        (folder/'memory_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        plot(result,folder)
        assert solver.success and len(solver.history)==sweeps>=50 and not hasattr(solver,'mpo')
        assert case['partners']==(1 if target=='A1g' else 2) and case['mpo_error']<1e-7
        assert max(case['covariance_error'],case['orthogonality_error'])<1e-8
        assert slope<64*1024 and np.ptp(tail)<4*1024**2,case
        assert len(set(samples[-10:,4]))==1 and samples[:,5].max()<=solver.nc._route_cache_limit
        solver.close()
    return result


def plot(result,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for case in result['cases']:
        a=np.asarray(case['samples']);label='Li2 '+case['target']
        for ax,column in zip(axes,(1,2,3)):
            ax.plot(a[:,0],(a[:,column]-a[0,column])/1024**2,label=label)
    for ax,label in zip(axes,('In-use malloc ranges','Allocator statistics','Retained NC boundaries')):
        ax.set_title(label);ax.set_xlabel('Full sweep cycle')
        ax.set_ylabel('Change since cycle 1 (MiB)');ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle('Li2 CAS(4e,25o) · actual allocations versus allocator-held storage')
    fig.savefig(folder/'nc_cbe_live_memory.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    p.add_argument('--sweeps',type=int,default=50)
    args=p.parse_args()
    with threadpool_limits(limits=1):run(args.folder,args.sweeps)
