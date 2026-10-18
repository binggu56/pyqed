"""Ground-state D2h H ladders: reduced spatial versus SU2-only MPS DMRG.

Native packed-s8 AO integrals; independent PySCF FCI contractions of the same
active integrals through --fci-max-orbitals (default 12). Larger spaces report
energy spread against the lowest computed DMRG energy, not an accuracy bound.
Equal input orbitals and ordering; both modes use the compiled complementary
engine and D caps total reduced bond size. First/warm runs share a process.
Profiles are separate instrumented runs and do not enter wall-time summaries.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/matplotlib-pyqed')
import argparse
import cProfile
import json
import pstats
from pathlib import Path
from time import perf_counter
import numpy as np
from examples.qchem.benchmark_su2_qcdmrg_scaling import peak_rss_mib
from pyqed import num_workers
from pyqed.qchem import Molecule, RHF
from pyqed.qchem.dmrg.dmrg import DMRG
from pyqed.qchem.symmetry import mo_symmetry_representations
from pyqed.qchem.mcscf.direct_ci import transform_active_space_spatial_integrals
from pyqed.mps.nonabelian.multiplet_sites import orbital_multiplets


def model(n):
    atom=[('H',(x,2.4*(i-(n//2-1)/2),0.)) for i in range(n//2) for x in (-.9,.9)]
    mol=Molecule(atom=atom,unit='bohr',basis='sto-3g',symmetry='D2h').build(eri='s8')
    mf=RHF(mol);mf.chkfile=None;mf.run(verbose=0,tol=1e-12)
    transform,_,_=orbital_multiplets('D2h',mo_symmetry_representations(mol,mf.mo_coeff))
    return mf,mf.mo_coeff@transform


def spatial_weights(state, mol, coeff):
    """Character-projector weights from MPS overlaps, without projecting the state.

    Diagnostic for SU(2)-only states in diagonal real Abelian orbital irreps.
    Exact orbital phase gates reuse the reduced orbital-transform machinery.
    """
    from pyqed.mps.nonabelian.orbital_transform import apply_spatial_orbital_transform
    group = mol.symmetry_info.group
    if not group.abelian:
        raise ValueError('This diagnostic requires Abelian orbital irreps.')
    representations = mo_symmetry_representations(mol, coeff)
    norm = state.overlap(state)
    expectations = []
    for name in group.operation_names:
        matrix = representations[name]
        phases = np.sign(np.diag(matrix).real)
        np.testing.assert_allclose(matrix, np.diag(phases), atol=1e-10, rtol=0)
        transformed = apply_spatial_orbital_transform(
            state, np.diag(phases), inverse=False, cutoff=0, max_bond=None)
        value = state.overlap(transformed) / norm
        assert abs(value.imag) < 1e-9
        expectations.append(value.real)
    weights = {label: float(np.vdot(characters, expectations).real / len(expectations))
               for label, characters in group.character_table.items()}
    assert min(weights.values()) > -1e-8
    assert abs(sum(weights.values()) - 1) < 1e-8
    return weights


def calculate(mf,coeff,n,method,D,tol=1e-7,workspace_mib=256,nsweeps=48,
              *, diagnose_symmetry=False, verbose=0, initial_guess=None, checkpoint=None,
              local_solver_schedule=None, fixed_sweeps=False):
    start=perf_counter()
    driver=DMRG(mf,n,n,D=D,symmetry='su2',spin=0,tol=tol,verbose=verbose)
    options = dict(local_solver_kwargs={'max_space':48, 'itermax':200,
        'tol':tol*tol, 'tol_residual':tol,
        'workspace_budget_bytes':workspace_mib*1024**2},
        verify_returned_mps_energy=True)
    if method=='reduced':
        options['wfnsym']='Ag'
    if local_solver_schedule is not None:
        options['local_solver_schedule'] = local_solver_schedule
    if fixed_sweeps:
        options.update(conv_tol=-1., require_convergence=False, cutoff=1e-20,
                       mixer_nsweeps=0)
        options['local_solver_kwargs'].update(itermax=100, max_space=50)
    solver=driver.run(mo_coeff=coeff,nsweeps=nsweeps,n_threads=num_workers(),
                      initial_guess=initial_guess,**options)
    elapsed=perf_counter()-start
    assert fixed_sweeps or driver.success,(n,method,driver.message)
    row=dict(norb=n,method=method,D=D,tolerance=tol,seconds=elapsed,energy=float(np.asarray(driver.e_tot).reshape(-1)[0]))
    row['sweeps']=solver.ncompleted
    row['half_sweeps']=solver.ncompleted_half_sweeps
    row['converged']=bool(driver.success)
    row['local_solver_schedule'] = local_solver_schedule
    row['process_peak_rss_mib']=peak_rss_mib()
    row['sweep_energies']=[float(entry['energy']) for entry in solver.history if entry.get('energy') is not None]
    owner=solver.history[-1]['moving_environment_stats']['su2_moving_environment']
    assert owner['half_sweep_python_bond_callbacks']==0
    row['fused_output_calls']=owner['raw_output_fusion_gemm_calls']
    row['phase_seconds']={key:float(value) for key,value in owner.items() if key.endswith('_seconds')}
    row['memory_bytes']={key:int(value) for key,value in owner.items() if key.endswith('_bytes')}
    row['operation_counts']={key:int(value) for key,value in owner.items()
                             if key.endswith(('_calls', '_iterations', '_hits', '_builds', '_rebuilds',
                                              '_refreshes', '_reuses'))}
    row['engine']='compiled complementary'
    row['output_fusion_enabled']=os.environ.get('PYQED_SU2_DISABLE_OUTPUT_FUSION', '')[:1] in ('', '0')
    row['contractions']='selectively fused reduced' if row['output_fusion_enabled'] else 'ordinary reduced'
    row['actual_bond_dimension'] = max(sum(site.legs[2].dims.values())
                                      for site in solver.ground_state.tensors)
    if diagnose_symmetry and method == 'off':
        start = perf_counter()
        row['spatial_weights'] = spatial_weights(solver.ground_state, mf.mol, coeff)
        row['diagnostic_seconds'] = perf_counter() - start
    if method=='reduced':
        assert tuple(driver.state_irreps)==('Ag',)
        assert solver.target_sector.point_group==0
    if checkpoint is not None:
        import pickle
        with Path(checkpoint).open('wb') as stream:
            pickle.dump(solver.ground_state, stream, protocol=pickle.HIGHEST_PROTOCOL)
        row['checkpoint'] = str(checkpoint)
    if fixed_sweeps and initial_guess is not None:
        from pyqed.qchem.dmrg.backends.nonabelian import _expectation_from_nonabelian_mps
        active = driver._active_hamiltonian
        row['initial_energy'] = _expectation_from_nonabelian_mps(
            initial_guess, active.mpo, moving_environment=active.moving_environment)
    if hasattr(solver,'close'):solver.close()
    return row


def plot(data,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=data['runs'];sizes=list(dict.fromkeys(r['norb'] for r in rows))
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for method in dict.fromkeys(r['method'] for r in rows):
        for first,ax in zip((True,False),axes[:2]):
            values=[np.median([r['seconds'] for r in rows if r['norb']==n and r['method']==method and
                (r['repeat']==0 if first else r['repeat']>0)]) for n in sizes]
            ax.plot(sizes,values,'o-',label=method)
        axes[2].plot(sizes,[max(1e-16,max(r['error'] for r in rows if r['norb']==n and r['method']==method)) for n in sizes],'o-',label=method)
    for ax in axes:
        ax.set_xlabel('Active orbitals = electrons');ax.set_yscale('log');ax.legend();ax.spines[['top','right']].set_visible(False)
    axes[0].set(title='First call',ylabel='Setup + solve / s');axes[1].set(title='Warm median',ylabel='Setup + solve / s')
    axes[2].set(title='FCI error / DMRG energy spread',ylabel='Energy error / Ha')
    note='D2h H ladders, STO-3G; same orbitals; compiled SU(2); D='+','.join(str(d) for d in sorted({r['D'] for r in rows}))
    if any(r.get('reference_kind')=='lowest DMRG' for r in rows):
        note+='\nDMRG spread measures agreement, not error against the exact energy.'
    fig.supxlabel(note,fontsize=10)
    for ext in ('png','pdf'):fig.savefig(directory/f'abelian-dmrg-scaling.{ext}',dpi=170)
    plt.close(fig)
    phases=('owned_prepare_seconds','owned_solve_seconds','owned_environment_seconds','owned_split_seconds')
    labels=('Prepare','Davidson','Environments','Split')
    selected=[r for r in rows if r['repeat']>0]
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    bottom=np.zeros(len(selected))
    for key,label in zip(phases,labels):
        values=np.array([r['phase_seconds'].get(key,0.) for r in selected])
        ax.bar(np.arange(len(selected)),values,bottom=bottom,label=label)
        bottom+=values
    ax.bar(np.arange(len(selected)),np.maximum(0.,[r['seconds'] for r in selected]-bottom),
           bottom=bottom,label='Other / verification')
    ax.set_xticks(np.arange(len(selected)),[f"CAS({r['norb']},{r['norb']})\n{r['method']} D={r['D']}" for r in selected])
    ax.set(ylabel='Wall time / s',title='Warm-run compiled sweep phases')
    ax.legend();ax.spines[['top','right']].set_visible(False)
    for ext in ('png','pdf'):fig.savefig(directory/f'abelian-dmrg-phases.{ext}',dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes',type=int,nargs='+',default=[4,6,8])
    parser.add_argument('--D',type=int,default=2048)
    parser.add_argument('--off-D',type=int,default=2048)
    parser.add_argument('--workspace-mib',type=int,default=256)
    parser.add_argument('--nsweeps',type=int,default=48)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--fci-max-orbitals',type=int,default=12,
        help='Larger spaces compare against the lowest DMRG energy, not FCI.')
    parser.add_argument('--methods',nargs='+',choices=('off','reduced'),default=['off','reduced'])
    parser.add_argument('--tolerance',type=float,default=1e-7)
    parser.add_argument('--output-dir',type=Path,default=Path('/private/tmp/abelian-dmrg-scaling'))
    args=parser.parse_args()
    if args.repeats<2 or min(args.D,args.off_D,args.workspace_mib,args.nsweeps)<1:
        parser.error('Use at least two repeats and positive bond budgets.')
    if any(n<4 or n%2 for n in args.sizes) or not np.isfinite(args.tolerance) or args.tolerance<=0:
        parser.error('Use even sizes >= 4 and a finite positive tolerance.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    from pyscf import fci,lib
    lib.num_threads(1)
    data=dict(workers=num_workers(),blas_threads=1,sweep_tol=args.tolerance,local_tol=args.tolerance,
        max_sweeps=args.nsweeps,fci_max_orbitals=args.fci_max_orbitals,
        workspace_mib=args.workspace_mib,runs=[])
    for n in args.sizes:
        print(f'CAS({n}e,{n}o): constructing native integrals', flush=True)
        mf,coeff=model(n)
        energy=None
        if n<=args.fci_max_orbitals:
            h,v,core=transform_active_space_spatial_integrals(mf,coeff,n,0,use_cholesky=False)
            ref=fci.direct_spin0.FCI();ref.conv_tol=1e-12;ref.max_cycle=200
            energy,_=ref.kernel(h,v,n,n,ecore=core)
            assert ref.converged
        for repeat in range(args.repeats):
            for method in (m for m in (('reduced','off') if repeat%2 else ('off','reduced')) if m in args.methods):
                print(f'CAS({n}e,{n}o): {method}, repeat {repeat+1}/{args.repeats}', flush=True)
                try:
                    row=calculate(mf,coeff,n,method,args.D if method=='reduced' else args.off_D,args.tolerance,args.workspace_mib,args.nsweeps)
                except RuntimeError as exc:
                    data.setdefault('failures',[]).append(dict(norb=n,method=method,
                        repeat=repeat,D=args.D if method=='reduced' else args.off_D,
                        message=str(exc)))
                    (args.output_dir/'abelian-dmrg-scaling.json').write_text(json.dumps(data,indent=2))
                    raise
                row.update(repeat=repeat,reference=None if energy is None else float(energy),
                    reference_kind='lowest DMRG' if energy is None else 'FCI',
                    error=None if energy is None else abs(row['energy']-energy))
                print(json.dumps(row),flush=True)
                data['runs'].append(row)
                (args.output_dir/'abelian-dmrg-scaling.json').write_text(json.dumps(data,indent=2))
                if energy is not None:
                    assert row['error']<1e-7,row
        if energy is None:
            selected=[r for r in data['runs'] if r['norb']==n]
            best=min(r['energy'] for r in selected)
            for row in selected:
                row.update(reference=best,error=abs(row['energy']-best))
            (args.output_dir/'abelian-dmrg-scaling.json').write_text(json.dumps(data,indent=2))
        if args.profile:
            for method in args.methods:
                profiler=cProfile.Profile();profiler.enable();calculate(mf,coeff,n,method,args.D if method=='reduced' else args.off_D,args.tolerance,args.workspace_mib,args.nsweeps);profiler.disable()
                path=args.output_dir/f'profile-{n}-{method}'
                profiler.dump_stats(str(path)+'.prof')
                with path.with_suffix('.txt').open('w') as stream:
                    pstats.Stats(profiler,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(65)
        plot(data,args.output_dir)


if __name__=='__main__':main()
