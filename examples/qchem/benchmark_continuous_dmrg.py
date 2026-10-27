"""Direct compiled spatial-rank NC versus reduced-MPO molecular DMRG.

Be/SO3, LiH/Coov and Li2/Dooh use 6-31G, packed s8 AO integrals, and
CAS(2e,8o), CAS(2e,10o), CAS(2e,16o), respectively. Both solvers retain
complete orbital multiplets, use the same D, seed and convergence settings,
and return every partner of each target. CASCI is an explicit validation
reference. These two-electron calculations do not establish many-electron
scaling. The NC run uses default dispatch and forbids Hamiltonian MPO construction and
Python numerical sweep fallbacks at runtime.

Run from the repository root with PYTHONPATH=. and BLAS threads set to one.
On macOS the extensions must share one compatible OpenMP runtime, e.g.
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib. The central num_workers()
budget sets compiled NC threads; outer Python calculations are serial.
JSON records and a reproducible timing/energy/partner figure are written
outside the repository. No active molecular database is created.
"""
import argparse
from contextlib import ExitStack
import json
import hashlib
from pathlib import Path
import platform
from time import perf_counter
from unittest.mock import patch
import numpy as np
from pyqed import num_workers
from pyqed.qchem import Molecule,RHF,CASCI
from pyqed.qchem.dmrg import DMRG
from pyqed.mps.nonabelian.multiplet_operator import MultipletMPO

CASES=[('Be','Be 0 0 0','SO3','L0','L1',1),
       ('LiH','Li 0 0 -1.5; H 0 0 1.5','Coov','A1','E1',1),
       ('Li2','Li 0 0 -2; Li 0 0 2','Dooh','A1g','E1u',2)]


def run(output,names,bond):
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    provenance=dict(basis='6-31g',unit='bohr',ao_integrals='s8',D=bond,nelec=2,
                    spin=0,seed=11,n_threads=num_workers(),outer_workers=1,
                    nsweeps=30,tolerance=1e-9,local_tolerance=1e-11,
                    python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
                    execution_order=["mpo","nc"],cache_policy="Shared process; structural CG caches are reused.")
    root=Path(__file__).resolve().parents[2]
    sources=[*sorted((root/'pyqed/mps/nonabelian').glob('nc_*')),
             root/'pyqed/mps/nonabelian/orbital_rank.hpp',
             root/'pyqed/mps/nonabelian/_su2_kernel.pyx',
             root/'pyqed/mps/nonabelian/su2_dmrg_engine.cpp',
             root/'pyqed/mps/nonabelian/su2_dmrg_engine.hpp',
             root/'pyqed/mps/nonabelian/multiplet_dmrg.py',
             root/'pyqed/mps/nonabelian/multiplet_sweep.py',
             root/'pyqed/qchem/dmrg/dmrg.py',Path(__file__).resolve()]
    provenance['implementation_sha256']={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in sources if path.is_file()}
    from pyqed.mps.nonabelian import _su2_kernel
    provenance['extension_sha256']=hashlib.sha256(Path(_su2_kernel.__file__).read_bytes()).hexdigest()
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    for name,atom,group,ground,partner,core in CASES:
        if name not in names:
            continue
        print(f'{name}: packed AO integrals and RHF',flush=True)
        mol=Molecule(atom=atom,unit='bohr',basis='6-31g',symmetry=group).build(eri='s8')
        mf=RHF(mol);mf.chkfile=None;mf.run(verbose=0,tol=1e-12)
        ncas=mf.mo_coeff.shape[1]-core
        for target in (ground,partner):
            print(f'{name}: CASCI validation, CAS(2e,{ncas}o), target {target}',flush=True)
            reference=CASCI(mf,ncas,2,ncore=core,multiplicity=1,verbose=0).run(wfnsym=target)
            expected=np.asarray(reference.e_tot).reshape(-1)
            energies={}
            for mode,method in (('mpo','multiplet'),('nc','complementary')):
                driver=DMRG(mf,ncas,2,D=bond,spin=0,symmetry='su2',tol=1e-9,verbose=0)
                print(f'{name}: {target}, {mode}',flush=True)
                start=perf_counter()
                with ExitStack() as guard:
                    if mode=='nc':
                        from pyqed.mps.nonabelian import multiplet_dmrg,multiplet_sweep,multiplet_operator,solver as local_solver
                        from pyqed.mps.nonabelian.nc_multiplet import NCMultipletDMRG
                        for owner,attribute in ((MultipletMPO,'__init__'),
                            (multiplet_dmrg.MultipletDMRG,'_run_multiplet'),
                            (multiplet_sweep.ReducedMultipletDMRG,'_optimize_site'),
                            (multiplet_sweep.ReducedMultipletDMRG,'_move_center'),
                            (multiplet_operator,'advance'),(local_solver,'_solve_packed_generalized_davidson'),
                            (NCMultipletDMRG,'_hamiltonian_action')):
                            guard.enter_context(patch.object(owner,attribute,side_effect=AssertionError(
                                f'Forbidden fallback {attribute} entered direct NC benchmark')))
                    solver=driver.run(wfnsym=target,spatial_method=method if mode=='mpo' else None,
                                      nsweeps=30,n_threads=num_workers())
                elapsed=perf_counter()-start
                diagnostic=solver.symmetry_diagnostics()
                energy=np.asarray(driver.e_tot).reshape(-1)
                dimension=mol.symmetry_info.group.irrep_dimensions[target]
                row=dict(molecule=name,atom=atom,group=group,target=target,ncas=ncas,ncore=core,
                    nelec=2,D=bond,mode=mode,n_threads=num_workers(),seconds=elapsed,
                    energies=energy.tolist(),reference=expected.tolist(),partners=len(solver.states),
                    error=float(np.max(abs(energy-expected))),converged=bool(driver.converged),
                    cycles=len(solver.history),covariance_error=float(diagnostic['covariance_error']),
                    orthogonality_error=float(diagnostic['orthogonality_error']),
                    partner_spread=float(np.ptp(energy)),hamiltonian_mpo=hasattr(solver,'mpo'))
                if mode=='nc':
                    local_records=[record for cycle in solver.history for record in cycle['sites']]
                    row.update(sweep_driver=solver.diagnostics['sweep_driver'],
                        local_solver=solver.diagnostics['local_solver'],
                        max_local_residual=max(record.get('residual_norm',0) for record in local_records),
                        local_solves=len(local_records))
                rows.append(row)
                print(json.dumps(row),flush=True)
                (output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
                assert row['converged'] and row['error']<1e-7,row
                assert len(solver.states)==dimension and tuple(solver.spatial_partner_rows)==tuple(range(dimension)),row
                assert row['covariance_error']<1e-8 and row['orthogonality_error']<1e-8,row
                np.testing.assert_allclose(diagnostic['partner_energies_hartree'],energy,atol=1e-8,rtol=0)
                if mode=='nc':
                    assert not row['hamiltonian_mpo'] and solver.contractions=='nc'
                energies[mode]=energy.copy()
                solver.close()
            np.testing.assert_allclose(energies['nc'],energies['mpo'],atol=1e-7,rtol=0)
        plot(rows,output)
        plot([r for r in rows if r["molecule"]==name],output,
             filename=f"{name.lower()}_cas2e{ncas}o_nc_mpo_benchmark.png")
    return rows


def plot(rows,output,filename="continuous_nc_mpo_benchmark.png"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=list(dict.fromkeys((r['molecule'],r['target']) for r in rows))
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for mode,color,offset,label in [('mpo','#577590',-.18,'Reduced MPO'),('nc','#e07a3f',.18,'Direct compiled NC')]:
        selected=[r for r in rows if r['mode']==mode]
        x=np.array([keys.index((r['molecule'],r['target'])) for r in selected])+offset
        for ax,key in zip(axes,('seconds','error','covariance_error')):
            ax.bar(x,[max(r[key],1e-15) for r in selected],width=.34,color=color,label=label)
            ax.set_xticks(range(len(keys)),[f'{name}\n{target}' for name,target in keys],fontsize=9)
            ax.set_yscale('log');ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Setup + sweeps (s)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
    axes[1].set_ylabel('Maximum error vs CASCI (hartree)');axes[1].axhline(1e-7,color='gray',ls='--',lw=1)
    axes[2].set_ylabel(r'Covariance error (display floor $10^{-15}$)')
    fig.suptitle('Complete orbital multiplets · CAS(2e) · 6-31G')
    fig.savefig(output/filename,dpi=160)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('/private/tmp/continuous-nc-mpo-benchmark'))
    p.add_argument('--molecules',nargs='+',choices=[x[0] for x in CASES],default=['Be','LiH','Li2'])
    p.add_argument('--D',type=int,default=16)
    a=p.parse_args();run(a.output,a.molecules,a.D)
