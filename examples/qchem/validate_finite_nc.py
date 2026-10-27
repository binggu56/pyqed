"""Reproduce finite-group NC/MPO partner validation and diagnostic figures.

Small explicit integral models validate the direct solver; no global projection
enters the NC calculation. Outputs default to /private/tmp/nc-finite-validation.
Run from the repository with PYTHONPATH=. and BLAS threads limited to one.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from pyqed.qchem.symmetry import irrep_representation_matrices
from pyqed.symmetry import PointGroupSymmetry
from pyqed.mps.nonabelian.multiplet_dmrg import MultipletDMRG


def model(group, orbital, target, shells=3, spin=0):
    sym=PointGroupSymmetry(group)
    d=sym.dimension(orbital)
    reps={g:np.kron(np.eye(shells),u) for g,u in irrep_representation_matrices(group,orbital).items()}
    rng=np.random.default_rng(81)
    t=rng.normal(size=(shells,shells));t=(t+t.T)*.2
    h=np.kron(t,np.eye(d))
    # Invariant pair densities include shell-changing two-electron routes.
    v=np.zeros((len(h),)*4)
    for _ in range(3):
        a=rng.normal(size=(shells,shells));a=np.kron((a+a.T)*.1,np.eye(d))
        v+=np.einsum('pq,rs->pqrs',a,a)
    return h,v,dict(nelec=2,spin=spin,D=8,point_group=group,wfnsym=target,orbital_representations=reps)



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('/private/tmp/nc-finite-validation'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    from examples.qchem.benchmark_nc_stability import guards
    from pyqed import num_workers
    from pyqed.mps.nonabelian import _su2_kernel
    import hashlib
    records=[]
    cases=[('D3h',"E'","E'",0),('D3h',"E''","E'",0),
           ('D6h','E1u','E2g',0),('D6h','E2g','E2g',0),('D6h','E2u','E2g',0),
           ('D3h',"E''","A2'",2),('D6h','E2u','A2g',2),('D2h','B1u','Ag',0)]
    for group,orbital,target,spin in cases:
        h,v,options=model(group,orbital,target,spin=spin)
        ref=MultipletDMRG.from_integrals(h,v,contractions='reduced',**options).run(nsweeps=12)
        with guards():
            nc=MultipletDMRG.from_integrals(h,v,**dict(options,D=1));nc.D=options['D']
            nc.vectors[0]=nc.vectors[0]*(1+.2j)
            nc.run(nsweeps=12,min_sweeps=3)
        report=nc.symmetry_diagnostics()
        energies=np.array([s.expectation(ref.mpo) for s in nc.states]).real
        error=float(np.max(np.abs(energies-np.asarray(ref.e_tot))))
        assert nc.success and ref.success and error<1e-9
        assert max(report['covariance_error'],report['orthogonality_error'])<1e-10
        records.append(dict(group=group,orbital=orbital,target=target,spin=spin,
            nc_energy=np.asarray(nc.e_tot).tolist(),mpo_energy=np.asarray(ref.e_tot).tolist(),
            partner_expectations=energies.tolist(),max_error=error,
            covariance=report['covariance_error'],orthogonality=report['orthogonality_error'],
            sweeps=len(nc.history),largest_multiplicity=max(max(leg.dims.values()) for leg in nc.bond_legs)))
    # Molecular validation uses the same native integrals for both solvers.
    from pyqed.qchem import Molecule, RHF, CASSCF
    from pyqed.qchem.dmrg.dmrg import DMRG
    from examples.qchem.validate_molecular_symmetry import geometry
    for molecule,group,target,ncas,nelec in [('H3+', 'D3h',"E'",3,2),
                                           ('benzene','D6h','E1u',6,6)]:
        if molecule=='H3+':
            mol=Molecule(atom='H 1 0 0; H -0.5 0.8660254037844386 0; H -0.5 -0.8660254037844386 0',
                unit='bohr',basis='sto-3g',charge=1,symmetry=group).build(eri='s8')
        else:
            atom,unit,charge,*_=geometry('benzene')
            mol=Molecule(atom=atom,unit=unit,basis='sto-3g',symmetry=group).build(eri='s8')
        mf=RHF(mol);mf.chkfile=None;mf.run(verbose=0,tol=1e-12,conv_tol_grad=1e-8)
        coeff=mf.mo_coeff
        if molecule=='benzene':
            pz=[i for i,label in enumerate(mol.ao_labels()) if '2pz' in label]
            active=np.flatnonzero(np.sum(coeff[pz]**2,axis=0)>.1)
            coeff=CASSCF(mf,6,6).reorder_mo_for_active_orbitals(coeff,active)
        ref=DMRG(mf,ncas,nelec,D=32,spin=0,symmetry='su2',verbose=0).run(
            mo_coeff=coeff,wfnsym=target,spatial_method='multiplet',nsweeps=12)
        with guards():
            nc=DMRG(mf,ncas,nelec,D=32,spin=0,symmetry='su2',verbose=0).run(
                mo_coeff=coeff,wfnsym=target,nsweeps=12)
        report=nc.symmetry_diagnostics()
        energies=np.array([state.expectation(ref.mpo)+ref.ecore for state in nc.states]).real
        error=float(np.max(np.abs(energies-np.asarray(ref.e_tot))))
        assert nc.success and ref.success and error<1e-9
        assert max(report['covariance_error'],report['orthogonality_error'])<1e-10
        records.append(dict(group=group,orbital=molecule,target=target,spin=0,
            nc_energy=np.asarray(nc.e_tot).tolist(),mpo_energy=np.asarray(ref.e_tot).tolist(),
            partner_expectations=energies.tolist(),max_error=error,
            covariance=report['covariance_error'],orthogonality=report['orthogonality_error'],
            sweeps=len(nc.history),largest_multiplicity=max(max(leg.dims.values()) for leg in nc.bond_legs)))
    (args.output/'results.json').write_text(json.dumps(records,indent=2))
    (args.output/'provenance.json').write_text(json.dumps(dict(threads=num_workers(),
        extension=_su2_kernel.__file__,extension_sha256=hashlib.sha256(Path(_su2_kernel.__file__).read_bytes()).hexdigest()),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    labels=[f"{r['group']} {r['orbital']}\n→ {r['target']}, 2S={r['spin']}" for r in records]
    x=np.arange(len(records))
    axes[0].bar(x,np.maximum([r['max_error'] for r in records],1e-16),color='#236b8e')
    axes[0].axhline(1e-9,color='#b54b36',linestyle='--',label='Acceptance: 1e-9 Ha')
    axes[0].set(yscale='log',ylabel='Maximum partner energy error (Ha)',title='Direct NC vs independent MPO')
    axes[0].legend(fontsize=8)
    axes[1].plot(x,np.maximum([r['covariance'] for r in records],1e-16),'o-',label='Covariance')
    axes[1].plot(x,np.maximum([r['orthogonality'] for r in records],1e-16),'s-',label='Orthogonality')
    axes[1].axhline(1e-10,color='#b54b36',linestyle='--')
    axes[1].set(yscale='log',ylabel='Absolute error',title='Complete partner structure')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x,labels,rotation=60,ha='right',fontsize=8)
        ax.grid(axis='y',alpha=.2)
    fig.savefig(args.output/'finite_nc_all_partner_validation.png',dpi=180)
    print(json.dumps(dict(cases=len(records),max_partner_error=max(r['max_error'] for r in records),output=str(args.output))))


if __name__=='__main__':
    main()
