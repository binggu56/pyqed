"""Independent PySCF validation of molecular symmetry (not a certification).

Packed AO integrals, reduced SU(2) MPS DMRG, and PySCF determinant FCI references.
Benzene uses the supported D2h subgroup, not full D6h. Run with PYTHONPATH=.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/matplotlib-codex')
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pyscf
from pyscf import scf, mcscf, fci, lib
from pyqed.qchem import Molecule, RHF, CASSCF
from pyqed.qchem.mcscf.direct_ci import CASCI, transform_active_space_spatial_integrals
from pyqed.qchem.dmrg.dmrg import DMRG
from examples.qchem.d3h_targeted_solvers import covariance_errors


def geometry(name):
    if name=='h3plus':
        return 'H 1 0 0; H -.5 .8660254037844386 0; H -.5 -.8660254037844386 0', 'bohr', 1, 3, 2, 'D3h', "A1'"
    if name=='water':
        return 'O 0 0 0; H .757 0 .587; H -.757 0 .587','angstrom',0,4,4,'C2v','A1'
    atoms=[]
    for symbol,radius in [('C',1.397),('H',2.479)]:
        for angle in np.arange(6)*np.pi/3:
            atoms.append(f'{symbol} {radius*np.cos(angle)} {radius*np.sin(angle)} 0')
    return '; '.join(atoms),'angstrom',0,6,6,'D2h','Ag'


def validate(name, include_dmrg=True):
    atom,unit,charge,ncas,nelecas,group,target=geometry(name)
    references={}
    for enabled in (False,True):
        mol=Molecule(atom=atom,unit=unit,charge=charge,basis='sto-3g', symmetry='D2h' if name=='benzene' and enabled else enabled).build(eri='s8')
        mf=RHF(mol); mf.chkfile=None
        mf.run(verbose=0,tol=1e-12,conv_tol_grad=1e-8)
        mf.cholesky_tol=1e-12
        references[enabled]=mf
    mf=references[True]; assert mf.mol.groupname==group
    pmol=mf.mol.topyscf(); pmol.verbose=0
    pmf=scf.RHF(pmol); pmf.conv_tol=1e-12; pmf.conv_tol_grad=1e-8; pmf.kernel()
    permutation=mf.mol.pyscf_ao_permutation(pmol)
    def to_reference(coeff):
        result=np.empty_like(coeff); result[permutation]=coeff
        return result
    coeff=mf.mo_coeff
    if name=='benzene':
        pz=[i for i,label in enumerate(mf.mol.ao_labels()) if '2pz' in label]
        active=np.flatnonzero(np.sum(coeff[pz]**2,axis=0)>.1).tolist()
        assert len(active)==6, active
        coeff=CASSCF(mf,ncas,nelecas).reorder_mo_for_active_orbitals(coeff,active)
    rows=[]
    def record(method,flag,energy,ref,elapsed,**extra):
        row=dict(molecule=name,group=group,method=method,symmetry=flag,energy=float(np.ravel(energy)[0]),reference=float(np.ravel(ref)[0]),seconds=elapsed,**extra)
        row['error']=abs(row['energy']-row['reference']); rows.append(row)
        print(json.dumps(row),flush=True)
        assert row['error']<1e-7, row
    for flag in (False,True): record('RHF',flag,references[flag].e_tot,pmf.e_tot,0)
    pref=mcscf.CASCI(pmf,ncas,nelecas); pref.fcisolver=fci.direct_spin0.FCI(pmol); pref.fcisolver.conv_tol=1e-12
    pref.kernel(to_reference(coeff))
    h1,eri,ecore=transform_active_space_spatial_integrals(mf,coeff,ncas,(mf.mol.nelec-nelecas)//2,use_cholesky=True)
    common=fci.direct_spin0.FCI();common.conv_tol=1e-12
    common_energy=common.kernel(h1,eri,ncas,nelecas,ecore=ecore)[0]
    for flag in (False,True):
        start=time.perf_counter()
        ci=CASCI(references[flag],ncas,nelecas,multiplicity=1,verbose=0).run(mo_coeff=coeff,wfnsym=target if flag else None)
        record('CASCI',flag,ci.e_tot,pref.e_tot,time.perf_counter()-start,shared_integral_error=abs(float(ci.e_tot[0])-common_energy))
        assert abs(float(ci.e_tot[0])-common_energy)<1e-8
        start=time.perf_counter()
        mc=CASSCF(references[flag],ncas,nelecas,multiplicity=1,verbose=0,use_cholesky=True,
                  max_cycle=80,auto_active_restarts=False,conv_tol=1e-10,conv_tol_grad=1e-6,conv_tol_grad_relaxed=1e-6).run(mo_coeff=coeff,wfnsym=target if flag else None)
        elapsed=time.perf_counter()-start
        assert mc.converged
        pc=mcscf.CASSCF(pmf,ncas,nelecas); pc.fcisolver=fci.direct_spin0.FCI(pmol); pc.fcisolver.conv_tol=1e-12
        pc.conv_tol=1e-10; pc.conv_tol_grad=1e-6
        pcoeff=to_reference(mc.mo_coeff)
        fixed=pc.casci(pcoeff)[0]
        gradient=float(np.linalg.norm(pc.get_grad(pcoeff)))
        record('CASSCF fixed orbitals',flag,mc.e_tot,fixed,elapsed,reference_gradient=gradient)
        assert gradient<3e-6, gradient
        pc.kernel(pcoeff)
        assert pc.converged
        record('CASSCF restart',flag,mc.e_tot,pc.e_tot,0)
        if include_dmrg:
            start=time.perf_counter()
            driver=DMRG(references[flag],ncas,nelecas,D=32,spin=0,symmetry='su2',tol=1e-9,verbose=0)
            solver=driver.run(mo_coeff=coeff,wfnsym=target if flag else None,nsweeps=10)
            record('MPS DMRG',flag,driver.e_tot,pref.e_tot,time.perf_counter()-start,solver_module=type(solver).__module__,shared_integral_error=abs(float(np.ravel(driver.e_tot)[0])-common_energy))
            assert abs(float(np.ravel(driver.e_tot)[0])-common_energy)<1e-8
            assert type(solver.states[0]).__module__.startswith('pyqed.mps')
            if hasattr(solver,'close'): solver.close()
    if name=='h3plus':
        # Compare both E rows to the independent unfiltered singlet spectrum.
        spectrum=fci.direct_spin0.FCI(pmol); spectrum.nroots=6
        probe=mcscf.CASCI(pmf,ncas,nelecas); probe.fcisolver=spectrum
        probe.kernel(to_reference(coeff))
        pair=CASCI(mf,ncas,nelecas,multiplicity=1,verbose=0).run(mo_coeff=coeff,wfnsym="E'")
        for energy in pair.e_tot:
            assert np.min(abs(probe.e_tot-energy))<1e-7
        if include_dmrg:
            driver=DMRG(mf,ncas,nelecas,D=8,spin=0,symmetry='su2',tol=1e-9,verbose=0)
            solver=driver.run(mo_coeff=coeff,wfnsym="E'",nsweeps=8)
            diagnostics=covariance_errors(solver)
            assert diagnostics['covariance_error']<1e-7
            assert diagnostics['orthogonality_error']<1e-7
            independent_pair=[float(probe.e_tot[np.argmin(abs(probe.e_tot-e))]) for e in pair.e_tot]
            record('MPS DMRG E pair',True,driver.e_tot,independent_pair,0,**diagnostics)
            np.testing.assert_allclose(driver.e_tot,pair.e_tot,atol=1e-7,rtol=0)
            solver.close()
    return rows


def plot(rows,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    molecules=list(dict.fromkeys(r['molecule'] for r in rows))
    fig,axes=plt.subplots(1,len(molecules),figsize=(4.2*len(molecules),4.5),layout='constrained',squeeze=False)
    for ax,name in zip(axes[0],molecules):
        subset=[r for r in rows if r['molecule']==name]
        methods=list(dict.fromkeys(r['method'] for r in subset))
        for flag,color,marker,offset in [(False,'#D55E00','x',-.1),(True,'#0072B2','o',.1)]:
            selected=[r for r in subset if r['symmetry']==flag]
            ax.scatter([methods.index(r['method'])+offset for r in selected],
                       [max(r['error'],1e-15) for r in selected],color=color,marker=marker,label='on' if flag else 'off')
        ax.axhline(1e-7,color='gray',linestyle='--',linewidth=1)
        ax.set(yscale='log',ylim=(3e-16,3e-7),title=f"{name} / {subset[0]['group']}",
               xticks=range(len(methods)),xticklabels=methods)
        ax.tick_params(axis='x',labelrotation=65,labelsize=8)
        ax.spines[['top','right']].set_visible(False)
    axes[0,0].set_ylabel('|Energy − reference| / Hartree')
    axes[0,-1].legend(title='Symmetry',frameon=False)
    for ext in ('png','pdf'): fig.savefig(directory/f'independent-symmetry-validation.{ext}',dpi=300)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--molecules',nargs='+',default=['h3plus','water','benzene'],choices=['h3plus','water','benzene'])
    parser.add_argument('--skip-dmrg',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=Path('/private/tmp/pyqed-symmetry-validation'))
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True);lib.num_threads(1)
    rows=[]
    for name in args.molecules:
        rows.extend(validate(name,not args.skip_dmrg))
        (args.output_dir/'validation.json').write_text(json.dumps(rows,indent=2));plot(rows,args.output_dir)
    (args.output_dir/'provenance.json').write_text(json.dumps(dict(pyscf_version=pyscf.__version__,basis='sto-3g',cholesky_tol=1e-12,energy_tolerance=1e-7,shared_integral_tolerance=1e-8),indent=2))

if __name__=='__main__': main()
