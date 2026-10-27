"""Compiled direct NC CBE: independent MPO actions, actual bond growth and partners."""
import numpy as np
import pytest
from pyqed.symmetry import OrbitalRotationSymmetry
from pyqed.mps.nonabelian.multiplet_dmrg import MultipletDMRG


def model(group, shells=3):
    orbital,target={'Coov':('E1','E2'),'Dooh':('E1u','E2g'),'SO3':('L1','L2')}[group]
    sym=OrbitalRotationSymmetry(group);d=sym.dimension(orbital)
    reps={g:np.kron(np.eye(shells),u) for g,u in sym.generators(orbital).items()}
    h=np.kron(np.diag(np.linspace(.1,.7,shells))+.3*(np.eye(shells,k=1)+np.eye(shells,k=-1)),np.eye(d))
    v=.2*np.einsum('pq,rs->pqrs',np.eye(shells*d),np.eye(shells*d))
    for i in range(shells):
        number=np.diag(np.repeat(np.arange(shells)==i,d).astype(float))
        v+=.8*np.einsum('pq,rs->pqrs',number,number)
    return h,v,dict(nelec=2,spin=0,D=1,point_group=group,orbital_representations=reps,wfnsym=target,seed=19)


@pytest.mark.parametrize('group',['Coov','Dooh','SO3'])
@pytest.mark.parametrize('direction',['lr','rl'])
@pytest.mark.parametrize('nelec',[2,4])
def test_nc_cbe_halves_match_mpo_and_preserve_complex_state(group,direction,nelec):
    from pyqed.mps.nonabelian.cbe import multiplet_pair,multiplet_pair_action
    from pyqed.mps.nonabelian import _su2_kernel as kernels
    h,v,settings=model(group,2)
    settings["nelec"]=nelec
    nc=MultipletDMRG.from_integrals(h,v,**settings)
    ref=MultipletDMRG.from_integrals(h,v,contractions='reduced',**settings)
    nc.vectors[0]=nc.vectors[0]+1j*np.random.default_rng(2).normal(size=nc.vectors[0].size)
    ref.vectors=[x.copy() for x in nc.vectors]
    left,nl=nc._edge();right,nr=nc._edge(True)
    lref,_=ref._edge();rref,_=ref._edge(True)
    matrices,layouts,lm,rm=multiplet_pair(nc,0)
    expected=multiplet_pair_action(ref,0,matrices,layouts,lm,rm,lref,rref)
    halves=kernels.spatial_nc_cbe_halves(nc,0,matrices,layouts,lm,rm,left,right)
    actual={q:np.zeros_like(m) for q,m in matrices.items()}
    for (out,inp,_),(a,b) in halves.items(): actual[out]+=a@matrices[inp]@b.T
    for q in matrices: np.testing.assert_allclose(actual[q],expected[q],atol=1e-11,rtol=0)
    nc._cbe_expansion=3;nc._cbe_preselection=None
    kernels.expand_spatial_nc_bond(nc,0,direction,left,right,nl,nr)
    after,*_=multiplet_pair(nc,0)
    for q in matrices: np.testing.assert_allclose(after[q],matrices[q],atol=1e-11,rtol=0)


@pytest.mark.parametrize('group',['Coov','Dooh','SO3'])
@pytest.mark.parametrize('nelec',[2,4])
def test_nc_cbe_grows_and_matches_mpo_all_partners(group,nelec,monkeypatch):
    from pyqed.mps.nonabelian import cbe,cbe_selection,multiplet_dmrg
    from pyqed.mps.nonabelian.multiplet_operator import MultipletMPO
    from pyqed.mps.nonabelian.fusion import TensorFusionMap
    from pyqed.mps.nonabelian.nc_multiplet import NCMultipletDMRG
    h,v,settings=model(group,4)
    settings["nelec"]=nelec
    limit=4 if nelec==2 else 12
    ref=MultipletDMRG.from_integrals(h,v,contractions='reduced',**dict(settings,D=limit)).run(nsweeps=15)
    def forbidden(*args,**kwargs): pytest.fail('Python/MPO/dense fallback entered compiled NC CBE')
    with monkeypatch.context() as patch:
        nc=MultipletDMRG.from_integrals(h,v,**settings)
        nc.D=limit
        for owner,name in [(MultipletMPO,'__init__'),(TensorFusionMap,'expand'),
                           (cbe,'expand_multiplet_bond'),(cbe,'multiplet_pair_action'),
                           (cbe,'truncate_multiplet_bond'),(cbe_selection,'shrewd_subspace'),
                           (cbe_selection,'multiplet_halves'),(multiplet_dmrg,'_contract'),
                           (np.linalg,'svd'),(np.linalg,'qr'),(np.linalg,'eigh'),
                           (NCMultipletDMRG,'_hamiltonian_action')]:
            patch.setattr(owner,name,forbidden)
        nc.run(algorithm='one_site_cbe',cbe_expansion=4,cbe_preselection=16,nsweeps=15,min_sweeps=4)
    assert nc.success and ref.success
    assert nc.diagnostics['cbe_added_multiplets']>0
    assert nc.diagnostics['cbe_compiled_update']
    assert nc.diagnostics['cbe_driver']=='compiled_nc'
    assert max(max(leg.dims.values()) for leg in nc.bond_legs)<=limit
    np.testing.assert_allclose(nc.e_tot,ref.e_tot,atol=1e-9,rtol=0)
    report=nc.symmetry_diagnostics()
    assert len(nc.states)==OrbitalRotationSymmetry(group).dimension(settings['wfnsym'])
    assert report['covariance_error']<1e-10 and report['orthogonality_error']<1e-10
    np.testing.assert_allclose([s.expectation(ref.mpo) for s in nc.states],nc.e_tot,atol=1e-9,rtol=0)


@pytest.mark.parametrize('options', [dict(cbe_expansion=0),dict(cbe_expansion=True),
    dict(cbe_preselection=0),dict(cbe_selection='exact'),dict(min_sweeps=0),
    dict(min_sweeps=3,nsweeps=2),dict(nsweeps='bad')])
def test_nc_cbe_rejects_invalid_controls(options):
    h,v,settings=model('Coov',2)
    solver=MultipletDMRG.from_integrals(h,v,**settings)
    with pytest.raises(ValueError): solver.run(algorithm='one_site_cbe',**options)


def test_large_basis_rhf_initialization_does_not_assign_core_occupations():
    from pyqed.qchem import Molecule,RHF
    from pyqed.qchem.symmetry import SymmetryEigensolver
    mol=Molecule(atom='Be 0 0 0',basis='6-311g',symmetry='SO3').build(eri='s8')
    # The bare core ordering splits a p shell; physical Fock occupations do not.
    with pytest.raises(ValueError,match='split a complete'):
        SymmetryEigensolver(mol).eigh(mol.hcore,nocc=2)
    mf=RHF(mol,init_guess='atom');mf.chkfile=None;mf.run(verbose=0,tol=1e-11)
    assert mf.converged
    np.testing.assert_allclose(np.trace(mf.dm@mol.overlap),4,atol=1e-9)
    representations=mol.symmetry_info.ao_representations
    # Continuous generators commute with the metric-weighted density.
    for name,g in representations.items():
        if name.startswith('L'):
            np.testing.assert_allclose(g@mf.dm,mf.dm@g.conj().T,atol=1e-8)


@pytest.mark.parametrize('group',['Coov','Dooh','SO3'])
def test_nc_cbe_repeated_run_and_close_bound_storage(group):
    h,v,settings=model(group,4)
    solver=MultipletDMRG.from_integrals(h,v,**settings);solver.D=4
    solver.run(algorithm='one_site_cbe',nsweeps=6,min_sweeps=6)
    before=solver.energy
    count=len(solver.nc._installed)
    solver.run(algorithm='one_site_cbe',nsweeps=20,min_sweeps=20)
    assert solver.success and len(solver.history)==20
    assert solver.energy==pytest.approx(before,abs=1e-10)
    assert len(solver.nc._installed)==count==2*(solver.nsites+1)
    assert solver.nc.owner.stats['boundary_count']==count
    assert solver.nc._route_cache_size<=solver.nc._route_cache_limit
    assert len(solver.nc._route_cache)<=64
    assert np.ptp([c['energy'] for c in solver.history])<1e-10
    solver.close()
    assert not solver.nc._installed and solver.nc.owner.stats['boundary_count']==0
    assert solver._cbe_decomposition is None
    assert not solver.nc._route_cache and solver.nc._route_cache_size==0


def test_cbe_svd_workspace_guard_precedes_allocation():
    from types import SimpleNamespace
    from pyqed.mps.nonabelian._su2_kernel import multiplet_cbe_selection
    matrix=SimpleNamespace(ndim=2,size=10**10,shape=(100000,100000))
    with pytest.raises(MemoryError,match='SVD workspace'):
        multiplet_cbe_selection({'q':matrix},{},direction='rl',expansion_size=1)


@pytest.mark.parametrize('group,target',[('Coov','A2'),('Dooh','A2g'),('SO3','L1')])
def test_nc_cbe_triplets_preserve_all_orbital_partners(group,target):
    h,v,settings=model(group,4);settings.update(spin=2,wfnsym=target)
    reference=MultipletDMRG.from_integrals(h,v,contractions='reduced',**dict(settings,D=4)).run(nsweeps=12)
    solver=MultipletDMRG.from_integrals(h,v,**settings);solver.D=4
    solver.run(algorithm='one_site_cbe',nsweeps=12)
    assert solver.success and reference.success
    assert solver.diagnostics['cbe_added_multiplets']>0
    np.testing.assert_allclose(solver.e_tot,reference.e_tot,atol=1e-10,rtol=0)
    diagnostics=solver.symmetry_diagnostics()
    assert diagnostics['covariance_error']<1e-10 and diagnostics['orthogonality_error']<1e-10
    np.testing.assert_allclose(diagnostics['partner_energies_hartree'],reference.e_tot,atol=1e-10,rtol=0)


def test_nc_cbe_validates_changed_bond_budget():
    h,v,settings=model('Coov',2)
    solver=MultipletDMRG.from_integrals(h,v,**settings);solver.D=0
    with pytest.raises(ValueError,match='D must be a positive integer'):
        solver.run(algorithm='one_site_cbe')
