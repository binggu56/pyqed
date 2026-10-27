"""Direct compiled NC sweeps against independent reduced-MPO sweeps."""
import numpy as np
import pytest
from pyqed.symmetry import OrbitalRotationSymmetry
from pyqed.mps.nonabelian.multiplet_dmrg import MultipletDMRG


@pytest.mark.parametrize('group,orbital,target',[('Coov','E1','E2'),
    ('Dooh','E1u','E2g'),('SO3','L1','L2')])
@pytest.mark.parametrize('nelec',[1,2])
def test_direct_nc_sweeps_match_mpo_preserving_all_partners(group,orbital,target,nelec,monkeypatch):
    from pyqed.mps.nonabelian.multiplet_operator import MultipletMPO
    from pyqed.mps.nonabelian.fusion import TensorFusionMap
    from pyqed.qchem.dmrg import dmrg
    spatial = OrbitalRotationSymmetry(group)
    d = spatial.dimension(orbital)
    reps = {name:np.kron(np.eye(2),matrix) for name,matrix in spatial.generators(orbital).items()}
    h = np.kron([[.1,.3],[.3,.7]],np.eye(d))
    v = (.2*np.einsum('pq,rs->pqrs',np.eye(2*d),np.eye(2*d))
         +.03*np.einsum('ps,qr->pqrs',np.eye(2*d),np.eye(2*d)))
    target = orbital if nelec == 1 else target
    settings = dict(nelec=nelec,spin=nelec%2,D=4,point_group=group,
        orbital_representations=reps,wfnsym=target,seed=19)
    reference = MultipletDMRG.from_integrals(h,v,contractions='reduced',**settings).run(nsweeps=8)
    def forbidden(*args,**kwargs):
        pytest.fail('MPO construction or component/global projection entered a direct NC sweep.')
    with monkeypatch.context() as patch:
        patch.setattr(MultipletMPO,'__init__',forbidden)
        patch.setattr(TensorFusionMap,'expand',forbidden)
        patch.setattr(TensorFusionMap,'matrix',property(forbidden))
        patch.setattr(dmrg,'_build_spatial_active_hamiltonian_matrix',forbidden)
        from pyqed.mps.nonabelian import multiplet_sweep,multiplet_operator,solver as local_solver
        patch.setattr(multiplet_sweep.ReducedMultipletDMRG,'_optimize_site',forbidden)
        patch.setattr(multiplet_sweep.ReducedMultipletDMRG,'_move_center',forbidden)
        patch.setattr(MultipletDMRG,'_run_multiplet',forbidden)
        patch.setattr(multiplet_operator,'advance',forbidden)
        patch.setattr(local_solver,'_solve_packed_generalized_davidson',forbidden)
        patch.setattr(multiplet_sweep,'eigh',forbidden)
        patch.setattr(multiplet_sweep,'eigsh',forbidden)
        patch.setattr(multiplet_sweep,'_diagonal_norm',forbidden)
        from pyqed.mps.nonabelian import fusion,multiplet_dmrg
        patch.setattr(fusion,'move_multiplet_center',forbidden)
        patch.setattr(multiplet_dmrg,'_contract',forbidden)
        from pyqed.mps.nonabelian.nc_multiplet import NCMultipletDMRG
        patch.setattr(NCMultipletDMRG,'_hamiltonian_action',forbidden)
        solver = MultipletDMRG.from_integrals(h,v,contractions='nc',**settings).run(nsweeps=8)
    assert reference.success and solver.success
    assert not hasattr(solver,'mpo')
    assert not solver.diagnostics['hamiltonian_mpo']
    assert solver.diagnostics['hamiltonian_representation'] == 'normal_complementary'
    assert len(solver.states) == spatial.dimension(target)
    np.testing.assert_allclose(solver.e_tot,reference.e_tot,atol=1e-10,rtol=0)
    report = solver.symmetry_diagnostics()
    assert report['orthogonality_error'] < 1e-10
    assert report['covariance_error'] < 1e-10
    np.testing.assert_allclose(report['partner_energies_hartree'],reference.e_tot,atol=1e-10,rtol=0)
    # Every actual returned partner also has the correct reference expectation.
    np.testing.assert_allclose([state.expectation(reference.mpo) for state in solver.states],
                               reference.e_tot,atol=1e-10,rtol=0)


@pytest.mark.parametrize('group,orbital,target',[('Coov','E1','A2'),
    ('Dooh','E1u','A2g'),('SO3','L1','L1')])
def test_direct_nc_triplets_match_mpo(group,orbital,target):
    spatial = OrbitalRotationSymmetry(group)
    d = spatial.dimension(orbital)
    reps = {name:np.kron(np.eye(2),matrix) for name,matrix in spatial.generators(orbital).items()}
    h = np.kron([[.1,.3],[.3,.7]],np.eye(d))
    v = .2*np.einsum('pq,rs->pqrs',np.eye(2*d),np.eye(2*d))
    solvers = [MultipletDMRG.from_integrals(h,v,nelec=2,spin=2,D=4,
        point_group=group,orbital_representations=reps,wfnsym=target,
        contractions=mode).run(nsweeps=8) for mode in ('reduced','nc')]
    assert all(s.success for s in solvers)
    np.testing.assert_allclose(solvers[0].e_tot,solvers[1].e_tot,atol=1e-10,rtol=0)
    assert len(solvers[1].states)==spatial.dimension(target)
    assert solvers[1].symmetry_diagnostics()['covariance_error']<1e-10


def test_direct_nc_excited_multiplets_preserve_all_rows():
    spatial = OrbitalRotationSymmetry('Dooh')
    reps = {name:np.kron(np.eye(2),matrix) for name,matrix in spatial.generators('E1u').items()}
    h = np.kron([[.1,.3],[.3,.7]],np.eye(2))
    solver = MultipletDMRG.from_integrals(h,np.zeros((4,)*4),nelec=1,spin=1,D=2,
        point_group='Dooh',orbital_representations=reps,wfnsym='E1u',contractions='nc').run(nstates=2,nsweeps=8,algorithm='one_site')
    assert solver.success and len(solver.states)==4
    np.testing.assert_allclose(solver.e_tot,np.linalg.eigvalsh(h),atol=1e-10,rtol=0)
    report = solver.symmetry_diagnostics()
    assert report['orthogonality_error']<1e-10 and report['covariance_error']<1e-10
    np.testing.assert_allclose(report['partner_energies_hartree'],solver.e_tot,atol=1e-10,rtol=0)


def test_direct_nc_zero_single_shell_has_a_valid_local_action():
    spatial = OrbitalRotationSymmetry('SO3')
    solver = MultipletDMRG.from_integrals(np.zeros((1,1)),np.zeros((1,)*4),
        nelec=2,spin=0,D=1,ecore=.43,point_group='SO3',
        orbital_representations=spatial.generators('L0'),wfnsym='L0',contractions='nc').run(nsweeps=2)
    assert solver.success
    np.testing.assert_allclose(solver.e_tot,.43,atol=1e-12,rtol=0)


@pytest.mark.parametrize('algorithm',[None,'one_site','one_site_cbe','two_site'])
def test_qchem_continuous_default_uses_direct_nc(monkeypatch,algorithm):
    from pyqed.qchem import Molecule,RHF
    from pyqed.qchem.dmrg import DMRG
    from pyqed.mps.nonabelian.multiplet_operator import MultipletMPO
    mol = Molecule(atom='H 0 0 -0.7; H 0 0 .7',unit='bohr',basis='sto-3g',symmetry='Dooh').build(eri='s8')
    mf = RHF(mol);mf.chkfile=None;mf.run(verbose=0,tol=1e-11)
    reference = DMRG(mf,2,2,D=4,spin=0,symmetry='su2',verbose=0).run(
        wfnsym='A1g',spatial_method='multiplet',nsweeps=6)
    def forbidden(*args,**kwargs):
        pytest.fail('The qchem continuous default constructed a Hamiltonian MPO.')
    monkeypatch.setattr(MultipletMPO,'__init__',forbidden)
    driver = DMRG(mf,2,2,D=4,spin=0,symmetry='su2',verbose=0)
    solver = driver.run(wfnsym='A1g',nsweeps=6,**({} if algorithm is None else dict(algorithm=algorithm)))
    assert driver.spatial_method=='complementary' and solver.contractions=='nc'
    assert not hasattr(solver,'mpo') and solver.success
    if algorithm in (None,'two_site'):
        assert solver.diagnostics['two_site_driver']=='compiled_nc'
    if algorithm=='one_site_cbe':
        assert solver.diagnostics['cbe_driver']=='compiled_nc'
    np.testing.assert_allclose(solver.e_tot,reference.e_tot,atol=1e-10,rtol=0)


@pytest.mark.parametrize('group,orbital',[('Coov','E1'),('Dooh','E1u'),('SO3','L1')])
@pytest.mark.parametrize('entry',['multiplet','reduced','spatial','nc'])
@pytest.mark.parametrize('algorithm',[None,'one_site','one_site_cbe','two_site'])
def test_every_continuous_factory_defaults_to_nc(group,orbital,entry,monkeypatch,algorithm):
    from pyqed.mps.nonabelian.multiplet_sweep import ReducedMultipletDMRG
    from pyqed.mps.nonabelian.spatial_symmetry import SpatialSymmetryDMRG
    from pyqed.mps.nonabelian.nc_multiplet import NCMultipletDMRG
    from pyqed.mps.nonabelian.multiplet_operator import MultipletMPO
    classes=dict(multiplet=MultipletDMRG,reduced=ReducedMultipletDMRG,
                 spatial=SpatialSymmetryDMRG,nc=NCMultipletDMRG)
    spatial=OrbitalRotationSymmetry(group);d=spatial.dimension(orbital)
    reps={name:np.kron(np.eye(2),matrix) for name,matrix in spatial.generators(orbital).items()}
    def forbidden(*args,**kwargs): pytest.fail('A continuous factory default constructed an MPO.')
    monkeypatch.setattr(MultipletMPO,'__init__',forbidden)
    solver=classes[entry].from_integrals(np.kron([[.1,.3],[.3,.7]],np.eye(d)),np.zeros((2*d,)*4),
        nelec=1,spin=1,D=2,point_group=group,orbital_representations=reps,wfnsym=orbital).run(nsweeps=6,**({} if algorithm is None else dict(algorithm=algorithm)))
    assert isinstance(solver,NCMultipletDMRG) and solver.success and solver.contractions=='nc'
    assert solver.diagnostics['sweep_driver']=='compiled'
    assert solver.diagnostics['sweep_control']=='shared'
    assert solver.diagnostics['max_bond_mode']=='per_sector'
    if algorithm in (None,'two_site'):
        assert solver.diagnostics['two_site_driver']=='compiled_nc'
    if algorithm=='one_site_cbe':
        assert solver.diagnostics['cbe_driver']=='compiled_nc'
    assert all(record['local_solver']=='davidson' for cycle in solver.history for record in cycle['sites'])
    assert len(solver.states)==d and not hasattr(solver,'mpo')
    np.testing.assert_allclose(solver.e_tot,np.linalg.eigvalsh([[.1,.3],[.3,.7]])[0],atol=1e-10)


def test_nc_noncanonical_metric_and_complex_gauges_match_reference():
    from scipy.linalg import eigh
    spatial=OrbitalRotationSymmetry('Dooh')
    reps={name:np.kron(np.eye(5),matrix) for name,matrix in spatial.generators('A1g').items()}
    random=np.random.default_rng(171).normal(size=(5,5))
    h=(random+random.T)*.1+np.diag(np.arange(5)*.2)
    v=.13*np.einsum('pq,rs->pqrs',np.eye(5),np.eye(5))
    settings=dict(nelec=4,spin=0,D=4,point_group='Dooh',orbital_representations=reps,wfnsym='A1g')
    ref=MultipletDMRG.from_integrals(h,v,contractions='reduced',**settings)
    nc=MultipletDMRG.from_integrals(h,v,**settings)
    rng=np.random.default_rng(72)
    for i,vector in enumerate(ref.vectors):
        vector=vector+.4j*rng.normal(size=len(vector))+.3*rng.normal(size=len(vector))
        ref.vectors[i]=vector.copy();nc.vectors[i]=vector.copy()
    ref._refresh();nc._refresh()
    lh,ln=ref._boundaries('rl');rh,rn=ref._boundaries('lr')
    site=2
    metric=ref._local_action(site,ref.identity[site],ln[site],rn[site+1],assemble=True)
    local=ref._hamiltonian_action(site,lh[site],rh[site+1],assemble=True)
    values,vectors=eigh((metric+metric.conj().T)/2)
    keep=values>1e-10*max(values.max(),1e-30)
    whitening=vectors[:,keep]/np.sqrt(values[keep])
    expected=eigh(whitening.conj().T@local@whitening,eigvals_only=True)[0]
    lh,ln=nc._boundaries('rl');rh,rn=nc._boundaries('lr')
    record=nc._optimize_site(site,lh[site],rh[site+1],ln[site],rn[site+1],metric_rtol=1e-10,local_tol=1e-11)
    assert not record['diagonal_metric']
    assert record['residual_norm']<1e-11
    np.testing.assert_allclose(record['energy'],expected,atol=1e-10)
    nc._move_center(2,'lr');nc._move_center(3,'rl')
    nc._refresh()
    np.testing.assert_allclose(nc.energy,expected,atol=1e-10)


def test_nc_large_local_davidson_is_matrix_free_and_matches_ritz():
    from scipy.sparse.linalg import LinearOperator,eigsh
    from pyqed.mps.nonabelian.multiplet_sweep import _diagonal_norm
    sym=OrbitalRotationSymmetry('SO3');n=7
    reps={name:np.kron(np.eye(n),matrix) for name,matrix in sym.generators('L0').items()}
    rng=np.random.default_rng(63)
    h=rng.normal(size=(n,n));h=(h+h.T)*.15+np.diag(np.arange(n)*.2)
    solver=MultipletDMRG.from_integrals(h,np.zeros((n,)*4),nelec=6,spin=0,D=5,
        point_group='SO3',orbital_representations=reps,wfnsym='L0')
    for i in range(3): solver._move_center(i,'lr')
    site=3;count=solver.fusion_maps[site].parameter_count
    assert count>256
    lh,ln=solver._boundaries('rl');rh,rn=solver._boundaries('lr')
    diagonal=_diagonal_norm(solver.fusion_maps[site],ln[site],rn[site+1])
    assert diagonal is not None
    keep=np.flatnonzero(diagonal>1e-10*diagonal.max());scale=1/np.sqrt(diagonal[keep])
    action=solver.nc.local_action(site,solver.fusion_maps[site],lh[site],rh[site+1])
    def apply(x):
        full=np.zeros(count,complex);full[keep]=scale*x
        return scale*action(full)[keep]
    initial=np.sqrt(diagonal[keep])*solver.vectors[site][keep]
    expected=eigsh(LinearOperator((len(keep),len(keep)),matvec=apply,dtype=complex),
        k=1,which='SA',v0=initial,tol=1e-12)[0][0]
    record=solver._optimize_site(site,lh[site],rh[site+1],ln[site],rn[site+1],metric_rtol=1e-10,local_tol=1e-10)
    assert record['residual_norm']<1e-10 and record['local_solver']=='davidson'
    np.testing.assert_allclose(record['energy'],expected,atol=1e-9,rtol=0)


@pytest.mark.parametrize('group,orbital',[('Coov','E1'),('Dooh','E1u'),('SO3','L1')])
def test_nc_excited_two_electron_multiplets_match_all_mpo_partners(group,orbital):
    from scipy.linalg import block_diag
    spatial=OrbitalRotationSymmetry(group);d=spatial.dimension(orbital)
    reps={name:block_diag(matrix,spatial.generators(spatial.identity)[name],
                         spatial.generators(spatial.identity)[name])
          for name,matrix in spatial.generators(orbital).items()}
    h=block_diag(.2*np.eye(d),[[.1,.17],[.17,.7]])
    v=.12*np.einsum('pq,rs->pqrs',np.eye(d+2),np.eye(d+2))
    settings=dict(nelec=2,spin=0,D=4,point_group=group,orbital_representations=reps,wfnsym=orbital)
    ref=MultipletDMRG.from_integrals(h,v,contractions='reduced',**settings).run(nstates=2,nsweeps=8)
    nc=MultipletDMRG.from_integrals(h,v,**settings).run(nstates=2,nsweeps=8,algorithm='one_site')
    assert ref.success and nc.success and len(nc.states)==2*d
    np.testing.assert_allclose(nc.e_tot,ref.e_tot,atol=1e-10,rtol=0)
    report=nc.symmetry_diagnostics()
    assert report['orthogonality_error']<1e-10 and report['covariance_error']<1e-10
    np.testing.assert_allclose([state.expectation(ref.mpo) for state in nc.states],ref.e_tot,atol=1e-10,rtol=0)
