"""Finite point-group direct NC: independent MPO actions and all partners."""
import numpy as np
import pytest
from pyqed.qchem.symmetry import irrep_representation_matrices
from pyqed.symmetry import PointGroupSymmetry
from pyqed.mps.nonabelian.multiplet_dmrg import MultipletDMRG
from pyqed.mps.nonabelian import _su2_kernel as kernels


from examples.qchem.validate_finite_nc import model


@pytest.mark.parametrize('group,orbital,target',[
    ('D3h',"E'","E'"),('D3h',"E''","E'"),
    ('D6h','E1u','E2g'),('D6h','E2g','E2g'),('D6h','E2u','E2g'),
    ('D2h','B1u','Ag'),('C1','A','A')])
def test_finite_two_site_all_partners(group,orbital,target):
    from examples.qchem.benchmark_nc_stability import guards
    h,v,options=model(group,orbital,target)
    ref=MultipletDMRG.from_integrals(h,v,contractions='reduced',**options).run(nsweeps=12)
    with guards():
        nc=MultipletDMRG.from_integrals(h,v,**dict(options,D=1))
        nc.D=options['D']
        nc.vectors[0]=nc.vectors[0]*(1+.2j)
        nc.run(nsweeps=12,min_sweeps=3)
    assert nc.success and ref.success
    assert nc.diagnostics['two_site_driver']=='compiled_nc'
    assert nc.diagnostics['hamiltonian_mpo'] is False
    assert len(nc.states)==PointGroupSymmetry(group).dimension(target)
    np.testing.assert_allclose(nc.e_tot,ref.e_tot,atol=1e-9,rtol=0)
    np.testing.assert_allclose([s.expectation(ref.mpo) for s in nc.states],nc.e_tot,atol=1e-9,rtol=0)
    report=nc.symmetry_diagnostics()
    assert report['covariance_error']<1e-10
    assert report['orthogonality_error']<1e-10


@pytest.mark.parametrize('group,orbital,target',[
    ('D3h',"E'","E'"),('D6h','E2u','E2g')])
def test_finite_pair_action_against_mpo(group,orbital,target):
    from pyqed.mps.nonabelian.cbe import multiplet_pair_action
    h,v,options=model(group,orbital,target,shells=2)
    nc=MultipletDMRG.from_integrals(h,v,**options)
    ref=MultipletDMRG.from_integrals(h,v,contractions='reduced',**options)
    left,nl=nc._edge();right,nr=nc._edge(True)
    action,matrices,layouts,weights=kernels.spatial_nc_pair_problem(nc,0,left,right,nl,nr)
    _,_,lm,rm=kernels.multiplet_pair(nc,0)
    lr,_=ref._edge();rr,_=ref._edge(True)
    def reference(x):
        offset=0;blocks={}
        for q,m in matrices.items():
            blocks[q]=x[offset:offset+m.size].reshape(m.shape)/weights[q];offset+=m.size
        out=multiplet_pair_action(ref,0,blocks,layouts,lm,rm,lr,rr)
        return np.concatenate([(out[q]/weights[q]).ravel() for q in matrices])
    size=sum(m.size for m in matrices.values())
    rng=np.random.default_rng(39)
    x=rng.normal(size=size)+1j*rng.normal(size=size)
    np.testing.assert_allclose(action(x),reference(x),atol=1e-11,rtol=0)
    matrix=np.column_stack([reference(e) for e in np.eye(size,dtype=complex)])
    np.testing.assert_allclose(action.diagonal(),matrix.diagonal(),atol=1e-11,rtol=0)


@pytest.mark.parametrize('group,orbital,target',[
    ('D3h',"E''","A2'"),('D6h','E2u','A2g')])
def test_finite_triplet_all_partners(group,orbital,target):
    h,v,options=model(group,orbital,target,spin=2)
    ref=MultipletDMRG.from_integrals(h,v,contractions='reduced',**options).run(nsweeps=12)
    nc=MultipletDMRG.from_integrals(h,v,**options).run(nsweeps=12)
    assert nc.success and ref.success
    np.testing.assert_allclose(nc.e_tot,ref.e_tot,atol=1e-10,rtol=0)
    np.testing.assert_allclose([s.expectation(ref.mpo) for s in nc.states],nc.e_tot,atol=1e-10,rtol=0)
    assert nc.symmetry_diagnostics()['covariance_error']<1e-10


@pytest.mark.parametrize('factory',['multiplet','spatial','reduced','nc'])
@pytest.mark.parametrize('group,orbital,target',[
    ('D3h',"E'","E'"),('D6h','E2u','E2g')])
def test_finite_default_factories_forbid_mpo(factory,group,orbital,target,monkeypatch):
    from examples.qchem.benchmark_nc_stability import guards
    from pyqed.mps.nonabelian.spatial_symmetry import SpatialSymmetryDMRG
    from pyqed.mps.nonabelian.nc_multiplet import NCMultipletDMRG
    from pyqed.qchem.dmrg import dmrg
    from pyqed.mps.nonabelian.fusion import TensorFusionMap
    def forbidden(*args,**kwargs):
        pytest.fail('A molecular dense projection entered the direct finite NC solver.')
    for name in ('_build_spatial_active_hamiltonian_matrix',
                 '_spatial_mps_to_spin_string_ci','_nonabelian_mps_to_dense_vector'):
        monkeypatch.setattr(dmrg,name,forbidden)
    monkeypatch.setattr(TensorFusionMap,'matrix',property(forbidden))
    h,v,options=model(group,orbital,target,shells=2)
    from pyqed.mps.nonabelian.multiplet_sweep import ReducedMultipletDMRG
    cls={'multiplet':MultipletDMRG,'spatial':SpatialSymmetryDMRG,
         'reduced':ReducedMultipletDMRG,'nc':NCMultipletDMRG}[factory]
    with guards():
        solver=cls.from_integrals(h,v,**options).run(nsweeps=6)
    assert isinstance(solver,NCMultipletDMRG) and solver.success
    assert solver.diagnostics['two_site_driver']=='compiled_nc'


from dmrg_molecule_fixtures import h3plus
from dmrg_molecule_fixtures import benzene


@pytest.mark.parametrize('molecule,target',[('h3plus',"E'"),('benzene','E1u')])
def test_molecular_default_all_partners_against_mpo(molecule,target,request):
    from examples.qchem.benchmark_nc_stability import guards
    from pyqed.qchem.dmrg.dmrg import DMRG
    from pyqed.mps.nonabelian.nc_multiplet import NCMultipletDMRG
    fixture=request.getfixturevalue(molecule)
    mf,coeff,ncas,nelec=(fixture,fixture.mo_coeff,3,2) if molecule=='h3plus' else (*fixture,6,6)
    ref=DMRG(mf,ncas,nelec,D=32,spin=0,symmetry='su2',verbose=0).run(
        mo_coeff=coeff,wfnsym=target,spatial_method='multiplet',nsweeps=12)
    with guards():
        driver=DMRG(mf,ncas,nelec,D=32,spin=0,symmetry='su2',verbose=0)
        nc=driver.run(mo_coeff=coeff,wfnsym=target,nsweeps=12)
    assert isinstance(nc,NCMultipletDMRG) and nc.success and ref.success
    assert driver.spatial_method=='complementary'
    assert nc.diagnostics['two_site_driver']=='compiled_nc'
    np.testing.assert_allclose(nc.e_tot,ref.e_tot,atol=1e-9,rtol=0)
    np.testing.assert_allclose([s.expectation(ref.mpo)+ref.ecore for s in nc.states],nc.e_tot,atol=1e-9,rtol=0)
    assert nc.symmetry_diagnostics()['covariance_error']<1e-10


def test_compiled_finite_table_rejects_missing_partners_and_forbidden_fusion():
    from pyqed.mps.nonabelian.nc_spatial import prepare_rank_schedule
    h,v,options=model('D3h',"E'","E'",shells=2)
    nc=MultipletDMRG.from_integrals(h,v,**options)
    class Capture:
        def spatial_normal_complementary_schedule(self):
            return nc.nc.owner.spatial_normal_complementary_schedule()
        def reduce_spatial_normal_complementary(self,*args):
            self.args=list(args)
    capture=Capture()
    prepare_rank_schedule(capture,nc.sites,[0,2,4])
    args=capture.args
    broken=list(args);broken[-2]=[1]*len(args[-2])
    with pytest.raises(ValueError,match='partner dimensions'):
        nc.nc.owner.reduce_spatial_normal_complementary(*broken)
    broken=list(args);broken[-1]=list(args[-1]);broken[-1][0]=2
    with pytest.raises(ValueError,match='multiplicity-free'):
        nc.nc.owner.reduce_spatial_normal_complementary(*broken)
    broken=list(args);broken[1]=list(args[1]);broken[1][0]=(0,0,100,1,1)
    with pytest.raises(ValueError,match='irrep index'):
        nc.nc.owner.reduce_spatial_normal_complementary(*broken)


@pytest.mark.parametrize('group,orbital',[
    ('D3h',"E'"),('D3h',"E''"),('D6h','E1g'),('D6h','E2g'),
    ('D6h','E1u'),('D6h','E2u')])
def test_every_finite_E_target_preserves_both_partners(group,orbital):
    h,v,options=model(group,orbital,orbital,shells=3,spin=1)
    options['nelec']=1
    ref=MultipletDMRG.from_integrals(h,v,contractions='reduced',**options).run(nsweeps=8)
    nc=MultipletDMRG.from_integrals(h,v,**options).run(nsweeps=8)
    assert nc.success and len(nc.states)==2
    np.testing.assert_allclose(nc.e_tot,ref.e_tot,atol=1e-10,rtol=0)
    np.testing.assert_allclose([s.expectation(ref.mpo) for s in nc.states],nc.e_tot,atol=1e-10,rtol=0)
    report=nc.symmetry_diagnostics()
    assert max(report['covariance_error'],report['orthogonality_error'])<1e-10


@pytest.mark.parametrize('group,orbital',[('D3h',"E'"),('D6h','E2g')])
def test_finite_only_integrals_match_mpo(group,orbital):
    """Group-averaged ERIs break continuous rotations but retain the finite group."""
    from scipy.linalg import block_diag
    from examples.qchem.benchmark_nc_stability import guards
    sym=PointGroupSymmetry(group)
    scalar=irrep_representation_matrices(group,sym.identity)
    e=irrep_representation_matrices(group,orbital)
    reps={g:block_diag(scalar[g],e[g],e[g]) for g in scalar}
    rng=np.random.default_rng(182)
    factors=rng.normal(size=(3,5,5));factors=(factors+factors.transpose(0,2,1))*.1
    v=np.zeros((5,)*4)
    for u in reps.values():
        rotated=np.einsum('pa,xpq,qb->xab',u,factors,u)
        v+=np.einsum('xpq,xrs->pqrs',rotated,rotated)/len(reps)
    theta=.137
    r=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    u=block_diag(np.ones((1,1)),r,r)
    assert np.linalg.norm(np.einsum('pa,qb,rc,sd,pqrs->abcd',u,u,u,u,v)-v)>1e-4
    h=block_diag([[-.4]],np.kron([[-.2,.17],[.17,.3]],np.eye(2)))
    options=dict(nelec=2,spin=0,D=8,point_group=group,wfnsym=orbital,orbital_representations=reps)
    ref=MultipletDMRG.from_integrals(h,v,contractions='reduced',**options).run(nsweeps=12)
    with guards():
        nc=MultipletDMRG.from_integrals(h,v,**options).run(nsweeps=12)
    assert nc.success and ref.success and len(nc.states)==2
    np.testing.assert_allclose(nc.e_tot,ref.e_tot,atol=1e-10,rtol=0)
    np.testing.assert_allclose([s.expectation(ref.mpo) for s in nc.states],nc.e_tot,atol=1e-10,rtol=0)
    assert nc.symmetry_diagnostics()['covariance_error']<1e-10
