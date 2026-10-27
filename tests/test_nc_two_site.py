"""Direct NC pair actions and full-multiplet two-site sweeps."""
import numpy as np
import pytest
from test_nc_cbe import model
from pyqed.mps.nonabelian import _su2_kernel as kernels
from pyqed.mps.nonabelian.multiplet_dmrg import MultipletDMRG
from pyqed.symmetry import OrbitalRotationSymmetry


@pytest.mark.parametrize('group',['Coov','Dooh','SO3'])
@pytest.mark.parametrize('nelec',[2,4])
def test_pair_action_complex_and_diagonal_against_mpo(group,nelec):
    from pyqed.mps.nonabelian.cbe import multiplet_pair_action
    h,v,settings=model(group,2);settings['nelec']=nelec
    nc=MultipletDMRG.from_integrals(h,v,**settings)
    ref=MultipletDMRG.from_integrals(h,v,contractions='reduced',**settings)
    left,nl=nc._edge();right,nr=nc._edge(True)
    action,matrices,layouts,weights=kernels.spatial_nc_pair_problem(nc,0,left,right,nl,nr)
    _,_,lm,rm=kernels.multiplet_pair(nc,0)
    lr,_=ref._edge();rr,_=ref._edge(True)
    rng=np.random.default_rng(9)
    x=rng.normal(size=sum(m.size for m in matrices.values()))+1j*rng.normal(size=sum(m.size for m in matrices.values()))
    def reference(x):
        offset=0;blocks={}
        for q,m in matrices.items():
            blocks[q]=x[offset:offset+m.size].reshape(m.shape)/weights[q];offset+=m.size
        out=multiplet_pair_action(ref,0,blocks,layouts,lm,rm,lr,rr)
        return np.concatenate([(out[q]/weights[q]).ravel() for q in matrices])
    np.testing.assert_allclose(action(x),reference(x),atol=1e-11,rtol=0)
    matrix=np.column_stack([reference(e) for e in np.eye(len(x),dtype=complex)])
    np.testing.assert_allclose(matrix,matrix.conj().T,atol=1e-11,rtol=0)
    np.testing.assert_allclose(action.diagonal(),matrix.diagonal(),atol=1e-11,rtol=0)
    energy,vector,info=action.solve(x,1e-11)
    np.testing.assert_allclose(energy,np.linalg.eigvalsh(matrix)[0],atol=1e-10)
    assert np.linalg.norm(reference(vector)-energy*vector)<1e-10


@pytest.mark.parametrize('group',['Coov','Dooh','SO3'])
@pytest.mark.parametrize('nelec',[2,4])
def test_two_site_growth_and_all_partners(group,nelec):
    from examples.qchem.benchmark_nc_stability import guards
    h,v,settings=model(group,4);settings['nelec']=nelec
    limit=4 if nelec==2 else 12
    ref=MultipletDMRG.from_integrals(h,v,contractions='reduced',**dict(settings,D=limit)).run(nsweeps=15)
    with guards():
        nc=MultipletDMRG.from_integrals(h,v,**settings);nc.D=limit
        nc.vectors[0]=nc.vectors[0]*(1+0.2j)
        nc.run(algorithm='two_site',nsweeps=15,min_sweeps=3)
    assert nc.success and ref.success
    assert nc.diagnostics['two_site_driver']=='compiled_nc'
    assert 1<max(max(leg.dims.values()) for leg in nc.bond_legs)<=limit
    np.testing.assert_allclose(nc.e_tot,ref.e_tot,atol=1e-9,rtol=0)
    assert len(nc.states)==OrbitalRotationSymmetry(group).dimension(settings['wfnsym'])
    report=nc.symmetry_diagnostics()
    assert report['covariance_error']<1e-10 and report['orthogonality_error']<1e-10
    np.testing.assert_allclose([s.expectation(ref.mpo) for s in nc.states],nc.e_tot,atol=1e-9,rtol=0)


@pytest.mark.parametrize('options',[{},dict(algorithm='two_site')])
def test_two_site_rejects_multiple_multiplets_before_sweeping(options):
    h,v,settings=model('Coov',3)
    nc=MultipletDMRG.from_integrals(h,v,**settings)
    with pytest.raises(ValueError,match='one complete'):
        nc.run(nstates=2,**options)
    assert not nc.history


@pytest.mark.parametrize('group,target',[('Coov','A2'),('Dooh','A2g'),('SO3','L1')])
def test_two_site_triplet_partners(group,target):
    h,v,settings=model(group,4);settings.update(spin=2,wfnsym=target)
    ref=MultipletDMRG.from_integrals(h,v,contractions='reduced',**dict(settings,D=4)).run(nsweeps=12)
    nc=MultipletDMRG.from_integrals(h,v,**settings);nc.D=4
    nc.run(algorithm='two_site',nsweeps=12)
    assert nc.success and ref.success
    np.testing.assert_allclose(nc.e_tot,ref.e_tot,atol=1e-10,rtol=0)
    diagnostics=nc.symmetry_diagnostics()
    assert diagnostics['covariance_error']<1e-10 and diagnostics['orthogonality_error']<1e-10
    np.testing.assert_allclose([s.expectation(ref.mpo) for s in nc.states],nc.e_tot,atol=1e-10,rtol=0)


def test_spatial_nc_rejects_total_bond_cap():
    h,v,settings=model('Coov',3)
    nc=MultipletDMRG.from_integrals(h,v,**settings)
    with pytest.raises(ValueError,match='multiplicities per sector'):
        nc.run(max_bond_mode='reduced')
    assert not nc.history


def test_pair_davidson_refines_a_nearly_converged_guess():
    h=np.array([[1.,.2,.3],[.2,2.,.4],[.3,.4,3.]])
    values,vectors=np.linalg.eigh(h)
    guess=vectors[:,0]+1e-11*vectors[:,1]
    action=kernels.NCPairHamiltonian({0:np.zeros((3,1))},{(0,0,0):(h,np.ones((1,1)))})
    energy,vector,info=action.solve(guess,1e-13)
    assert info['converged']
    assert np.linalg.norm(h@vector-energy*vector)<1e-13
    assert abs(energy-values[0])<1e-13
