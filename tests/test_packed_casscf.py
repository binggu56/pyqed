"""Lossless real pair storage, independent of molecular point-group symmetry."""
import numpy as np
import pytest

from pyqed.qchem.basis import PackedRIFactors, mo_pair_factors
from pyqed.qchem.mcscf import casscf, orbopt


@pytest.mark.parametrize('auxiliary',[0,1,17])
@pytest.mark.parametrize('output',[3,7])
def test_packed_transform_and_selection(auxiliary,output,monkeypatch):
    rng=np.random.default_rng(176)
    b=rng.normal(size=(auxiliary,7,7)); b+=b.transpose(0,2,1).copy()
    c,_=np.linalg.qr(rng.normal(size=(7,7)))
    c=c[:,:output]
    packed=mo_pair_factors(b,np.eye(7),packed=True)
    monkeypatch.setattr(PackedRIFactors,'__array__',lambda *a,**kw:
                        (_ for _ in ()).throw(AssertionError('Full packed tensor expansion')))
    actual=mo_pair_factors(packed,c,packed=True)
    expected=c.T@(b@c)
    np.testing.assert_allclose(actual.block(0,auxiliary),expected,atol=2e-12)
    np.testing.assert_allclose(actual.select([output-1,0],[1,0]),
                               expected[:,[output-1,0]][:,:,[1,0]],atol=2e-12)
    assert actual.pair_factors.nbytes == auxiliary*output*(output+1)//2*8


def test_packing_rejects_asymmetry_and_complex_coefficients():
    b=np.arange(18.).reshape(2,3,3)
    with pytest.raises(ValueError,match='symmetric'):
        mo_pair_factors(b,np.eye(3),packed=True)
    with pytest.raises(ValueError,match='real'):
        mo_pair_factors(b,np.eye(3,dtype=complex),packed=True)


@pytest.mark.parametrize('complex_density',[False,True])
@pytest.mark.parametrize('n',[8,12])
def test_packed_hessian_matches_full_storage(monkeypatch,complex_density,n):
    from pyqed.qchem import _casscf_cpp
    rng=np.random.default_rng(211)
    o=5
    b=rng.normal(size=(19,n,n)); b+=b.transpose(0,2,1).copy()
    h=rng.normal(size=(n,n)); h+=h.T.copy()
    d1=np.eye(o); d1[:2,:2]*=2
    density=orbopt.CoreActiveRDM(2,np.eye(3),rng.normal(size=(3,)*4))
    if complex_density:
        density.dm2=density.dm2+1j*rng.normal(size=density.dm2.shape)
        density.dtype=density.dm2.dtype
    k=rng.normal(size=(3,n,n)); k-=k.transpose(0,2,1).copy()
    k[0,o:,o:]=0  # CASSCF rotations omit the virtual/virtual block.
    solver=object.__new__(casscf.SecondOrderCASSCF)
    expected=solver._orbital_hessian_action_from_factors_batch(h,b,d1,density,k,{})
    packed=mo_pair_factors(b,np.eye(n),packed=True)
    monkeypatch.setattr(PackedRIFactors,'__array__',lambda *a,**kw:
                        (_ for _ in ()).throw(AssertionError('Full packed tensor expansion')))
    cache={}
    actual=solver._orbital_hessian_action_from_factors_batch(h,packed,d1,density,k,cache)
    if complex_density:
        assert cache['native_workspace'] is None
    else:
        assert cache['native_workspace'] is not None
        info=_casscf_cpp.factor_hessian_workspace_info(cache['native_workspace'])
        assert info['packed_pairs'] and info['factor_storage_bytes']==packed.pair_factors.nbytes
    np.testing.assert_allclose(actual,expected,atol=2e-11)
    with monkeypatch.context() as context:
        context.setattr(orbopt,'_cpp_attr',lambda *a:None)
        fallback=solver._orbital_hessian_action_from_factors_batch(h,packed,d1,density,k,{})
    np.testing.assert_allclose(fallback,expected,atol=2e-11)
    f=orbopt.generalized_fock_from_factors(h,b[:,:,:o],d1,density)
    np.testing.assert_allclose(orbopt.generalized_fock_from_factors(h,packed,d1,density),f,atol=2e-11)


def test_packed_block_offsets_and_scratch_bound():
    from pyqed.qchem import _casscf_cpp
    rng=np.random.default_rng(631)
    p,n,o=600,80,60
    b=rng.normal(size=(p,n,n)); b+=b.transpose(0,2,1).copy()
    density=orbopt.CoreActiveRDM(56,np.eye(4),np.ones((4,)*4))
    one=2*np.eye(o); one[56:,56:]=np.eye(4)
    k=rng.normal(size=(n,n))*.01; k-=k.T.copy()
    k[o:,o:]=0
    solver=object.__new__(casscf.SecondOrderCASSCF)
    expected=solver._orbital_hessian_action_from_factors(np.eye(n),b,one,density,k,{})
    packed=mo_pair_factors(b,np.eye(n),packed=True)
    cache={}
    actual=solver._orbital_hessian_action_from_factors(np.eye(n),packed,one,density,k,cache)
    info=_casscf_cpp.factor_hessian_workspace_info(cache['native_workspace'])
    assert info['packed_pairs'] and 0 < info['auxiliary_block_size'] < p
    assert p%info['auxiliary_block_size'] != 0
    assert info['scratch_bytes'] <= (64 << 20)+8*(n*n+n*o)
    assert info['factor_storage_bytes'] == p*n*(n+1)//2*8
    np.testing.assert_allclose(actual,expected,atol=5e-9,rtol=2e-12)


def test_asymmetric_molecule_workflow_never_expands_packed_storage(monkeypatch):
    from pyqed.qchem import Molecule
    atom='; '.join(f'H {i*.9} {(.07 if i%2 else .01*i)} {.025*(i%3)}' for i in range(6))
    mf=Molecule(atom=atom,basis='sto-3g',unit='angstrom').build(
        eri='ri',auxbasis='def2-universal-jkfit').RHF().run()
    np.random.seed(531)
    reference=casscf.SecondOrderCASSCF(mf,4,4,multiplicity=1,ci_method='ci',
        conv_tol_grad=1e-4,conv_tol_grad_relaxed=1e-4)
    transform=casscf.transform_eri_factors_to_mo_pair
    with monkeypatch.context() as context:
        context.setattr(casscf,'transform_eri_factors_to_mo_pair',
            lambda *args,**kwargs: transform(*args,**{**kwargs,'packed':False}))
        reference.state_average(np.ones(6)/6).run(nstates=6)
    monkeypatch.setattr(PackedRIFactors,'__array__',lambda *a,**kw:
                        (_ for _ in ()).throw(AssertionError('Full packed tensor expansion')))
    np.random.seed(531)
    solver=casscf.SecondOrderCASSCF(mf,4,4,multiplicity=1,ci_method='ci',
        conv_tol_grad=1e-4,conv_tol_grad_relaxed=1e-4)
    make_ci=solver._make_factor_integral_casci
    calls=[]
    def checked(h,pair,*args,**kwargs):
        assert isinstance(pair,PackedRIFactors)
        result=make_ci(h,pair,*args,**kwargs)
        assert isinstance(result.mf.eri_factors,PackedRIFactors)
        assert np.shares_memory(result.mf.eri_factors.pair_factors,pair.pair_factors)
        assert not result.mf.eri_factors.pair_factors.flags.writeable
        calls.append(1)
        return result
    monkeypatch.setattr(solver,'_make_factor_integral_casci',checked)
    solver.state_average(np.ones(6)/6).run(nstates=6)
    assert solver.converged and calls
    np.testing.assert_allclose(solver.e_tot,reference.e_tot,atol=1e-10,rtol=0)
    np.testing.assert_allclose(solver.spin_square(),0,atol=1e-9)
